"""The training path, exercised end to end on a tiny model.

This is the only part of the study that cannot run in the CPU environment, and
therefore the only part nobody had executed before B's first fit. The optimiser
group construction in particular is easy to get subtly wrong -- a parameter in
two groups is updated twice per step, a parameter in none is frozen, and either
one produces a model that trains to a plausible-looking score.

So the whole path runs here against a three-layer BERT built from a config, with
no download and no GPU: the architecture is irrelevant to what is being checked,
and everything that *is* being checked is architecture-independent.

Skipped unless torch and transformers are installed, which is what the frozen
conda environment provides. Run it on the GPU box before starting P0 -- it costs
seconds and the alternative is finding out fifteen fits later.
"""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import BertConfig, BertModel, BertTokenizer  # noqa: E402

from paper.artifacts import _check_distribution  # noqa: E402
from paper.data import load_dev  # noqa: E402
from paper.labels import EVAL_FIELDS, NUM_LABELS  # noqa: E402
from paper.train_config import (  # noqa: E402
    BACKBONE_LR,
    HEAD_LR,
    LLRD_DECAY,
    NO_DECAY,
    WEIGHT_DECAY,
)

N_LAYERS = 3


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A complete local checkpoint, so the test never touches the network."""
    d = tmp_path_factory.mktemp("tiny-bert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += [chr(0x4E00 + i) for i in range(300)]
    (d / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    config = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=N_LAYERS,
        num_attention_heads=2, intermediate_size=64,
    )
    BertModel(config).save_pretrained(d)
    BertTokenizer(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def model(tiny_model_dir):
    from paper.model import MultiTaskEncoder

    return MultiTaskEncoder(str(tiny_model_dir), NUM_LABELS)


# --- the optimiser groups -------------------------------------------------

def test_every_parameter_is_optimised_exactly_once(model):
    """Twice and it takes a double step; not at all and it never learns.
    Neither shows up as anything but a slightly worse number."""
    groups = model.get_optimizer_groups(BACKBONE_LR, HEAD_LR)

    seen, duplicated = set(), []
    for group in groups:
        for p in group["params"]:
            if id(p) in seen:
                duplicated.append(id(p))
            seen.add(id(p))

    assert not duplicated, f"{len(duplicated)} parameters are in two groups"
    assert seen == {id(p) for p in model.parameters()}, "some parameters are unoptimised"


def test_weight_decay_follows_the_frozen_recipe(model):
    """Biases and LayerNorm gains are exempt; nothing falls through to torch's
    default, which train_config deliberately does not own."""
    name_of = {id(p): n for n, p in model.named_parameters()}
    groups = model.get_optimizer_groups(BACKBONE_LR, HEAD_LR)

    assert all("weight_decay" in g for g in groups)
    for group in groups:
        for p in group["params"]:
            name = name_of[id(p)]
            expected = 0.0 if any(k in name for k in NO_DECAY) else WEIGHT_DECAY
            assert group["weight_decay"] == expected, name


def test_layerwise_decay_produces_a_ladder(model):
    groups = model.get_optimizer_groups(BACKBONE_LR, HEAD_LR)
    lrs = {round(g["lr"], 12) for g in groups}

    assert round(HEAD_LR, 12) in lrs, "the heads must keep their own learning rate"
    assert round(BACKBONE_LR, 12) in lrs, "the top encoder layer trains at the base rate"
    assert round(BACKBONE_LR * LLRD_DECAY ** N_LAYERS, 12) in lrs, "embeddings decay furthest"
    assert min(lrs) < BACKBONE_LR


# --- the loop and its output ----------------------------------------------

@pytest.fixture(scope="module")
def probs(tiny_model_dir):
    """One epoch of real training, then inference, exactly as the driver does."""
    import paper.train_fold as tf
    from torch.utils.data import DataLoader

    def serial(*args, **kwargs):        # workers add nothing at this size and
        kwargs["num_workers"] = 0       # multiprocessing start methods differ
        return DataLoader(*args, **kwargs)

    rows = load_dev()
    tokenizer = BertTokenizer.from_pretrained(str(tiny_model_dir))
    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp.setattr(tf, "DataLoader", serial)
        mp.setattr(tf, "EPOCHS", 1)
        model, _ = tf.train_rotation(
            rows[:24], tokenizer, seed=42, model_name=str(tiny_model_dir), revision=None,
        )
        return tf.predict_probs(model, rows[24:36], tokenizer)


def test_training_produces_contract_shaped_probabilities(probs):
    assert set(probs) == set(EVAL_FIELDS)
    for field, arr in probs.items():
        assert arr.shape == (12, len(EVAL_FIELDS[field])), field
        assert arr.dtype == np.float32, field


def test_the_probabilities_survive_the_bundle_writer(probs):
    """float32 softmax has to clear the write-time distribution check, or every
    real run fails at the last step of the rotation it just spent an hour on."""
    for field, arr in probs.items():
        _check_distribution(arr, field)


def test_inference_preserves_row_order(tiny_model_dir, probs):
    """Row order is the contract: the arrays are aligned to the manifest's id
    list by position, so a reordering here is a silent relabelling."""
    import paper.train_fold as tf
    from torch.utils.data import DataLoader

    def serial(*args, **kwargs):
        kwargs["num_workers"] = 0
        return DataLoader(*args, **kwargs)

    rows = load_dev()[24:36]
    tokenizer = BertTokenizer.from_pretrained(str(tiny_model_dir))
    from paper.model import MultiTaskEncoder

    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp.setattr(tf, "DataLoader", serial)
        model = MultiTaskEncoder(str(tiny_model_dir), NUM_LABELS).to(tf.DEVICE)
        model.eval()
        whole = tf.predict_probs(model, rows, tokenizer)
        reversed_ = tf.predict_probs(model, rows[::-1], tokenizer)

    for field in EVAL_FIELDS:
        assert np.allclose(whole[field], reversed_[field][::-1], atol=1e-5), field
