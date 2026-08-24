"""Selecting a corpus, and the guard that makes selecting the wrong one safe.

``--corpus`` is the only switch that decides which four hundred or two thousand
rows a 30-fit campaign trains on. Getting it wrong has no visible symptom at
launch: the model loads, the folds parse, and the fits run to completion
against the wrong data. The checksum comparison in ``run_training`` is what
turns that into an immediate failure, so it is tested here rather than trusted.
"""

import json

import pytest

from paper import corpus
from paper.data import data_checksum as zh_checksum
from paper.data import load_dev
from paper.data_en import load_english


def test_the_two_corpora_are_the_ones_the_study_has():
    assert set(corpus.CORPORA) == {"aicup_zh", "mlpromise_en"}
    assert corpus.DEFAULT == "aicup_zh"


def test_the_default_is_the_frozen_study():
    """Every existing invocation keeps its meaning. A default that had to be
    passed would make the frozen path the special case."""
    assert corpus.load_rows() == load_dev()
    assert corpus.splits_dir() == "splits"
    assert corpus.probs_dir() == "probs"


def test_english_artifacts_land_somewhere_else():
    """Contract section 4 names ``splits/`` and ``probs/``. Writing English
    folds or bundles into them would be silent and unrecoverable from the tree."""
    assert corpus.splits_dir("mlpromise_en") == "splits_en"
    assert corpus.splits_dir("mlpromise_en") != corpus.splits_dir("aicup_zh")
    assert corpus.decisions_root("mlpromise_en") == "runs_en"


def test_english_has_arms_rather_than_one_probs_directory():
    """Seven (backbone, lambda) arms over one set of rows.

    Every arm writes bundles named ``{protocol}_seed{seed}_r{k}`` -- the same
    names the other six use -- so they cannot share a directory. The frozen
    corpus has one, because contract section 4 names it.
    """
    assert corpus.probs_dir("aicup_zh") == "probs"
    assert corpus.probs_dir("mlpromise_en") is None


def test_an_unknown_corpus_is_refused():
    with pytest.raises(ValueError, match="unknown corpus"):
        corpus.load_rows("mlpromise_fr")


def test_both_corpora_speak_the_frozen_vocabulary():
    """The point of translating English at load time: nothing downstream of
    ``load_rows`` can tell which corpus it was handed."""
    from paper.labels import EVAL_FIELDS, FIELDS, is_valid_tuple

    for name in corpus.CORPORA:
        rows = corpus.load_rows(name)
        for row in rows:
            for field in FIELDS:
                assert row[field] in EVAL_FIELDS[field], (name, field, row["id"])
            assert is_valid_tuple(*(row[field] for field in FIELDS)), (name, row["id"])


def test_the_two_corpora_share_no_row_id():
    """A wrong-corpus join must return nothing rather than something."""
    assert not {r["id"] for r in load_dev()} & {r["id"] for r in load_english()}


def test_a_split_manifest_identifies_the_corpus_it_was_built_from():
    """The guard ``run_training`` relies on.

    It does not read a corpus name -- it compares checksums -- so this asserts
    the property that makes the comparison decisive: the two corpora cannot
    produce the same ``data_checksum``.
    """
    from paper.data_en import data_checksum as en_checksum

    assert zh_checksum() != en_checksum()


@pytest.mark.parametrize("name", sorted(corpus.CORPORA))
def test_the_shipped_manifests_match_their_corpus(name):
    """Each committed split manifest states the checksum of the rows it folds.

    A manifest that matched neither corpus would be unusable; one that matched
    the other would be worse.
    """
    from pathlib import Path

    from paper.data import REPO_ROOT

    directory = Path(REPO_ROOT) / corpus.CORPORA[name]["splits_dir"]
    manifests = sorted(directory.glob("*.json"))
    assert manifests, directory

    expected = data_checksum_of(name)
    for path in manifests:
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        assert split["data_checksum"] == expected, path.name


def data_checksum_of(name: str) -> str:
    if name == "aicup_zh":
        return zh_checksum()
    from paper.data_en import data_checksum as en_checksum

    return en_checksum()


@pytest.mark.parametrize("name", sorted(corpus.CORPORA))
def test_every_row_is_tested_exactly_once_per_seed(name):
    """The rotating design's defining property, asserted for both corpora.

    Nine reports fold less evenly than forty-nine, so this is worth checking on
    the English manifests rather than assuming the generator carried over.
    """
    from pathlib import Path

    from paper.data import REPO_ROOT

    rows = corpus.load_rows(name)
    directory = Path(REPO_ROOT) / corpus.CORPORA[name]["splits_dir"]
    for path in sorted(directory.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        tested = [i for rot in split["rotations"] for i in rot["test_ids"]]
        assert len(tested) == len(rows), path.name
        assert set(tested) == {r["id"] for r in rows}, path.name


def test_the_english_document_disjoint_folds_are_uneven_and_that_is_the_data():
    """Nine reports into five folds cannot be balanced, and the paper has to
    say so rather than present the fold sizes as a design choice."""
    from pathlib import Path

    from paper.data import REPO_ROOT

    path = Path(REPO_ROOT) / "splits_en" / "pdf_group_seed42.json"
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    sizes = sorted(len(rot["test_ids"]) for rot in split["rotations"])
    assert sum(sizes) == 400
    assert max(sizes) - min(sizes) > 10, sizes


# --------------------------------------------------------------------------
# Keeping the two corpora's artifacts apart on disk
# --------------------------------------------------------------------------

def test_the_english_decisions_root_is_not_the_repository_root():
    """The frozen predictions would be overwritten, not merely joined.

    A decision run writes ``{protocol}_seed{seed}_{method}.csv.gz`` whatever
    corpus it was fit on, so the two corpora produce byte-different files under
    forty-two identical names. Defaulting the English output to the root would
    replace the frozen study's predictions file by file.
    """
    assert corpus.decisions_root("aicup_zh") == "."
    assert corpus.decisions_root("mlpromise_en") != "."
    assert corpus.decisions_root("mlpromise_en") == "runs_en"


def _dirs(name):
    return {d for d in (corpus.splits_dir(name), corpus.probs_dir(name),
                        corpus.decisions_root(name)) if d}


@pytest.mark.parametrize("name", sorted(corpus.CORPORA))
def test_every_output_directory_of_a_corpus_is_its_own(name):
    """No corpus writes anywhere another one does."""
    mine = _dirs(name)
    others = {d for other in corpus.CORPORA if other != name for d in _dirs(other)}
    assert not mine & others, sorted(mine & others)


PLANNED_ENGLISH_ARMS = (
    ("roberta-large", 0.0), ("roberta-large", 0.3),
    ("microsoft/deberta-v3-large", 0.0), ("microsoft/deberta-v3-large", 0.3),
    ("google/electra-large-discriminator", 0.0),
    ("google/electra-large-discriminator", 0.3),
    ("roberta-base", 0.0),
)


def test_every_planned_arm_gets_its_own_directory():
    """Seven arms, seven paths, no collisions.

    A collision has no error and no symptom: the second arm's bundles simply
    replace the first's under identical names, and afterwards nothing in the
    tree says which fits produced them.
    """
    paths = [corpus.arm_dir("mlpromise_en", m, lam)
             for m, lam in PLANNED_ENGLISH_ARMS]
    assert len(set(paths)) == len(PLANNED_ENGLISH_ARMS), sorted(paths)
    assert all(p.startswith("runs_en/") for p in paths), paths


def test_the_arm_directory_spells_lambda_the_way_the_chinese_screens_do():
    """``lambda_0.0``, not ``lambda_0``: one spelling per arm, and it matches
    runs/deberta_v2_320m/lambda_0.0 so the two languages read alike."""
    assert corpus.arm_dir("mlpromise_en", "roberta-large", 0) \
        == "runs_en/roberta_large/lambda_0.0"
    assert corpus.arm_dir("mlpromise_en", "roberta-large", 0.3) \
        == "runs_en/roberta_large/lambda_0.3"


def test_the_frozen_corpus_has_no_arm_structure():
    """Its paths are named by contract section 4 and do not move."""
    assert corpus.arm_dir("aicup_zh", "hfl/chinese-roberta-wwm-ext-large", 0.0) == "."


def test_the_model_slug_drops_the_owner_and_is_path_safe():
    assert corpus.model_slug("microsoft/deberta-v3-large") == "deberta_v3_large"
    assert corpus.model_slug("roberta-large") == "roberta_large"
    assert "/" not in corpus.model_slug("google/electra-large-discriminator")


@pytest.mark.parametrize("name", sorted(corpus.CORPORA))
def test_a_new_probability_array_would_be_committed_not_silently_ignored(name):
    """``*.npy`` in .gitignore swallows a new probs tree unless negated.

    The failure has no symptom until someone clones: the bundle keeps its
    meta.json, so every validator that reads metadata still passes, and the
    probabilities are simply absent. Each corpus's probs directory needs its
    own negation, and this asserts the negation is there before the bundles
    are.
    """
    import subprocess

    from paper.data import REPO_ROOT

    probs = corpus.probs_dir(name)
    array = (f"{probs}/pdf_group_seed42_r0/test_promise_status.npy" if probs
             else f"{corpus.arm_dir(name, 'roberta-large', 0.0)}/probs/"
                  "pdf_group_seed42_r0/test_promise_status.npy")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", array], cwd=REPO_ROOT, check=False,
    ).returncode == 0
    assert not ignored, (
        f"{array} is gitignored, so a run of --corpus {name} would commit a "
        "bundle carrying meta.json and no probabilities. Add a negation for "
        "this tree to .gitignore."
    )
