"""Class-bias estimation on the Calibration partition.

Three things have to hold for M2/M3/M5/M6 to mean what the paper says they
mean: the objective is the metric the paper reports, the estimator only ever
sees Calibration rows, and the conditional variant really is conditional. Each
is asserted against something independent of the implementation -- the scorer
itself, the split manifest, and an invariance property respectively.
"""

import numpy as np
import pytest

from paper.artifacts import read_predictions
from paper.calibration import (
    GRID,
    _macro_f1_grid,
    as_json,
    fit_biases,
)
from paper.data import REPO_ROOT, index_by_id, load_dev
from paper.labels import (
    CONDITIONAL_PINNED_CLASSES,
    CONDITIONING_SUBSET,
    EVAL_FIELDS,
    FIELDS,
    LABEL2ID,
    is_valid_tuple,
)
from paper.methods import log_scores
from paper.run_decisions import load_bundle, load_run, run_method
from paper.score import macro_f1
from paper.validate import load_split

ROWS = load_dev()
SPLIT = load_split("pdf_group", 42)
FIXTURES = REPO_ROOT / "contracts" / "examples" / "probs"


def _rotation(k):
    return next(r for r in SPLIT["rotations"] if r["k"] == k)


@pytest.fixture(scope="module")
def bundle():
    """Rotation 1: its calibration partition is missing ``Misleading``."""
    meta, probs = load_bundle(FIXTURES / "pdf_group_seed42_r1")
    return meta, probs


def _fit(bundle, mode, probs=None):
    meta, loaded = bundle
    return fit_biases(mode, probs or loaded["calibration"], meta["calibration_ids"],
                      _rotation(meta["rotation"]), ROWS)


# --- the objective is the metric --------------------------------------------

def test_the_search_objective_agrees_with_the_scorer():
    """The vectorised macro-F1 exists only for speed. If it drifted from
    paper/score.py the biases would be tuned for a metric the paper does not
    report -- and every calibrated number would be quietly off."""
    rng = np.random.default_rng(0)
    for field in FIELDS:
        labels = EVAL_FIELDS[field]
        gold = rng.integers(0, len(labels), 400)
        preds = rng.integers(0, len(labels), (25, 400))

        fast = _macro_f1_grid(preds, gold)
        slow = [macro_f1([labels[i] for i in gold], [labels[i] for i in row], labels)
                for row in preds]
        assert fast == pytest.approx(slow), field


def test_the_objective_uses_the_present_labels_convention():
    """A class absent from the gold must not drag the average down, matching
    paper/score.py; scoring it as 0.0 would reward predicting it never."""
    gold = np.zeros(10, dtype=int)          # one class only
    perfect = np.zeros((1, 10), dtype=int)
    assert _macro_f1_grid(perfect, gold)[0] == pytest.approx(1.0)


# --- only Calibration rows ---------------------------------------------------

def test_the_test_partition_is_refused(bundle):
    """Plan §6.2: the API takes calibration labels and refuses test labels.
    Tuning on the rows a method is scored on is the one failure here that would
    make every calibrated number meaningless while looking excellent."""
    meta, probs = bundle
    with pytest.raises(ValueError, match="Calibration partition"):
        fit_biases("global", probs["test"], meta["test_ids"],
                   _rotation(meta["rotation"]), ROWS)


def test_an_unknown_mode_is_refused(bundle):
    with pytest.raises(ValueError, match="calibration mode"):
        _fit(bundle, "temperature")


def test_a_manifest_that_disagrees_with_the_data_is_caught(bundle):
    meta, probs = bundle
    rot = dict(_rotation(meta["rotation"]))
    rot["calibration_absent_classes"] = {}          # the data say otherwise
    with pytest.raises(ValueError, match="absent"):
        fit_biases("global", probs["calibration"], meta["calibration_ids"], rot, ROWS)


# --- what the estimator may and may not move ---------------------------------

def test_every_bias_lands_on_the_grid(bundle):
    for mode in ("global", "conditional"):
        biases, _ = _fit(bundle, mode)
        for field, vec in biases.items():
            assert np.isin(vec, GRID).all(), (mode, field)


def test_the_fit_never_lowers_the_objective_it_optimises(bundle):
    """Coordinate ascent from all zeros accepts strict improvements only, so
    the calibrated rule is at least as good as the uncalibrated one *on the
    calibration partition*. Nothing here claims that carries to Test."""
    meta, probs = bundle
    by_id = index_by_id(ROWS)
    scores = log_scores(probs["calibration"])
    biases, _ = _fit(bundle, "global")

    for field in FIELDS:
        labels = EVAL_FIELDS[field]
        gold = [by_id[i][field] for i in meta["calibration_ids"]]
        before = macro_f1(gold, [labels[i] for i in np.argmax(scores[field], axis=1)], labels)
        after = macro_f1(gold, [labels[i] for i in
                                np.argmax(scores[field] + biases[field], axis=1)], labels)
        assert after >= before, field


def test_an_absent_class_is_pinned_and_recorded(bundle):
    """Rotation 1's calibration partition contains no ``Misleading``. Contract
    §2: its bias is fixed at 0.0 and the fallback is recorded per rotation."""
    biases, fallback = _fit(bundle, "global")
    assert fallback == {"evidence_quality": ["Misleading"]}
    assert biases["evidence_quality"][LABEL2ID["evidence_quality"]["Misleading"]] == 0.0


def test_a_class_absent_from_train_is_pinned_but_not_a_calibration_fallback(bundle):
    """Korean has no Misleading rows anywhere; that is not fold-specific."""
    meta, probs = bundle
    rows = [dict(row) for row in ROWS]
    for row in rows:
        if row["evidence_quality"] == "Misleading":
            row["evidence_quality"] = "Not Clear"
    rot = dict(_rotation(meta["rotation"]))
    rot["calibration_absent_classes"] = {}

    biases, fallback = fit_biases(
        "global", probs["calibration"], meta["calibration_ids"], rot, rows,
    )

    assert fallback == {}
    assert biases["evidence_quality"][LABEL2ID["evidence_quality"]["Misleading"]] == 0.0


def test_the_structural_na_classes_are_pinned_but_are_not_fallback(bundle):
    """Plan §3.2: unidentifiable, not unsupported. Recording them as fallback
    would conflate a property of the hierarchy with a property of this split."""
    biases, fallback = _fit(bundle, "conditional")
    for field, pinned in CONDITIONAL_PINNED_CLASSES.items():
        for label in pinned:
            assert biases[field][LABEL2ID[field][label]] == 0.0
            assert label not in fallback.get(field, [])


def test_conditional_leaves_promise_status_exactly_where_global_does(bundle):
    """PS has no parent, so both estimators fit it on the whole partition with
    the same free classes. Any difference would mean one of them is not doing
    what §3.2 defines."""
    g, _ = _fit(bundle, "global")
    c, _ = _fit(bundle, "conditional")
    assert np.array_equal(g["promise_status"], c["promise_status"])


def test_the_estimator_is_deterministic(bundle):
    a, _ = _fit(bundle, "conditional")
    b, _ = _fit(bundle, "conditional")
    assert all(np.array_equal(a[f], b[f]) for f in FIELDS)


# --- the conditional variant really is conditional ---------------------------

def test_conditional_ignores_rows_outside_the_conditioning_subset(bundle):
    """The defining property, asserted as an invariance: scrambling the child
    fields' probabilities on rows their parent excludes must not move a
    conditional bias by a single grid step, while a global bias is free to
    follow them."""
    meta, probs = bundle
    by_id = index_by_id(ROWS)
    rng = np.random.default_rng(7)

    scrambled = {f: probs["calibration"][f].copy() for f in FIELDS}
    for field, (parent, value) in CONDITIONING_SUBSET.items():
        outside = np.array([by_id[i][parent] != value for i in meta["calibration_ids"]])
        noise = rng.random((outside.sum(), len(EVAL_FIELDS[field]))).astype(np.float32)
        scrambled[field][outside] = noise / noise.sum(axis=1, keepdims=True)

    before, _ = _fit(bundle, "conditional")
    after, _ = _fit(bundle, "conditional", probs=scrambled)
    for field in CONDITIONING_SUBSET:
        assert np.array_equal(before[field], after[field]), field

    g_before, _ = _fit(bundle, "global")
    g_after, _ = _fit(bundle, "global", probs=scrambled)
    assert any(not np.array_equal(g_before[f], g_after[f]) for f in CONDITIONING_SUBSET), \
        "the scramble must be large enough for the contrast to mean something"


# --- through the runner ------------------------------------------------------

@pytest.fixture(scope="module")
def run():
    return load_run(FIXTURES, "pdf_group", 42, SPLIT)


@pytest.mark.parametrize("projection_arm, decoder_arm", [("M2", "M5"), ("M3", "M6")])
def test_paired_methods_are_handed_one_bias_set(run, projection_arm, decoder_arm):
    """The two arms of one calibration mode share one estimate. The factorial
    reading of the table depends on it: were the decoder arm re-fitted, M6 vs
    M3 would mix the decoder's effect with a bias tuned for the decoder."""
    cache = {}
    first = run_method(projection_arm, run, ROWS, SPLIT, cache)[1]
    fitted = dict(cache)
    second = run_method(decoder_arm, run, ROWS, SPLIT, cache)[1]

    assert list(cache) == list(fitted), "the second arm re-fitted an existing bias"
    assert all(cache[k] is fitted[k] for k in fitted), "the shared object was replaced"
    for rotation in first:
        assert first[rotation]["calibration_biases"] == second[rotation]["calibration_biases"]
        assert first[rotation]["fallback_applied"] == second[rotation]["fallback_applied"]


def test_every_calibrated_method_emits_legal_tuples(run):
    cache = {}
    for method_id in ("M2", "M3", "M5", "M6"):
        records, params = run_method(method_id, run, ROWS, SPLIT, cache)
        assert len(records) == 2000
        assert all(is_valid_tuple(*(r[f"pred_{a}"] for a in ("ps", "vt", "es", "eq")))
                   for r in records), method_id
        assert set(params) == {"0", "1", "2", "3", "4"}
        assert params["1"]["fallback_applied"] == {"evidence_quality": ["Misleading"]}


@pytest.mark.parametrize("seed", [123, 456])
def test_committed_chinese_conditional_outputs_match_current_calibrator(seed):
    """Frozen M3/M6 predictions must be regenerated when calibration changes."""
    split = load_split("pdf_group", seed)
    run = load_run(REPO_ROOT / "probs", "pdf_group", seed, split)
    cache = {}

    for method_id in ("M3", "M6"):
        regenerated, _ = run_method(method_id, run, ROWS, split, cache)
        committed = read_predictions(
            REPO_ROOT / "predictions" / f"pdf_group_seed{seed}_{method_id}.csv.gz"
        )
        assert regenerated == committed, f"{method_id} seed {seed} is stale"


def test_the_recorded_biases_are_json_shaped(bundle):
    biases, _ = _fit(bundle, "global")
    recorded = as_json(biases)
    assert set(recorded) == set(FIELDS)
    assert recorded["promise_status"].keys() == {"Yes", "No"}
    assert all(isinstance(v, float) for v in recorded["promise_status"].values())
