"""The seven decision rules and the driver that runs them.

The properties asserted here are the ones the results table's validity rests
on: that the method set is the one the plan defines, that the uncalibrated
methods cannot accidentally acquire a bias, that moving to score space did not
change what M0 and M1 output, and that a run covers every row exactly once.
"""

import numpy as np
import pytest

from paper.data import REPO_ROOT, index_by_id, load_dev
from paper.labels import EVAL_FIELDS, FIELDS, is_valid_tuple
from paper.methods import METHOD_IDS, METHODS, PROB_FLOOR, decide, log_scores
from paper.projection import independent_argmax, project
from paper.run_decisions import _check_complete, load_run, run_method
from paper.validate import load_split

ROWS = load_dev()
SPLIT = load_split("pdf_group", 42)
FIXTURES = REPO_ROOT / "contracts" / "examples" / "probs"


def _random_probs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    probs = {}
    for field in FIELDS:
        raw = rng.random((n, len(EVAL_FIELDS[field])))
        probs[field] = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    return probs


# --- the method table -----------------------------------------------------

def test_the_method_set_is_the_one_the_plan_defines():
    assert METHOD_IDS == ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
    assert [(m.calibration, m.output_rule) for m in METHODS.values()] == [
        (None, "independent"),
        (None, "projection"),
        ("global", "projection"),
        ("conditional", "projection"),
        (None, "decoder"),
        ("global", "decoder"),
        ("conditional", "decoder"),
    ]
    assert not METHODS["M0"].guarantees_valid_state
    assert all(METHODS[m].guarantees_valid_state for m in METHOD_IDS[1:])


def test_an_uncalibrated_method_cannot_be_handed_biases():
    """M0/M1/M4 are the no-calibration arms; silently accepting a bias would
    make them a different method with the same name."""
    probs = _random_probs(4)
    biases = {f: np.zeros(len(EVAL_FIELDS[f])) for f in FIELDS}
    with pytest.raises(ValueError, match="calibration"):
        decide("M1", probs, biases)


def test_a_calibrated_method_cannot_run_without_biases():
    with pytest.raises(ValueError, match="calibration"):
        decide("M3", _random_probs(4), None)


# --- score space ----------------------------------------------------------

def test_scoring_in_log_space_leaves_m0_and_m1_unchanged():
    """The move to ``log p + b`` is what lets the decoder sum four fields. It
    must not have altered the two rules that predate it, or every M0/M1 number
    would silently depend on a refactor rather than on the rule."""
    probs = _random_probs(500)
    scores = log_scores(probs)
    assert decide("M0", probs) == independent_argmax(probs) == independent_argmax(scores)
    assert decide("M1", probs) == project(probs) == project(scores)


def test_an_underflowed_probability_still_scores_finite():
    """Real float32 softmax reaches exactly 0 on losing classes; -inf there
    would make any state containing that class unreachable for the decoder."""
    probs = _random_probs(3)
    probs["evidence_quality"][0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scores = log_scores(probs)
    assert np.isfinite(scores["evidence_quality"]).all()
    assert scores["evidence_quality"][0, 1] == pytest.approx(np.log(PROB_FLOOR))
    assert scores["evidence_quality"][0, 0] > scores["evidence_quality"][0, 1]


def test_a_bias_of_the_wrong_length_is_rejected():
    biases = {f: np.zeros(len(EVAL_FIELDS[f])) for f in FIELDS}
    biases["evidence_status"] = np.zeros(2)
    with pytest.raises(ValueError, match="bias has shape"):
        log_scores(_random_probs(4), biases)


def test_a_bias_moves_the_decision_it_is_supposed_to_move():
    """Guards the wiring M2/M3/M5/M6 will depend on: a bias added in score
    space has to reach the output rule, not be computed and dropped."""
    probs = _random_probs(200, seed=3)
    unbiased = [p["promise_status"] for p in decide("M1", probs)]
    assert "No" in unbiased, "the unbiased run should not already be all-Yes"

    biases = {f: np.zeros(len(EVAL_FIELDS[f])) for f in FIELDS}
    biases["promise_status"] = np.array([50.0, 0.0])   # overwhelming push to Yes
    assert all(p["promise_status"] == "Yes" for p in project(log_scores(probs, biases)))


# --- the driver -----------------------------------------------------------

@pytest.fixture(scope="module")
def run():
    return load_run(FIXTURES, "pdf_group", 42, SPLIT)


def test_loading_a_run_validates_every_bundle(run):
    assert len(run) == 5
    assert [meta["rotation"] for meta, _ in run] == [0, 1, 2, 3, 4]


def test_loading_a_run_forwards_the_selected_splits_directory(monkeypatch, tmp_path):
    """The English driver must not validate 400 test ids against Chinese folds."""
    from paper import run_decisions

    for k in range(5):
        bundle = tmp_path / f"pdf_group_seed42_r{k}"
        bundle.mkdir()
        (bundle / "meta.json").write_text("{}", encoding="utf-8")

    seen = {}

    def validate_run(dirs, *, splits_dir):
        seen["splits_dir"] = splits_dir
        return []

    monkeypatch.setattr(run_decisions, "validate_probs_run", validate_run)
    monkeypatch.setattr(run_decisions, "validate_probs_bundle", lambda *args: [])
    monkeypatch.setattr(run_decisions, "load_bundle", lambda path: ({}, {}))

    selected = tmp_path / "splits_en"
    loaded = load_run(tmp_path, "pdf_group", 42, {}, splits_dir=selected)

    assert len(loaded) == 5
    assert seen["splits_dir"] == selected


def test_a_run_covers_every_row_exactly_once(run):
    records, _ = run_method("M1", run, ROWS, SPLIT)
    _check_complete(records, ROWS, "M1")
    assert len(records) == 2000
    assert len({r["id"] for r in records}) == 2000


def test_predictions_are_attached_to_the_right_rows(run):
    """The arrays are aligned to the manifest's id list by position, so this is
    the check that the concatenation did not shift anything."""
    records, _ = run_method("M0", run, ROWS, SPLIT)
    by_id = index_by_id(ROWS)
    rotation_of = {i: rot["k"] for rot in SPLIT["rotations"] for i in rot["test_ids"]}
    for r in records:
        assert r["gold_ps"] == by_id[r["id"]]["promise_status"]
        assert r["pdf_url"] == by_id[r["id"]]["pdf_url"]
        assert r["rotation"] == rotation_of[r["id"]]


def test_m1_is_always_legal_and_m0_is_not(run):
    m0, _ = run_method("M0", run, ROWS, SPLIT)
    m1, _ = run_method("M1", run, ROWS, SPLIT)

    def illegal(records):
        return sum(not is_valid_tuple(*(r[f"pred_{a}"] for a in ("ps", "vt", "es", "eq")))
                   for r in records)

    assert illegal(m1) == 0
    assert illegal(m0) > 0, "the fixtures must exercise the baseline's failure mode"


def test_projection_never_revisits_the_parent(run):
    """M1 is greedy and top-down: a confident child cannot overturn the parent.
    That is precisely what the joint decoder will be able to do, so the
    difference between M1 and M4 rests on this holding."""
    m0, _ = run_method("M0", run, ROWS, SPLIT)
    m1, _ = run_method("M1", run, ROWS, SPLIT)
    by_id = {r["id"]: r for r in m1}
    assert all(r["pred_ps"] == by_id[r["id"]]["pred_ps"] for r in m0)


def test_the_driver_records_a_bias_for_exactly_the_calibrated_methods(run):
    """The results file is the audit trail for what a method actually did. A
    calibrated arm with no recorded bias, or an uncalibrated one carrying a
    bias, would be a different method from the one the table names."""
    cache: dict = {}
    for method_id in METHOD_IDS:
        _, params = run_method(method_id, run, ROWS, SPLIT, cache)
        recorded = [params[str(k)]["calibration_biases"] for k in range(5)]
        if METHODS[method_id].calibration is None:
            assert all(r is None for r in recorded), method_id
        else:
            assert all(r for r in recorded), method_id


def test_a_short_run_is_refused(tmp_path):
    """Four rotations cannot produce a 2,000-row file, and a file that is
    quietly short would still score."""
    import shutil

    for k in range(4):
        shutil.copytree(FIXTURES / f"pdf_group_seed42_r{k}",
                        tmp_path / f"pdf_group_seed42_r{k}")
    with pytest.raises(SystemExit, match="missing probability bundles"):
        load_run(tmp_path, "pdf_group", 42, SPLIT)
