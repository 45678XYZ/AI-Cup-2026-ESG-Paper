"""The lambda-selection criterion.

Its one load-bearing property is negative: the selection must not be able to
see the Test partition. Everything else here is arithmetic; that one is the
reason the sweep can be called pre-registered at all.
"""

import json

import numpy as np
import pytest

from paper.data import index_by_id, load_dev
from paper.labels import EVAL_FIELDS, FIELDS
from paper.score import compute_weighted_macro_f1
from paper.select_lambda import INDISTINGUISHABLE, calibration_score, sweep_scores
from paper.structure_loss import LAMBDA_GRID

REAL = "probs/pdf_group_seed42_r0"


@pytest.fixture(scope="module")
def by_id():
    return index_by_id(load_dev())


def test_it_never_opens_a_test_array(monkeypatch, by_id):
    """The selection is only honest if it structurally cannot consult the rows
    the study is scored on. Patching the loader is the only way to assert that
    the code does not merely refrain from using them."""
    opened = []
    real_load = np.load

    def spy(path, *a, **kw):
        opened.append(str(path))
        return real_load(path, *a, **kw)

    monkeypatch.setattr(np, "load", spy)
    calibration_score(REAL, by_id)

    assert opened, "nothing was loaded; the spy is not wired up"
    assert not [p for p in opened if "test_" in p], \
        f"the selection read Test arrays: {[p for p in opened if 'test_' in p]}"


def test_the_score_is_the_official_metric_on_the_calibration_rows(by_id):
    """Recomputed here from the raw arrays, so a change to the scorer or to the
    argmax convention shows up as a disagreement rather than as a shifted
    lambda nobody can explain."""
    with open(f"{REAL}/meta.json", encoding="utf-8") as f:
        ids = json.load(f)["calibration_ids"]
    probs = {f: np.load(f"{REAL}/calibration_{f}.npy") for f in FIELDS}
    pred = [{f: EVAL_FIELDS[f][int(np.argmax(probs[f][n]))] for f in FIELDS}
            for n in range(len(ids))]
    gold = [{f: by_id[i][f] for f in FIELDS} for i in ids]

    assert calibration_score(REAL, by_id) == pytest.approx(
        compute_weighted_macro_f1(gold, pred), abs=1e-12)


def test_no_decision_rule_is_interposed(by_id):
    """Independent argmax, deliberately: projection or a bias would confound
    the probabilities a lambda produces with how well a rule repairs them."""
    import paper.projection as projection

    called = []
    for name in ("project", "independent_argmax"):
        fn = getattr(projection, name)
        setattr(projection, name, lambda *a, _n=name, **k: called.append(_n))
    try:
        calibration_score(REAL, by_id)
    finally:
        pass
    assert not called, f"a decision rule ran during selection: {called}"


def test_it_groups_the_frozen_study_as_one_arm(by_id):
    """The 30 committed bundles predate structure_lambda; absent must read as
    the frozen setting, or the control arm looks like a sweep of its own."""
    table = sweep_scores("probs", by_id)
    assert list(table) == [0.0], table
    assert table[0.0]["rotations"] == 30


def test_the_fallback_threshold_is_below_the_spread_it_must_catch():
    """Seed-to-seed std of the official metric is about 0.004; a sweep that
    lands inside that has not resolved anything."""
    assert 0 < INDISTINGUISHABLE < 0.004


def test_the_grid_has_a_median_for_the_fallback_to_take():
    assert len(LAMBDA_GRID) % 2 == 1, "an even grid has no single median"
