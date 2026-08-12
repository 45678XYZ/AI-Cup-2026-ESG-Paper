"""The deterministic projection, and how it relates to the M0 baseline.

The claims asserted here are the ones the paper makes about M1, so they are
checked exhaustively where the space allows rather than on samples.
"""

import itertools
import json

import numpy as np
import pytest

from paper.data import REPO_ROOT
from paper.labels import EVAL_FIELDS, FIELDS, LABEL2ID, STATES, is_valid_tuple
from paper.projection import ADMISSIBLE, independent_argmax, project

ALL_TUPLES = list(itertools.product(*(EVAL_FIELDS[f] for f in FIELDS)))


def _one_hot(tup):
    """Probabilities whose independent argmax is exactly ``tup``."""
    probs = {}
    for field, label in zip(FIELDS, tup):
        row = np.full(len(EVAL_FIELDS[field]), 0.01, dtype=np.float32)
        row[LABEL2ID[field][label]] = 0.99
        probs[field] = row[None, :]
    return probs


# --- the admissible sets --------------------------------------------------

def test_admissible_sets_come_from_the_state_space():
    assert ADMISSIBLE[()] == ["Yes", "No"]
    assert ADMISSIBLE[("No",)] == ["N/A"]
    assert ADMISSIBLE[("Yes",)] == [
        "already", "within_2_years", "between_2_and_5_years", "more_than_5_years",
    ]
    assert ADMISSIBLE[("Yes", "already")] == ["Yes", "No"]
    assert ADMISSIBLE[("Yes", "already", "No")] == ["N/A"]
    assert ADMISSIBLE[("Yes", "already", "Yes")] == ["Clear", "Not Clear", "Misleading"]


def test_no_dead_end_prefix():
    """Every prefix the projection can reach must offer at least one option."""
    for prefix, allowed in ADMISSIBLE.items():
        assert allowed, prefix


# --- the two properties M1 relies on --------------------------------------

def test_every_possible_argmax_projects_to_a_legal_state():
    """All 120 outcomes, including the 103 illegal ones."""
    assert len(ALL_TUPLES) == 120
    for tup in ALL_TUPLES:
        pred = project(_one_hot(tup))[0]
        assert is_valid_tuple(*(pred[f] for f in FIELDS)), (tup, pred)


def test_projection_leaves_a_legal_argmax_untouched():
    """M1 may only differ from M0 where M0 broke the hierarchy."""
    for state in STATES:
        tup = (state.ps, state.vt, state.es, state.eq)
        assert tuple(project(_one_hot(tup))[0][f] for f in FIELDS) == tup


def test_projection_changes_exactly_the_illegal_rows():
    rng = np.random.default_rng(0)
    probs = {}
    for field in FIELDS:
        raw = rng.random((500, len(EVAL_FIELDS[field])))
        probs[field] = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    base, projected = independent_argmax(probs), project(probs)
    changed = [i for i, (a, b) in enumerate(zip(base, projected)) if a != b]
    illegal = [
        i for i, a in enumerate(base)
        if not is_valid_tuple(*(a[f] for f in FIELDS))
    ]
    assert changed == illegal
    assert illegal, "random probabilities should produce some illegal tuples"


# --- determinism ----------------------------------------------------------

def test_ties_break_toward_the_canonically_earlier_class():
    """Uniform probabilities must still give one reproducible answer."""
    probs = {
        f: np.full((1, len(EVAL_FIELDS[f])), 1 / len(EVAL_FIELDS[f]), dtype=np.float32)
        for f in FIELDS
    }
    assert project(probs)[0] == dict(
        zip(FIELDS, ("Yes", "already", "Yes", "Clear"))
    )


def test_argmax_and_projection_agree_on_how_to_break_a_tie():
    """Otherwise an M0/M1 difference could be an artefact of tie handling."""
    probs = {f: np.zeros((1, len(EVAL_FIELDS[f])), dtype=np.float32) for f in FIELDS}
    probs["promise_status"][0] = [0.5, 0.5]
    probs["verification_timeline"][0] = [0.2, 0.2, 0.2, 0.2, 0.2]
    probs["evidence_status"][0] = [0.4, 0.4, 0.2]
    probs["evidence_quality"][0] = [0.25, 0.25, 0.25, 0.25]
    assert independent_argmax(probs)[0] == project(probs)[0]


def test_row_count_mismatch_is_rejected():
    probs = {f: np.zeros((3, len(EVAL_FIELDS[f])), dtype=np.float32) for f in FIELDS}
    probs["evidence_quality"] = probs["evidence_quality"][:2]
    with pytest.raises(ValueError, match="row count"):
        project(probs)


# --- against a real contract bundle ---------------------------------------

def test_runs_on_a_contract_shaped_bundle():
    """The fixtures are the input M1 will actually receive."""
    bundle = REPO_ROOT / "contracts" / "examples" / "probs" / "pdf_group_seed42_r0"
    meta = json.load(open(bundle / "meta.json", encoding="utf-8"))
    probs = {f: np.load(bundle / f"test_{f}.npy") for f in FIELDS}

    preds = project(probs)
    assert len(preds) == len(meta["test_ids"])
    assert all(is_valid_tuple(*(p[f] for f in FIELDS)) for p in preds)
