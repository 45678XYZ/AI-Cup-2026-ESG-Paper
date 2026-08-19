"""The 17-state joint decoder, and how it differs from the projection.

The claims the paper makes about M4 are that its output is always legal, that
it maximises the summed field score over the legal set, and that it can revise
a parent decision the greedy projection is stuck with. Each is asserted here
against a brute-force reference rather than against the implementation's own
logic.
"""

import itertools

import numpy as np
import pytest

from paper.decoder import STATE_COLUMNS, decode_state_ids, decode_valid_states, state_scores
from paper.labels import EVAL_FIELDS, FIELDS, STATES, is_valid_tuple
from paper.methods import decide, log_scores
from paper.projection import project

ALL_TUPLES = list(itertools.product(*(EVAL_FIELDS[f] for f in FIELDS)))


def _random_scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {f: rng.normal(size=(n, len(EVAL_FIELDS[f]))) for f in FIELDS}


def _brute_force(scores, row, alpha=None):
    """The definition, evaluated directly: best legal tuple by summed score."""
    alpha = [1.0] * len(FIELDS) if alpha is None else alpha
    best, best_score = None, -np.inf
    for tup in ALL_TUPLES:
        if not is_valid_tuple(*tup):
            continue
        total = sum(
            a * scores[f][row][EVAL_FIELDS[f].index(lab)]
            for a, f, lab in zip(alpha, FIELDS, tup)
        )
        if total > best_score:
            best, best_score = tup, total
    return dict(zip(FIELDS, best))


# --- the state table ------------------------------------------------------

def test_the_state_table_is_the_frozen_one():
    assert STATE_COLUMNS.shape == (17, 4)
    for i, state in enumerate(STATES):
        tup = (state.ps, state.vt, state.es, state.eq)
        assert [EVAL_FIELDS[f][c] for f, c in zip(FIELDS, STATE_COLUMNS[i])] == list(tup)


# --- correctness against the definition -----------------------------------

def test_it_finds_the_same_tuple_as_brute_force():
    scores = _random_scores(200)
    decoded = decode_valid_states(scores)
    for row in range(200):
        assert decoded[row] == _brute_force(scores, row), row


def test_every_output_is_legal():
    decoded = decode_valid_states(_random_scores(500, seed=7))
    assert all(is_valid_tuple(*(p[f] for f in FIELDS)) for p in decoded)


def test_the_chosen_state_really_is_the_maximum():
    scores = _random_scores(100, seed=2)
    table = state_scores(scores)
    chosen = decode_state_ids(scores)
    assert table[np.arange(100), chosen].max() == table.max()
    assert np.all(table[np.arange(100), chosen] == table.max(axis=1))


def test_alpha_is_one_unless_asked_otherwise():
    scores = _random_scores(50, seed=5)
    assert np.allclose(state_scores(scores), state_scores(scores, alpha=[1, 1, 1, 1]))


def test_a_malformed_alpha_is_rejected():
    with pytest.raises(ValueError, match="alpha has shape"):
        state_scores(_random_scores(4), alpha=[1.0, 1.0])


# --- what separates the decoder from the projection -----------------------

def test_the_decoder_can_overturn_a_parent_the_projection_cannot():
    """A marginal promise_status against three children that all point the
    other way. The greedy rule locks in the parent; the joint rule does not."""
    scores = {f: np.zeros((1, len(EVAL_FIELDS[f]))) for f in FIELDS}
    scores["promise_status"][0] = [-0.1, 0.0]              # No wins, barely
    scores["verification_timeline"][0] = [5.0, 0, 0, 0, -5.0]   # strongly "already"
    scores["evidence_status"][0] = [5.0, 0.0, -5.0]             # strongly Yes
    scores["evidence_quality"][0] = [5.0, 0, 0, -5.0]           # strongly Clear

    assert project(scores)[0]["promise_status"] == "No"
    assert decode_valid_states(scores)[0] == {
        "promise_status": "Yes", "verification_timeline": "already",
        "evidence_status": "Yes", "evidence_quality": "Clear",
    }


def test_the_two_rules_agree_when_the_parent_is_not_marginal():
    """They must not differ everywhere, or the contrast measures noise."""
    scores = _random_scores(500, seed=11)
    agree = sum(a == b for a, b in zip(project(scores), decode_valid_states(scores)))
    assert 0 < agree < 500, f"agreement was {agree}/500"


def test_ties_resolve_to_the_lower_state_id():
    scores = {f: np.zeros((1, len(EVAL_FIELDS[f]))) for f in FIELDS}
    assert decode_state_ids(scores)[0] == 0


# --- through the method dispatcher ----------------------------------------

def test_m4_routes_to_the_decoder_and_needs_no_biases():
    rng = np.random.default_rng(0)
    probs = {}
    for f in FIELDS:
        raw = rng.random((100, len(EVAL_FIELDS[f])))
        probs[f] = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    assert decide("M4", probs) == decode_valid_states(log_scores(probs))
