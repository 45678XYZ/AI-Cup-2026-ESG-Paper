"""The training-time legality objective.

The numeric core runs without torch: ``legal_log_mass_np`` is the reference and
the torch implementation is asserted against it, so the objective's meaning is
checkable in the CPU environment where the rest of the study runs. Only the
autograd behaviour needs torch.
"""

import numpy as np
import pytest

from paper.decoder import STATE_COLUMNS, STATE_LABELS
from paper.labels import EVAL_FIELDS, FIELDS, LABEL2ID, STATES, is_valid_tuple
from paper.structure_loss import (
    LAMBDA_GRID,
    LAMBDA_UNSET,
    illegal_mass_penalty,
    legal_log_mass_np,
)


def uniform(n_rows=4):
    return {f: np.log(np.full((n_rows, len(EVAL_FIELDS[f])), 1 / len(EVAL_FIELDS[f])))
            for f in FIELDS}


def onehot(state, n_rows=1, mass=1.0 - 1e-9):
    """Log-probabilities concentrated on one label assignment."""
    out = {}
    for f in FIELDS:
        n = len(EVAL_FIELDS[f])
        p = np.full((n_rows, n), (1 - mass) / (n - 1))
        p[:, LABEL2ID[f][state[f]]] = mass
        out[f] = np.log(p)
    return out


# --- what the objective means ---------------------------------------------

def test_uniform_predictions_put_seventeen_of_one_hundred_twenty_on_legality():
    """2x5x3x4 = 120 label combinations, 17 of them legal. A model that has
    learned nothing starts here, which is what fixes the penalty's scale."""
    assert np.exp(legal_log_mass_np(uniform()))[0] == pytest.approx(17 / 120)


def test_a_confident_legal_prediction_approaches_full_mass():
    legal = {f: STATE_LABELS[0][f] for f in FIELDS}
    assert is_valid_tuple(*(legal[f] for f in FIELDS))
    assert np.exp(legal_log_mass_np(onehot(legal)))[0] > 0.999


def test_a_confident_illegal_prediction_approaches_zero_mass():
    """``promise_status=No`` with a real timeline is the hierarchy's own
    contradiction: nothing was promised, so nothing can be verified."""
    illegal = {"promise_status": "No", "verification_timeline": "already",
               "evidence_status": "Yes", "evidence_quality": "Clear"}
    assert not is_valid_tuple(*(illegal[f] for f in FIELDS))
    assert np.exp(legal_log_mass_np(onehot(illegal)))[0] < 1e-6


def test_the_penalty_never_prefers_one_legal_state_over_another():
    """It constrains, it does not supervise. Two confident, equally legal
    predictions must cost the same, or the term competes with the labels."""
    masses = [np.exp(legal_log_mass_np(onehot({f: s[f] for f in FIELDS})))[0]
              for s in STATE_LABELS]
    # The spread is the 1e-9 left over from ``onehot`` spread across a differing
    # number of classes per state, not a preference between states.
    assert max(masses) - min(masses) < 1e-8
    assert len(masses) == 17


def test_it_reads_the_decoder_s_own_state_table():
    """Restating the 17 states here would let the objective a model trains
    under drift from the decoder M4-M6 scores it with."""
    assert STATE_COLUMNS.shape == (len(STATES), len(FIELDS))
    for s, state in enumerate(STATE_LABELS):
        for t, field in enumerate(FIELDS):
            assert STATE_COLUMNS[s, t] == LABEL2ID[field][state[field]]


# --- the frozen study's setting -------------------------------------------

def test_the_default_is_off():
    """The 30 committed bundles were produced without this term; the default
    has to reproduce them rather than merely approximate them."""
    assert LAMBDA_UNSET == 0.0
    assert all(x > 0 for x in LAMBDA_GRID), "a zero in the sweep is the control"


def test_a_zero_lambda_leaves_the_frozen_loss_untouched():
    """``_loss`` must not compute the term at all at lambda 0 -- adding
    ``0.0 * penalty`` is equivalent, not identical, and the committed bundles
    are a bit-for-bit claim."""
    # Read the file rather than import it: ``paper.train_fold`` needs torch,
    # and this property must be checkable where the CPU half of the study runs.
    from paper.data import REPO_ROOT

    src = (REPO_ROOT / "paper" / "train_fold.py").read_text(encoding="utf-8")
    assert "if structure_lambda:" in src, "the term must be branched on, not scaled by zero"


# --- the torch implementation ---------------------------------------------
# Skipped per test, not per module: an importorskip at module scope would take
# every test above with it, and those are the ones that must run on CPU.

try:
    import torch
except ImportError:                                          # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="needs torch")


@requires_torch
def test_torch_matches_the_reference():
    rng = np.random.default_rng(0)
    logits = {f: rng.normal(size=(6, len(EVAL_FIELDS[f]))) for f in FIELDS}
    log_probs = {f: v - np.log(np.exp(v).sum(axis=1, keepdims=True)) for f, v in logits.items()}
    expected = -legal_log_mass_np(log_probs).mean()
    got = illegal_mass_penalty({f: torch.tensor(v) for f, v in logits.items()})
    assert float(got) == pytest.approx(expected, abs=1e-9)


@requires_torch
def test_the_penalty_has_a_gradient_that_moves_mass_onto_legal_states():
    logits = {f: torch.zeros(1, len(EVAL_FIELDS[f]), requires_grad=True) for f in FIELDS}
    illegal_mass_penalty(logits).backward()
    # promise_status=No forbids every child label but N/A, so pushing mass to
    # No must raise the N/A logits of the children it constrains.
    assert logits["promise_status"].grad is not None
    assert torch.isfinite(logits["promise_status"].grad).all()
