"""The gradient-accumulation contract, checked without torch.

These are the arithmetic properties the training loop depends on. They decide
how many optimiser steps a fit takes and how each window's loss is scaled, so
getting one wrong moves every number in the study -- and nothing downstream can
detect it, because the probabilities that come out are still well-formed.

``paper/train_fold.py`` cannot be imported without torch, which the CPU half of
the study does not install; that is why the arithmetic lives in
``paper/accumulation.py`` and is tested here rather than beside the loop.
"""

import math

import pytest

from paper.accumulation import (
    EFFECTIVE_BATCH,
    accumulation_window,
    loss_scale,
    optimiser_steps_per_epoch,
)
from paper.train_config import BATCH_SIZE, GRAD_ACCUM_STEPS

# Spans the real rotations (1184-1222 train rows over batches of 8, i.e. 148-153
# batches) plus the degenerate small cases.
BATCH_COUNTS = [1, 2, 3, 4, 5, 7, 8, 148, 149, 150, 151, 152, 153]


@pytest.mark.parametrize("n_batches", BATCH_COUNTS)
def test_every_batch_belongs_to_exactly_one_window(n_batches):
    """No batch may be dropped or counted twice: the leftover batch at the end
    of an epoch used to be discarded, which quietly shrank the training set."""
    covered = 0
    for step in range(n_batches):
        size, _ = accumulation_window(step, n_batches)
        assert 1 <= size <= GRAD_ACCUM_STEPS
        covered += 1 / size
    assert covered == pytest.approx(optimiser_steps_per_epoch(n_batches))


@pytest.mark.parametrize("n_batches", BATCH_COUNTS)
def test_the_scheduler_horizon_matches_the_real_step_count(n_batches):
    """``train_rotation`` sizes the LR schedule with
    ``optimiser_steps_per_epoch``. If the loop takes more steps than that, the
    tail of every epoch runs past the end of the schedule."""
    closes = sum(accumulation_window(s, n_batches)[1] for s in range(n_batches))
    assert closes == optimiser_steps_per_epoch(n_batches)


@pytest.mark.parametrize("n_batches", BATCH_COUNTS)
def test_a_window_closes_on_its_last_step_and_only_there(n_batches):
    for step in range(n_batches):
        size, closes = accumulation_window(step, n_batches)
        start = (step // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
        assert closes == (step == start + size - 1)


def test_a_short_final_window_is_kept_and_scaled_by_its_own_size():
    """Three batches at accumulation 2: two full windows' worth is one window
    of 2 and one of 1, and the odd one still takes a step."""
    assert [accumulation_window(i, 3) for i in range(3)] == [
        (2, False), (2, True), (1, True),
    ]


def test_the_frozen_recipe_still_accumulates():
    """If either constant moved, the windows above stop describing the study."""
    assert (BATCH_SIZE, GRAD_ACCUM_STEPS) == (8, 2)
    assert EFFECTIVE_BATCH == 16


# --- how much each row weighs ---------------------------------------------

@pytest.mark.parametrize("n_batches", BATCH_COUNTS)
def test_every_row_in_an_epoch_carries_the_same_weight(n_batches):
    """The property the scaling exists for. Losses are reduction="mean", so a
    short batch has already inflated its own rows; scaling by the real row
    count against a fixed denominator is what cancels that. Whatever the batch
    layout, one row is worth 1/EFFECTIVE_BATCH of a step."""
    for rows_in_batch in range(1, BATCH_SIZE + 1):
        per_row = loss_scale(rows_in_batch) / rows_in_batch
        assert per_row == pytest.approx(1 / EFFECTIVE_BATCH)


def test_a_full_window_still_weighs_exactly_one_step():
    """The common case must be untouched, or all 30 bundles are invalidated
    rather than the 29 with a short final batch."""
    assert loss_scale(BATCH_SIZE) * GRAD_ACCUM_STEPS == pytest.approx(1.0)


def test_a_lone_short_batch_is_not_amplified():
    """The defect this replaces: dividing by the window's batch count gave a
    single-row final window 1/1 of a step -- that row at 16x a normal row's
    weight, twelve times per fit."""
    assert loss_scale(1) == pytest.approx(1 / EFFECTIVE_BATCH)
    assert loss_scale(1) < loss_scale(2) < loss_scale(BATCH_SIZE)


@pytest.mark.parametrize("n_train", [1184, 1185, 1202, 1222])
def test_an_epoch_weighs_the_same_as_its_row_count(n_train):
    """Summed over a real rotation, the epoch is worth exactly n_train rows --
    no row dropped (the pre-PR behaviour) and none counted twice."""
    n_batches = math.ceil(n_train / BATCH_SIZE)
    total = sum(
        loss_scale(min(BATCH_SIZE, n_train - i * BATCH_SIZE))
        for i in range(n_batches)
    )
    assert total == pytest.approx(n_train / EFFECTIVE_BATCH)
