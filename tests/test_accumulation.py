"""The gradient-accumulation contract, checked without torch.

These are the arithmetic properties the training loop depends on. They decide
how many optimiser steps a fit takes and how each window's loss is scaled, so
getting one wrong moves every number in the study -- and nothing downstream can
detect it, because the probabilities that come out are still well-formed.

``paper/train_fold.py`` cannot be imported without torch, which the CPU half of
the study does not install; that is why the arithmetic lives in
``paper/accumulation.py`` and is tested here rather than beside the loop.
"""

import pytest

from paper.accumulation import accumulation_window, optimiser_steps_per_epoch
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
