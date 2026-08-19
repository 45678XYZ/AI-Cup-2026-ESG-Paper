"""Gradient-accumulation arithmetic, kept apart from the training loop.

This is pure integer arithmetic over ``GRAD_ACCUM_STEPS``, but it decides how
many optimiser steps a fit takes and how each window's loss is scaled -- both
of which change the numbers. ``paper/train_fold.py`` imports torch at module
level, so a test living beside the loop can only run where torch is installed,
which is not where the CPU half of the study runs. Housing the arithmetic here
lets ``tests/test_accumulation.py`` check it in every environment.
"""

import math

from paper.train_config import GRAD_ACCUM_STEPS


def accumulation_window(step, n_batches):
    """Return this window's size in batches and whether ``step`` closes it.

    The final window of an epoch may hold fewer than ``GRAD_ACCUM_STEPS``
    batches. It still takes an optimiser step, so its loss is scaled by the
    number of batches it actually contains rather than by the nominal count.
    """
    start = (step // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
    size = min(GRAD_ACCUM_STEPS, n_batches - start)
    return size, step + 1 == start + size


def optimiser_steps_per_epoch(n_batches):
    """How many times the optimiser steps in one epoch over ``n_batches``.

    The scheduler is built for ``this * EPOCHS`` steps. Deriving both from one
    function is what keeps the horizon and the real step count from drifting:
    a scheduler sized for fewer steps than the loop takes silently runs the
    tail of every epoch at the wrong learning rate.
    """
    return math.ceil(n_batches / GRAD_ACCUM_STEPS)
