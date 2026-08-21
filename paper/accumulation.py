"""Gradient-accumulation arithmetic, kept apart from the training loop.

This is pure integer arithmetic over ``GRAD_ACCUM_STEPS``, but it decides how
many optimiser steps a fit takes and how each window's loss is scaled -- both
of which change the numbers. ``paper/train_fold.py`` imports torch at module
level, so a test living beside the loop can only run where torch is installed,
which is not where the CPU half of the study runs. Housing the arithmetic here
lets ``tests/test_accumulation.py`` check it in every environment.
"""

import math

from paper.train_config import BATCH_SIZE, GRAD_ACCUM_STEPS

# What one optimiser step is defined to represent, in rows.
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS


def accumulation_window(step, n_batches):
    """Return this window's size in batches and whether ``step`` closes it.

    The final window of an epoch may hold fewer than ``GRAD_ACCUM_STEPS``
    batches. It still takes an optimiser step; how much that step weighs is
    ``loss_scale``'s business, not this function's.
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


def loss_scale(n_rows):
    """The fraction of one optimiser step a batch of ``n_rows`` carries.

    The losses are ``reduction="mean"``, i.e. already averaged over the batch,
    so a batch holding fewer than ``BATCH_SIZE`` rows has *already* inflated
    each of its rows before any accumulation divisor is applied. Scaling by the
    real row count against a fixed denominator undoes that: every row carries
    ``1 / EFFECTIVE_BATCH`` of a step wherever it lands, and a short window
    simply takes a proportionally smaller step.

    Dividing by the window's batch count instead -- or by GRAD_ACCUM_STEPS
    unconditionally -- leaves the inflation in place. A rotation whose last
    window is one batch of one row would apply that row at a full effective
    batch's weight, 16x a normal row, twelve times per fit.
    """
    return n_rows / EFFECTIVE_BATCH
