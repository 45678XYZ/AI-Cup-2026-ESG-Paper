"""Training-time structural objective: the negative log-likelihood of legality.

Plan section 10 lists "training-time structural objectives" among the things
this study does *not* compare, and the committed runs bear out why that gap
matters: the four heads softmax independently, and across the official test
partitions they place, on average, **50.6% of their probability mass on joint
states the hierarchy forbids**. Projection recovers the argmax after the fact;
it cannot recover the capacity spent getting there.

The penalty here is the semantic loss of Xu et al. (2018), "A Semantic Loss
Function for Deep Learning with Symbolic Knowledge": under the model's own
independence assumption, the probability that its output is legal is the sum
over the 17 admissible states of the product of the four per-field
probabilities, and ``-log`` of that is a proper objective. It says nothing
about *which* legal state is right -- that is the label's job -- so it adds a
constraint rather than a second, competing supervision signal.

The state table is imported from ``paper.decoder`` rather than restated, so the
objective a model is trained under and the decoder M4-M6 score it with cannot
drift apart.

**Off by default.** ``LAMBDA_UNSET`` is the frozen study's setting; with it the
loss is the one that produced the 30 committed bundles, unchanged. Nothing here
touches ``paper/train_config.py``, whose sha256 every bundle records as its
proof of recipe.
"""

import numpy as np

from paper.decoder import STATE_COLUMNS
from paper.labels import FIELDS

# The frozen study's setting: no structural term at all.
LAMBDA_UNSET = 0.0

# Candidate strengths for the pre-registered selection sweep
# (docs/pre_registration_structural_training.md). Chosen so the penalty spans
# well below to slightly above the base loss at initialisation, where the two
# terms are about 1.20 and 1.95 respectively.
LAMBDA_GRID = (0.1, 0.3, 1.0)


def legal_log_mass_np(log_probs) -> np.ndarray:
    """``log P(output is legal)`` per row, from per-field log-probabilities.

    ``log_probs`` maps each field to an ``(N, C)`` array of log-probabilities
    whose columns follow ``EVAL_FIELDS[field]``. The reference implementation:
    the torch version below is asserted against it, so the objective's meaning
    is checkable in an environment without torch.
    """
    total = None
    for t, field in enumerate(FIELDS):
        picked = log_probs[field][:, STATE_COLUMNS[:, t]]        # (N, 17)
        total = picked if total is None else total + picked
    m = total.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(total - m).sum(axis=1, keepdims=True))).ravel()


def illegal_mass_penalty(logits):
    """``-log P(output is legal)``, averaged over the batch. Torch, differentiable.

    torch is imported here rather than at module scope so the reference
    implementation above stays importable in the CPU environment, where the
    rest of the study runs.
    """
    import torch

    total = None
    for t, field in enumerate(FIELDS):
        lp = torch.log_softmax(logits[field], dim=-1)
        cols = torch.as_tensor(STATE_COLUMNS[:, t], device=lp.device)
        picked = lp.index_select(1, cols)                        # (B, 17)
        total = picked if total is None else total + picked
    return -torch.logsumexp(total, dim=1).mean()
