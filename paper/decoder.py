"""Joint decoding over the 17 legal states (M4-M6).

Where the projection decides ``promise_status`` first and never reconsiders it,
this rule scores all 17 legal tuples and takes the best:

    y_hat = argmax_{s in S}  sum_t  alpha_t * s_t(x, s_t)

with ``s_t`` the field score ``log p + b`` from ``paper/methods.py``. Because
every legal tuple is scored whole, a confident ``evidence_quality`` *can*
overturn a marginal ``promise_status`` — which the greedy projection can never
do, and which is the entire content of the M4-vs-M1 contrast.

``alpha`` is fixed at 1 for the main comparison (plan §3.3), deliberately: a
decoder carrying tunable task weights would beat the projection partly on the
strength of having more parameters, and the table could not tell the two causes
apart. The argument exists so an exploratory row can vary it, but every M4-M6
run in the main table passes nothing.

The state table is built from ``labels.STATES``, so the decoder cannot drift
from the frozen state space or from what the projection considers legal.
"""

import numpy as np

from paper.labels import FIELDS, LABEL2ID, STATES

# (17, 4) column indices: STATE_COLUMNS[s, t] is the class index that state s
# takes in field t. Reduces decoding to one gather-and-sum per row.
STATE_COLUMNS = np.array(
    [[LABEL2ID[field][getattr(state, alias)] for field, alias in zip(FIELDS, ("ps", "vt", "es", "eq"))]
     for state in STATES],
    dtype=np.intp,
)

STATE_LABELS = [
    dict(zip(FIELDS, (s.ps, s.vt, s.es, s.eq))) for s in STATES
]


def state_scores(scores, alpha=None) -> np.ndarray:
    """``(N, 17)`` score of every legal state for every row.

    ``scores`` maps each field to an ``(N, C)`` array of ``log p + b``.
    """
    n_rows = len(next(iter(scores.values())))
    alpha = np.ones(len(FIELDS)) if alpha is None else np.asarray(alpha, dtype=np.float64)
    if alpha.shape != (len(FIELDS),):
        raise ValueError(f"alpha has shape {alpha.shape}, expected {(len(FIELDS),)}")

    total = np.zeros((n_rows, len(STATES)), dtype=np.float64)
    for t, field in enumerate(FIELDS):
        field_scores = np.asarray(scores[field], dtype=np.float64)
        if len(field_scores) != n_rows:
            raise ValueError(f"field arrays disagree on row count at {field}")
        # Column STATE_COLUMNS[:, t] picks, for each state, the class this
        # field takes in it; the gather is over states, not over rows.
        total += alpha[t] * field_scores[:, STATE_COLUMNS[:, t]]
    return total


def decode_valid_states(scores, alpha=None) -> list[dict]:
    """Best legal tuple per row, as a dict of field->label.

    Ties go to the lower ``state_id``, matching ``numpy.argmax`` and the
    canonically-earlier tie-break the projection uses, so an M1/M4 difference
    can never be an artefact of tie handling.
    """
    best = np.argmax(state_scores(scores, alpha), axis=1)
    return [STATE_LABELS[i] for i in best]


def decode_state_ids(scores, alpha=None) -> np.ndarray:
    """The chosen ``state_id`` per row, for analyses that want the integer."""
    return np.argmax(state_scores(scores, alpha), axis=1)
