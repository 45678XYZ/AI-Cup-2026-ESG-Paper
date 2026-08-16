"""Joint decoding over the frozen 17-state label space."""

import numpy as np

from paper.labels import FIELD_ALIAS, FIELDS, LABEL2ID, STATES
from paper.projection import _n_rows

DEFAULT_ALPHA = (1.0, 1.0, 1.0, 1.0)


def decode(scores, alpha=DEFAULT_ALPHA) -> list[dict]:
    """Choose the maximum summed score among the 17 legal states.

    ``scores`` are per-field arrays, normally ``log(p)`` or ``log(p) + bias``.
    The main comparison fixes ``alpha`` to four ones.  State order is canonical,
    so ``numpy.argmax`` gives deterministic state-id tie breaking.
    """
    n = _n_rows(scores)
    alpha = tuple(float(x) for x in alpha)
    if len(alpha) != len(FIELDS):
        raise ValueError(f"alpha has {len(alpha)} entries, expected {len(FIELDS)}")

    state_scores = np.zeros((n, len(STATES)), dtype=float)
    for field_index, field in enumerate(FIELDS):
        state_columns = [
            LABEL2ID[field][getattr(state, FIELD_ALIAS[field])] for state in STATES
        ]
        state_scores += alpha[field_index] * np.asarray(scores[field])[:, state_columns]

    winners = np.argmax(state_scores, axis=1)
    predictions = []
    for state_id in winners:
        state = STATES[int(state_id)]
        predictions.append(dict(zip(FIELDS, (state.ps, state.vt, state.es, state.eq))))
    return predictions
