"""Deterministic projection of independent field predictions onto the 17 states.

M0 takes each field's argmax independently, so its output can be any of the 120
tuples -- 103 of which the hierarchy forbids. M1-M3 apply the rule defined here
instead: each field's argmax is **restricted to the values the already-decided
parent fields still allow**.

Two properties make the M0/M1 pair a clean measurement:

  * the output is one of the 17 legal states for any input whatsoever;
  * where the independent argmax is already legal, the projection returns
    exactly that tuple, so M1 differs from M0 only on the rows M0 got wrong
    structurally.

What this is deliberately *not* is the 17-state decoder (M4-M6). This rule is
greedy and top-down: ``promise_status`` is decided first and never revisited, so
a confident ``evidence_quality`` cannot overturn a marginal ``promise_status``.
The joint decoder can. Keeping the two in separate modules is what lets the
results table attribute a difference to exactly that distinction.

The admissible sets are derived from ``labels.STATES`` rather than restated
here, so they cannot drift from the frozen state space.
"""

import numpy as np

from paper.labels import EVAL_FIELDS, FIELDS, LABEL2ID


def _build_admissible() -> dict:
    """Map a prefix of decided labels to what the next field may still take.

    ``()`` -> promise_status options, ``("Yes",)`` -> the timelines legal under
    PS=Yes, and so on. Built by walking the 17 states, so a change to the state
    space propagates here automatically.
    """
    from paper.labels import STATES

    table: dict = {}
    for state in STATES:
        tup = (state.ps, state.vt, state.es, state.eq)
        for depth in range(len(FIELDS)):
            allowed = table.setdefault(tup[:depth], [])
            if tup[depth] not in allowed:
                allowed.append(tup[depth])

    # Canonical order inside each set, so ties break identically everywhere.
    for prefix, allowed in table.items():
        field = FIELDS[len(prefix)]
        allowed.sort(key=lambda lab: LABEL2ID[field][lab])
    return table


ADMISSIBLE = _build_admissible()


def _n_rows(probs) -> int:
    """Row count shared by all four arrays; disagreement is a caller bug."""
    counts = {field: len(probs[field]) for field in FIELDS}
    if len(set(counts.values())) != 1:
        raise ValueError(f"field arrays disagree on row count: {counts}")
    return next(iter(counts.values()))


def _project_row(row_probs) -> dict:
    chosen: tuple = ()
    for field in FIELDS:
        p = row_probs[field]
        allowed = ADMISSIBLE[chosen]
        chosen += (max(allowed, key=lambda lab: p[LABEL2ID[field][lab]]),)
    return dict(zip(FIELDS, chosen))


def project(probs) -> list[dict]:
    """Hierarchy-respecting predictions, one dict of field->label per row.

    ``probs`` maps each field to an ``(N, C)`` array whose columns follow
    ``EVAL_FIELDS[field]``. Ties go to the canonically earlier class, matching
    ``numpy.argmax`` so that M0 and M1 resolve a tie the same way and the
    comparison is never about tie-breaking.
    """
    n = _n_rows(probs)
    return [_project_row({f: probs[f][i] for f in FIELDS}) for i in range(n)]


def independent_argmax(probs) -> list[dict]:
    """M0's rule: each field's argmax, with no regard for the hierarchy.

    It lives beside the projection because the two differ in exactly one
    respect -- whether the parent's decision constrains the child -- and reading
    them side by side is the clearest statement of what M1 adds to M0.
    """
    n = _n_rows(probs)
    return [
        {f: EVAL_FIELDS[f][int(np.argmax(probs[f][i]))] for f in FIELDS}
        for i in range(n)
    ]
