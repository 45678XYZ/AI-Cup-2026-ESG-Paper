"""Metric-aware class-bias estimation on the Calibration partition (M2/M3/M5/M6).

The bias enters the same score space the decisions are made in:

    s_t,c(x) = log p_t,c(x) + b_t,c

and is chosen to maximise the official metric on held-out rows the model was
not trained on. Two estimators are compared:

* ``global``      every class of every field is estimated on the whole
                  Calibration partition.
* ``conditional`` ``promise_status`` still uses the whole partition, but the
                  child fields use only the rows their parent admits: VT and ES
                  on gold ``PS=Yes``, EQ on gold ``ES=Yes`` (plan §3.2).

**The objective is each field's own macro-F1, evaluated before the output rule
is applied.** That is a deliberate choice, and the reason the results table can
be read as a factorial. One fitted bias set therefore serves both output rules:
M2 and M5 share the global biases, M3 and M6 share the conditional ones, so the
M6-vs-M3 contrast isolates the decoder rather than mixing it with a bias that
was re-fitted for the decoder. Optimising the weighted total *through* the
projection or the decoder would score higher on the calibration partition and
would make that contrast uninterpretable; the cost of not doing it is that a
bias which helps its own field can still be undone downstream, which the paper
states as a limitation rather than hides.

Because each field's macro-F1 depends only on that field's bias here, the four
fields decouple completely, and maximising them separately maximises the
official weighted sum exactly -- the weights only matter where one knob trades
one field against another, and no such knob exists in this space.

Three kinds of coordinate are never estimated:

* classes seen during Train but absent from the objective's own Calibration
  gold, which are fixed at 0.0 and recorded per rotation as
  ``fallback_applied`` (contract §2);
* classes absent from Train itself, which are also fixed at 0.0 but are not a
  calibration fallback -- the model had no supervised example to calibrate;
* the three child-field ``N/A`` classes under conditional estimation, which are
  unidentifiable rather than merely unsupported and are pinned at 0.0 by
  definition (``labels.CONDITIONAL_PINNED_CLASSES``, plan §3.2). Those are
  *not* recorded as fallback: nothing fell back.

Only the Calibration partition may be seen here. ``fit_biases`` checks the ids
it is handed against the split manifest and refuses anything else, so passing
the Test partition raises instead of quietly producing a better-looking number.
"""

import numpy as np

from paper.data import index_by_id
from paper.labels import (
    CONDITIONAL_FREE_CLASSES,
    CONDITIONING_SUBSET,
    EVAL_FIELDS,
    FIELDS,
    LABEL2ID,
)
from paper.methods import log_scores

# The search grid, in log-probability units. +-3 covers a factor of e^3 ~ 20 in
# relative odds, well past the point where a class stops being competitive at
# all; 0.05 is finer than the gap between adjacent decision boundaries on 371
# calibration rows. Both are frozen constants of the protocol, not tuned.
GRID = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.05), 2)
MAX_PASSES = 20

# A strict improvement is required to move a coordinate, so a pass that changes
# nothing ends the search. The tolerance only keeps float noise from looking
# like progress; macro-F1 values are ratios of small integers.
_TOL = 1e-12


def _macro_f1_grid(pred_ids, gold_ids) -> np.ndarray:
    """Macro-F1 of each row of ``pred_ids`` (G, N) against ``gold_ids`` (N,).

    A vectorised restatement of ``paper.score.macro_f1`` -- same present-labels
    -only convention, same ``zero_division=0`` -- because the search evaluates
    it tens of thousands of times per rotation and the scorer takes about a
    millisecond per call. ``tests/test_calibration.py`` asserts the two agree,
    so the objective cannot drift from the metric the paper reports.
    """
    present = np.unique(gold_ids)
    hit = pred_ids[:, None, :] == present[None, :, None]     # (G, K, N)
    is_gold = gold_ids[None, None, :] == present[None, :, None]

    tp = (hit & is_gold).sum(axis=2)
    predicted = hit.sum(axis=2)
    actual = is_gold.sum(axis=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    return f1.mean(axis=1)


def _grid_predictions(scores, bias, klass, grid) -> np.ndarray:
    """(G, N) argmax class ids as coordinate ``klass`` sweeps ``grid``."""
    trial = np.broadcast_to(scores + bias, (len(grid), *scores.shape)).copy()
    trial[:, :, klass] = scores[:, klass] + grid[:, None]
    return np.argmax(trial, axis=2)


def _fit_field(scores, gold_ids, free, grid) -> np.ndarray:
    """Coordinate ascent over one field's free class biases, from all zeros.

    Deterministic by construction: the start point is the uncalibrated rule,
    coordinates are visited in the frozen class order, only a strict
    improvement moves one, and among equally good grid values the smallest
    ``|b|`` wins (then the more negative one). A coordinate therefore stays at
    0.0 unless moving it demonstrably helps.
    """
    bias = np.zeros(scores.shape[1])
    best = _macro_f1_grid(np.argmax(scores, axis=1)[None, :], gold_ids)[0]

    for _ in range(MAX_PASSES):
        improved = False
        for klass in free:
            f1s = _macro_f1_grid(_grid_predictions(scores, bias, klass, grid), gold_ids)
            pick = np.lexsort((grid, np.abs(grid), -f1s))[0]
            if f1s[pick] > best + _TOL:
                bias[klass], best = grid[pick], f1s[pick]
                improved = True
        if not improved:
            break
    return bias


def _subset_mask(field, mode, gold) -> np.ndarray:
    """Rows the objective for ``field`` is evaluated on."""
    n_rows = len(next(iter(gold.values())))
    if mode == "global" or field not in CONDITIONING_SUBSET:
        return np.ones(n_rows, dtype=bool)
    parent, value = CONDITIONING_SUBSET[field]
    return gold[parent] == LABEL2ID[parent][value]


def fit_biases(mode, probs, ids, rotation, rows, grid=GRID) -> tuple[dict, dict]:
    """Estimate one rotation's class biases. Returns ``(biases, fallback)``.

    ``probs`` are the Calibration partition's probabilities, ``ids`` the row
    ids they correspond to, ``rotation`` that rotation's entry in the split
    manifest, ``rows`` the full dev set.

    ``biases`` maps each field to a length-C vector in the score space of
    ``paper.methods.log_scores``; ``fallback`` maps each field to the classes
    fixed at 0.0 for want of any gold example.
    """
    if mode not in ("global", "conditional"):
        raise ValueError(f"unknown calibration mode: {mode!r}")

    expected = rotation["calibration_ids"]
    if list(ids) != list(expected):
        raise ValueError(
            f"biases may only be estimated on rotation {rotation['k']}'s "
            f"Calibration partition ({len(expected)} rows); got {len(ids)} rows "
            "that are not it. Passing the Test partition here would tune the "
            "decision rule on the rows it is scored on."
        )

    by_id = index_by_id(rows)
    gold = {
        field: np.array([LABEL2ID[field][by_id[i][field]] for i in ids])
        for field in FIELDS
    }
    train_gold = {
        field: np.array([
            LABEL2ID[field][by_id[i][field]] for i in rotation["train_ids"]
        ])
        for field in FIELDS
    }
    scores = log_scores(probs)
    for field in FIELDS:
        if len(scores[field]) != len(ids):
            raise ValueError(
                f"{field}: {len(scores[field])} probability rows for {len(ids)} ids"
            )

    biases, fallback = {}, {}
    for field in FIELDS:
        labels = EVAL_FIELDS[field]
        estimable = (labels if mode == "global"
                     else CONDITIONAL_FREE_CLASSES[field])

        mask = _subset_mask(field, mode, gold)
        observed = set(gold[field][mask].tolist())
        train_mask = _subset_mask(field, mode, train_gold)
        trained = set(train_gold[field][train_mask].tolist())
        eligible = [c for c in estimable if LABEL2ID[field][c] in trained]
        absent = [c for c in eligible if LABEL2ID[field][c] not in observed]
        free = [LABEL2ID[field][c] for c in eligible if c not in absent]

        biases[field] = _fit_field(scores[field][mask], gold[field][mask], free, grid)
        if absent:
            fallback[field] = absent

    _check_fallback_matches_manifest(mode, fallback, rotation)
    return biases, fallback


def _check_fallback_matches_manifest(mode, fallback, rotation) -> None:
    """The global fit sees the whole partition, so its absent classes must be
    exactly the ones the split manifest recorded. A disagreement means the
    manifest and the data are not the same vintage -- the sort of thing that
    otherwise surfaces as a slightly different number."""
    if mode != "global":
        return
    recorded = {f: sorted(cs) for f, cs in rotation["calibration_absent_classes"].items()}
    found = {f: sorted(cs) for f, cs in fallback.items()}
    if recorded != found:
        raise ValueError(
            f"rotation {rotation['k']}: the split manifest records absent "
            f"calibration classes {recorded}, the data give {found}"
        )


def as_json(biases) -> dict:
    """``{field: {class: bias}}`` for the results file's decision_params."""
    return {
        field: dict(zip(EVAL_FIELDS[field], (round(float(b), 4) for b in vec)))
        for field, vec in biases.items()
    }
