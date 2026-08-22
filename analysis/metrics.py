"""Subset-aware weighted macro-F1, fast enough for 10,000 resamples.

``paper/score.py`` stays the authority on the official metric. This module is a
vectorised restatement of it, and its correctness is not argued but pinned:
``tests/test_analysis_metrics.py`` asserts exact agreement on the full 2,000
rows and on 50 random subsets, including the resampling-with-replacement case
the bootstrap actually produces.

Why it is needed at all: the paired PDF-cluster bootstrap evaluates the metric
10,000 times per contrast per seed. ``sklearn.metrics.f1_score`` rebuilds label
bookkeeping on every call, which turns a five-contrast run into tens of
minutes; the bincount formulation below runs it in about a second.
"""

import numpy as np

from paper.labels import (
    CONDITIONING_SUBSET,
    EVAL_FIELDS,
    FIELD_ALIAS,
    FIELD_WEIGHTS,
    FIELDS,
    LABEL2ID,
)

N_CLASSES = tuple(len(EVAL_FIELDS[field]) for field in FIELDS)
WEIGHTS = tuple(FIELD_WEIGHTS[field] for field in FIELDS)

# Ancestor conditions of the task hierarchy, for the path-constrained metrics
# below. ``verification_timeline`` and ``evidence_status`` are admitted only by
# a predicted ``promise_status = Yes``; ``evidence_quality`` additionally needs
# a predicted ``evidence_status = Yes``.
_PS, _VT, _ES, _EQ = range(len(FIELDS))
_PS_YES = LABEL2ID["promise_status"]["Yes"]
_ES_YES = LABEL2ID["evidence_status"]["Yes"]
_NA = tuple(LABEL2ID[field].get("N/A") for field in FIELDS)


def encode(records, order=None):
    """``(N, 4)`` gold and predicted class-id arrays, column order ``FIELDS``.

    ``order`` reindexes the records by ``id`` string. Reindexing by id rather
    than trusting file order is contract section 1.1: the 42 prediction files
    each run in their own rotation order, and aligning them by position would
    silently pair different paragraphs.
    """
    if order is not None:
        by_id = {r["id"]: r for r in records}
        missing = [i for i in order if i not in by_id]
        if missing:
            raise KeyError(f"{len(missing)} ids absent, e.g. {missing[:3]}")
        records = [by_id[i] for i in order]

    gold = np.empty((len(records), len(FIELDS)), dtype=np.int64)
    pred = np.empty_like(gold)
    for j, field in enumerate(FIELDS):
        alias, lookup = FIELD_ALIAS[field], LABEL2ID[field]
        gold[:, j] = [lookup[r[f"gold_{alias}"]] for r in records]
        pred[:, j] = [lookup[r[f"pred_{alias}"]] for r in records]
    return gold, pred


def _macro_f1(g, p, n_classes) -> float:
    """Macro-F1 over the classes present in ``g``.

    Uses the identity ``2*tp / (support_gold + support_pred)``, which equals
    ``2*tp / (2*tp + fp + fn)``. Restricting the mean to classes present in the
    gold is the competition convention implemented by ``paper.score.macro_f1``;
    it matters here because a bootstrap resample routinely drops ``Misleading``
    (n=2) entirely, changing which classes are averaged.
    """
    # One slot past the real classes holds the path-constrained metrics'
    # sentinel. It never occurs in the gold, so it is never averaged over and
    # never contributes to another class's denominator: a field masked to it
    # simply loses its true positive, which is what "counted as a false
    # prediction" means. On sentinel-free input the extra slot is all zeros and
    # the result is bit-for-bit the official metric.
    width = n_classes + 1
    gold_counts = np.bincount(g, minlength=width)
    present = gold_counts > 0
    if not present.any():
        return 0.0
    pred_counts = np.bincount(p, minlength=width)
    tp = np.bincount(g[g == p], minlength=width)
    denom = gold_counts + pred_counts
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(width, dtype=float),
                   where=denom > 0)
    return float(f1[present].mean())


def field_macro_f1(gold, pred, idx=None) -> dict:
    """Per-field macro-F1, keyed by field name."""
    if idx is not None:
        gold, pred = gold[idx], pred[idx]
    return {
        field: _macro_f1(gold[:, j], pred[:, j], N_CLASSES[j])
        for j, field in enumerate(FIELDS)
    }


def conditional_field_macro_f1(gold, pred, idx=None) -> dict:
    """Per-field macro-F1 on the rows each field's parent admits (plan §4.5).

    ``verification_timeline`` and ``evidence_status`` are scored on gold
    ``promise_status = Yes``; ``evidence_quality`` on gold
    ``evidence_status = Yes``; ``promise_status`` has no parent and is scored on
    everything. The subsets come from ``paper.labels.CONDITIONING_SUBSET``, the
    same table the conditional calibration uses, so the two cannot drift apart.

    The point of reporting it beside the unconditioned score: a child field's
    unconditioned macro-F1 mixes two very different questions -- can the model
    choose the right timeline, and can it repeat the ``N/A`` the hierarchy
    already fixes -- and the second is easy for every method. Conditioning
    removes it. **The subset is chosen by the gold parent, never the predicted
    one**, so the rows scored are identical for all seven methods and the
    numbers stay comparable.
    """
    if idx is not None:
        gold, pred = gold[idx], pred[idx]
    out = {}
    for j, field in enumerate(FIELDS):
        keep = slice(None)
        if field in CONDITIONING_SUBSET:
            parent, value = CONDITIONING_SUBSET[field]
            keep = gold[:, FIELDS.index(parent)] == LABEL2ID[parent][value]
        out[field] = _macro_f1(gold[keep, j], pred[keep, j], N_CLASSES[j])
    return out


def weighted_macro_f1(gold, pred, idx=None) -> float:
    """The official primary metric, optionally on a subset or resample."""
    if idx is not None:
        gold, pred = gold[idx], pred[idx]
    return float(sum(
        WEIGHTS[j] * _macro_f1(gold[:, j], pred[:, j], N_CLASSES[j])
        for j in range(len(FIELDS))
    ))


def tuple_accuracy(gold, pred, idx=None) -> float:
    """Rows where all four fields are correct.

    The complement of the official metric on a hierarchical task: weighted
    macro-F1 scores each field independently and therefore rewards a prediction
    that is right about three fields while being a combination the label space
    forbids. This one only counts a row when the whole tuple is right, so an
    illegal prediction scores zero by construction rather than by penalty.
    """
    if idx is not None:
        gold, pred = gold[idx], pred[idx]
    return float(np.all(gold == pred, axis=1).mean())


def enforce_ancestors(pred) -> np.ndarray:
    """``pred`` with every ancestor-unsupported field replaced by a sentinel.

    The masking rule of the hierarchical-classification literature's
    path-constrained (C-) metrics: *a node predicted as true counts as a valid
    prediction if and only if all its ancestors are also predicted as true;
    otherwise it counts as a false prediction.* Here the sentinel is the class
    id ``N_CLASSES[j]``, one past the last real class of field ``j``. It cannot
    occur in the gold, so scoring it is exactly scoring a miss.

    ``N/A`` is left alone: predicting "no timeline" under "no promise" is the
    hierarchy agreeing with itself, not a claim its ancestors fail to support.
    On any of the 17 legal states this function is the identity, which is why
    a method that guarantees legal output scores the same under both metrics.
    """
    pred = np.asarray(pred)
    out = pred.copy()
    ps_ok = pred[:, _PS] == _PS_YES
    for j in (_VT, _ES):
        out[~ps_ok & (pred[:, j] != _NA[j]), j] = N_CLASSES[j]
    es_ok = ps_ok & (pred[:, _ES] == _ES_YES)
    out[~es_ok & (pred[:, _EQ] != _NA[_EQ]), _EQ] = N_CLASSES[_EQ]
    return out


def consistent_weighted_macro_f1(gold, pred, idx=None) -> float:
    """The official metric's own path-constrained variant.

    Identical to ``weighted_macro_f1`` in every respect -- same four fields,
    same macro average over the classes present in the gold, same task weights
    -- except that a field whose ancestors were not predicted counts as a false
    prediction rather than earning partial credit. The two therefore differ by
    exactly one factor, which is what makes the comparison between them an
    argument about consistency and not about the shape of the metric.

    Masking is row-wise, so masking before subsetting and subsetting before
    masking give the same answer; doing it first keeps ``idx`` meaning the same
    thing it means everywhere else.
    """
    return weighted_macro_f1(gold, enforce_ancestors(pred), idx)
