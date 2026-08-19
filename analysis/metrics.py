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

from paper.labels import EVAL_FIELDS, FIELD_ALIAS, FIELD_WEIGHTS, FIELDS, LABEL2ID

N_CLASSES = tuple(len(EVAL_FIELDS[field]) for field in FIELDS)
WEIGHTS = tuple(FIELD_WEIGHTS[field] for field in FIELDS)


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
    gold_counts = np.bincount(g, minlength=n_classes)
    present = gold_counts > 0
    if not present.any():
        return 0.0
    pred_counts = np.bincount(p, minlength=n_classes)
    tp = np.bincount(g[g == p], minlength=n_classes)
    denom = gold_counts + pred_counts
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(n_classes, dtype=float),
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


def weighted_macro_f1(gold, pred, idx=None) -> float:
    """The official primary metric, optionally on a subset or resample."""
    if idx is not None:
        gold, pred = gold[idx], pred[idx]
    return float(sum(
        WEIGHTS[j] * _macro_f1(gold[:, j], pred[:, j], N_CLASSES[j])
        for j in range(len(FIELDS))
    ))
