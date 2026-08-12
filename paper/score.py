"""The official AI CUP weighted macro-F1.

The single implementation source for the primary metric; no other module may
re-implement it.

Note the competition convention preserved here: only labels that actually occur
in the ground truth are scored. Folds differing in which rare classes are
present therefore produce scores that are not directly comparable, which is why
per-fold F1 must never be averaged.
"""

from sklearn.metrics import f1_score

from paper.labels import EVAL_FIELDS, FIELD_WEIGHTS


def present_labels(y_true, labels):
    """The subset of ``labels`` that actually occurs in the gold."""
    seen = set(y_true)
    return [l for l in labels if l in seen]


def macro_f1(y_true, y_pred, labels) -> float:
    """Macro-F1 under the competition's present-labels-only convention.

    The one place that convention is implemented. Anything computing a
    field-level or subset-level macro-F1 calls this rather than reaching for
    ``f1_score`` directly, so the convention cannot diverge between modules.
    """
    present = present_labels(y_true, labels)
    if not present:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def compute_weighted_macro_f1(gt_data, pred_data) -> float:
    """gt_data, pred_data: lists of dicts keyed by the four EVAL_FIELDS."""
    return sum(
        macro_f1([x[f] for x in gt_data], [x[f] for x in pred_data], labels) * FIELD_WEIGHTS[f]
        for f, labels in EVAL_FIELDS.items()
    )


def compute_per_field_f1(gt_data, pred_data) -> dict:
    """Per-field macro-F1 plus the weighted overall score."""
    results = {
        f: macro_f1([x[f] for x in gt_data], [x[f] for x in pred_data], labels)
        for f, labels in EVAL_FIELDS.items()
    }
    results["overall"] = sum(results[f] * FIELD_WEIGHTS[f] for f in EVAL_FIELDS)
    return results
