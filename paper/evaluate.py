"""Turn per-row predictions into the contract-3 aggregate summary.

Every number in ``results/*.json`` comes from here, computed from the
predictions CSV and nothing else, so the summary is always reproducible from
the per-row file it accompanies.
"""

from collections import Counter

from sklearn.metrics import f1_score

from paper.labels import EVAL_FIELDS, FIELD_ALIAS, FIELDS, INVALID_STATE_ID
from paper.score import compute_per_field_f1, macro_f1, present_labels

# Conditional metrics: the subset on which each field is actually meaningful.
CONDITIONAL_SUBSETS = {
    "verification_timeline": ("promise_status", "Yes"),
    "evidence_status": ("promise_status", "Yes"),
    "evidence_quality": ("evidence_status", "Yes"),
}


def _columns(records, field):
    alias = FIELD_ALIAS[field]
    return (
        [r[f"gold_{alias}"] for r in records],
        [r[f"pred_{alias}"] for r in records],
    )


def summarise_predictions(records) -> dict:
    """Aggregate metrics for one (protocol, seed, method) prediction set."""
    # The weighted score and the four field scores come from paper/score.py,
    # which owns the official metric; nothing is re-implemented here.
    gold = [{f: r[f"gold_{FIELD_ALIAS[f]}"] for f in FIELDS} for r in records]
    pred = [{f: r[f"pred_{FIELD_ALIAS[f]}"] for f in FIELDS} for r in records]
    field_scores = compute_per_field_f1(gold, pred)
    per_field = {f: field_scores[f] for f in FIELDS}

    per_class_f1, per_class_support = {}, {}
    for field, labels in EVAL_FIELDS.items():
        y_true, y_pred = _columns(records, field)
        present = present_labels(y_true, labels)
        scores = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)
        counts = Counter(y_true)
        per_class_f1[field] = {lab: float(s) for lab, s in zip(present, scores)}
        per_class_support[field] = {lab: counts.get(lab, 0) for lab in labels}

    conditional = {}
    for field, (parent, parent_value) in CONDITIONAL_SUBSETS.items():
        parent_alias = FIELD_ALIAS[parent]
        subset = [r for r in records if r[f"gold_{parent_alias}"] == parent_value]
        key = f"{field}_given_{parent_alias}_{parent_value.lower()}"
        if not subset:
            conditional[key] = None
            continue
        y_true, y_pred = _columns(subset, field)
        conditional[key] = macro_f1(y_true, y_pred, EVAL_FIELDS[field])

    n = len(records)
    return {
        "n_rows": n,
        "weighted_macro_f1": field_scores["overall"],
        "per_field_macro_f1": per_field,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "conditional_f1": conditional,
        "tuple_exact_match": sum(
            r["pred_state_id"] == r["gold_state_id"] for r in records
        ) / n if n else 0.0,
        "invalid_tuple_rate": sum(
            r["pred_state_id"] == INVALID_STATE_ID for r in records
        ) / n if n else 0.0,
    }
