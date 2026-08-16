"""Calibration-only metric-aware class-bias optimisation.

The fitted values are decision parameters, not probability calibration.  For
each field we add a class-specific constant to ``log(p)`` and choose constants
that maximise that field's macro-F1 on the Calibration partition.  No Test
labels are accepted by this API.

Global calibration uses every Calibration row.  Conditional calibration uses
the gold-parent subsets frozen in :mod:`paper.labels`; structurally impossible
child ``N/A`` parameters and data-absent classes stay exactly zero.
"""

from collections import Counter

import numpy as np

from paper.labels import (
    CONDITIONAL_FREE_CLASSES,
    CONDITIONAL_PINNED_CLASSES,
    CONDITIONING_SUBSET,
    EVAL_FIELDS,
    FIELDS,
)
from paper.score import macro_f1

BIAS_MIN = -5.0
BIAS_MAX = 5.0
BIAS_STEP = 0.25
MAX_PASSES = 10
_EPS = 1e-12


def log_scores(probs) -> dict[str, np.ndarray]:
    """Return finite log probabilities without changing argmax decisions."""
    return {
        field: np.log(np.clip(np.asarray(probs[field], dtype=float), _EPS, 1.0))
        for field in FIELDS
    }


def apply_biases(probs, biases) -> dict[str, np.ndarray]:
    """Convert probabilities to ``log(p) + b`` score arrays."""
    scores = log_scores(probs)
    for field, labels in EVAL_FIELDS.items():
        values = np.asarray([biases[field][label] for label in labels], dtype=float)
        scores[field] = scores[field] + values
    return scores


def _predicted_labels(scores, labels):
    indices = np.argmax(scores, axis=1)
    return [labels[int(i)] for i in indices]


def _fit_field(scores, gold, labels, movable):
    """Deterministic coordinate ascent over the frozen bias grid."""
    biases = np.zeros(len(labels), dtype=float)
    movable_indices = [labels.index(label) for label in movable]
    grid = np.arange(BIAS_MIN, BIAS_MAX + BIAS_STEP / 2, BIAS_STEP)

    def objective(candidate):
        pred = _predicted_labels(scores + candidate, labels)
        return macro_f1(gold, pred, labels)

    best = objective(biases)
    for _ in range(MAX_PASSES):
        improved = False
        for index in movable_indices:
            current = biases[index]
            candidate_best, value_best = best, current
            for value in grid:
                trial = biases.copy()
                trial[index] = value
                score = objective(trial)
                if score > candidate_best + 1e-12:
                    candidate_best, value_best = score, float(value)
                elif abs(score - candidate_best) <= 1e-12:
                    # Keep the current point when it lies on the same plateau;
                    # otherwise use the smallest move, then the smaller value.
                    old_key = (abs(value_best - current), abs(value_best), value_best)
                    new_key = (abs(float(value) - current), abs(float(value)), float(value))
                    if new_key < old_key:
                        value_best = float(value)
            if candidate_best > best + 1e-12:
                biases[index] = value_best
                best = candidate_best
                improved = True
        if not improved:
            break
    return {label: float(biases[i]) for i, label in enumerate(labels)}, best


def fit_biases(probs, rows, *, partition, mode):
    """Fit global or conditional biases from Calibration labels only.

    ``partition`` is deliberately mandatory.  Passing ``"test"`` (or any
    other value) raises before labels are inspected, making the protocol
    boundary an executable invariant rather than a calling convention.
    """
    if partition != "calibration":
        raise ValueError("class biases may only be fitted on the calibration partition")
    if mode not in ("global", "conditional"):
        raise ValueError(f"unknown calibration mode: {mode!r}")
    counts = {field: len(np.asarray(probs[field])) for field in FIELDS}
    if set(counts.values()) != {len(rows)}:
        raise ValueError(f"probability rows do not align with calibration labels: {counts}")

    all_scores = log_scores(probs)
    fitted, objectives, absent = {}, {}, {}
    for field, labels in EVAL_FIELDS.items():
        if mode == "conditional" and field in CONDITIONING_SUBSET:
            parent, value = CONDITIONING_SUBSET[field]
            mask = np.asarray([row[parent] == value for row in rows], dtype=bool)
            free = CONDITIONAL_FREE_CLASSES[field]
        else:
            mask = np.ones(len(rows), dtype=bool)
            free = labels

        gold = [row[field] for row, keep in zip(rows, mask) if keep]
        support = Counter(gold)
        data_absent = [label for label in free if support[label] == 0]
        movable = [label for label in free if support[label] > 0]
        field_biases, objective = _fit_field(all_scores[field][mask], gold, labels, movable)

        # Both kinds of pinned parameter are explicit and exact, not merely
        # values the optimiser happened to leave near zero.
        pinned = list(CONDITIONAL_PINNED_CLASSES[field]) if mode == "conditional" else []
        for label in pinned + data_absent:
            field_biases[label] = 0.0

        fitted[field] = field_biases
        objectives[field] = objective
        if data_absent:
            absent[field] = data_absent

    return fitted, {
        "mode": mode,
        "objective": "per-field macro_f1",
        "grid": {"min": BIAS_MIN, "max": BIAS_MAX, "step": BIAS_STEP},
        "max_passes": MAX_PASSES,
        "calibration_objective": objectives,
        "fallback_applied": absent,
        "structural_pinned": (
            CONDITIONAL_PINNED_CLASSES if mode == "conditional" else {}
        ),
    }
