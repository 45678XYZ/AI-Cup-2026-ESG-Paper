import numpy as np
import pytest

from paper.calibration import apply_biases, fit_biases
from paper.labels import CONDITIONAL_PINNED_CLASSES, EVAL_FIELDS, FIELDS


def _uniform(n):
    return {
        field: np.full((n, len(labels)), 1 / len(labels), dtype=np.float32)
        for field, labels in EVAL_FIELDS.items()
    }


def _rows():
    return [
        {"promise_status": "Yes", "verification_timeline": "already",
         "evidence_status": "Yes", "evidence_quality": "Clear"},
        {"promise_status": "No", "verification_timeline": "N/A",
         "evidence_status": "N/A", "evidence_quality": "N/A"},
        {"promise_status": "Yes", "verification_timeline": "within_2_years",
         "evidence_status": "No", "evidence_quality": "N/A"},
        {"promise_status": "Yes", "verification_timeline": "already",
         "evidence_status": "Yes", "evidence_quality": "Not Clear"},
    ]


def test_calibration_api_rejects_test_labels():
    with pytest.raises(ValueError, match="only.*calibration"):
        fit_biases(_uniform(4), _rows(), partition="test", mode="global")


def test_conditional_structural_na_biases_are_exactly_zero():
    biases, meta = fit_biases(
        _uniform(4), _rows(), partition="calibration", mode="conditional",
    )
    for field, labels in CONDITIONAL_PINNED_CLASSES.items():
        for label in labels:
            assert biases[field][label] == 0.0
    assert meta["structural_pinned"] == CONDITIONAL_PINNED_CLASSES


def test_data_absent_classes_take_the_documented_zero_fallback():
    biases, meta = fit_biases(
        _uniform(4), _rows(), partition="calibration", mode="global",
    )
    assert biases["evidence_quality"]["Misleading"] == 0.0
    assert "Misleading" in meta["fallback_applied"]["evidence_quality"]


def test_applying_biases_preserves_shapes_and_moves_scores():
    probs = _uniform(4)
    biases = {field: {label: 0.0 for label in labels}
              for field, labels in EVAL_FIELDS.items()}
    biases["promise_status"]["No"] = 1.0
    scores = apply_biases(probs, biases)
    assert set(scores) == set(FIELDS)
    assert scores["promise_status"].shape == probs["promise_status"].shape
    assert np.all(scores["promise_status"][:, 1] > scores["promise_status"][:, 0])
