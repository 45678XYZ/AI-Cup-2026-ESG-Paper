"""The post-run summaries for the three additional ML-Promise languages."""

import json

import pytest

from analysis.multilingual_replication import _contrast, build_report
from paper.data import REPO_ROOT


def test_contrasts_are_paired_by_seed():
    def method(values):
        return {"per_seed": [
            {
                "seed": seed,
                "weighted_macro_f1": value,
                "tuple_accuracy": value / 2,
                "invalid_tuple_rate": 0.0,
            }
            for seed, value in zip((42, 123, 456), values)
        ]}

    out = _contrast({"M1": method((0.4, 0.6, 0.8)),
                     "M0": method((0.3, 0.4, 0.5))}, "M1", "M0")
    assert out["weighted_macro_f1"]["per_seed_delta"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert out["weighted_macro_f1"]["mean_delta"] == pytest.approx(0.2)


@pytest.mark.parametrize("language,rows,reports", [
    ("fr", 400, 9),
    ("ja", 400, 19),
    ("ko", 500, 32),
])
def test_checked_in_summary_is_reproducible(language, rows, reports):
    report = build_report(f"mlpromise_{language}")
    with open(REPO_ROOT / f"runs_{language}" / "summary.json", encoding="utf-8") as f:
        checked_in = json.load(f)

    assert report == checked_in
    assert report["design"]["n_rows"] == rows
    assert report["design"]["n_report_clusters"] == reports
    assert report["artifact_counts"] == {
        "probability_bundles": 150,
        "prediction_files": 210,
        "result_files": 210,
    }
    assert report["observations"]["m0_invalid_positive_in_every_arm"]
    assert report["observations"]["m1_to_m6_max_invalid_tuple_rate"] == 0.0
    assert report["observations"]["m1_minus_m0_tuple_nonnegative_arm_count"] == 8
