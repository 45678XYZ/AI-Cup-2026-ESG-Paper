"""The post-run summaries for the three additional ML-Promise languages."""

import json
import math

import pytest

from analysis.multilingual_replication import _contrast, build_report
from paper.data import REPO_ROOT

# Floats are compared with a tolerance; everything else exactly.
#
# ``==`` on the whole nested structure made the reproducibility check depend on
# summation order, which follows the BLAS build and the CPU rather than the
# code: it passed where the runs were produced and failed on other machines,
# with every leaf differing in its last bits (~1e-16 relative).
#
# The tolerance is seven orders of magnitude above that noise and still far
# below any change the check exists to catch -- a stale summary moves numbers
# in the third or fourth decimal, not the sixteenth. Structure stays exact, so
# a renamed key, a dropped contrast or a changed type still fails.
REL_TOL = 1e-9
ABS_TOL = 1e-12


def _mismatches(actual, expected, path=""):
    """Every place the two structures disagree, floats up to a tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            yield f"{path}: expected object, got {type(actual).__name__}"
            return
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                yield f"{path}/{key}: missing from the recomputed report"
            elif key not in expected:
                yield f"{path}/{key}: absent from the checked-in summary"
            else:
                yield from _mismatches(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            yield f"{path}: expected list, got {type(actual).__name__}"
        elif len(actual) != len(expected):
            yield f"{path}: length {len(actual)} against {len(expected)}"
        else:
            for i, (a, e) in enumerate(zip(actual, expected)):
                yield from _mismatches(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) or isinstance(actual, float):
        # Both sides are checked: a value that turned into a string on either
        # side is a mismatch to report, not a TypeError out of ``isclose``.
        # ``bool`` is excluded deliberately -- it is an ``int`` subclass, and
        # ``True == 1.0`` would let a flag silently become a number.
        pair = [v for v in (actual, expected)
                if isinstance(v, bool) or not isinstance(v, (int, float))]
        if pair:
            yield f"{path}: expected numbers, got {actual!r} against {expected!r}"
        elif not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            yield f"{path}: {actual!r} against {expected!r}"
    elif actual != expected:
        yield f"{path}: {actual!r} against {expected!r}"

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

    diffs = list(_mismatches(report, checked_in))
    assert not diffs, (
        f"{len(diffs)} difference(s) between the recomputed report and "
        f"runs_{language}/summary.json; first few: {diffs[:5]}"
    )
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
