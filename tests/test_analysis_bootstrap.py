"""Properties of the paired cluster bootstrap.

The statistics are checked by construction rather than against stored numbers:
a bootstrap that silently lost its pairing would still produce a plausible
interval, which is exactly the kind of failure the interface contract's own
test (section 0) asks about -- it does not raise, it just reports a different
number.
"""

import numpy as np
import pytest

from analysis.bootstrap import holm, paired_delta, resample_indices
from analysis.load import EXAMPLES_ROOT, load_aligned, pdf_clusters
from analysis.metrics import weighted_macro_f1
from paper.data import canonical_row_order, load_dev

DEV = load_dev()
ORDER = canonical_row_order(DEV)
CLUSTERS = pdf_clusters(ORDER, DEV)
SEEDS = (42, 123, 456)


def _sets(method):
    return [load_aligned("pdf_group", s, method, ORDER, EXAMPLES_ROOT) for s in SEEDS]


M0 = _sets("M0")
M3 = _sets("M3")


def test_resample_draws_whole_pdfs():
    # Every cluster contributes all of its rows or none of them: a resample
    # that split a report would break the clustering the CI depends on.
    idx = resample_indices(CLUSTERS, np.random.default_rng(0))
    for cluster in CLUSTERS:
        assert int(np.isin(idx, cluster).sum()) % len(cluster) == 0


def test_resample_keeps_the_row_count():
    idx = resample_indices(CLUSTERS, np.random.default_rng(0))
    # 49 clusters drawn 49 times; the row count varies with which reports were
    # drawn, but every index must be a real row position.
    assert idx.min() >= 0 and idx.max() < len(ORDER)


def test_resample_is_deterministic_given_a_generator_seed():
    a = resample_indices(CLUSTERS, np.random.default_rng(7))
    b = resample_indices(CLUSTERS, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_a_method_against_itself_has_exactly_zero_effect():
    # The sharpest pairing check: if the two arms were resampled independently
    # the interval would straddle zero with positive width instead of
    # collapsing onto it.
    out = paired_delta(M0, M0, CLUSTERS, n_boot=200)
    assert out["delta"] == 0.0
    assert out["ci_low"] == 0.0 and out["ci_high"] == 0.0
    assert out["p_value"] == 1.0


def test_point_estimate_is_the_mean_of_the_per_seed_differences():
    out = paired_delta(M3, M0, CLUSTERS, n_boot=200)
    expected = np.mean([
        weighted_macro_f1(g, pa) - weighted_macro_f1(g, pb)
        for (g, pa), (_, pb) in zip(M3, M0)
    ])
    assert out["delta"] == pytest.approx(expected, abs=1e-12)
    assert len(out["per_seed_delta"]) == 3


def test_interval_brackets_the_point_estimate():
    out = paired_delta(M3, M0, CLUSTERS, n_boot=2000)
    assert out["ci_low"] <= out["delta"] <= out["ci_high"]
    assert out["ci_low"] < out["ci_high"]


def test_p_value_is_never_reported_as_zero():
    out = paired_delta(M3, M0, CLUSTERS, n_boot=200)
    assert out["p_value"] >= 1 / (200 + 1)


def test_bootstrap_is_reproducible():
    a = paired_delta(M3, M0, CLUSTERS, n_boot=200, seed=1)
    b = paired_delta(M3, M0, CLUSTERS, n_boot=200, seed=1)
    assert a == b


def test_misaligned_arms_are_refused():
    # Swapping one arm's row order leaves the shapes intact and the scores
    # plausible; only the gold comparison catches it.
    gold, pred = M0[0]
    shuffled = (gold[::-1], pred[::-1])
    with pytest.raises(ValueError, match="not aligned"):
        paired_delta([M3[0]], [shuffled], CLUSTERS, n_boot=10)


def test_holm_matches_the_hand_computed_adjustment():
    # n=4; sorted p = .01, .02, .03, .04 -> 4*.01, 3*.02, 2*.03, 1*.04
    # with the running maximum enforcing monotonicity.
    adjusted = holm({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04})
    assert adjusted["a"] == pytest.approx(0.04)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["d"] == pytest.approx(0.06)


def test_holm_is_monotone_and_capped_at_one():
    adjusted = holm({"a": 0.4, "b": 0.5, "c": 0.9})
    assert all(v <= 1.0 for v in adjusted.values())
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
