"""Cross-seed aggregation and the pre-specified contrasts."""

import numpy as np
import pytest

from analysis.aggregate import (
    CONTRASTS,
    mean_std,
    misleading_free_index,
    protocol_summary,
)
from analysis.load import EXAMPLES_ROOT, METHODS, pdf_clusters
from paper.data import canonical_row_order, load_dev

DEV = load_dev()
ORDER = canonical_row_order(DEV)
CLUSTERS = pdf_clusters(ORDER, DEV)
SEEDS = (42, 123, 456)

# n_boot is small here on purpose: this file tests structure, not intervals.
SUMMARY = protocol_summary("pdf_group", ORDER, EXAMPLES_ROOT, CLUSTERS,
                           n_boot=200, seeds=SEEDS, dev=DEV)


def test_the_five_contrasts_are_exactly_the_pre_specified_ones():
    assert [(a, b) for a, b, _ in CONTRASTS] == [
        ("M1", "M0"), ("M3", "M2"), ("M4", "M1"), ("M6", "M3"), ("M6", "M5"),
    ]


def test_mean_std_uses_the_sample_convention():
    mean, std = mean_std([0.1, 0.2, 0.3])
    assert mean == pytest.approx(0.2)
    assert std == pytest.approx(0.1)  # ddof=1, not the population 0.0816


def test_every_method_gets_a_summary_row():
    assert set(SUMMARY["methods"]) == set(METHODS)
    row = SUMMARY["methods"]["M3"]
    assert set(row) >= {
        "weighted_macro_f1_mean", "weighted_macro_f1_std",
        "per_field_mean", "tuple_exact_match_mean", "invalid_tuple_rate_mean",
    }


def test_per_seed_scores_are_kept_not_just_the_mean():
    # Table 2 reports mean+/-std over seeds, and the plan forbids averaging
    # per-fold scores; keeping the three seed scores is what lets that be
    # checked after the fact.
    assert len(SUMMARY["methods"]["M3"]["weighted_macro_f1_per_seed"]) == 3


def test_only_m0_is_allowed_to_emit_invalid_tuples():
    for method in METHODS:
        rate = SUMMARY["methods"][method]["invalid_tuple_rate_mean"]
        if method == "M0":
            assert rate > 0
        else:
            assert rate == 0.0


def test_all_five_contrasts_carry_an_interval_and_an_adjusted_p():
    assert len(SUMMARY["contrasts"]) == 5
    for row in SUMMARY["contrasts"].values():
        assert row["ci_low"] <= row["delta"] <= row["ci_high"]
        assert 0.0 < row["p_value"] <= 1.0
        assert row["p_holm"] >= row["p_value"]
        assert row["description"]


def test_sensitivity_drops_exactly_the_two_misleading_rows():
    idx = misleading_free_index(ORDER, DEV)
    assert len(idx) == len(ORDER) - 2


def test_sensitivity_scores_are_reported_beside_the_main_ones():
    row = SUMMARY["methods"]["M3"]
    assert "weighted_macro_f1_mean_no_misleading" in row
    # Dropping 2 of 2,000 rows cannot move the score much, but it must move
    # something: EQ's macro-F1 loses a class from the average.
    assert row["weighted_macro_f1_mean_no_misleading"] != row["weighted_macro_f1_mean"]


def test_per_field_means_cover_all_four_fields():
    per_field = SUMMARY["methods"]["M3"]["per_field_mean"]
    assert set(per_field) == {
        "promise_status", "verification_timeline",
        "evidence_status", "evidence_quality",
    }
    assert all(0.0 <= v <= 1.0 for v in per_field.values())


# --------------------------------------------- two Holm families, not one
# The five contrasts are pre-specified once and evaluated on two metrics that
# answer different questions: weighted macro-F1 is the competition's ranking
# rule, its path-constrained variant is the same metric with ancestor-
# unsupported predictions counted as misses. Correcting them as one family of
# ten would treat them as ten shots at the same question, which they are not;
# correcting each family of five keeps the pre-specified structure intact and
# is stated as such in the caption.


FAMILIES = ("contrasts", "consistent_contrasts", "tuple_contrasts",
            "hierarchical_contrasts")


def test_summary_carries_every_contrast_family():
    """Three metrics, one pre-specified set of five contrasts. tuple accuracy
    keeps its family because the analysis plan named it: retiring it after
    seeing the numbers would drop M4-M1, a result that runs against us."""
    out = protocol_summary("pdf_group", ORDER, EXAMPLES_ROOT, CLUSTERS,
                           n_boot=120, dev=DEV)
    for family in FAMILIES:
        assert set(out[family]) == set(out["contrasts"])
        assert len(out[family]) == 5


def test_each_family_is_corrected_over_five_hypotheses_not_ten():
    out = protocol_summary("pdf_group", ORDER, EXAMPLES_ROOT, CLUSTERS,
                           n_boot=120, dev=DEV)
    for family in FAMILIES:
        rows = out[family]
        raw = {k: r["p_value"] for k, r in rows.items()}
        smallest = min(raw.values())
        # Holm 對最小的 p 乘以家族大小；家族是 5 就不能是 10
        assert rows[min(raw, key=raw.get)]["p_holm"] <= min(1.0, 5 * smallest) + 1e-12


def test_the_secondary_family_measures_a_different_thing():
    """If both families produced the same deltas, one of them is wired wrong."""
    out = protocol_summary("pdf_group", ORDER, EXAMPLES_ROOT, CLUSTERS,
                           n_boot=120, dev=DEV)
    primary = {k: r["delta"] for k, r in out["contrasts"].items()}
    secondary = {k: r["delta"] for k, r in out["consistent_contrasts"].items()}
    assert primary != secondary


def test_summary_carries_the_path_constrained_score_per_method():
    """The property that makes the secondary metric self-checking: a method that
    guarantees legal output scores the same under both, so only M0 may move."""
    out = protocol_summary("pdf_group", ORDER, EXAMPLES_ROOT, CLUSTERS,
                           n_boot=120, dev=DEV)
    methods = out["methods"]
    for method, row in methods.items():
        official = row["weighted_macro_f1_mean"]
        constrained = row["consistent_weighted_macro_f1_mean"]
        if method == "M0":
            assert constrained < official
        else:
            assert constrained == pytest.approx(official, abs=1e-12), method


def test_summary_carries_the_conditional_field_scores():
    """Plan section 4.5 lists conditional F1 among the secondary metrics: each
    child field scored only on the rows its gold parent admits."""
    out = protocol_summary("pdf_group", ORDER, EXAMPLES_ROOT, CLUSTERS,
                           n_boot=120, dev=DEV)
    for method, row in out["methods"].items():
        cond = row["conditional_per_field_mean"]
        assert set(cond) == set(row["per_field_mean"]), method
        # the root field has no parent, so conditioning cannot move it
        assert cond["promise_status"] == pytest.approx(
            row["per_field_mean"]["promise_status"], abs=1e-12), method
        assert cond["evidence_quality"] != pytest.approx(
            row["per_field_mean"]["evidence_quality"], abs=1e-6), method


def test_summary_carries_the_hierarchical_metric_per_method():
    """The metric a hierarchical-classification reviewer asks about first. It
    ranks the methods differently from the official one, which is the point."""
    out = protocol_summary("pdf_group", ORDER, EXAMPLES_ROOT, CLUSTERS,
                           n_boot=120, dev=DEV)
    for method, row in out["methods"].items():
        h = row["hierarchical_mean"]
        assert set(h) == {"hP", "hR", "hF"}, method
        assert all(0.0 <= v <= 1.0 for v in h.values()), method
