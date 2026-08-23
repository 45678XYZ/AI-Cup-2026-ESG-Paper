"""The structural-arm comparison keeps the two training arms paired."""

import pytest

from analysis.load import EXAMPLES_ROOT
from analysis.structural_arm import compare_arms


@pytest.fixture(scope="module")
def self_comparison():
    return compare_arms(
        EXAMPLES_ROOT, EXAMPLES_ROOT, n_boot=100, bootstrap_seed=7,
    )


def test_identical_arms_have_no_h1_change(self_comparison):
    for row in self_comparison["protocols"].values():
        h1 = row["h1_invalid_tuple_rate"]
        assert h1["delta"] == 0.0
        assert not h1["lower_in_every_seed"]


def test_identical_arms_have_no_h2_change(self_comparison):
    h2 = self_comparison["h2"]
    assert h2["delta"] == 0.0
    assert h2["ci_low"] == 0.0
    assert h2["ci_high"] == 0.0
    assert h2["p_value"] == 1.0
    assert not h2["supported"]


def test_full_two_by_seven_design_is_retained(self_comparison):
    for row in self_comparison["protocols"].values():
        methods = row["all_method_weighted_macro_f1"]
        assert set(methods) == {f"M{i}" for i in range(7)}
        assert all(method["delta"] == pytest.approx(0.0) for method in methods.values())


def test_named_safety_classes_are_explicit(self_comparison):
    named = self_comparison["safety_check"]["named_rare_classes"]
    assert set(named) == {
        "verification_timeline:within_2_years",
        "evidence_quality:Not Clear",
    }
    assert all(row["delta"] == pytest.approx(0.0) for row in named.values())
