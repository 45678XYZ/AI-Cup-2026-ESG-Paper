"""Architecture-screen comparisons preserve pairing and contrast labels."""

import pytest

from analysis.architecture_screen import compare_architecture
from analysis.load import EXAMPLES_ROOT


@pytest.fixture(scope="module")
def self_comparison():
    return compare_architecture(
        EXAMPLES_ROOT, EXAMPLES_ROOT, EXAMPLES_ROOT,
        n_boot=100, bootstrap_seed=7,
    )


def test_identical_screen_does_not_pass_seed42_gate(self_comparison):
    gate = self_comparison["seed42_expansion_gate"]
    assert not gate["invalid_rate_lower"]
    assert not gate["m1_weighted_macro_f1_higher"]
    assert not gate["passed"]


def test_identical_arms_have_zero_paired_effect(self_comparison):
    effect = self_comparison["structural_effect_m1"]
    assert effect["delta"] == 0.0
    assert effect["ci_low"] == 0.0
    assert effect["ci_high"] == 0.0
    assert effect["p_value"] == 1.0
    assert not effect["supported"]


def test_backbone_difference_is_separately_labelled(self_comparison):
    contrast = self_comparison["backbone_total_difference_m1"]
    assert contrast["delta"] == 0.0
    assert contrast["interpretation"] == "total_backbone_difference_not_structural_effect"


def test_all_methods_and_named_safety_classes_are_retained(self_comparison):
    methods = self_comparison["protocols"]["pdf_group"]["all_method_weighted_macro_f1"]
    assert set(methods) == {f"M{i}" for i in range(7)}
    named = self_comparison["safety_check"]["named_rare_classes"]
    assert set(named) == {
        "verification_timeline:within_2_years",
        "evidence_quality:Not Clear",
    }
    assert all(row["delta"] == pytest.approx(0.0) for row in named.values())


def test_single_seed_screen_is_supported_for_failed_gate_reporting():
    report = compare_architecture(
        EXAMPLES_ROOT, EXAMPLES_ROOT, EXAMPLES_ROOT,
        n_boot=20, bootstrap_seed=7, seeds=[42],
    )
    assert report["design"]["seeds"] == [42]
    assert report["seed42_expansion_gate"]["evaluated_on_seed"] == 42
