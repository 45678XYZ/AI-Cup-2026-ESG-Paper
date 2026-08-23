"""RBT-base comparison keeps the two pre-registered conditions explicit."""

import json

import pytest

from analysis.rbt_base import summarize_protocol


def _write_result(root, protocol, seed, method, score, invalid=0.0):
    path = root / "results" / f"{protocol}_seed{seed}_{method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "weighted_macro_f1": score,
        "invalid_tuple_rate": invalid,
    }), encoding="utf-8")


def test_summary_requires_both_generality_conditions(tmp_path):
    run = tmp_path / "run"
    anchor = tmp_path / "anchor"
    for seed in (42, 123, 456):
        _write_result(run, "pdf_group", seed, "M0", 0.50, invalid=0.20)
        _write_result(run, "pdf_group", seed, "M1", 0.53)
        _write_result(anchor, "pdf_group", seed, "M0", 0.60, invalid=0.10)
        _write_result(anchor, "pdf_group", seed, "M1", 0.61)

    report = summarize_protocol(run, anchor, "pdf_group")

    assert report["architecture"]["m0_invalid_mean"] == pytest.approx(0.20)
    assert report["architecture"]["m1_minus_m0_mean"] == pytest.approx(0.03)
    assert report["generality_hypothesis"] == {
        "architecture_invalid_higher_than_anchor": True,
        "architecture_m1_minus_m0_higher_than_anchor": True,
        "both_conditions_met": True,
    }


def test_summary_reports_a_failed_condition_without_relabelling(tmp_path):
    run = tmp_path / "run"
    anchor = tmp_path / "anchor"
    for seed in (42, 123, 456):
        _write_result(run, "row_strat", seed, "M0", 0.50, invalid=0.05)
        _write_result(run, "row_strat", seed, "M1", 0.49)
        _write_result(anchor, "row_strat", seed, "M0", 0.60, invalid=0.10)
        _write_result(anchor, "row_strat", seed, "M1", 0.60)

    report = summarize_protocol(run, anchor, "row_strat")

    assert report["differences"]["m0_invalid_mean"] == pytest.approx(-0.05)
    assert report["differences"]["m1_minus_m0_mean"] == pytest.approx(-0.01)
    assert not report["generality_hypothesis"]["both_conditions_met"]
