"""What D is allowed to write, derived from what the intervals actually say.

C is the only person who knows which differences survived resampling. Handing
D a table and letting them phrase the claims themselves is how "no detectable
difference" becomes "no difference" in a submitted paper. This module makes
the distinction a computed output rather than a conversation.
"""

import pytest

from analysis.findings import build_findings, classify_contrasts

AUDIT_STUB = {"development": {"paragraphs": 2000, "pdfs": 49},
              "splits": {"misleading_rows": [{"pdf_url": "a"}, {"pdf_url": "b"}],
                         "calibration_without_misleading": {"n_without": 18,
                                                            "n_rotations": 30}}}


def _contrast(delta, low, high, desc="d"):
    return {"delta": delta, "ci_low": low, "ci_high": high,
            "description": desc, "p_value": 0.1, "p_holm": 0.2}


def test_classify_splits_on_whether_the_interval_excludes_zero():
    rows = {
        "M1-M0": _contrast(-0.001, -0.006, 0.003),   # 跨 0
        "M6-M5": _contrast(-0.009, -0.017, -0.001),  # 排除 0，負向
        "MX-MY": _contrast(+0.020, 0.004, 0.031),    # 排除 0，正向
    }
    out = classify_contrasts(rows)
    assert out["undetermined"] == ["M1-M0"]
    assert out["worse"] == ["M6-M5"]
    assert out["better"] == ["MX-MY"]


def test_brief_refuses_to_call_an_undetermined_contrast_absent():
    """"未偵測到差異" 與 "沒有差異" 不是同一句話。"""
    rows = {"M1-M0": _contrast(-0.001, -0.006, 0.003)}
    text = build_findings(AUDIT_STUB, rows, regimes={}, cases=None)

    line = next(l for l in text.splitlines() if "M1-M0" in l)
    assert "no detectable difference" in line.lower()
    # 並且要明文禁止把它改寫成更強的說法
    assert "never as *no difference*" in text


def test_brief_carries_the_misleading_prohibition():
    text = build_findings(AUDIT_STUB, {"M1-M0": _contrast(0, -0.01, 0.01)},
                          regimes={}, cases=None)
    assert "Misleading" in text
    assert "must not" in text.lower() or "禁止" in text


def test_brief_reports_the_regime_gap_when_it_is_significant():
    regimes = {"M0": {"same_document": 0.585, "document_disjoint": 0.572,
                      "delta": 0.012, "ci_low": 0.004, "ci_high": 0.022}}
    text = build_findings(AUDIT_STUB, {}, regimes=regimes, cases=None)
    assert "0.012" in text and "[0.004, 0.022]" in text


def test_brief_numbers_come_from_the_input():
    rows = {"M9-M8": _contrast(-0.042, -0.099, -0.011, "made up")}
    text = build_findings(AUDIT_STUB, rows, regimes={}, cases=None)
    assert "-0.042" in text and "[-0.099, -0.011]" in text and "made up" in text


def test_brief_separates_the_two_metric_families():
    """The headline is the disagreement, so both families must be visible and
    labelled — a reader must never mistake the secondary metric for the one the
    competition ranks on."""
    primary = {"M1-M0": _contrast(-0.001, -0.006, 0.003, "legalisation")}
    secondary = {"M1-M0": _contrast(+0.035, 0.028, 0.043, "legalisation")}
    text = build_findings(AUDIT_STUB, primary, regimes={}, cases=None,
                          consistent_contrasts=secondary)

    assert "weighted macro-F1" in text
    assert "path-constrained" in text.lower()
    assert "[0.028, 0.043]" in text
    assert "[-0.006, 0.003]" in text
    # 官方排名依據必須被明確指出
    assert "ranks" in text.lower() or "official" in text.lower()


def test_brief_flags_a_contrast_that_disagrees_across_metrics():
    """M1-M0 is the whole paper: undetectable on one metric, significant on the
    other. If the brief does not say so explicitly, D has to notice it alone."""
    primary = {"M1-M0": _contrast(-0.001, -0.006, 0.003, "legalisation")}
    secondary = {"M1-M0": _contrast(+0.035, 0.028, 0.043, "legalisation")}
    text = build_findings(AUDIT_STUB, primary, regimes={}, cases=None,
                          consistent_contrasts=secondary)
    assert "disagree" in text.lower()
    assert "M1-M0" in text


def test_brief_reports_the_pre_specified_family_it_did_not_headline():
    """tuple accuracy no longer carries the argument, but it was named in the
    analysis plan and one of its contrasts runs against us. Dropping it from
    the brief would be selective reporting."""
    primary = {"M1-M0": _contrast(-0.001, -0.006, 0.003, "legalisation")}
    tuples = {"M4-M1": _contrast(-0.006, -0.010, -0.002, "decoding")}
    text = build_findings(AUDIT_STUB, primary, regimes={}, cases=None,
                          tuple_contrasts=tuples)
    assert "tuple accuracy" in text.lower()
    assert "[-0.010, -0.002]" in text
    assert "pre-specified" in text.lower()


def test_brief_localises_the_disagreement_to_a_single_factor():
    """The reason this metric was chosen over whole-row accuracy: it differs
    from the official one in exactly one respect, so a disagreement is evidence
    about that respect and not about the shape of the metric."""
    primary = {"M1-M0": _contrast(-0.001, -0.006, 0.003, "legalisation")}
    secondary = {"M1-M0": _contrast(+0.004, 0.001, 0.007, "legalisation")}
    text = build_findings(AUDIT_STUB, primary, regimes={}, cases=None,
                          consistent_contrasts=secondary)
    section = text.split("### Where the two metrics disagree")[1]
    assert "exactly one" in section.lower()


def test_brief_reports_the_conditional_field_scores():
    """The conditioned numbers have no column in Table 2, so the brief is where
    they reach D. Without them the plan's secondary-metric list is incomplete."""
    methods = {
        "M0": {"per_field_mean": {"promise_status": 0.8, "evidence_quality": 0.42},
               "conditional_per_field_mean": {"promise_status": 0.8,
                                              "evidence_quality": 0.31}},
    }
    text = build_findings(AUDIT_STUB, {}, regimes={}, cases=None, methods=methods)
    assert "conditional" in text.lower()
    assert "0.31" in text
    assert "evidence_quality" in text


def test_brief_reports_the_misleading_free_sensitivity():
    """The plan asks for a score computed without the two Misleading rows. It
    was being computed and thrown away."""
    methods = {
        "M0": {"per_field_mean": {"promise_status": 0.8},
               "conditional_per_field_mean": {"promise_status": 0.8},
               "weighted_macro_f1_mean": 0.5723,
               "weighted_macro_f1_mean_no_misleading": 0.5881},
    }
    text = build_findings(AUDIT_STUB, {}, regimes={}, cases=None, methods=methods)
    assert "0.588" in text
    assert "sensitivity" in text.lower()


def test_brief_records_the_two_misleading_instances_one_by_one():
    cases = {"totals": {"n_rows": 4, "n_invalid": 0, "invalid_rate": 0.0,
                        "by_rule": {"promise_no_children_set": 0},
                        "fields_repaired": 0, "fields_destroyed": 0,
                        "net_fields": 0},
             "runs": [{"misleading_cases": [
                 {"id": "42", "pdf_url": "http://x", 
                  "gold": {"evidence_quality": "Misleading"},
                  "predicted": {"M0": "Clear", "M1": "Clear"}}]}]}
    text = build_findings(AUDIT_STUB, {}, regimes={}, cases=cases)
    assert "42" in text and "Clear" in text


def test_brief_forbids_the_systematically_claim():
    """One benchmark, one backbone, seven decision rules. The evidence supports
    "can substantially understate" on this task; it does not support a claim
    about the metric in general, and that is the single easiest thing for a
    reviewer to attack."""
    text = build_findings(AUDIT_STUB, {}, regimes={}, cases=None)
    prohibitions = text.split("## Prohibitions")[1]
    assert "systematically" in prohibitions
    assert "substantially understate" in prohibitions


def test_brief_reports_the_override_ledger():
    """The mechanism behind M4-M1 has to arrive as counts. Stated as prose it
    was not only unverifiable, it was wrong: the decoder makes the parent field
    *more* accurate and still loses whole rows."""
    cases = {"totals": {"n_rows": 6000, "n_invalid": 1527, "invalid_rate": 0.25,
                        "by_rule": {"promise_no_children_set": 755},
                        "fields_repaired": 822, "fields_destroyed": 637,
                        "net_fields": 185,
                        "parent_overrides": {
                            "n_rows": 6000, "n_changed": 160,
                            "to_correct": 104, "to_wrong": 56,
                            "wrong_to_wrong": 0,
                            "tuple_correct_before": 56,
                            "tuple_correct_after": 20}},
             "runs": []}
    text = build_findings(AUDIT_STUB, {}, regimes={}, cases=cases)
    assert "160" in text and "104" in text
    assert "56" in text and "20" in text
    assert "promise_status" in text
