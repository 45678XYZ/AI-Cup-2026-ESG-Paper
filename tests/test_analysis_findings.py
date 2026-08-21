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
                          tuple_contrasts=secondary)

    assert "weighted macro-F1" in text and "tuple accuracy" in text.lower()
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
                          tuple_contrasts=secondary)
    assert "disagree" in text.lower()
    assert "M1-M0" in text
