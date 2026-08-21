"""The qualitative half of C's W3: what the invalid tuples actually are.

Table 2 says structured decoding barely moves weighted macro-F1 while removing
every invalid tuple. That is a puzzle on its face, and the numbers here are
what resolve it: which hierarchy rule the model breaks, and whether projection
repairs a field or destroys it. Both are counted, never described.
"""

import numpy as np
import pytest

from analysis.cases import (
    case_analysis,
    projection_ledger,
    violation_breakdown,
)
from analysis.load import EXAMPLES_ROOT, load_aligned
from paper.data import canonical_row_order, load_dev
from paper.labels import FIELDS, LABEL2ID

DEV = load_dev()
ORDER = canonical_row_order(DEV)


def _row(ps, vt, es, eq):
    return [LABEL2ID["promise_status"][ps],
            LABEL2ID["verification_timeline"][vt],
            LABEL2ID["evidence_status"][es],
            LABEL2ID["evidence_quality"][eq]]


def test_violation_breakdown_names_the_rule_each_row_breaks():
    pred = np.array([
        _row("No", "already", "N/A", "N/A"),      # PS=No 卻給了時程
        _row("Yes", "already", "No", "Clear"),    # ES=No 卻評了品質
        _row("Yes", "already", "Yes", "N/A"),     # ES=Yes 卻沒評品質
        _row("Yes", "N/A", "Yes", "Clear"),       # PS=Yes 卻缺時程
        _row("No", "N/A", "N/A", "N/A"),          # 合法
    ])
    out = violation_breakdown(pred)
    assert out["n_invalid"] == 4
    assert out["by_rule"]["promise_no_children_set"] == 1
    assert out["by_rule"]["evidence_no_quality_set"] == 1
    assert out["by_rule"]["evidence_yes_quality_missing"] == 1
    assert out["by_rule"]["promise_yes_children_absent"] == 1
    assert sum(out["by_rule"].values()) == out["n_invalid"]


def test_violation_breakdown_reports_nothing_for_legal_predictions():
    pred = np.array([_row("No", "N/A", "N/A", "N/A")])
    out = violation_breakdown(pred)
    assert out["n_invalid"] == 0
    assert out["invalid_rate"] == 0.0


def test_projection_ledger_separates_repairs_from_damage():
    """A field the rule changes is repaired, destroyed, or was wrong either way.
    Collapsing the three hides exactly what Table 2 needs explained."""
    gold   = np.array([_row("Yes", "already", "Yes", "Clear")])
    before = np.array([_row("Yes", "already", "No", "Clear")])   # 非法
    after  = np.array([_row("Yes", "already", "No", "N/A")])     # 投影：EQ 改壞了
    led = projection_ledger(gold, before, after)
    assert led["fields_repaired"] == 0
    assert led["fields_destroyed"] == 1
    assert led["fields_wrong_either_way"] == 0
    assert led["net_fields"] == -1


def test_projection_ledger_counts_a_genuine_repair():
    gold   = np.array([_row("No", "N/A", "N/A", "N/A")])
    before = np.array([_row("No", "already", "N/A", "N/A")])
    after  = np.array([_row("No", "N/A", "N/A", "N/A")])
    led = projection_ledger(gold, before, after)
    assert led["fields_repaired"] == 1 and led["net_fields"] == 1


def test_case_analysis_runs_on_the_example_set():
    out = case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV)
    assert out["protocol"] == "pdf_group" and out["seed"] == 42
    assert out["independent"]["n_invalid"] >= 0
    assert set(out["projection"]) >= {"fields_repaired", "fields_destroyed", "net_fields"}
    # 投影後必然合法 —— 這是結構保證，不是實驗結果
    assert out["after_projection"]["n_invalid"] == 0


def test_case_analysis_rejects_a_method_that_cannot_be_invalid():
    """M1-M6 are legal by construction; asking for their breakdown is a bug."""
    with pytest.raises(ValueError, match="M0"):
        case_analysis("pdf_group", 42, ORDER, EXAMPLES_ROOT, dev=DEV, baseline="M3")


def test_case_analysis_is_written_as_a_deliverable(tmp_path):
    """It has to land in a file: D writes the Discussion from it, and a number
    that only exists in a console is not evidence anyone can check."""
    from analysis.cases import write_case_analysis

    out = write_case_analysis(tmp_path, ORDER, EXAMPLES_ROOT, dev=DEV,
                              protocols=("pdf_group",), seeds=(42, 123))
    assert out.exists() and out.name == "case_analysis.json"

    import json
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["runs"]) == 2
    assert doc["totals"]["n_invalid"] == sum(r["independent"]["n_invalid"]
                                             for r in doc["runs"])
    # 規則分布必須加總回非法列數，否則有列沒有被歸類
    assert sum(doc["totals"]["by_rule"].values()) == doc["totals"]["n_invalid"]
