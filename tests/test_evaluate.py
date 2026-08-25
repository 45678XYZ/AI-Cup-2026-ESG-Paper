"""Per-row predictions and the aggregate summary computed from them."""

import json

import pytest

from contracts.make_examples import make_contract3, make_contract4
from paper.artifacts import (
    prediction_row,
    read_predictions,
    read_predictions_df,
    write_predictions,
)
from paper.data import file_sha256, load_dev
from paper.evaluate import build_results, summarise_predictions, write_results
from paper.labels import FIELDS
from paper.score import compute_weighted_macro_f1
from paper.splits import build_split

ROWS = load_dev()
SPLIT = build_split("pdf_group", 42, ROWS)


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    out = tmp_path_factory.mktemp("examples")
    make_contract3(SPLIT, ROWS, out, methods=["M0", "M3"])
    make_contract4(out)
    return out


def _perfect_records():
    """Predictions equal to gold, so every metric has a known value."""
    return [prediction_row(r, 0, {f: r[f] for f in FIELDS}) for r in ROWS]


# --- the summary ----------------------------------------------------------

def test_perfect_predictions_score_one():
    s = summarise_predictions(_perfect_records())
    assert abs(s["weighted_macro_f1"] - 1.0) < 1e-9
    assert s["tuple_exact_match"] == 1.0
    assert s["invalid_tuple_rate"] == 0.0


def test_summary_agrees_with_the_official_scorer():
    """paper/score.py owns the primary metric; evaluate must not diverge."""
    records = _perfect_records()
    gold = [{f: r[f] for f in FIELDS} for r in ROWS]
    pred = [{f: r[f] for f in FIELDS} for r in ROWS]
    assert abs(
        summarise_predictions(records)["weighted_macro_f1"]
        - compute_weighted_macro_f1(gold, pred)
    ) < 1e-12


def test_invalid_tuples_are_counted():
    records = _perfect_records()
    # PS=No with a substantive timeline is not one of the 17 legal states.
    records[0]["pred_ps"], records[0]["pred_vt"] = "No", "already"
    records[0]["pred_state_id"] = -1
    assert summarise_predictions(records)["invalid_tuple_rate"] == 1 / len(records)


def test_conditional_metrics_use_the_gold_parent_subset():
    s = summarise_predictions(_perfect_records())
    assert set(s["conditional_f1"]) == {
        "verification_timeline_given_ps_yes",
        "evidence_status_given_ps_yes",
        "evidence_quality_given_es_yes",
    }
    assert all(abs(v - 1.0) < 1e-9 for v in s["conditional_f1"].values())


def test_support_counts_come_from_gold():
    s = summarise_predictions(_perfect_records())
    assert s["per_class_support"]["evidence_quality"]["Misleading"] == 2
    assert sum(s["per_class_support"]["promise_status"].values()) == 2000


# --- the predictions CSV --------------------------------------------------

def test_predictions_roundtrip_keeps_ids_as_strings(tmp_path):
    path = write_predictions(tmp_path / "p.csv.gz", _perfect_records()[:50])
    back = read_predictions(path)
    assert len(back) == 50
    assert all(isinstance(r["id"], str) for r in back)
    assert back[0]["id"] == ROWS[0]["id"]
    assert isinstance(back[0]["gold_state_id"], int)


def test_writing_the_same_predictions_twice_gives_the_same_bytes(tmp_path):
    """results/*.json binds a summary to its predictions by sha256, so the
    checksum has to mean "these predictions" and not "this write"."""
    records = _perfect_records()[:50]
    a = write_predictions(tmp_path / "a.csv.gz", records)
    b = write_predictions(tmp_path / "b.csv.gz", records)
    assert file_sha256(a) == file_sha256(b)


def test_pandas_needs_the_helper_to_keep_ids_as_strings(tmp_path):
    """Quoting does not survive pandas type inference, so C reading these files
    the obvious way gets int64 ids and every id join silently matches nothing.
    Contract §4.1 points at read_predictions_df for exactly this reason."""
    pd = pytest.importorskip("pandas")
    path = write_predictions(tmp_path / "p.csv.gz", _perfect_records()[:50])

    # Asserted on the values, not the dtype: pandas 2 stores strings as object
    # and pandas 3 as str, and the property that matters is that an id still
    # compares equal to the id in the split manifest.
    assert pd.read_csv(path)["id"].dtype.kind == "i"   # the trap, pinned
    df = read_predictions_df(path)
    assert isinstance(df["id"].iloc[0], str)
    assert df["id"].iloc[0] == ROWS[0]["id"]
    assert df["rotation"].dtype.kind == "i"            # only id is coerced back


# --- the generated example files -----------------------------------------

def test_examples_are_internally_consistent(examples):
    for method in ("M0", "M3"):
        summary = json.load(
            open(examples / "results" / f"pdf_group_seed42_{method}.json", encoding="utf-8")
        )
        records = read_predictions(examples / summary["predictions_file"])
        recomputed = summarise_predictions(records)
        for key in ("weighted_macro_f1", "tuple_exact_match", "invalid_tuple_rate"):
            assert abs(summary[key] - recomputed[key]) < 1e-12


def test_examples_reproduce_the_properties_downstream_code_must_handle(examples):
    m0 = json.load(open(examples / "results" / "pdf_group_seed42_M0.json", encoding="utf-8"))
    m3 = json.load(open(examples / "results" / "pdf_group_seed42_M3.json", encoding="utf-8"))
    assert m0["invalid_tuple_rate"] > 0      # independent argmax breaks the hierarchy
    assert m3["invalid_tuple_rate"] == 0     # the structured rules never do
    assert m0["n_rows"] == m3["n_rows"] == 2000
    assert m0["per_class_support"] == m3["per_class_support"]
    assert m0["synthetic"] is True


def test_contract4_placeholders_are_bare_tabulars(examples):
    for name in ("table1_dataset", "table2_main", "table6_regimes"):
        tex = (examples / "tables" / f"{name}.tex").read_text(encoding="utf-8")
        assert tex.startswith(r"\begin{tabular}")
        # D controls float placement and captions under the 8-page budget, so C
        # must not ship the wrapper.
        assert r"\begin{table}" not in tex
        assert r"\caption" not in tex
        assert (examples / "tables" / f"{name}_caption.txt").exists()


def test_table2_placeholder_carries_every_column_the_plan_specifies(examples):
    """D measures the 8-page budget against this file. A column that appears
    only when the real numbers arrive is a layout problem discovered in W4."""
    tex = (examples / "tables" / "table2_main.tex").read_text(encoding="utf-8")
    header = [c.strip() for c in tex.splitlines()[2].removesuffix(r"\\").split("&")]
    assert header == ["ID", "Calibration", "Decoding", "Weighted F1",
                      "PS", "VT", "ES", "EQ", "Tuple Acc.", r"Invalid \%"]
    assert tex.splitlines()[0] == r"\begin{tabular}{ll" + "r" * 8 + "}"
    for line in tex.splitlines():
        if line.startswith("M"):
            assert len(line.removesuffix(r"\\").split("&")) == len(header)


# --- the results envelope -------------------------------------------------

def test_results_envelope_binds_a_summary_to_its_own_predictions_file(tmp_path):
    """build_results is the single writer of the contract-3 object, so the
    envelope C validated against is the envelope the real runner emits."""
    records = _perfect_records()
    path = write_predictions(tmp_path / "pdf_group_seed42_M6.csv.gz", records)
    results = build_results(
        records, protocol="pdf_group", seed=42, method="M6",
        predictions_path=path, data_checksum="sha256:abc", synthetic=True,
    )
    assert results["predictions_file"] == "predictions/pdf_group_seed42_M6.csv.gz"
    assert results["predictions_sha256"] == file_sha256(path)
    assert results["weighted_macro_f1"] == summarise_predictions(records)["weighted_macro_f1"]

    written = json.load(open(write_results(tmp_path / "r.json", results), encoding="utf-8"))
    assert written == results
