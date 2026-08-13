"""Emitted tables must match the placeholder structure D laid out against."""

import json
import re

from analysis.aggregate import protocol_summary, regime_comparison
from analysis.audit import full_audit
from analysis.load import EXAMPLES_ROOT, pdf_clusters
from analysis.tables import (
    TABLE_FILES,
    render_table1,
    render_table2,
    render_table3,
    write_tables,
)
from paper.data import REPO_ROOT, canonical_row_order, load_dev

DEV = load_dev()
ORDER = canonical_row_order(DEV)
CLUSTERS = pdf_clusters(ORDER, DEV)
PLACEHOLDERS = REPO_ROOT / "contracts" / "examples" / "tables"

AUDIT = full_audit(DEV)
SUMMARIES = {
    p: protocol_summary(p, ORDER, EXAMPLES_ROOT, CLUSTERS, n_boot=200, dev=DEV)
    for p in ("pdf_group", "row_strat")
}
REGIMES = regime_comparison(SUMMARIES, ORDER, EXAMPLES_ROOT, CLUSTERS, n_boot=200)


def _column_count(tex):
    spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", tex).group(1)
    return len(re.findall(r"[lrc]", spec))


def test_tables_carry_only_a_tabular_environment():
    rendered = (render_table1(AUDIT), render_table2(SUMMARIES["pdf_group"]),
                render_table3(REGIMES))
    for tex in rendered:
        assert tex.lstrip().startswith(r"\begin{tabular}")
        # Layout is D's call under the 8-page budget (contract section 5).
        for forbidden in (r"\begin{table}", r"\caption", r"\label"):
            assert forbidden not in tex


def test_column_counts_match_the_placeholders():
    for name, rendered in (
        ("table1_dataset.tex", render_table1(AUDIT)),
        ("table2_main.tex", render_table2(SUMMARIES["pdf_group"])),
        ("table3_regimes.tex", render_table3(REGIMES)),
    ):
        placeholder = (PLACEHOLDERS / name).read_text(encoding="utf-8")
        assert _column_count(rendered) == _column_count(placeholder), name


def test_table1_leaves_the_unlabelled_test_column_empty():
    tex = render_table1(AUDIT)
    misleading = [l for l in tex.splitlines() if "Misleading" in l][0]
    assert misleading.count("n/a") == 1   # Test column only
    assert "2" in misleading              # Development column carries n=2


def test_table1_prints_the_audited_counts():
    tex = render_table1(AUDIT)
    assert "2000" in tex and "49" in tex and "50" in tex


def test_table2_has_one_row_per_method_in_order():
    body = render_table2(SUMMARIES["pdf_group"])
    ids = [l.split("&")[0].strip() for l in body.splitlines()
           if l.strip().startswith("M")]
    assert ids == ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]


def test_table2_reports_zero_invalid_for_every_structured_method():
    for line in render_table2(SUMMARIES["pdf_group"]).splitlines():
        cells = [c.strip() for c in line.split("&")]
        if cells and cells[0] in {"M1", "M2", "M3", "M4", "M5", "M6"}:
            assert cells[-1].replace(r"\\", "").strip() == "0.0"


def test_written_manifest_records_every_input_checksum(tmp_path):
    inputs = sorted((EXAMPLES_ROOT / "results").glob("*.json"))
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, inputs)

    for name in TABLE_FILES:
        assert (tmp_path / name).exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for name in TABLE_FILES:
        entry = manifest["tables"][name]
        assert entry["source_script"] == "analysis/tables.py"
        assert len(entry["input_sha256"]) == len(entry["input_files"])
        assert all(v.startswith("sha256:") for v in entry["input_sha256"].values())


def test_captions_are_written_beside_the_tables(tmp_path):
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, [])
    for stem in ("table1_dataset", "table2_main", "table3_regimes"):
        caption = (tmp_path / f"{stem}_caption.txt").read_text(encoding="utf-8")
        assert caption.strip()
    # The two disclosures the plan requires captions to carry.
    assert "Misleading" in (tmp_path / "table1_dataset_caption.txt").read_text(
        encoding="utf-8")
    assert "not a bias" in (tmp_path / "table3_regimes_caption.txt").read_text(
        encoding="utf-8")
