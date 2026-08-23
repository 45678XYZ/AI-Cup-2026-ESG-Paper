import json
from pathlib import Path

from pypdf import PdfWriter
from paper.data import REPO_ROOT

from manuscript.check import (
    asset_errors,
    font_errors,
    log_errors,
    pdf_errors,
    source_errors,
    source_warnings,
    source_text,
    main,
)


def write_minimal(root: Path, body: str, metadata: str = "") -> None:
    root.mkdir()
    (root / "main.tex").write_text(body, encoding="utf-8")
    (root / "metadata.tex").write_text(metadata, encoding="utf-8")


def test_draft_rejects_prohibited_hardware_and_unqualified_null_claim(tmp_path):
    write_minimal(tmp_path / "m", "\\input{../tables/table4_contrasts.tex}\n"
                  "Path-constrained wF1 and hierarchical F1 are post hoc. "
                  "There was no difference on an L" + "40S run.")
    errors = source_errors(tmp_path / "m")
    assert any("prohibited hardware" in error for error in errors)
    assert any("no detectable difference" in error for error in errors)


def test_draft_rejects_an_equivalence_claim(tmp_path):
    write_minimal(tmp_path / "m", "\\input{../tables/table4_contrasts.tex}\n"
                  "Path-constrained wF1 and hierarchical F1 are post hoc. "
                  "The two decision rules are equivalent.")
    assert any("equivalence" in error for error in source_errors(tmp_path / "m"))


def test_draft_requires_full_table4_and_post_hoc_disclosure(tmp_path):
    write_minimal(tmp_path / "m", "Path-constrained wF1 and hierarchical F1.")
    errors = source_errors(tmp_path / "m")
    assert any("table4_contrasts.tex" in error for error in errors)
    assert any("post hoc" in error for error in errors)


def test_author_metadata_is_draft_warning_but_final_error(tmp_path):
    write_minimal(tmp_path / "m", "\\input{../tables/table4_contrasts.tex}\n"
                  "Path-constrained wF1 and hierarchical F1 were adopted post hoc.")
    assert not any("author metadata" in error for error in source_errors(tmp_path / "m"))
    assert any("author metadata" in warning for warning in source_warnings(tmp_path / "m"))
    assert any("author metadata" in error for error in source_errors(tmp_path / "m", final=True))


def test_pdf_page_limit(tmp_path):
    path = tmp_path / "nine-pages.pdf"
    writer = PdfWriter()
    for _ in range(9):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as stream:
        writer.write(stream)
    assert any("9 pages" in error for error in pdf_errors(path, max_pages=8))


def test_log_rejects_unresolved_references(tmp_path):
    log = tmp_path / "main.log"
    log.write_text("LaTeX Warning: There were undefined references.", encoding="utf-8")
    assert log_errors(log)


def test_asset_check_names_missing_generated_files(tmp_path):
    errors = asset_errors(tmp_path)
    assert any("table4_contrasts.tex" in error for error in errors)


def test_source_text_reads_root_sections_and_bib(tmp_path):
    (tmp_path / "main.tex").write_text("root", encoding="utf-8")
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "method.tex").write_text("section", encoding="utf-8")
    (tmp_path / "refs.bib").write_text("bib", encoding="utf-8")
    assert source_text(tmp_path) == "root\nbib\nsection"


def test_equivalence_language_and_bound_metric_names_require_disclosure(tmp_path):
    write_minimal(tmp_path / "m", "C-wF1 and hF establish equivalence.")
    errors = source_errors(tmp_path / "m")
    assert any("equivalence" in error for error in errors)
    assert any("post hoc" in error for error in errors)


def test_table4_comment_spoof_is_not_an_inclusion(tmp_path):
    write_minimal(tmp_path / "m", "% \\input{../tables/table4_contrasts.tex}")
    assert any("table4_contrasts.tex" in error for error in source_errors(tmp_path / "m"))


def test_asset_check_detects_tampered_manifested_table(tmp_path):
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    tables.mkdir()
    figures.mkdir()
    names = ["table1_dataset.tex", "table2_main.tex", "table3_regimes.tex", "table4_contrasts.tex"]
    for name in names:
        (tables / name).write_text(name, encoding="utf-8")
    (figures / "figure1_hierarchy.pdf").write_bytes(b"pdf")
    from paper.data import file_sha256
    outputs = {name: file_sha256(tables / name) for name in names}
    (tables / "manifest.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"outputs": {"tables": outputs}}), encoding="utf-8")
    (tables / "table4_contrasts.tex").write_text("tampered", encoding="utf-8")
    assert any("manifest" in error or "provenance" in error for error in asset_errors(tmp_path))


def test_font_checker_accepts_pdf_without_font_resources(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as stream:
        writer.write(stream)
    assert font_errors(path) == []


def test_cli_returns_nonzero_for_policy_error(tmp_path, monkeypatch, capsys):
    write_minimal(tmp_path / "m", "no difference")
    monkeypatch.setattr("sys.argv", ["check", "--root", str(tmp_path / "m"), "--repo-root", str(tmp_path)])
    assert main() == 1
    assert "ERROR:" in capsys.readouterr().out


def test_canonical_repository_assets_have_clean_provenance():
    assert asset_errors(REPO_ROOT) == []


def test_missing_required_table_manifest_entry_is_rejected(tmp_path):
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    tables.mkdir(); figures.mkdir()
    for name in ("table1_dataset.tex", "table2_main.tex", "table3_regimes.tex",
                 "table4_contrasts.tex", "table5_metrics.tex"):
        (tables / name).write_text(name, encoding="utf-8")
    (figures / "figure1_hierarchy.pdf").write_bytes(b"pdf")
    (tables / "manifest.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": []}), encoding="utf-8")
    errors = asset_errors(tmp_path)
    assert any("manifest entry" in error for error in errors)


def test_tampered_table_input_is_rejected(tmp_path):
    tables = tmp_path / "tables"; tables.mkdir()
    source = tmp_path / "input.json"; source.write_text("original", encoding="utf-8")
    (tables / "manifest.json").write_text(json.dumps({"tables": {
        name: {"source_script": "script.py", "input_files": ["input.json"],
               "input_sha256": {"input.json": "sha256:bad"}}
        for name in ("table1_dataset.tex", "table2_main.tex", "table3_regimes.tex",
                     "table4_contrasts.tex", "table5_metrics.tex")
    }}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": []}), encoding="utf-8")
    errors = asset_errors(tmp_path)
    assert any("input mismatch" in error for error in errors)


def test_missing_or_failed_run_manifest_consistency_is_rejected(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "manifest.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": [{"status": "fail"}]}), encoding="utf-8")
    assert any("consistency" in error for error in asset_errors(tmp_path))


def test_untracked_figure_is_rejected(tmp_path):
    (tmp_path / "tables").mkdir(); (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "figure1_hierarchy.pdf").write_bytes(b"pdf")
    assert any("figure" in error for error in asset_errors(tmp_path))


def test_asset_check_rejects_malformed_but_valid_manifest_shapes(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tmp_path / "run_manifest.json").write_text(json.dumps({"consistency": []}), encoding="utf-8")
    cases = [
        ([], "manifest top level"),
        ({"tables": []}, "tables section"),
        ({"tables": {"table1_dataset.tex": []}}, "table manifest entry"),
        ({"tables": {"table1_dataset.tex": {"input_sha256": []}}}, "input_sha256"),
    ]
    for index, (payload, label) in enumerate(cases):
        path = tables / f"manifest-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.rename(tables / "manifest.json")
        errors = asset_errors(tmp_path)
        assert any(label in error for error in errors), (label, errors)


def test_asset_check_rejects_malformed_consistency_shapes(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "manifest.json").write_text(json.dumps({"tables": {}}), encoding="utf-8")
    for payload in ({"consistency": {}}, {"consistency": [{}]}, {"consistency": ["bad"]}):
        (tmp_path / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        errors = asset_errors(tmp_path)
        assert any("consistency" in error for error in errors)
