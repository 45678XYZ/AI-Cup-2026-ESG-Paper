from pathlib import Path

from pypdf import PdfWriter

from manuscript.check import (
    asset_errors,
    font_errors,
    log_errors,
    pdf_errors,
    source_errors,
    source_warnings,
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

