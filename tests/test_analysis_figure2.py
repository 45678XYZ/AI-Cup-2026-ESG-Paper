"""Figure 2 must be a faithful vector view of the generated multilingual table."""

import re

import pytest
from pypdf import PdfReader

from analysis.figure1 import tex_available
from analysis.figure2 import (
    DEFS_NAME,
    SOURCE,
    TABLE,
    build,
    parse_table,
    write_defs,
)


FIXTURE = r"""\begin{tabular}{lrrrrrr}
\toprule
Corpus & Rows & M0 illegal \% & $N\!/\!A$ net & Subst.\ net & $\Delta$ official & $\Delta$ whole-row \\
\midrule
Chinese (AI CUP) & 2,000 & 12.55 & +395 & -264 & -0.0011 & \textbf{+0.0350} \\
English & 400 & 23.17 & +148 & -86 & +0.0032 & \textbf{+0.0725} \\
French & 400 & 21.08 & +150 & -107 & +0.0036 & \textbf{+0.0625} \\
Japanese & 400 & 31.25 & +183 & -245 & -0.0131 & \textbf{+0.0708} \\
Korean & 500 & 22.73 & +103 & -148 & -0.0192 & \textbf{+0.0293} \\
\bottomrule
\end{tabular}
"""


def fixture_table(tmp_path):
    path = tmp_path / "table7_multilingual_mechanism.tex"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def test_parser_reads_every_printed_quantity_exactly(tmp_path):
    """Changing a table column or sign must not silently move another marker."""
    rows = parse_table(fixture_table(tmp_path))

    assert [row["language"] for row in rows] == [
        "Chinese", "English", "French", "Japanese", "Korean"
    ]
    assert rows[0] == {
        "language": "Chinese",
        "corpus": "AI CUP",
        "n_rows": 2000,
        "m0_invalid_percent": 12.55,
        "na_net": 395,
        "substantive_net": -264,
        "weighted_f1_delta": -0.0011,
        "tuple_delta": 0.0350,
    }
    assert rows[-1]["m0_invalid_percent"] == 22.73
    assert rows[-1]["weighted_f1_delta"] == -0.0192
    assert rows[-1]["tuple_delta"] == 0.0293


def test_parser_rejects_a_missing_language(tmp_path):
    """A four-row figure would make the 5/5 takeaway false by omission."""
    path = fixture_table(tmp_path)
    four_rows = "\n".join(
        line for line in FIXTURE.splitlines()
        if not line.startswith("Korean ")
    )
    path.write_text(four_rows, encoding="utf-8")

    with pytest.raises(ValueError, match="five|language|Korean"):
        parse_table(path)


def test_defs_preserve_values_and_derive_sign_counts(tmp_path):
    """The headline counts are computed from deltas, never typed into TikZ."""
    rows = parse_table(fixture_table(tmp_path))
    defs = write_defs(rows, tmp_path / DEFS_NAME).read_text(encoding="utf-8")

    assert r"\newcommand{\ChineseInvalid}{12.55}" in defs
    assert r"\newcommand{\ChineseNaNet}{+395}" in defs
    assert r"\newcommand{\ChineseSubstNet}{-264}" in defs
    assert r"\newcommand{\ChineseWfDelta}{-0.0011}" in defs
    assert r"\newcommand{\ChineseTupleDelta}{+0.0350}" in defs
    assert r"\newcommand{\LanguageCount}{5}" in defs
    assert r"\newcommand{\TuplePositiveCount}{5}" in defs
    assert r"\newcommand{\WfPositiveCount}{2}" in defs


def test_committed_table_and_defs_are_current(tmp_path):
    rows = parse_table(TABLE)
    assert len(rows) == 5
    fresh = write_defs(rows, tmp_path / DEFS_NAME)
    committed = SOURCE.parent / DEFS_NAME
    assert committed.is_file()
    assert committed.read_text(encoding="utf-8") == fresh.read_text(encoding="utf-8")


def test_source_does_not_copy_result_values_by_hand():
    """Deleting generated definitions must leave no publishable score behind."""
    body = "\n".join(
        line for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("%")
    )
    for literal in ("12.55", "+395", "-264", "-0.0011", "+0.0350"):
        assert literal not in body


@pytest.mark.skipif(not tex_available(), reason="no TeX installation")
def test_build_emits_a_reproducible_vector_pdf_with_embedded_fonts(tmp_path):
    table = fixture_table(tmp_path)
    first = build(table, tmp_path / "a" / "figure2_multilingual.pdf")
    second = build(table, tmp_path / "b" / "figure2_multilingual.pdf")

    assert first.read_bytes() == second.read_bytes()
    page = PdfReader(str(first)).pages[0]
    resources = page["/Resources"]
    assert "/XObject" not in resources or not any(
        item.get_object().get("/Subtype") == "/Image"
        for item in resources["/XObject"].values()
    )

    descriptors = []
    for font_ref in resources["/Font"].values():
        font = font_ref.get_object()
        for descendant in font.get("/DescendantFonts", []):
            descriptors.append(descendant.get_object()["/FontDescriptor"].get_object())
        if "/FontDescriptor" in font:
            descriptors.append(font["/FontDescriptor"].get_object())
    assert descriptors
    assert all(
        any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))
        for descriptor in descriptors
    )

    text = re.sub(r"\s+", " ", page.extract_text())
    for label in ("Chinese", "English", "French", "Japanese", "Korean"):
        assert label in text
