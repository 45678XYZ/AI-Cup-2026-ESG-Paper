"""The preview is a convenience, but a broken one is worse than none.

It exists because contract section 5 keeps the delivered `.tex` files to a bare
``tabular``: correct for D, who needs to place the floats, and useless for
anyone who just wants to look at a table.
"""

import pytest
from pypdf import PdfReader

from analysis.figure1 import tex_available
from analysis.preview import _body, _escape, build
from analysis.tables import ALL_TABLE_FILES
from paper.data import REPO_ROOT


def test_escape_leaves_the_tex_the_captions_already_contain():
    # Captions are written for LaTeX and legitimately carry maths and commands.
    assert _escape(r"$\pm$ is seed spread") == r"$\pm$ is seed spread"
    assert _escape(r"95\% percentile") == r"95\% percentile"


def test_escape_protects_the_characters_that_would_abort_the_build():
    assert _escape("a & b") == r"a \& b"
    assert _escape("table1_dataset") == r"table1\_dataset"


def test_a_delivered_table_missing_from_disk_fails_the_preview(tmp_path):
    """Skipping it is how a paper float goes missing between two rebuilds.

    ``ALL_TABLE_FILES`` is the delivered set. The assembly used to ``continue``
    past an entry that was not on disk, so a table whose writer nobody had
    wired into ``python -m analysis`` simply stopped appearing -- no error, a
    preview one page shorter, and every command still reporting success. That
    happened to table 7. Refusing to build is the only outcome a reader can
    notice.

    Checked here rather than through ``build`` because ``build`` needs TeX and
    this invariant does not.
    """
    missing, *present = ALL_TABLE_FILES
    for name in present:
        (tmp_path / name).write_text(
            "\\begin{tabular}{l}\na \\\\\n\\end{tabular}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
        _body(tmp_path)


def test_the_assembly_covers_every_delivered_table(tmp_path):
    for name in ALL_TABLE_FILES:
        (tmp_path / name).write_text(
            "\\begin{tabular}{l}\na \\\\\n\\end{tabular}\n", encoding="utf-8")
    body = "\n".join(_body(tmp_path))
    for name in ALL_TABLE_FILES:
        assert name.removesuffix(".tex").replace("_", r"\_") in body, name


@pytest.mark.skipif(not tex_available(), reason="needs latexmk")
def test_preview_renders_every_delivered_table(tmp_path):
    out = build(REPO_ROOT / "tables", tmp_path / "preview.pdf")
    assert out.exists() and out.stat().st_size > 10_000
    # One page per table plus two figure pages; a silently omitted Figure 2
    # would pass a mere "file exists" check.
    assert len(PdfReader(str(out)).pages) >= len(ALL_TABLE_FILES) + 2


@pytest.mark.skipif(not tex_available(), reason="needs latexmk")
def test_two_builds_are_byte_identical(tmp_path):
    """The preview is only committable if rebuilding it is a no-op.

    pdfTeX stamps a creation date and a document ID into every build, so two
    runs of the same source differ in bytes while being identical as documents.
    That is why the file was gitignored. Pinning SOURCE_DATE_EPOCH removes the
    only source of variation, which is what lets the rendered tables live in
    the repository for anyone without a local TeX installation.
    """
    first = build(REPO_ROOT / "tables", tmp_path / "a.pdf").read_bytes()
    second = build(REPO_ROOT / "tables", tmp_path / "b.pdf").read_bytes()
    assert first == second, (
        "two builds of the same tables differ; the preview cannot be "
        "committed until the build is deterministic")


@pytest.mark.skipif(not tex_available(), reason="needs latexmk")
def test_the_preview_embeds_no_absolute_path(tmp_path):
    """Two clones of the same commit must produce the same preview.

    `\\includegraphics` makes pdfTeX record the *absolute* path of the file it
    embedded as /PTEX.FileName, so a preview built in a clone differed from
    the committed one in 376,401 bytes while being the same document. Byte
    equality across two directories is what a committed artifact has to mean.
    """
    pdf = build(REPO_ROOT / "tables", tmp_path / "preview.pdf").read_bytes()
    assert str(REPO_ROOT).encode() not in pdf, (
        "the preview records the absolute path it was built from; it cannot "
        "be committed until pdfTeX is told to omit it")
    assert b"/PTEX.FileName" not in pdf
