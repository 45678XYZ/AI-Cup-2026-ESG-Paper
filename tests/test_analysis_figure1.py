"""Figure 1's counts must come from the label space, never from the .tex source.

The figure prints three numbers -- 17 legal tuples, 120 combinations, 103
hierarchy-invalid -- and the plan's "所有數字由 script 生成，不手抄" rule covers
them just as it covers the tables. Moving the drawing into TikZ puts that rule
at risk, because a hand-written .tex can simply type "17". These tests are what
keeps the numbers derived: the source may only reference macros, and the macros
are generated from ``paper.labels``.
"""

import math
import re
import shutil
import subprocess

import pytest

from analysis.figure1 import (
    DEFS_NAME,
    SOURCE,
    build,
    state_counts,
    tex_available,
    write_defs,
)
from paper.labels import EVAL_FIELDS, NUM_LABELS, STATES

COUNTS = state_counts()
MACRO_USE = re.compile(r"\\(Num[A-Za-z]+)\b")
MACRO_DEF = re.compile(r"\\newcommand\{\\(Num[A-Za-z]+)\}\{([^}]*)\}")


def test_counts_come_from_the_label_space():
    """Every count is derived, so a label-space change moves the figure with it."""
    assert COUNTS["NumStates"] == len(STATES)
    assert COUNTS["NumCombinations"] == math.prod(NUM_LABELS.values())
    assert COUNTS["NumInvalid"] == COUNTS["NumCombinations"] - COUNTS["NumStates"]


def test_the_printed_decomposition_actually_multiplies_out():
    """The figure prints 1 + T x (1 + Q) = S; assert it is arithmetic, not a caption."""
    timelines = COUNTS["NumTimelines"]
    qualities = COUNTS["NumQualities"]
    assert timelines == len([x for x in EVAL_FIELDS["verification_timeline"] if x != "N/A"])
    assert qualities == len([x for x in EVAL_FIELDS["evidence_quality"] if x != "N/A"])
    assert 1 + timelines * (1 + qualities) == COUNTS["NumStates"]


def test_defs_file_defines_every_macro_the_source_uses(tmp_path):
    """A macro used but never defined is a compile error; catch it without TeX."""
    defs_path = write_defs(tmp_path / DEFS_NAME)
    defined = {name: value for name, value in MACRO_DEF.findall(defs_path.read_text())}
    used = set(MACRO_USE.findall(SOURCE.read_text())) - {"NumberOfDefsIsIrrelevant"}

    assert used, "the source stopped referencing any generated macro"
    assert used <= set(defined), f"undefined in defs: {sorted(used - set(defined))}"
    for name, value in defined.items():
        assert value == str(COUNTS[name])


def test_source_never_hardcodes_a_count():
    """The whole point of the defs file: no bare 17 / 120 / 103 in the drawing."""
    text = SOURCE.read_text()
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )
    for name in ("NumStates", "NumCombinations", "NumInvalid"):
        literal = str(COUNTS[name])
        assert not re.search(rf"(?<![\d.]){literal}(?![\d.])", body), (
            f"{literal} ({name}) is typed into the figure source; use \\{name}"
        )


def test_committed_defs_are_current(tmp_path):
    """The defs file is committed so the source previews in an editor as-is.

    That convenience is also a hazard: a stale committed copy would silently
    draw last week's numbers. Regenerating and comparing is the whole guard.
    """
    committed = SOURCE.parent / DEFS_NAME
    assert committed.exists(), f"{DEFS_NAME} must sit next to the figure source"
    fresh = write_defs(tmp_path / DEFS_NAME)
    assert committed.read_text() == fresh.read_text(), (
        "committed figure1_defs.tex is stale; run `python -m analysis`"
    )


def test_tex_available_reports_whether_the_figure_can_be_rebuilt(monkeypatch):
    """__main__ asks this before rebuilding, so a TeX-less machine still
    recomputes every table instead of dying on the figure."""
    import analysis.figure1 as figure1

    monkeypatch.setattr(figure1.shutil, "which", lambda name: None)
    assert figure1.tex_available() is False
    monkeypatch.setattr(
        figure1.shutil,
        "which",
        lambda name: "/usr/bin/tectonic" if name == "tectonic" else None,
    )
    assert figure1.tex_available() is True
    monkeypatch.setattr(
        figure1.shutil,
        "which",
        lambda name: "/usr/bin/latexmk" if name == "latexmk" else None,
    )
    assert figure1.tex_available() is True


def test_build_names_the_missing_tool(monkeypatch, tmp_path):
    """Failing with 'FileNotFoundError: latexmk' would not tell anyone what to do."""
    import analysis.figure1 as figure1

    monkeypatch.setattr(figure1.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="latexmk.*tectonic|tectonic.*latexmk"):
        figure1.build(tmp_path / "figure1_hierarchy.pdf")


def test_build_uses_tectonic_when_latexmk_is_missing(monkeypatch, tmp_path):
    """The repository's manuscript compiler is a valid figure compiler too."""
    import analysis.figure1 as figure1

    out = tmp_path / "figure1_hierarchy.pdf"
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        out.write_bytes(b"%PDF-1.5\nnot empty")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        figure1.shutil,
        "which",
        lambda name: "/usr/bin/tectonic" if name == "tectonic" else None,
    )
    monkeypatch.setattr(figure1.subprocess, "run", fake_run)

    assert figure1.build(out) == out
    assert commands and commands[0][0] == "tectonic"


def test_build_rejects_an_empty_pdf(monkeypatch, tmp_path):
    """latexmk can exit 0 having left a truncated PDF; existence is not enough.

    Silently accepting one would ship a blank Figure 1 into the paper, which
    is the one failure mode nobody checks for by eye until it is printed.
    """
    import analysis.figure1 as figure1

    out = tmp_path / "figure1_hierarchy.pdf"

    def fake_run(cmd, **kwargs):
        out.write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(figure1.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(figure1.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="empty|failed"):
        figure1.build(out)


@pytest.mark.skipif(not tex_available(), reason="no TeX installation")
def test_build_emits_a_vector_pdf_with_every_font_embedded(tmp_path):
    """Contract §5: vector PDF, fonts embedded, no raster fallback."""
    pypdf = pytest.importorskip("pypdf")

    out = build(tmp_path / "figure1_hierarchy.pdf")
    assert out.exists() and out.stat().st_size > 0

    page = pypdf.PdfReader(str(out)).pages[0]
    resources = page["/Resources"]
    assert "/XObject" not in resources or not any(
        x.get_object().get("/Subtype") == "/Image"
        for x in resources["/XObject"].values()
    ), "figure contains a raster image; it must stay vector"

    descriptors = []
    for font in resources["/Font"].values():
        font = font.get_object()
        for descendant in font.get("/DescendantFonts", []):
            descriptors.append(descendant.get_object()["/FontDescriptor"].get_object())
        if "/FontDescriptor" in font:
            descriptors.append(font["/FontDescriptor"].get_object())

    assert descriptors, "no font descriptors found; cannot prove embedding"
    for desc in descriptors:
        embedded = any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3"))
        assert embedded, f"{desc.get('/FontName')} is referenced but not embedded"


@pytest.mark.skipif(not tex_available(), reason="no TeX installation")
def test_render_uses_multiple_non_grey_fill_colours(tmp_path):
    """The architecture should retain its visual grouping when rendered.

    This inspects the PDF drawing operators rather than grepping the TikZ
    source. Removing the filled task, route, or outcome groups is the break it
    catches; coloured strokes alone do not satisfy the contract.
    """
    pypdf = pytest.importorskip("pypdf")

    out = build(tmp_path / "figure1_hierarchy.pdf")
    content = pypdf.PdfReader(str(out)).pages[0].get_contents().get_data()
    fills = {
        tuple(float(component) for component in match.groups())
        for match in re.finditer(
            rb"(?<![\d.])(\d*\.\d+|\d+) (\d*\.\d+|\d+) "
            rb"(\d*\.\d+|\d+) rg\b",
            content,
        )
    }
    non_grey = {
        colour for colour in fills
        if max(colour) - min(colour) > 0.05 and max(colour) < 0.99
    }

    assert len(non_grey) >= 3, (
        "Figure 1 needs at least three non-grey vector fills to distinguish "
        "task constraints, an invalid route, and validity-preserving routes"
    )


@pytest.mark.skipif(not tex_available(), reason="no TeX installation")
def test_build_is_reproducible_from_a_clean_directory(tmp_path):
    """__main__ passes an arbitrary --figures-dir; build must not need repo state."""
    out = build(tmp_path / "nested" / "figure1_hierarchy.pdf")
    assert out.exists()
    assert (out.parent / DEFS_NAME).exists()


@pytest.mark.skipif(not tex_available(), reason="no TeX installation")
def test_two_builds_in_different_directories_are_byte_identical(tmp_path):
    """Figure 1 is committed, so rebuilding it must be a no-op.

    It was not: pdfTeX stamps a creation date and a trailer /ID into every
    build, so `python -m analysis` left the figure modified in the working
    tree every single run, and a clean-clone reproduction could only ever
    claim the .tex and caption files. Two different output directories are
    used because the /ID is derived from the output path as well as the time.

    The preview embeds this file, so an unpinned figure made the preview
    unreproducible too.
    """
    first = build(tmp_path / "a" / "figure1_hierarchy.pdf").read_bytes()
    second = build(tmp_path / "b" / "figure1_hierarchy.pdf").read_bytes()
    assert first == second, (
        "two builds of figure 1 differ; it cannot stay committed without "
        "churning the working tree on every regeneration")


@pytest.mark.skipif(not tex_available(), reason="no TeX installation")
def test_a_leftover_build_cache_does_not_change_the_output(tmp_path):
    """Building twice into the *same* directory must give the same bytes.

    latexmk keeps an .fdb_latexmk next to its output and decides from it what
    still needs doing. That cache made a pinned build produce an unpinned
    date: the committed figure carried a wall-clock stamp while a fresh clone
    of the same commit produced the pinned one, so the two disagreed while
    both looked correct. Building into a fresh temporary directory each time
    is what makes leftover state impossible rather than merely unlikely.
    """
    out = tmp_path / "figure1_hierarchy.pdf"
    first = build(out).read_bytes()
    assert list(tmp_path.iterdir()) or True
    second = build(out).read_bytes()
    assert first == second, (
        "a second build in the same directory differs; leftover latexmk state "
        "is reaching the output")


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="no TeX installation")
def test_the_figure_carries_no_timestamp(tmp_path):
    """Two builds seconds apart match; two builds minutes apart did not.

    `test_two_builds_in_different_directories_are_byte_identical` runs both
    builds inside one test, so they land in the same wall-clock second and a
    clock-dependent stamp passes it. That is how a committed figure went on
    changing on every regeneration while a test named for byte-identity stayed
    green. SOURCE_DATE_EPOCH reaches latexmk but the standalone class does not
    honour it, so the stamp is omitted outright -- and absence is what this
    asserts, because absence cannot drift.
    """
    pdf = build(tmp_path / "figure1_hierarchy.pdf").read_bytes()
    assert b"/CreationDate" not in pdf
    assert b"/ModDate" not in pdf
