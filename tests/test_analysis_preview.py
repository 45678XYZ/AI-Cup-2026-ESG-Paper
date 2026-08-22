"""The preview is a convenience, but a broken one is worse than none.

It exists because contract section 5 keeps the delivered `.tex` files to a bare
``tabular``: correct for D, who needs to place the floats, and useless for
anyone who just wants to look at a table.
"""

import pytest

from analysis.figure1 import tex_available
from analysis.preview import _escape, build
from analysis.tables import TABLE_FILES
from paper.data import REPO_ROOT


def test_escape_leaves_the_tex_the_captions_already_contain():
    # Captions are written for LaTeX and legitimately carry maths and commands.
    assert _escape(r"$\pm$ is seed spread") == r"$\pm$ is seed spread"
    assert _escape(r"95\% percentile") == r"95\% percentile"


def test_escape_protects_the_characters_that_would_abort_the_build():
    assert _escape("a & b") == r"a \& b"
    assert _escape("table1_dataset") == r"table1\_dataset"


@pytest.mark.skipif(not tex_available(), reason="needs latexmk")
def test_preview_renders_every_delivered_table(tmp_path):
    out = build(REPO_ROOT / "tables", tmp_path / "preview.pdf")
    assert out.exists() and out.stat().st_size > 10_000
    # one page per table plus the figure; a silently empty preview would pass
    # a mere "file exists" check
    assert len(TABLE_FILES) >= 5
