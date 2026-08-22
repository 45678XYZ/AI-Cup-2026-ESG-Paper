"""The writing report is hand-written prose quoting generated numbers.

`docs/study_report.md` is 82% background, method and protocol -- text a rerun
cannot invalidate -- and 18% figures lifted from `tables/`. The two halves fail
differently. Prose goes stale only when the study design changes, and a reader
can tell. A quoted interval goes stale the moment anything is recomputed, and
it goes stale *silently*: the sentence around it still reads perfectly well.

That already happened once. An earlier draft concluded "the evidence points to
route 2" on the strength of five intervals computed on the official metric
alone; adding the tuple-accuracy family reversed it. Nothing in the repository
would have caught that, because a wrong conclusion in prose is not a failing
assertion.

So the report stays hand-written and these tests hold its numbers to the
deliverables. They compare against `tables/` rather than against constants:
pinning expected values here would only move the staleness problem into the
test file.
"""

import json
import re

from paper.data import REPO_ROOT

DOCS = REPO_ROOT / "docs"
TABLES = REPO_ROOT / "tables"

REPORT = (DOCS / "study_report.md").read_text(encoding="utf-8")
INTERVAL = re.compile(r"\[[-+]?\d+\.\d+,\s*[-+]?\d+\.\d+\]")


def _paper_intervals():
    """Intervals that reach the paper: the table 2 caption and table 3."""
    text = ((TABLES / "table2_main_caption.txt").read_text(encoding="utf-8")
            + (TABLES / "table3_regimes.tex").read_text(encoding="utf-8"))
    return set(INTERVAL.findall(text))


def _delivered_intervals():
    """Every interval any deliverable carries, the brief included.

    The two directions are deliberately asymmetric. The report must quote
    everything that reaches the paper, so a moved number cannot slip past. It
    need not quote every interval in the brief -- the brief carries secondary
    analyses at full length -- but it may not contain one that exists nowhere,
    which is how a typo or an invented figure would show up.
    """
    return _paper_intervals() | set(INTERVAL.findall(
        (TABLES / "findings.md").read_text(encoding="utf-8")))


def test_report_quotes_every_interval_that_reaches_the_paper():
    """Table 2's caption and Table 3 -- if one moves, the report moves."""
    missing = sorted(ci for ci in _paper_intervals() if ci not in REPORT)
    assert not missing, (
        f"{len(missing)} interval(s) in tables/ are absent from the report; "
        f"rerun `python -m analysis`, then update docs/study_report.md: {missing}"
    )


def test_report_invents_no_interval_of_its_own():
    """The reverse direction: an interval the deliverables do not contain is
    either a typo or a number someone made up."""
    delivered = _delivered_intervals()
    unknown = sorted(ci for ci in set(INTERVAL.findall(REPORT))
                     if ci not in delivered)
    assert not unknown, f"report contains intervals absent from tables/: {unknown}"


def test_report_quotes_the_case_analysis_counts():
    totals = json.loads(
        (TABLES / "case_analysis.json").read_text(encoding="utf-8"))["totals"]
    numbers = set(re.findall(r"[-+]?[\d,]+", REPORT))
    for label, value in (
        ("invalid tuples", totals["n_invalid"]),
        ("fields repaired", totals["fields_repaired"]),
        ("fields destroyed", totals["fields_destroyed"]),
        ("N/A net", totals["na"]["net"]),
        ("substantive net", totals["substantive"]["net"]),
    ):
        rendered = {f"{value:,}", str(value), f"{value:+,}", f"{value:+}"}
        assert rendered & numbers, f"{label} = {value} is absent from the report"


def test_report_defers_to_the_generated_brief():
    """The report is read once; findings.md is regenerated every run. When the
    two disagree the generated one wins, and the report must say so."""
    assert "findings.md" in REPORT
    assert "為準" in REPORT or "takes precedence" in REPORT.lower()
