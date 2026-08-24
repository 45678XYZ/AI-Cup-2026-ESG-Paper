"""The writing report is hand-written prose quoting generated numbers.

`docs/writing/study_report.md` is 82% background, method and protocol -- text a rerun
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

DOCS = REPO_ROOT / "docs" / "writing"
TABLES = REPO_ROOT / "tables"

REPORT = (DOCS / "study_report.md").read_text(encoding="utf-8")
INTERVAL = re.compile(r"\[[-+]?\d+\.\d+,\s*[-+]?\d+\.\d+\]")
# ``p_Holm=0.025`` in the brief, ``| 0.025 |`` in a report table: match the
# number itself and compare the sets.
P_HOLM = re.compile(r"p_?Holm[=\s]*([01]\.\d{3})", re.IGNORECASE)


PAPER_SOURCES = ("table2_main_caption.txt", "table6_regimes.tex",
                 "table4_contrasts.tex", "table4_contrasts_caption.txt")


def _paper_intervals():
    """Every interval printed in a tabular or caption that reaches the paper.

    Table 4 carries the contrast families that used to live in table 2's
    caption; both are listed so moving a number between them cannot let it
    escape the report.
    """
    text = "".join((TABLES / name).read_text(encoding="utf-8")
                   for name in PAPER_SOURCES)
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
        f"rerun `python -m analysis`, then update docs/writing/study_report.md: {missing}"
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


def _quoted_p_values():
    """Corrected p-values the report states, and only those.

    Two forms appear: ``p_Holm=0.025`` inline, and a ``p_Holm`` column in a
    markdown table. The column is located from its header rather than by
    position, so a reordered table does not silently stop being checked -- and
    F1 scores in neighbouring columns are never mistaken for p-values.
    """
    found = set(P_HOLM.findall(REPORT))
    column, header = None, None
    for line in REPORT.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            column, header = None, None
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(c and set(c) <= set("-: ") for c in cells):
            # A separator row starts a table: the column is re-derived from the
            # header just above it, so the previous table's index cannot leak
            # into this one and pick up an F1 score.
            column = next((i for i, c in enumerate(header or [])
                           if "p_holm" in c.lower()), None)
            continue
        if column is not None and column < len(cells):
            found |= set(re.findall(r"([01]\.\d{3})", cells[column]))
        header = cells
    return found


def test_report_invents_no_corrected_p_value():
    """The interval guard cannot see p-values, and a mistyped one is exactly as
    wrong as a mistyped interval -- more so now that every verdict rests on it.

    This caught a real error: the report carried 0.482 for a contrast whose
    corrected p is 0.643, copied from the neighbouring family where the same Δ
    and interval get a different rank under Holm.
    """
    delivered = set(P_HOLM.findall(
        (TABLES / "findings.md").read_text(encoding="utf-8")))
    delivered |= set(re.findall(r"([01]\.\d{3})", "".join(
        (TABLES / name).read_text(encoding="utf-8") for name in PAPER_SOURCES)))

    unknown = sorted(p for p in _quoted_p_values() if p not in delivered)
    assert not unknown, (
        "the report quotes corrected p-values that no deliverable contains: "
        f"{unknown}"
    )


def test_report_quotes_the_mechanism_rates():
    """Two figures the structural argument rests on -- what the metric pays for
    an illegal row, and how well the model already makes the call the hierarchy
    determines. Both were hand-computed into an earlier draft before any script
    produced them, which is how a load-bearing number goes stale in silence."""
    totals = json.loads(
        (TABLES / "case_analysis.json").read_text(encoding="utf-8"))["totals"]
    credit = f"{totals['partial_credit_on_invalid'] * 100:.1f}"
    assert credit in REPORT, f"partial credit {credit}% is absent from the report"

    na = [row["na_determination"]
          for row in totals["hierarchy_information"].values()]
    lo, hi = f"{min(na) * 100:.0f}", f"{max(na) * 100:.0f}"
    assert f"{lo}–{hi}%" in REPORT or f"{lo}-{hi}%" in REPORT, (
        f"the N/A-determination range {lo}-{hi}% is absent from the report")
