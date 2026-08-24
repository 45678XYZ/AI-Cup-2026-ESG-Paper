"""Emitted tables must match the placeholder structure D laid out against."""

import copy
import json
import re
from pathlib import Path

import pytest

from analysis.aggregate import protocol_summary, regime_comparison
from analysis.audit import full_audit
from analysis.load import EXAMPLES_ROOT, pdf_clusters
from analysis.tables import (
    FAMILY_LABELS,
    _word,
    TABLE_FILES,
    render_table4,
    build_captions,
    render_table1,
    render_table2,
    render_table6_regimes,
    table_inputs,
    write_tables,
)
from paper.data import REPO_ROOT, canonical_row_order, file_sha256, load_dev

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
INPUTS = table_inputs(EXAMPLES_ROOT)


def _column_count(tex):
    spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", tex).group(1)
    return len(re.findall(r"[lrc]", spec))


def test_tables_carry_only_a_tabular_environment():
    rendered = (render_table1(AUDIT), render_table2(SUMMARIES["pdf_group"]),
                render_table6_regimes(REGIMES))
    for tex in rendered:
        assert tex.lstrip().startswith(r"\begin{tabular}")
        # Layout is D's call under the 8-page budget (contract section 5).
        for forbidden in (r"\begin{table}", r"\caption", r"\label"):
            assert forbidden not in tex


def test_column_counts_match_the_placeholders():
    for name, rendered in (
        ("table1_dataset.tex", render_table1(AUDIT)),
        ("table2_main.tex", render_table2(SUMMARIES["pdf_group"])),
        ("table6_regimes.tex", render_table6_regimes(REGIMES)),
    ):
        placeholder = (PLACEHOLDERS / name).read_text(encoding="utf-8")
        assert _column_count(rendered) == _column_count(placeholder), name


def test_table1_leaves_the_unlabelled_test_column_empty():
    tex = render_table1(AUDIT)
    misleading = [l for l in tex.splitlines() if "Misleading" in l][0]
    assert misleading.count("n/a") == 1   # Test column only
    assert "2" in misleading              # Development column carries n=2


def test_table1_prints_the_audited_counts():
    """Row by row rather than by substring. The previous version asserted that
    "50" appeared somewhere in the table, which passed for two months while the
    Companies cell was wrong: the release spells one company two ways, so the
    raw count read 50 against 49 reports. A substring check cannot tell a right
    number from a wrong one that happens to be present."""
    rows = {
        line.split("&")[0].strip(): [c.strip() for c in line.split("&")[1:]]
        for line in render_table1(AUDIT).splitlines() if "&" in line
    }
    assert rows["Paragraphs"][:2] == ["2000", r"2000 \\"]
    assert rows["Source reports (PDFs)"][:2] == ["49", r"49 \\"]
    assert rows["Companies"][:2] == ["49", r"49 \\"]


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
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)

    for name in TABLE_FILES:
        assert (tmp_path / name).exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for name in TABLE_FILES:
        entry = manifest["tables"][name]
        assert entry["input_files"], name
        assert len(entry["input_sha256"]) == len(entry["input_files"])
        assert all(v.startswith("sha256:") for v in entry["input_sha256"].values())


def test_captions_are_written_beside_the_tables(tmp_path):
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)
    for stem in ("table1_dataset", "table2_main", "table6_regimes"):
        caption = (tmp_path / f"{stem}_caption.txt").read_text(encoding="utf-8")
        assert caption.strip()
    # The two disclosures the plan requires captions to carry.
    assert "Misleading" in (tmp_path / "table1_dataset_caption.txt").read_text(
        encoding="utf-8")
    assert "not a bias" in (tmp_path / "table6_regimes_caption.txt").read_text(
        encoding="utf-8")


# ---------------------------------------------------------------- provenance
# Two failures the manifest and the captions used to allow silently: the
# manifest checksummed results/*.json while every score came from
# predictions/*.csv.gz, and the captions transcribed audited counts as literal
# text. Neither made a number wrong on the day it was written, which is exactly
# why both need a test rather than a reader's attention.


def test_manifest_records_the_files_each_table_actually_reads(tmp_path):
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    scored = manifest["tables"]["table2_main.tex"]["input_files"]
    assert scored, "table 2 must record the files its numbers came from"
    assert all("predictions" in p for p in scored)
    assert not any(p.endswith(".json") and "results" in p for p in scored)

    audited = manifest["tables"]["table1_dataset.tex"]["input_files"]
    assert any("dataset" in p for p in audited)
    assert any("splits" in p for p in audited)
    assert not any("predictions" in p for p in audited), (
        "table 1 counts come from the dataset audit, not from predictions"
    )


def test_manifest_records_repo_relative_paths(tmp_path):
    """The manifest is committed and travels with the paper, so an absolute
    path would bake one machine's directory layout into a published artifact
    and be uncheckable from a fresh clone. It would also contradict the
    relative paths the contract's own example specifies.
    """
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for name in TABLE_FILES:
        entry = manifest["tables"][name]
        for path in entry["input_files"]:
            assert not Path(path).is_absolute(), f"{name}: {path}"
            assert not path.startswith(".."), f"{name} escapes the repo: {path}"
        # The checksums stay keyed by the same strings that are listed.
        assert sorted(entry["input_sha256"]) == sorted(entry["input_files"])


def test_manifest_hashes_the_file_rather_than_the_recorded_string(tmp_path):
    """Recording a relative path must not change which bytes were hashed."""
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["tables"]["table2_main.tex"]["input_sha256"]
    for path, digest in recorded.items():
        assert digest == file_sha256(REPO_ROOT / path), path


def test_each_table_names_the_script_that_computed_it(tmp_path):
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    scripts = {n: manifest["tables"][n]["source_script"] for n in TABLE_FILES}
    assert scripts["table1_dataset.tex"] == "analysis/audit.py"
    assert scripts["table2_main.tex"] == "analysis/aggregate.py"


def test_write_tables_refuses_to_emit_an_empty_audit_trail(tmp_path):
    """A root holding predictions/ but no results/ used to produce all three
    tables with an empty input_files and no warning at all."""
    empty = {name: [] for name in TABLE_FILES}
    with pytest.raises(ValueError, match="no input files recorded"):
        write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, empty)

    # A partially-populated map is the likelier accident, and must fail too.
    partial = dict(INPUTS, **{"table2_main.tex": []})
    with pytest.raises(ValueError, match="table2_main.tex"):
        write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, partial)


def test_table1_caption_reports_the_audited_counts():
    caption = build_captions(AUDIT)["table1_dataset"]
    absent = AUDIT["splits"]["calibration_without_misleading"]
    assert f"{absent['n_without']} of the {absent['n_rotations']} rotations" in caption


def test_captions_move_with_a_resampled_split():
    """The whole point: a caption that stays put while the audit moves is a
    transcribed number wearing a generated number's clothes."""
    altered = copy.deepcopy(AUDIT)
    altered["splits"]["calibration_without_misleading"]["n_without"] = 7
    caption = build_captions(altered)["table1_dataset"]
    assert "7 of the" in caption
    assert "18 of the" not in caption


def test_table2_and_3_captions_follow_the_audit_too():
    """reviewer flagged table 1; these two carry transcribed counts as well."""
    altered = copy.deepcopy(AUDIT)
    altered["development"]["paragraphs"] = 1234
    altered["development"]["pdfs"] = 7
    captions = build_captions(altered, seeds=(1, 2))

    assert "1,234" in captions["table2_main"]
    assert "2,000" not in captions["table2_main"]
    assert "two seeds" in captions["table2_main"]
    assert "three seeds" not in captions["table2_main"]
    assert "same 7 reports" in captions["table6_regimes"]
    assert "49" not in captions["table6_regimes"]


# ------------------------------------------------------- table 2 contrasts
# Plan §5 asks for the paired Δ and 95% CI "under the table". aggregate.py
# computes them for the five pre-specified contrasts and Holm-corrects the
# family, but nothing used to carry them into a deliverable: the table shipped
# with seed spread and no interval, so a reader could not tell whether any
# difference in it survives resampling.


def test_table4_carries_every_pre_specified_contrast():
    """Each contrast must appear once per family, or a reader cannot tell that
    the same hypotheses were tested under every metric."""
    summary = SUMMARIES["pdf_group"]
    tex = render_table4(summary)
    for key in summary["contrasts"]:
        assert key in tex, f"{key} missing from table 4"


def test_table4_caption_counts_what_survives_the_correction_not_the_intervals():
    """The five contrasts are tested together. A caption that counts intervals
    excluding zero states an uncorrected result under a sentence that says
    "Holm-corrected", which is the multiplicity error the correction exists to
    prevent."""
    contrasts = SUMMARIES["pdf_group"]["contrasts"]
    caption = build_captions(AUDIT, contrasts=contrasts)["table4_contrasts"]
    n = sum(1 for r in contrasts.values() if r["p_holm"] < 0.05)
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    assert f"{words[n]} on Weighted macro-F1" in caption
    assert "uncorrected" in caption.lower()   # the intervals must be labelled


def test_table4_follows_the_numbers_rather_than_repeating_them():
    """A transcribed interval would not move when the study is re-run."""
    summary = copy.deepcopy(SUMMARIES["pdf_group"])
    key = next(iter(summary["contrasts"]))
    summary["contrasts"][key].update(delta=0.123, ci_low=0.111, ci_high=0.222)
    tex = render_table4(summary)
    assert "+0.123" in tex
    assert "[0.111, 0.222]" in tex


def test_captions_still_build_without_contrasts():
    """Table 1 and 3 never needed them; omitting them must not break the run."""
    assert build_captions(AUDIT)["table2_main"].strip()


def test_written_caption_includes_the_contrasts(tmp_path):
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)
    caption = (tmp_path / "table2_main_caption.txt").read_text(encoding="utf-8")
    assert "Holm" in caption, "write_tables must pass the contrasts through"


def test_survival_summary_counts_each_family_separately():
    """The caption is copied verbatim into the paper, so the count has to come
    from the corrected p-values of each family, not from a stored sentence."""
    from analysis.tables import _survival_summary

    def rows(n_surviving, total=5):
        return {f"M{i+1}-M{i}": {
            "delta": -0.01, "ci_low": -0.02, "ci_high": -0.001,
            "description": "d", "p_value": 0.01,
            "p_holm": 0.01 if i < n_surviving else 0.4,
        } for i in range(total)}

    sentence = _survival_summary({"contrasts": rows(0),
                                  "tuple_contrasts": rows(2)})
    assert "none on Weighted macro-F1" in sentence
    assert "two on Tuple accuracy" in sentence
    assert "Of the five contrasts" in sentence


def test_table4_caption_names_the_metrics_and_their_provenance():
    """The secondary metric has no column of its own -- the column count is what
    D measured the page budget against -- so the caption is the only place its
    intervals can appear, and it must say which family the competition ranks on."""
    primary = SUMMARIES["pdf_group"]["contrasts"]
    secondary = SUMMARIES["pdf_group"]["consistent_contrasts"]
    cap = build_captions(AUDIT, contrasts=primary,
                         consistent_contrasts=secondary)["table4_contrasts"]
    assert "path-constrained" in cap.lower()
    assert "hierarchical" in cap.lower()
    assert cap.count("Holm") >= 1


def test_table4_caption_discloses_which_family_was_not_pre_specified():
    """The path-constrained metric was adopted after the primary analysis. A
    caption that reports its intervals without saying so presents a post-hoc
    choice as a planned one."""
    cap = build_captions(
        AUDIT, contrasts=SUMMARIES["pdf_group"]["contrasts"],
        consistent_contrasts=SUMMARIES["pdf_group"]["consistent_contrasts"],
        tuple_contrasts=SUMMARIES["pdf_group"]["tuple_contrasts"],
    )["table4_contrasts"]
    assert "not pre-specified" in cap
    assert "named in the analysis plan in advance" in cap


def test_table4_caption_reports_the_pre_specified_secondary_family():
    """tuple accuracy is a printed column and a planned family; its intervals
    belong under the table whatever the newer metric says."""
    tuples = SUMMARIES["pdf_group"]["tuple_contrasts"]
    cap = build_captions(AUDIT, contrasts=SUMMARIES["pdf_group"]["contrasts"],
                         tuple_contrasts=tuples)["table4_contrasts"]
    assert "tuple accuracy" in cap.lower()
    assert "named in the" in cap.lower()


def test_table2_caption_computes_the_gap_between_the_two_metrics():
    """The secondary metric is absent from the table, so the caption has to
    locate it: how far M0 falls under it, and that the constrained arms do not
    move at all. Both are computed, never transcribed."""
    primary = SUMMARIES["pdf_group"]["contrasts"]
    secondary = SUMMARIES["pdf_group"]["consistent_contrasts"]
    methods = SUMMARIES["pdf_group"]["methods"]
    cap = build_captions(AUDIT, contrasts=primary, consistent_contrasts=secondary,
                         methods=methods)["table2_main"]
    assert f"{methods['M0']['consistent_weighted_macro_f1_mean']:.3f}" in cap
    assert "M1--M6" in cap


def test_table1_flags_that_every_report_is_shared_with_the_test_split():
    """C's remit names this explicitly: the 100% dev/test PDF overlap must be
    prominent, not buried in another table's caption. It is the reason Table 3
    exists at all."""
    tex = render_table1(AUDIT)
    shared = [l for l in tex.splitlines() if "shared" in l.lower()]
    assert len(shared) == 1, "no row states the overlap"
    n = AUDIT["pdf_overlap"]["n_shared"]
    assert shared[0].count(str(n)) == 2       # both columns, not a footnote
    assert "n/a" not in shared[0]             # this one IS known for test


def test_the_caption_discloses_the_known_cross_split_duplicate():
    """Plan section 4.1 requires the duplicate to be disclosed, not made
    prominent -- unlike the report overlap, which section 7 does require in the
    tabular. One duplicated paragraph out of 2,000 is a disclosure, so it reads
    better as a clause than as a row whose two columns both say 1."""
    cap = build_captions(AUDIT)["table1_dataset"]
    assert "duplicat" in cap.lower()
    assert _word(len(AUDIT["duplicates"]["dev_test"])) in cap


def test_table1_caption_states_the_overlap_and_the_duplicate():
    cap = build_captions(AUDIT)["table1_dataset"]
    assert str(AUDIT["pdf_overlap"]["n_shared"]) in cap
    assert "duplicate" in cap.lower()


# --- table 4: the contrast families ----------------------------------------
#
# The study's central result is four metrics disagreeing about the same five
# pre-specified contrasts. Until now it lived only in Table 2's caption, which
# had grown to some 500 words and still had no room for the fourth family. A
# statistical result of that size belongs in a table.


def test_table4_lists_every_contrast_of_every_family():
    """Derived from FAMILY_LABELS rather than a hard-coded list of four.

    The list used to be written out here, which meant demoting two families
    broke this test for the wrong reason -- it was asserting the old design
    rather than the property "every contrast of every reported family appears
    exactly once".
    """
    summary = SUMMARIES["pdf_group"]
    tex = render_table4(summary)
    families = [k for k, _ in FAMILY_LABELS if summary.get(k)]
    body = [l for l in tex.splitlines() if l.endswith(r"\\") and "midrule" not in l]
    # one header row plus one row per (family, contrast)
    assert len(body) == 1 + sum(len(summary[k]) for k in families)
    for key in summary["contrasts"]:
        assert tex.count(key) == len(families)


def test_table4_marks_what_survived_the_correction():
    """The reader has to be able to see the verdict without recomputing it."""
    summary = SUMMARIES["pdf_group"]
    tex = render_table4(summary)
    survivors = sum(1 for family, _ in FAMILY_LABELS if summary.get(family)
                    for row in summary[family].values() if row["p_holm"] < 0.05)
    assert tex.count(r"\textbf{") >= survivors


def test_table4_carries_only_a_tabular():
    tex = render_table4(SUMMARIES["pdf_group"])
    assert tex.lstrip().startswith(r"\begin{tabular}")
    for forbidden in (r"\begin{table}", r"\caption", r"\label"):
        assert forbidden not in tex


# --- table 5: the metric study ----------------------------------------------


def test_table5_shows_the_two_metrics_disagreeing_about_the_winner():
    """If the columns ever agreed on the best method the table would have no
    reason to exist, and the paper would need a different
 the gold itself contained
    illegal tuples, "the metric pays for combinations the labels call
    impossible" would be our opinion rather than an internal inconsistency.

    It lives in the caption rather than the tabular because it is a single
    fact about the label space, not a development-versus-test statistic -- but
    a load-bearing fact left unchecked is how it stops being true, so it is
    asserted here against the audit rather than trusted to prose."""
    caption = build_captions(AUDIT)["table1_dataset"]
    assert f"{AUDIT['development']['invalid_rows']} " in caption or "no development row violates" in caption
    assert "120" in caption and "17" in caption


def test_table1_caption_gives_the_size_of_the_legal_space():
    """"15 / 17" in the tabular says nothing about how much the hierarchy
    excludes; 17 of 120 does."""
    caption = build_captions(AUDIT)["table1_dataset"]
    assert "17" in caption and "120" in caption


def test_table4_reports_only_the_two_pre_specified_families():
    """docs/inference_families.md settles this: the official metric and tuple
    exact-match are the pre-specified families; path-constrained wF1 and hF
    were added after the primary analysis and drop to prose.

    Rendering four families side by side presents all four as equally planned,
    which is the impression the whole document exists to prevent -- and it puts
    ten exploratory tests on the page for a reader to count.
    """
    body = render_table4(SUMMARIES["pdf_group"])
    assert "Weighted macro-F1 (official)" in body
    assert "Tuple accuracy" in body
    assert "Path-constrained" not in body
    assert "Hierarchical F1" not in body


def test_table4_caption_says_where_the_demoted_metrics_went():
    """Silently dropping two families a previous draft carried is the kind of
    edit that reads as tidying and lands as selective reporting."""
    caption = build_captions(AUDIT, contrasts=SUMMARIES["pdf_group"]["contrasts"],
                             tuple_contrasts=SUMMARIES["pdf_group"]["tuple_contrasts"],
                             methods=SUMMARIES["pdf_group"]["methods"],
                             )["table4_contrasts"]
    assert "exploratory" in caption.lower()



def test_table8_caption_names_the_class_that_is_structurally_unreachable():
    """`Misleading` has two instances and is never predicted, so a quarter of
    the highest-weighted field's macro-F1 is out of reach by construction. A
    reader given the shortfall without that fact will over-estimate what is
    available."""
    caption = build_captions(AUDIT, methods=SUMMARIES["pdf_group"]["methods"],
                             )["table5_headroom"]
    assert "Misleading" in caption


def test_table4_caption_carries_the_resolution_the_deleted_table_showed():
    """table7 used to say this by re-presenting table 4's official block sorted
    by effect size. Everything it showed was already in those rows, so it is a
    sentence now -- but a sentence that vanishes is how a deleted table becomes
    a deleted finding."""
    caption = build_captions(AUDIT, contrasts=SUMMARIES["pdf_group"]["contrasts"],
                             methods=SUMMARIES["pdf_group"]["methods"],
                             )["table4_contrasts"]
    family = SUMMARIES["pdf_group"]["contrasts"]
    shaped = [k for k, r in family.items()
              if (r["ci_low"] > 0 or r["ci_high"] < 0) and r["p_holm"] >= 0.05]
    if shaped:
        assert "excluding zero" in caption
        assert "range across the seven" in caption

