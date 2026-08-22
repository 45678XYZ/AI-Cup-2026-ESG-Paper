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
    TABLE_FILES,
    build_captions,
    render_table1,
    render_table2,
    render_table3,
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
    for stem in ("table1_dataset", "table2_main", "table3_regimes"):
        caption = (tmp_path / f"{stem}_caption.txt").read_text(encoding="utf-8")
        assert caption.strip()
    # The two disclosures the plan requires captions to carry.
    assert "Misleading" in (tmp_path / "table1_dataset_caption.txt").read_text(
        encoding="utf-8")
    assert "not a bias" in (tmp_path / "table3_regimes_caption.txt").read_text(
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
    assert "same 7 reports" in captions["table3_regimes"]
    assert "49" not in captions["table3_regimes"]


# ------------------------------------------------------- table 2 contrasts
# Plan §5 asks for the paired Δ and 95% CI "under the table". aggregate.py
# computes them for the five pre-specified contrasts and Holm-corrects the
# family, but nothing used to carry them into a deliverable: the table shipped
# with seed spread and no interval, so a reader could not tell whether any
# difference in it survives resampling.


def test_table2_caption_carries_every_pre_specified_contrast():
    contrasts = SUMMARIES["pdf_group"]["contrasts"]
    caption = build_captions(AUDIT, contrasts=contrasts)["table2_main"]

    assert len(contrasts) == 5, "the frozen family is five contrasts"
    for key, row in contrasts.items():
        assert key in caption, f"{key} missing from the caption"
        assert row["description"] in caption
        assert f"{row['ci_low']:.3f}" in caption and f"{row['ci_high']:.3f}" in caption
    assert "Holm" in caption
    assert "10,000" in caption


def test_table2_caption_states_how_many_intervals_exclude_zero():
    """The one sentence a reader needs before reading any individual row."""
    contrasts = SUMMARIES["pdf_group"]["contrasts"]
    caption = build_captions(AUDIT, contrasts=contrasts)["table2_main"]
    n = sum(1 for r in contrasts.values() if r["ci_low"] > 0 or r["ci_high"] < 0)
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    assert words[n] in caption.lower()


def test_table2_caption_follows_the_numbers_rather_than_repeating_them():
    """A transcribed interval would not move when the study is re-run."""
    contrasts = copy.deepcopy(SUMMARIES["pdf_group"]["contrasts"])
    key = next(iter(contrasts))
    contrasts[key].update(delta=0.123, ci_low=0.111, ci_high=0.222)
    caption = build_captions(AUDIT, contrasts=contrasts)["table2_main"]
    assert "0.123" in caption
    assert "[0.111, 0.222]" in caption


def test_captions_still_build_without_contrasts():
    """Table 1 and 3 never needed them; omitting them must not break the run."""
    assert build_captions(AUDIT)["table2_main"].strip()


def test_written_caption_includes_the_contrasts(tmp_path):
    write_tables(tmp_path, AUDIT, SUMMARIES, REGIMES, INPUTS)
    caption = (tmp_path / "table2_main_caption.txt").read_text(encoding="utf-8")
    assert "Holm" in caption, "write_tables must pass the contrasts through"


def test_contrast_sentence_agrees_in_number():
    """The caption is copied verbatim into the paper, so "one ... exclude" is
    not a typo anyone downstream is expected to catch."""
    from analysis.tables import _contrast_sentence

    def rows(n_excluding, total=5):
        out = {}
        for i in range(total):
            excl = i < n_excluding
            out[f"M{i+1}-M{i}"] = {
                "delta": -0.01, "ci_low": -0.02 if excl else -0.02,
                "ci_high": -0.001 if excl else 0.01,
                "description": "d", "p_value": 0.1, "p_holm": 0.1,
            }
        return out

    assert "One of the five intervals excludes zero" in _contrast_sentence(rows(1), "m")
    assert "Two of the five intervals exclude zero" in _contrast_sentence(rows(2), "m")
    assert "None of the five intervals exclude zero" in _contrast_sentence(rows(0), "m")


def test_table2_caption_carries_both_metric_families():
    """The secondary metric has no column of its own -- the column count is what
    D measured the page budget against -- so the caption is the only place its
    intervals can appear, and it must say which family the competition ranks on."""
    primary = SUMMARIES["pdf_group"]["contrasts"]
    secondary = SUMMARIES["pdf_group"]["consistent_contrasts"]
    cap = build_captions(AUDIT, contrasts=primary,
                         consistent_contrasts=secondary)["table2_main"]
    assert "path-constrained" in cap.lower()
    for key in secondary:
        assert key in cap
    assert cap.count("Holm") >= 1


def test_table2_caption_discloses_which_family_was_not_pre_specified():
    """The path-constrained metric was adopted after the primary analysis. A
    caption that reports its intervals without saying so presents a post-hoc
    choice as a planned one."""
    cap = build_captions(
        AUDIT, contrasts=SUMMARIES["pdf_group"]["contrasts"],
        consistent_contrasts=SUMMARIES["pdf_group"]["consistent_contrasts"],
        tuple_contrasts=SUMMARIES["pdf_group"]["tuple_contrasts"],
    )["table2_main"]
    assert "not pre-specified" in cap
    assert "named in advance by the analysis plan" in cap


def test_table2_caption_reports_the_pre_specified_secondary_family():
    """tuple accuracy is a printed column and a planned family; its intervals
    belong under the table whatever the newer metric says."""
    tuples = SUMMARIES["pdf_group"]["tuple_contrasts"]
    cap = build_captions(AUDIT, contrasts=SUMMARIES["pdf_group"]["contrasts"],
                         tuple_contrasts=tuples)["table2_main"]
    assert "tuple accuracy" in cap.lower()
    for key in tuples:
        assert key in cap


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
