"""What enforcing legality costs, measured the same way in every arm.

The claim the paper rests on is that projecting onto the 17 legal states is
free. That is a statement about a *contrast within an arm* -- M1 against M0 --
and the study currently has no artifact that reports it: ``structural_arm`` and
``runs`` both compare arms to each other on a fixed method, which
is a different question. This module supplies the missing one, for every
(backbone, lambda) arm that exists, so the claim can be read off a single table
rather than assembled by hand from four documents.
"""

import pytest

from analysis.legality_cost import ARMS, arm_legality_cost, build_report
from analysis.load import pdf_clusters
from paper.data import REPO_ROOT, canonical_row_order, load_dev

DEV = load_dev()
ORDER = canonical_row_order(DEV)
CLUSTERS = pdf_clusters(ORDER, DEV)


def test_every_declared_arm_has_predictions_on_disk():
    """An arm named here but absent on disk would silently drop a row from the
    table, and the table's whole point is that it is exhaustive."""
    for arm in ARMS:
        root = REPO_ROOT / arm.root if arm.root else REPO_ROOT
        path = root / "predictions" / "pdf_group_seed42_M0.csv.gz"
        assert path.exists(), f"{arm.label}: {path} missing"


def test_the_arms_cover_every_backbone_and_lambda_that_was_run():
    labels = {(a.backbone, a.structure_lambda) for a in ARMS}
    assert ("Chinese RoBERTa-large", 0.0) in labels     # frozen anchor
    assert ("Chinese RoBERTa-large", 0.3) in labels     # pre-registered structural arm
    assert ("Chinese DeBERTa-v2-320M", 0.0) in labels
    assert ("Chinese DeBERTa-v2-320M", 0.3) in labels
    assert ("Chinese ELECTRA-large", 0.0) in labels
    assert ("Chinese ELECTRA-large", 0.3) in labels
    assert ("Chinese RoBERTa-base", 0.0) in labels      # generality check, no lambda arm
    assert len(ARMS) == 7


def test_projection_drives_the_invalid_rate_to_zero_in_every_arm():
    """This is the constructive half of the claim and it must hold for all of
    them: M1's output space *is* the 17 legal states, whatever produced the
    probabilities."""
    for arm in ARMS:
        root = REPO_ROOT / arm.root if arm.root else REPO_ROOT
        cost = arm_legality_cost(root, ORDER, DEV, clusters=CLUSTERS, n_boot=50)
        assert cost["invalid_rate"]["M1"] == 0.0, arm.label
        assert cost["invalid_rate"]["M0"] > 0.0, arm.label


def test_both_contrasts_are_reported_with_intervals_on_both_metrics():
    cost = arm_legality_cost(REPO_ROOT, ORDER, DEV, clusters=CLUSTERS, n_boot=50)
    for contrast in ("legality_cost", "decoder_vs_projection"):
        for metric in ("official_weighted_macro_f1", "tuple_accuracy"):
            row = cost[contrast][metric]
            assert row["ci_low"] <= row["delta"] <= row["ci_high"], (contrast, metric)
            assert 0.0 <= row["p_value"] <= 1.0


def test_both_constrained_rules_are_legal_in_every_arm():
    """M1 and M4 differ in which legal tuple they pick, never in whether it is
    legal. If either ever emitted an invalid tuple the table's whole framing --
    two equally legal rules that the metrics rank differently -- would be false."""
    for arm in ARMS:
        root = REPO_ROOT / arm.root if arm.root else REPO_ROOT
        cost = arm_legality_cost(root, ORDER, DEV, clusters=CLUSTERS, n_boot=50)
        assert cost["invalid_rate"]["M1"] == 0.0, arm.label
        assert cost["invalid_rate"]["M4"] == 0.0, arm.label


def test_every_contrast_averages_to_its_own_per_seed_deltas():
    """``delta`` is by construction the mean over seeds of the same difference,
    under whichever metric produced it. This report is where that identity is
    worth asserting: the two metrics sit side by side under keys that look
    interchangeable, and a per-seed list built from the wrong one is still
    three small signed floats filed under the right name.
    """
    report = build_report(n_boot=50)
    for arm in report["arms"]:
        for contrast in ("legality_cost", "decoder_vs_projection"):
            for metric, row in arm[contrast].items():
                mean = sum(row["per_seed_delta"]) / len(row["per_seed_delta"])
                assert mean == pytest.approx(row["delta"], abs=1e-12), (
                    arm["label"], contrast, metric)


def test_the_report_records_what_it_read():
    """Every number has to trace to a file, as the rest of tables/ does."""
    report = build_report(n_boot=50)
    assert len(report["arms"]) == len(ARMS)
    for entry in report["arms"]:
        assert entry["input_sha256"], entry["label"]


# --- rendering -------------------------------------------------------------

from analysis.legality_cost import build_caption, render_table  # noqa: E402


def _contrast(delta, low, high):
    return {"delta": delta, "ci_low": low, "ci_high": high, "p_value": 0.1}


def _arm(backbone, lam, invalid, cost_off, cost_row, dec_off, dec_row):
    return {
        "label": f"{backbone} (lambda={lam:g})", "backbone": backbone,
        "structure_lambda": lam, "seeds": [42, 123, 456],
        "invalid_rate": {"M0": invalid, "M1": 0.0, "M4": 0.0},
        "legality_cost": {"official_weighted_macro_f1": cost_off,
                          "tuple_accuracy": cost_row},
        "decoder_vs_projection": {"official_weighted_macro_f1": dec_off,
                                  "tuple_accuracy": dec_row},
    }


STUB = {
    "protocol": "pdf_group", "n_boot": 10000, "bootstrap_seed": 20260814,
    "arms": [
        # DeBERTa: every column's interval clears zero except the decoder's
        # whole-row cell, which is the one cell that must not be bold.
        _arm("Chinese DeBERTa-v2-320M", 0.0, 0.1975,
             _contrast(-0.0080, -0.0158, -0.0007), _contrast(0.0502, 0.041, 0.060),
             _contrast(0.0130, 0.003, 0.023), _contrast(-0.0033, -0.010, 0.003)),
        _arm("Chinese RoBERTa-large", 0.3, 0.0518,
             _contrast(-0.0009, -0.0040, 0.0017), _contrast(0.0138, 0.010, 0.018),
             _contrast(-0.0008, -0.004, 0.002), _contrast(-0.0013, -0.004, 0.001)),
    ],
}


def test_the_table_has_one_row_per_arm():
    rows = [l for l in render_table(STUB).splitlines()
            if "&" in l and "Backbone" not in l and "multicolumn" not in l]
    assert len(rows) == len(STUB["arms"])


def test_both_contrasts_get_their_own_column_pair():
    header = render_table(STUB)
    assert "M1$-$M0" in header and "M4$-$M1" in header
    # The same two column names tables 2 and 7 print, so a reader carrying the
    # official-versus-tuple comparison between them need not re-map it.
    assert header.count("wF1") == 2 and header.count("Tuple acc.") == 2


def test_only_cells_whose_interval_clears_zero_are_bold():
    """The reader is meant to read the table by its bold pattern, so a cell
    that is bold without clearing zero would mislead more than a missing one."""
    line = [l for l in render_table(STUB).splitlines() if "DeBERTa" in l][0]
    cells = [c.strip() for c in line.split("&")]
    assert cells[3].startswith(r"\textbf")     # legality cost, official
    assert cells[4].startswith(r"\textbf")     # legality cost, whole-row
    assert cells[5].startswith(r"\textbf")     # decoder, official
    assert not cells[6].startswith(r"\textbf")  # decoder, whole-row: crosses 0


def test_the_caption_states_that_both_constrained_rules_are_legal():
    """The invalid column reports M0 only. Without the caption a reader could
    conclude the table never checked M4's legality."""
    caption = build_caption(STUB)
    assert "M1 and M4 both emit an invalid tuple on 0" in caption


def test_the_caption_marks_the_family_as_exploratory():
    assert "exploratory" in build_caption(STUB).lower()


# --- the caption must not contradict the cells beside it -------------------
#
# The table is read by its bold pattern and the caption tells the reader what
# the pattern means. A caption sentence that denies an effect the tabular marks
# as present is worse than no sentence: the reader trusts the prose.


def _sentences(caption):
    return [s.strip() for s in caption.replace("\\%", "%").split(". ") if s.strip()]


def _replaced(arm_index, contrast, metric, cell):
    """STUB with one cell swapped, so a hardcoded count shows up as a count
    that does not move when the arms do."""
    import copy
    stub = copy.deepcopy(STUB)
    stub["arms"][arm_index][contrast][metric] = cell
    return stub


def test_the_caption_names_a_metric_whenever_it_talks_about_the_trained_arms():
    """STUB's structurally trained arm gains +0.0138 whole-row with an interval
    clearing zero, and ``render_table`` prints that cell bold. An unscoped
    'the trained arms show neither effect' contradicts the cell beside it, so
    every sentence about that subset has to say which metric it holds for."""
    said = [s for s in _sentences(build_caption(STUB))
            if "structurally trained" in s]
    assert said, "the caption no longer distinguishes the trained arms"
    for sentence in said:
        assert "official" in sentence.lower(), sentence


def test_the_whole_row_count_follows_the_arms_rather_than_the_prose():
    """Both stub arms gain whole-row accuracy; flip one and the caption has to
    say so. A sentence carrying 'all seven arms' as text would not move."""
    assert "in 2 of 2 arms" in build_caption(STUB)
    flipped = _replaced(1, "legality_cost", "tuple_accuracy",
                        _contrast(-0.0138, -0.018, -0.010))
    assert "in 1 of 2 arms" in build_caption(flipped)


def test_the_caption_states_a_direction_only_when_the_detectable_arms_agree():
    """One stub arm's decoder cell clears zero on the official metric and is
    positive, which the caption may report. Make the second one clear zero in
    the opposite direction and there is no shared direction left to state --
    the caption has to drop the claim, not average it."""
    assert "detectable cells is positive" in build_caption(STUB)
    split = _replaced(1, "decoder_vs_projection", "official_weighted_macro_f1",
                      _contrast(-0.0080, -0.013, -0.003))
    caption = build_caption(split)
    assert "detectable cells is positive" not in caption
    assert "detectable cells is negative" not in caption


def test_the_direction_is_not_attached_to_the_subgroup_counted_last():
    """The direction describes every detectable cell, which the tally counts
    across all arms; the clause it used to trail counts the structurally
    trained ones, which in the real report have none. Attached there it reads
    as a claim about the trained arms -- and about them it is exactly wrong,
    since their cells are negative. It has to be its own sentence.
    """
    trailing = [s for s in _sentences(build_caption(STUB))
                if "trained ones" in s]
    assert trailing, "the caption no longer counts the trained arms separately"
    for sentence in trailing:
        assert "positive" not in sentence, sentence
        assert "negative" not in sentence, sentence


def test_the_preview_renders_the_new_table():
    from analysis.tables import ALL_TABLE_FILES
    assert "table3_legality_cost.tex" in ALL_TABLE_FILES
