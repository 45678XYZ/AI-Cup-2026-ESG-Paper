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
    assert ("RoBERTa-large", 0.0) in labels     # frozen anchor
    assert ("RoBERTa-large", 0.3) in labels     # pre-registered structural arm
    assert ("DeBERTa-v2-320M", 0.0) in labels
    assert ("DeBERTa-v2-320M", 0.3) in labels
    assert ("ELECTRA-large", 0.0) in labels
    assert ("ELECTRA-large", 0.3) in labels
    assert ("RoBERTa-base", 0.0) in labels      # generality check, no lambda arm
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
        _arm("DeBERTa-v2-320M", 0.0, 0.1975,
             _contrast(-0.0080, -0.0158, -0.0007), _contrast(0.0502, 0.041, 0.060),
             _contrast(0.0130, 0.003, 0.023), _contrast(-0.0033, -0.010, 0.003)),
        _arm("RoBERTa-large", 0.3, 0.0518,
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
    assert header.count("Official") == 2 and header.count("Whole-row") == 2


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


def test_the_preview_renders_the_new_table():
    from analysis.tables import ALL_TABLE_FILES
    assert "table3_legality_cost.tex" in ALL_TABLE_FILES
