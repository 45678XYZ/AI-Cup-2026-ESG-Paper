"""What enforcing legality costs, measured the same way in every arm.

The claim the paper rests on is that projecting onto the 17 legal states is
free. That is a statement about a *contrast within an arm* -- M1 against M0 --
and the study currently has no artifact that reports it: ``structural_arm`` and
``architecture_screen`` both compare arms to each other on a fixed method, which
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


def test_the_cost_is_reported_with_an_interval_on_both_metrics():
    arm = ARMS[0]
    cost = arm_legality_cost(REPO_ROOT, ORDER, DEV, clusters=CLUSTERS, n_boot=50)
    for metric in ("official_weighted_macro_f1", "tuple_accuracy"):
        row = cost[metric]
        assert row["ci_low"] <= row["delta"] <= row["ci_high"], (arm.label, metric)
        assert 0.0 <= row["p_value"] <= 1.0


def test_the_report_records_what_it_read():
    """Every number has to trace to a file, as the rest of tables/ does."""
    report = build_report(n_boot=50)
    assert len(report["arms"]) == len(ARMS)
    for entry in report["arms"]:
        assert entry["input_sha256"], entry["label"]
