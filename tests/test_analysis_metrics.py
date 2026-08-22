"""The fast scorer must be the official scorer, bit for bit.

analysis/metrics.py exists only for speed. Its licence to exist is this file:
if it ever disagrees with paper/score.py -- the contract's single source of
truth for the metric -- the bootstrap is measuring something the paper does not
report.
"""

import numpy as np
import pytest

from analysis.load import EXAMPLES_ROOT, METHODS, load_aligned, pdf_clusters
from analysis.metrics import (
    conditional_field_macro_f1,
    hierarchical_f1,
    hierarchical_prf,
    consistent_weighted_macro_f1,
    encode,
    enforce_ancestors,
    field_macro_f1,
    weighted_macro_f1,
)
from paper.artifacts import read_predictions
from paper.data import canonical_row_order, load_dev
from paper.labels import (
    CONDITIONING_SUBSET,
    EVAL_FIELDS,
    FIELD_ALIAS,
    FIELDS,
    LABEL2ID,
    STATES,
)
from paper.score import compute_per_field_f1, compute_weighted_macro_f1, macro_f1

DEV = load_dev()
ORDER = canonical_row_order(DEV)
RECORDS = read_predictions(
    EXAMPLES_ROOT / "predictions" / "pdf_group_seed42_M3.csv.gz"
)
# The unconstrained arm: the only one whose predictions break the hierarchy.
RECORDS_M0 = read_predictions(
    EXAMPLES_ROOT / "predictions" / "pdf_group_seed42_M0.csv.gz"
)


def _reference(records, idx=None):
    """paper/score.py's answer, from the raw label strings."""
    if idx is not None:
        records = [records[i] for i in idx]
    gold = [{f: r[f"gold_{FIELD_ALIAS[f]}"] for f in FIELDS} for r in records]
    pred = [{f: r[f"pred_{FIELD_ALIAS[f]}"] for f in FIELDS} for r in records]
    return gold, pred


def test_matches_paper_score_on_the_full_set():
    gold, pred = encode(RECORDS)
    g, p = _reference(RECORDS)
    assert weighted_macro_f1(gold, pred) == pytest.approx(
        compute_weighted_macro_f1(g, p), abs=1e-12
    )


def test_matches_paper_score_on_random_subsets():
    # Subsets are where the present-labels-only convention bites: Misleading
    # (n=2) drops in and out, changing which classes are averaged over.
    gold, pred = encode(RECORDS)
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = rng.choice(len(RECORDS), size=int(rng.integers(50, len(RECORDS))),
                         replace=True)
        g, p = _reference(RECORDS, idx)
        assert weighted_macro_f1(gold, pred, idx) == pytest.approx(
            compute_weighted_macro_f1(g, p), abs=1e-12
        )


def test_per_field_matches_paper_score():
    gold, pred = encode(RECORDS)
    g, p = _reference(RECORDS)
    reference = compute_per_field_f1(g, p)
    mine = field_macro_f1(gold, pred)
    for field in FIELDS:
        assert mine[field] == pytest.approx(reference[field], abs=1e-12)


def test_empty_subset_scores_zero_like_the_official_scorer():
    gold, pred = encode(RECORDS)
    assert weighted_macro_f1(gold, pred, np.array([], dtype=np.int64)) == 0.0


def test_aligned_load_puts_every_method_on_one_row_order():
    # The property the paired bootstrap rests on: one resample index array is
    # valid for every method and every seed.
    a_gold, a_pred = load_aligned("pdf_group", 42, "M0", ORDER, EXAMPLES_ROOT)
    b_gold, b_pred = load_aligned("pdf_group", 42, "M6", ORDER, EXAMPLES_ROOT)
    assert a_gold.shape == (len(ORDER), len(FIELDS))
    assert np.array_equal(a_gold, b_gold)      # same rows, same order
    assert not np.array_equal(a_pred, b_pred)  # different decision rules


def test_aligned_load_is_consistent_across_protocols_and_seeds():
    reference, _ = load_aligned("pdf_group", 42, "M0", ORDER, EXAMPLES_ROOT)
    for protocol in ("pdf_group", "row_strat"):
        for seed in (42, 123, 456):
            gold, _ = load_aligned(protocol, seed, "M0", ORDER, EXAMPLES_ROOT)
            assert np.array_equal(gold, reference)


def test_all_forty_two_example_sets_load():
    for protocol in ("pdf_group", "row_strat"):
        for seed in (42, 123, 456):
            for method in METHODS:
                gold, pred = load_aligned(protocol, seed, method, ORDER,
                                          EXAMPLES_ROOT)
                assert gold.shape == pred.shape == (2000, 4)


def test_pdf_clusters_partition_every_row_exactly_once():
    clusters = pdf_clusters(ORDER, DEV)
    assert len(clusters) == 49
    covered = np.concatenate(clusters)
    assert sorted(covered.tolist()) == list(range(len(ORDER)))


def test_encode_reindexes_by_id_not_by_position():
    # The contract's central failure mode (section 1.1): a file whose rows are
    # in a different order must still align, and must do so by id.
    shuffled = list(reversed(RECORDS))
    gold_a, pred_a = encode(RECORDS, order=ORDER)
    gold_b, pred_b = encode(shuffled, order=ORDER)
    assert np.array_equal(gold_a, gold_b)
    assert np.array_equal(pred_a, pred_b)


# --- path-constrained (C-)metrics ------------------------------------------
#
# The consistency-aware variant of the official metric: identical in every
# respect -- per field, macro over present gold labels, same weights -- except
# that a field whose ancestors were not predicted true is counted as a false
# prediction. Adding it required widening ``_macro_f1``'s bincount to hold a
# sentinel class, which is why the first test below scores every method
# against paper/score.py: if that widening perturbed the official metric by so
# much as a float ulp, Table 2, Table 3 and every interval would be wrong.


def test_official_metric_is_unchanged_for_every_method():
    # The regression net for the sentinel-capable scorer. M0 is the method
    # whose predictions include illegal tuples, so it is the one that would
    # move first if a sentinel ever leaked into the official metric.
    for method in METHODS:
        records = read_predictions(
            EXAMPLES_ROOT / "predictions" / f"pdf_group_seed42_{method}.csv.gz"
        )
        gold, pred = encode(records)
        g, p = _reference(records)
        assert weighted_macro_f1(gold, pred) == pytest.approx(
            compute_weighted_macro_f1(g, p), abs=1e-12
        ), method


def test_enforce_ancestors_invalidates_the_children_of_a_declined_promise():
    # PS=No admits no timeline, no evidence and no quality, so all three
    # substantive predictions below are unsupported by their ancestors.
    pred = np.array([[1, 0, 0, 0]], dtype=np.int64)     # No / already / Yes / Clear
    assert np.array_equal(enforce_ancestors(pred), [[1, 5, 3, 4]])


def test_enforce_ancestors_invalidates_quality_when_evidence_is_not_yes():
    pred = np.array([[0, 0, 1, 0]], dtype=np.int64)     # Yes / already / No / Clear
    assert np.array_equal(enforce_ancestors(pred), [[0, 0, 1, 4]])


def test_enforce_ancestors_leaves_every_legal_state_untouched():
    # The 17 legal states are exactly the tuples whose every field is
    # supported, so masking must be the identity on all of them.
    legal = np.array(
        [[LABEL2ID[f][lab] for f, lab in zip(FIELDS, (s.ps, s.vt, s.es, s.eq))]
         for s in STATES],
        dtype=np.int64,
    )
    assert np.array_equal(enforce_ancestors(legal), legal)


def test_enforce_ancestors_does_not_mutate_its_argument():
    pred = np.array([[1, 0, 0, 0]], dtype=np.int64)
    enforce_ancestors(pred)
    assert np.array_equal(pred, [[1, 0, 0, 0]])


def test_consistent_metric_is_the_official_scorer_on_masked_labels():
    # Pins the C-metric to paper/score.py the same way the official metric is
    # pinned: an unsupported field becomes a label string absent from the gold,
    # which sklearn scores as a miss for the true class and a false prediction
    # for nobody. No second implementation of macro-F1 is introduced.
    gold, pred = encode(RECORDS_M0)
    masked = enforce_ancestors(pred)
    g, _ = _reference(RECORDS_M0)
    p = [
        {f: ("INVALID" if row[j] == len(EVAL_FIELDS[f]) else EVAL_FIELDS[f][row[j]])
         for j, f in enumerate(FIELDS)}
        for row in masked
    ]
    assert consistent_weighted_macro_f1(gold, pred) == pytest.approx(
        compute_weighted_macro_f1(g, p), abs=1e-12
    )


def test_consistent_metric_agrees_with_the_official_one_exactly_when_output_is_legal():
    # The literature's own check on a path-constrained metric: a method that
    # guarantees label consistency scores identically under both. M1-M6 do;
    # unconstrained argmax does not, and must score strictly lower.
    order, root = ORDER, EXAMPLES_ROOT
    for method in METHODS:
        gold, pred = load_aligned("pdf_group", 42, method, order, root)
        official = weighted_macro_f1(gold, pred)
        constrained = consistent_weighted_macro_f1(gold, pred)
        if method == "M0":
            assert constrained < official
        else:
            assert constrained == pytest.approx(official, abs=1e-12), method


def test_consistent_metric_honours_the_subset_index():
    gold, pred = encode(RECORDS_M0)
    idx = np.arange(0, len(RECORDS_M0), 3)
    assert consistent_weighted_macro_f1(gold, pred, idx) == pytest.approx(
        consistent_weighted_macro_f1(gold[idx], pred[idx]), abs=1e-12
    )


# --- conditional field F1 (plan section 4.5) --------------------------------
#
# A child field is only meaningful on the rows its parent admits: scoring
# verification_timeline over rows whose gold promise_status is No measures how
# well the model reproduces a label the hierarchy fixes to N/A, which flatters
# every method equally. The plan asks for the conditioned version alongside the
# unconditioned one; the conditioning is on gold, never on the prediction.


def test_conditional_f1_matches_the_official_scorer_on_the_conditioned_rows():
    gold, pred = encode(RECORDS)
    mine = conditional_field_macro_f1(gold, pred)
    g, p = _reference(RECORDS)
    for field in FIELDS:
        if field in CONDITIONING_SUBSET:
            parent, value = CONDITIONING_SUBSET[field]
            keep = [i for i, row in enumerate(g) if row[parent] == value]
        else:
            keep = list(range(len(g)))
        expected = macro_f1([g[i][field] for i in keep],
                            [p[i][field] for i in keep],
                            EVAL_FIELDS[field])
        assert mine[field] == pytest.approx(expected, abs=1e-12), field


def test_conditional_f1_leaves_the_root_field_unconditioned():
    gold, pred = encode(RECORDS)
    assert (conditional_field_macro_f1(gold, pred)["promise_status"]
            == pytest.approx(field_macro_f1(gold, pred)["promise_status"], abs=1e-12))


def test_conditional_f1_actually_changes_the_child_fields():
    # If conditioning were a no-op the metric would be reporting nothing new.
    gold, pred = encode(RECORDS)
    conditioned = conditional_field_macro_f1(gold, pred)
    plain = field_macro_f1(gold, pred)
    for field in CONDITIONING_SUBSET:
        assert conditioned[field] != pytest.approx(plain[field], abs=1e-6), field


def test_conditional_f1_honours_the_subset_index():
    gold, pred = encode(RECORDS)
    idx = np.arange(0, len(RECORDS), 3)
    a = conditional_field_macro_f1(gold, pred, idx)
    b = conditional_field_macro_f1(gold[idx], pred[idx])
    for field in FIELDS:
        assert a[field] == pytest.approx(b[field], abs=1e-12), field


# --- ancestor-based hierarchical F1 -----------------------------------------
#
# The metric the hierarchical-classification literature reaches for first, and
# the one a reviewer asks about the moment we claim the official metric suits
# this task badly. Each row's label is a path: every field holding a value
# other than N/A contributes one node, N/A contributes none. hP, hR and hF are
# then the standard micro-averaged set overlaps.


def test_hierarchical_prf_on_a_hand_computed_pair_of_rows():
    # row 1: gold Yes/already/Yes/Clear, pred Yes/already/No/N/A
    #        |Y|=4, |P|=3, overlap 2
    # row 2: gold No/N/A/N/A/N/A, pred No/already/Yes/Clear  (hierarchy-invalid)
    #        |Y|=1, |P|=4, overlap 1
    gold = np.array([[0, 0, 0, 0], [1, 4, 2, 3]])
    pred = np.array([[0, 0, 1, 3], [1, 0, 0, 0]])
    out = hierarchical_prf(gold, pred)
    assert out["hP"] == pytest.approx(3 / 7, abs=1e-12)
    assert out["hR"] == pytest.approx(3 / 5, abs=1e-12)
    assert out["hF"] == pytest.approx(0.5, abs=1e-12)
    assert hierarchical_f1(gold, pred) == pytest.approx(out["hF"], abs=1e-12)


def test_hierarchical_f1_is_one_for_a_perfect_prediction():
    gold, _ = encode(RECORDS)
    assert hierarchical_f1(gold, gold) == pytest.approx(1.0, abs=1e-12)


def test_hierarchical_f1_penalises_a_node_whose_parent_disagrees():
    # Same row scored twice: once legal, once with a child asserted under a
    # parent that forbids it. The invalid version must score lower.
    gold = np.array([[1, 4, 2, 3]])
    legal = np.array([[1, 4, 2, 3]])
    invalid = np.array([[1, 0, 0, 0]])
    assert hierarchical_f1(gold, invalid) < hierarchical_f1(gold, legal)


def test_hierarchical_f1_honours_the_subset_index():
    gold, pred = encode(RECORDS)
    idx = np.arange(0, len(RECORDS), 3)
    assert hierarchical_f1(gold, pred, idx) == pytest.approx(
        hierarchical_f1(gold[idx], pred[idx]), abs=1e-12)
