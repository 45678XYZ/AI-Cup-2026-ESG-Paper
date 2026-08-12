from paper.score import compute_per_field_f1, compute_weighted_macro_f1

GT = [
    {"promise_status": "Yes", "verification_timeline": "already",
     "evidence_status": "Yes", "evidence_quality": "Clear"},
    {"promise_status": "No", "verification_timeline": "N/A",
     "evidence_status": "N/A", "evidence_quality": "N/A"},
]


def test_perfect_score():
    assert abs(compute_weighted_macro_f1(GT, GT) - 1.0) < 1e-6


def test_score_range():
    preds = [GT[1], GT[0]]  # both rows wrong
    assert 0.0 <= compute_weighted_macro_f1(GT, preds) <= 1.0


def test_score_returns_float():
    assert isinstance(compute_weighted_macro_f1(GT, GT), float)


def test_per_field_overall_matches_weighted():
    per_field = compute_per_field_f1(GT, GT)
    assert abs(per_field["overall"] - compute_weighted_macro_f1(GT, GT)) < 1e-12
