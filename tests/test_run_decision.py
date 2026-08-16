import json

from contracts.make_fixtures import make_rotation_fixture
from paper.artifacts import read_predictions
from paper.data import load_dev
from paper.run_decision import run_one
from paper.splits import build_split
from paper.validate import validate_predictions


def test_runner_assembles_five_test_folds_and_writes_real_contract(tmp_path):
    rows = load_dev()
    split = build_split("pdf_group", 42, rows)
    probs = tmp_path / "probs"
    for k in range(5):
        make_rotation_fixture(
            split, k, probs / f"pdf_group_seed42_r{k}", concentration=0.4, rows=rows,
        )

    methods = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
    written = run_one(
        "pdf_group", 42, methods=methods, probs_dir=probs,
        predictions_dir=tmp_path / "predictions", results_dir=tmp_path / "results",
    )
    assert len(written) == 7
    for method in methods:
        pred_path = tmp_path / "predictions" / f"pdf_group_seed42_{method}.csv.gz"
        records = read_predictions(pred_path)
        assert len(records) == 2000
        assert validate_predictions(pred_path, rows=rows, split=split, method=method) == []
        result = json.load(open(
            tmp_path / "results" / f"pdf_group_seed42_{method}.json", encoding="utf-8",
        ))
        assert result["n_rows"] == 2000
        assert result["method"] == method
        if method != "M0":
            assert result["invalid_tuple_rate"] == 0.0
