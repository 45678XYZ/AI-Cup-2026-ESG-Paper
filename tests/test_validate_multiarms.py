"""Whole-corpus validation keeps deliberate model/lambda arms separate."""

from paper import validate


def test_validate_paths_groups_repeated_run_names_by_probs_directory(monkeypatch, tmp_path):
    paths = []
    for arm in ("model_a/lambda_0.0", "model_b/lambda_0.3"):
        probs = tmp_path / arm / "probs"
        for seed in (42, 123):
            for rotation in range(5):
                path = probs / f"pdf_group_seed{seed}_r{rotation}"
                path.mkdir(parents=True)
                paths.append(path)

    run_groups = []
    study_groups = []
    monkeypatch.setattr(validate, "validate_probs_bundle", lambda *a, **k: [])
    monkeypatch.setattr(
        validate, "validate_probs_run",
        lambda dirs, **kwargs: run_groups.append(tuple(dirs)) or [],
    )
    monkeypatch.setattr(
        validate, "validate_probs_study",
        lambda dirs: study_groups.append(tuple(dirs)) or [],
    )

    assert validate._validate_paths(paths, rows=[]) == []
    assert len(run_groups) == 4
    assert all(len(group) == 5 for group in run_groups)
    assert len(study_groups) == 2
    assert all(len(group) == 10 for group in study_groups)
    assert all(len({path.parent for path in group}) == 1 for group in study_groups)
