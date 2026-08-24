"""The queued multilingual campaign is finite, pinned and non-colliding."""

from paper.corpus import arm_dir
from paper.multilingual_config import (
    LANGUAGES,
    LAMBDAS,
    MODELS,
    SEEDS,
    expected_bundle_count,
)
from scripts import queue_after_english


def test_every_multilingual_model_is_immutable_and_unique():
    assert len(MODELS) == 4
    assert len({model["name"] for model in MODELS}) == 4
    assert all(len(model["revision"]) == 40 for model in MODELS)
    assert all(model["worker"] in {"large", "base"} for model in MODELS)


def test_the_frozen_matrix_is_150_fits_per_language():
    assert LANGUAGES == ("mlpromise_fr", "mlpromise_ja", "mlpromise_ko")
    assert LAMBDAS == (0.0, 0.3)
    assert SEEDS == (42, 123, 456)
    assert expected_bundle_count("mlpromise_fr") == 150
    assert expected_bundle_count() == 450


def test_all_twenty_four_language_model_lambda_arms_are_disjoint():
    paths = {
        arm_dir(language, model["name"], structure_lambda)
        for language in LANGUAGES
        for model in MODELS
        for structure_lambda in LAMBDAS
    }
    assert len(paths) == 3 * 4 * 2


def test_queue_points_at_two_distinct_existing_worktrees():
    assert queue_after_english.MULTI_ROOT.exists()
    assert queue_after_english.ENGLISH_ROOT.exists()
    assert queue_after_english.MULTI_ROOT != queue_after_english.ENGLISH_ROOT
    assert (queue_after_english.ENGLISH_ROOT / "runs_en").exists()


def test_queue_launches_multilingual_workers_as_importable_modules():
    commands = queue_after_english.multilingual_worker_commands()
    assert len(commands) == 2
    for command in commands:
        assert command[1:4] == ["-u", "-m", "scripts.run_multilingual_experiments"]
