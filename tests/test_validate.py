"""The inbound checks in paper/validate.py.

Each test breaks exactly one thing and asserts it is caught, because the value
of a validator is entirely in what it refuses. A validator that only ever sees
good input is untested.
"""

import json

import numpy as np
import pytest

from contracts.make_examples import make_contract3
from contracts.make_fixtures import make_rotation_fixture
from paper.data import load_dev
from paper.splits import build_split
from paper.validate import (
    validate_predictions,
    validate_probs_bundle,
    validate_probs_run,
    validate_probs_study,
)

ROWS = load_dev()
SPLIT = build_split("pdf_group", 42, ROWS)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("probs")
    return make_rotation_fixture(SPLIT, 0, out / "pdf_group_seed42_r0", rows=ROWS)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return [
        make_rotation_fixture(SPLIT, k, out / f"pdf_group_seed42_r{k}", rows=ROWS)
        for k in range(5)
    ]


def _retouch(bundle, tmp_path, mutate_meta=None, mutate_array=None):
    """Copy a bundle, apply one mutation, return the copy."""
    import shutil

    dst = tmp_path / bundle.name
    shutil.copytree(bundle, dst)
    if mutate_meta:
        meta = json.load(open(dst / "meta.json", encoding="utf-8"))
        mutate_meta(meta)
        with open(dst / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
    if mutate_array:
        name, fn = mutate_array
        np.save(dst / name, fn(np.load(dst / name)))
    return dst


# --- contract 2 -----------------------------------------------------------

def test_a_conforming_bundle_has_no_problems(bundle):
    assert validate_probs_bundle(bundle, SPLIT) == []


def test_non_probabilities_are_caught(bundle, tmp_path):
    """The failure the whole check exists for: B emitting logits, or applying
    a softmax that never ran. Shape and row count stay perfectly correct."""
    bad = _retouch(bundle, tmp_path,
                   mutate_array=("test_promise_status.npy", lambda a: a * 4.0))
    assert any("sum to 1" in p for p in validate_probs_bundle(bad, SPLIT))


def test_a_modified_array_no_longer_matches_its_checksum(bundle, tmp_path):
    bad = _retouch(bundle, tmp_path,
                   mutate_array=("test_evidence_status.npy", lambda a: a[::-1]))
    assert any("sha256" in p for p in validate_probs_bundle(bad, SPLIT))


def test_an_unpinned_revision_is_caught_on_a_real_run(bundle, tmp_path):
    """Fixtures are exempt; a bundle claiming to be a real fit is not."""
    def unpin(meta):
        meta.pop("synthetic")
        meta["model_revision"] = "main"
    problems = validate_probs_bundle(_retouch(bundle, tmp_path, unpin), SPLIT)
    assert any("model_revision is not pinned" in p for p in problems)
    assert any("provenance" in p for p in problems)


def test_a_bundle_built_against_another_partition_is_caught(bundle, tmp_path):
    """Invariant 1b: each rotation looks fine alone, the set does not."""
    bad = _retouch(bundle, tmp_path,
                   mutate_meta=lambda m: m.update(split_fingerprint="sha256:0000"))
    assert any("split_fingerprint" in p for p in validate_probs_bundle(bad, SPLIT))


def test_reordered_ids_are_caught(bundle, tmp_path):
    """Same ids, different order — the arrays are aligned by position, so this
    is a silent row shuffle rather than a missing-data error."""
    bad = _retouch(bundle, tmp_path,
                   mutate_meta=lambda m: m.update(test_ids=m["test_ids"][::-1]))
    assert any("test_ids" in p for p in validate_probs_bundle(bad, SPLIT))


def test_a_wrong_data_checksum_is_caught(bundle, tmp_path):
    bad = _retouch(bundle, tmp_path,
                   mutate_meta=lambda m: m.update(data_checksum="sha256:0000"))
    assert any("data_checksum" in p for p in validate_probs_bundle(bad, SPLIT))


# --- contract 2, across a whole run ---------------------------------------

def test_a_complete_run_has_no_problems(run):
    assert validate_probs_run(run) == []


def test_a_run_uses_the_supplied_corpus_geometry(tmp_path):
    """Decision validation must not reload the default 2,000-row manifest."""
    ids = [f"fr-{i}" for i in range(400)]
    split = {"canonical_row_order": ids}
    bundles = []
    for rotation in range(5):
        bundle = tmp_path / f"pdf_group_seed42_r{rotation}"
        bundle.mkdir()
        meta = {
            "protocol": "pdf_group",
            "seed": 42,
            "rotation": rotation,
            "test_ids": ids[rotation::5],
        }
        with open(bundle / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        bundles.append(bundle)

    assert validate_probs_run(bundles, split=split) == []


def test_a_missing_rotation_is_caught(run):
    assert any("rotations present" in p for p in validate_probs_run(run[:4]))


def test_rotations_from_different_partitions_are_caught(run, tmp_path):
    mixed = list(run[:4]) + [
        _retouch(run[4], tmp_path,
                 mutate_meta=lambda m: m.update(split_fingerprint="sha256:0000"))
    ]
    assert any("split_fingerprint" in p for p in validate_probs_run(mixed))


def test_rotations_from_different_training_recipes_are_caught(run, tmp_path):
    """Contract §3 invariant 6. Each such bundle is internally valid; the five
    are concatenated into one score, so a rotation re-run under a different
    recipe puts two models into one number and nothing per-bundle sees it."""
    mixed = list(run[:4]) + [
        _retouch(run[4], tmp_path,
                 mutate_meta=lambda m: m.update(model_revision="deadbeef"))
    ]
    assert any("model_revision" in p for p in validate_probs_run(mixed))


def _second_run(run, tmp_path, mutate_meta=None):
    """The five bundles copied under a second (protocol, seed) name.

    Only the recipe keys are touched, so these copies still carry seed 42's id
    lists — enough for the study-level check, which reads nothing else, and not
    a valid bundle set for anything that does.
    """
    import shutil

    dirs = []
    for src in run:
        dst = tmp_path / src.name.replace("seed42", "seed123")
        shutil.copytree(src, dst)
        meta = json.load(open(dst / "meta.json", encoding="utf-8"))
        meta["seed"] = 123
        if mutate_meta:
            mutate_meta(meta)
        with open(dst / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        dirs.append(dst)
    return dirs


def test_a_recipe_that_moved_between_two_runs_is_caught(run, tmp_path):
    """The gap the per-run check cannot see: each run of five is internally
    consistent, so both pass, while the 3-seed std silently becomes a mixture
    of pipeline variance and a config change."""
    other = _second_run(run, tmp_path, mutate_meta=lambda m: m.update(epochs=10))
    problems = validate_probs_study(list(run) + other)
    assert any("epochs" in p for p in problems)
    assert any("pdf_group_seed123" in p for p in problems)


def test_two_runs_of_one_recipe_are_clean(run, tmp_path):
    assert validate_probs_study(list(run) + _second_run(run, tmp_path)) == []


def test_the_study_check_defers_to_the_run_check(run, tmp_path):
    """A run that is internally inconsistent is reported by the per-run check,
    which names the rotation; the study check must not restate it less
    precisely."""
    broken = list(run[:4]) + [
        _retouch(run[4], tmp_path / "broken",
                 mutate_meta=lambda m: m.update(epochs=10))
    ]
    assert validate_probs_study(broken) == []
    assert any("epochs" in p for p in validate_probs_run(broken))


def test_a_second_gpu_or_a_pull_mid_run_is_not_an_error(run, tmp_path):
    """The complement of the check above: git_sha and hardware are excluded on
    purpose, so rotations spread over a pull or a second card still validate."""
    spread = list(run[:4]) + [
        _retouch(run[4], tmp_path,
                 mutate_meta=lambda m: m.update(git_sha="0" * 40,
                                                hardware="RTX 3090 (idx 0)"))
    ]
    assert validate_probs_run(spread) == []


# --- contract 3 -----------------------------------------------------------

@pytest.fixture(scope="module")
def predictions(tmp_path_factory):
    out = tmp_path_factory.mktemp("preds")
    make_contract3(SPLIT, ROWS, out, methods=["M0", "M3"])
    return out / "predictions"


def test_conforming_predictions_have_no_problems(predictions):
    assert validate_predictions(predictions / "pdf_group_seed42_M3.csv.gz",
                                rows=ROWS, split=SPLIT, method="M3") == []


def test_m0_may_emit_invalid_tuples_but_m1_to_m6_may_not(predictions):
    """M0's invalid tuples are the baseline's defining property, not an error;
    the same file read as M3 must fail."""
    path = predictions / "pdf_group_seed42_M0.csv.gz"
    assert validate_predictions(path, rows=ROWS, split=SPLIT, method="M0") == []
    problems = validate_predictions(path, rows=ROWS, split=SPLIT, method="M3")
    assert any("hierarchy-invalid" in p for p in problems)


def test_row_misalignment_is_caught_by_the_redundant_gold_columns(predictions, tmp_path):
    """The gold columns are duplicated from the dataset on purpose; rotating
    them past the ids is exactly the silent failure §1.1 warns about."""
    from paper.artifacts import read_predictions, write_predictions

    records = read_predictions(predictions / "pdf_group_seed42_M3.csv.gz")
    shifted = records[1:] + records[:1]
    for dst, src in zip(records, shifted):
        dst["gold_ps"], dst["gold_state_id"] = src["gold_ps"], src["gold_state_id"]
    path = write_predictions(tmp_path / "pdf_group_seed42_M3.csv.gz", records)

    problems = validate_predictions(path, rows=ROWS, split=SPLIT, method="M3")
    assert any("misalignment" in p for p in problems)
