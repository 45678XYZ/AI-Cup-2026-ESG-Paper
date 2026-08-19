"""The study-level index.

Its whole value is that it is generated rather than written, so the tests are
about what it refuses to let slide: a predictions file that no longer matches
the results summarising it, a split built against other data, and the fact that
a manifest of the synthetic fixtures looks exactly like a manifest of a real
run unless it says otherwise.
"""

import gzip
import json
import shutil

import pytest

from paper.data import REPO_ROOT
from paper.run_manifest import build_manifest, warnings_for

EXAMPLES = REPO_ROOT / "contracts" / "examples"


@pytest.fixture(scope="module")
def manifest():
    return build_manifest(EXAMPLES)


def _verdict(manifest, name):
    return next(c for c in manifest["consistency"] if c["check"] == name)


@pytest.fixture
def copied(tmp_path):
    """A writable copy of the example artifacts."""
    dst = tmp_path / "examples"
    shutil.copytree(EXAMPLES, dst)
    return dst


# --- what it indexes --------------------------------------------------------

def test_it_indexes_the_whole_example_set(manifest):
    assert len(manifest["splits"]) == 6
    assert manifest["probs"]["present"] == 5
    assert len(manifest["probs"]["missing"]) == 25
    assert len(manifest["decisions"]["predictions"]) == 42
    assert len(manifest["decisions"]["results"]) == 42
    assert manifest["decisions"]["incomplete"] == []


def test_every_verdict_is_clean_on_the_examples(manifest):
    assert [c["status"] for c in manifest["consistency"]] == ["pass"] * 4


def test_it_is_json_serialisable(manifest):
    assert json.loads(json.dumps(manifest))["manifest_version"] == "1.0"


def test_it_transcribes_no_scores(manifest):
    """The per-row files are the authority on every number. A score copied here
    could disagree with them, and a reader would have no way to know which one
    the paper used."""
    text = json.dumps(manifest)
    for forbidden in ("weighted_macro_f1", "tuple_exact_match", "per_field_macro_f1"):
        assert forbidden not in text


def test_absent_bundles_are_recorded_rather_than_refused(tmp_path):
    """Before B delivers there are no bundles at all. A manifest that failed
    until the study was finished would be useless exactly when it is needed."""
    (tmp_path / "results").mkdir()
    manifest = build_manifest(tmp_path)
    assert manifest["probs"]["present"] == 0
    assert _verdict(manifest, "probability bundles")["status"] == "skipped"
    assert _verdict(manifest, "one training recipe across the study")["status"] == "skipped"


# --- what it refuses to let slide -------------------------------------------

def test_a_predictions_file_replaced_after_the_fact_is_caught(copied):
    """The link results/*.json -> predictions/*.csv.gz is written once and
    never verified again, so a file swapped afterwards leaves every table
    attributable to something that no longer exists in that form."""
    target = copied / "predictions" / "pdf_group_seed42_M3.csv.gz"
    rows = gzip.decompress(target.read_bytes())
    target.write_bytes(gzip.compress(rows + b"\n"))

    verdict = _verdict(build_manifest(copied), "results still point at the predictions they summarise")
    assert verdict["status"] == "fail"
    assert any("pdf_group_seed42_M3" in p for p in verdict["problems"])


def test_a_results_file_naming_an_absent_predictions_file_is_caught(copied):
    (copied / "predictions" / "row_strat_seed456_M6.csv.gz").unlink()
    verdict = _verdict(build_manifest(copied), "results still point at the predictions they summarise")
    assert verdict["status"] == "fail"
    assert any("absent" in p for p in verdict["problems"])


def test_a_split_built_against_other_data_is_caught(copied, tmp_path):
    splits = tmp_path / "splits"
    shutil.copytree(REPO_ROOT / "splits", splits)
    path = splits / "row_strat_seed123.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    split["data_checksum"] = "sha256:0000"
    path.write_text(json.dumps(split), encoding="utf-8")

    manifest = build_manifest(copied, splits_dir=splits)
    verdict = _verdict(manifest, "splits carry the data checksum")
    assert verdict["status"] == "fail"
    assert any("row_strat_seed123" in p for p in verdict["problems"])


# --- what it says out loud --------------------------------------------------

def test_synthetic_inputs_are_stated_not_implied(manifest):
    assert any("synthetic" in note and "no number derived" in note
               for note in manifest["warnings"])


def test_a_clean_real_run_carries_no_synthetic_note():
    fabricated = {
        "git_sha": "abc123",
        "probs": {"missing": [], "runs": {"pdf_group_seed42": {"bundles": {
            f"pdf_group_seed42_r{k}": {"synthetic": False} for k in range(5)}}}},
    }
    assert warnings_for(fabricated) == []
