"""Public contract for the standalone paper-results reproduction capsule."""

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from paper.data import REPO_ROOT


RELEASE_ROOT = REPO_ROOT / "ntcir19-esg-validity-layer"
EXPECTED_FIRST_LEVEL_DIRECTORIES = {
    "analysis",
    "figures",
    "inputs",
    "paper",
    "reference",
}


def test_release_groups_all_curated_inputs_by_corpus():
    """A release must expose one input boundary, not per-language roots."""
    load_reproduce_module()
    directories = {
        path.name for path in RELEASE_ROOT.iterdir()
        if path.is_dir() and path.name != "outputs"
    }
    assert directories == EXPECTED_FIRST_LEVEL_DIRECTORIES

    manifest = json.loads(
        (RELEASE_ROOT / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["artifacts"]) == 311
    assert all(path.startswith("inputs/") for path in manifest["artifacts"])


def load_reproduce_module():
    spec = importlib.util.spec_from_file_location(
        "esg_validity_layer_reproduce", RELEASE_ROOT / "reproduce.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def run_release(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RELEASE_ROOT / "reproduce.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_release_verifies_the_exact_curated_input_capsule():
    """Missing, extra, or unpinned paper inputs must make the public check fail."""
    completed = run_release("--verify-inputs")

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report == {
        "artifact_files": 311,
        "prediction_files": 300,
        "raw_text_files": 0,
        "source_commit": "babd85e61368037d235607ac5ebc798378a7ef75",
        "status": "ok",
    }


def test_release_rejects_an_artifact_whose_bytes_do_not_match_the_manifest(tmp_path):
    """A present filename with changed bytes must not pass provenance checks."""
    root = tmp_path / "release"
    prediction = (
        root / "inputs" / "aicup_zh" / "runs" / "main"
        / "predictions" / "paper.csv.gz"
    )
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"changed prediction bytes")
    corpus_index = root / "inputs" / "aicup_zh" / "corpus_index.json.gz"
    corpus_index.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(corpus_index, "wt", encoding="utf-8") as stream:
        json.dump({"development": [], "test": []}, stream)
    (root / "release_manifest.json").write_text(
        json.dumps({
            "source_commit": "fixture",
            "prediction_files": 1,
            "artifacts": {
                "inputs/aicup_zh/runs/main/predictions/paper.csv.gz":
                    "sha256:" + "0" * 64,
                "inputs/aicup_zh/corpus_index.json.gz": "sha256:" + "0" * 64,
            },
        }),
        encoding="utf-8",
    )

    reproduce = load_reproduce_module()
    with pytest.raises(ValueError, match="checksum mismatch"):
        reproduce.verify_inputs(root)


def test_release_rebuilds_and_verifies_every_manuscript_table(tmp_path):
    """The public command must recompute paper tables, not copy reference files."""
    output = tmp_path / "outputs"
    completed = run_release("--output-dir", str(output))

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report == {
        "figure_source_verified": True,
        "status": "ok",
        "tables_verified": 8,
    }
    expected_stems = (
        "table1_dataset",
        "table2_main",
        "table3_legality_cost",
        "table4_contrasts",
        "table6_regimes",
        "table7_multilingual_mechanism",
        "table8_invalid_anatomy",
        "table9_external_arms",
    )
    for stem in expected_stems:
        assert (output / "tables" / f"{stem}.tex").read_bytes() == (
            RELEASE_ROOT / "reference" / "tables" / f"{stem}.tex"
        ).read_bytes()
        assert (output / "tables" / f"{stem}_caption.txt").read_bytes() == (
            RELEASE_ROOT / "reference" / "tables" / f"{stem}_caption.txt"
        ).read_bytes()
