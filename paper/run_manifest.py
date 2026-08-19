"""One index for the whole study: what exists, from what, produced by which code.

    python -m paper.run_manifest                          # -> run_manifest.json
    python -m paper.run_manifest --root contracts/examples --out /tmp/m.json

``tables/manifest.json`` (C, contract §5) traces every printed number back to
the script and the input files behind it. This is the layer underneath: where
those input files themselves came from -- which commit, which environment,
which data, which model revision -- and whether they are mutually consistent.
Plan §6.1 lists what an official run must preserve; today it is spread across
30 bundle ``meta.json`` files, 6 split manifests and 42 results files, with no
single place that says whether the set as a whole hangs together.

Two rules keep it from becoming a second source of truth:

* **It records checksums and verdicts, never scores.** A score copied here
  could disagree with the per-row file it came from, and the contract is
  explicit that the per-row file wins. Nothing is transcribed.
* **It calls the existing validators rather than restating their logic.** The
  consistency section is a summary of ``paper/validate.py``, plus the one check
  no one else performs: that each results file's ``predictions_sha256`` still
  matches the predictions file on disk. That link is written once at generation
  time and never verified again, so a predictions file replaced afterwards
  would leave every downstream number attributable to a file that no longer
  exists in that form.

Incompleteness is normal and is recorded rather than treated as failure: before
B delivers, ``probs`` is simply 0 of 30. A manifest that refused to generate
until the study was finished would be useless exactly when it is needed.
"""

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from paper.data import (
    REPO_ROOT,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    data_checksum,
    file_sha256,
    load_dev,
)
from paper.labels import FIELDS
from paper.methods import METHOD_IDS
from paper.provenance import git_sha, now_iso
from paper.train_config import N_FOLDS, SEEDS
from paper.validate import (
    validate_predictions,
    validate_probs_bundle,
    validate_probs_run,
    validate_probs_study,
)

MANIFEST_VERSION = "1.0"
PROTOCOLS = ("pdf_group", "row_strat")

# Recorded because a version change here moves the numbers without touching a
# line of this repository's code. torch is optional: A's half of the study runs
# without it, and its absence is a fact worth recording rather than an error.
PACKAGES = ("numpy", "scikit-learn", "pandas", "torch", "transformers")


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def data_section(rows) -> dict:
    return {
        "data_checksum": data_checksum(rows),
        "n_rows": len(rows),
        "files": {
            _relative(p): file_sha256(p)
            for p in (TRAIN_PATH, VAL_PATH, TEST_PATH) if p.exists()
        },
    }


def splits_section(splits_dir) -> dict:
    out = {}
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            path = Path(splits_dir) / f"{protocol}_seed{seed}.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                split = json.load(f)
            out[path.name] = {
                "sha256": file_sha256(path),
                "split_fingerprint": split.get("split_fingerprint"),
                "data_checksum": split.get("data_checksum"),
                "resample_attempts": split.get("resample_attempts"),
            }
    return out


def probs_section(probs_dir) -> dict:
    """Bundles present, by run.

    Only ``meta.json`` is checksummed: it carries the sha256 of all eight
    arrays, and ``paper/validate.py`` checks the arrays against it, so hashing
    the meta covers the bundle transitively without duplicating eight digests.
    """
    runs, missing = {}, []
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            run_id = f"{protocol}_seed{seed}"
            bundles = {}
            for k in range(N_FOLDS):
                path = Path(probs_dir) / f"{run_id}_r{k}" / "meta.json"
                if not path.exists():
                    missing.append(f"{run_id}_r{k}")
                    continue
                with open(path, encoding="utf-8") as f:
                    meta = json.load(f)
                bundles[f"{run_id}_r{k}"] = {
                    "meta_sha256": file_sha256(path),
                    "split_fingerprint": meta.get("split_fingerprint"),
                    "git_sha": meta.get("git_sha"),
                    "synthetic": bool(meta.get("synthetic")),
                }
            if bundles:
                first = next(iter(bundles))
                with open(Path(probs_dir) / first / "meta.json", encoding="utf-8") as f:
                    meta = json.load(f)
                runs[run_id] = {
                    "recipe": {
                        key: meta.get(key) for key in
                        ("model_name", "model_revision", "train_config_sha256",
                         "checkpoint_rule", "checkpoint_last_k", "epochs")
                    },
                    "hardware": sorted({
                        m for m in (_meta_of(probs_dir, b).get("hardware") for b in bundles)
                        if m
                    }),
                    "bundles": bundles,
                }
    return {
        "expected": len(PROTOCOLS) * len(SEEDS) * N_FOLDS,
        "present": sum(len(r["bundles"]) for r in runs.values()),
        "missing": missing,
        "runs": runs,
    }


def _meta_of(probs_dir, bundle_name) -> dict:
    with open(Path(probs_dir) / bundle_name / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def decisions_section(root) -> dict:
    """The contract-3 files: per-row predictions and their results summaries."""
    predictions, results, missing = {}, {}, []
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            for method in METHOD_IDS:
                stem = f"{protocol}_seed{seed}_{method}"
                pred = Path(root) / "predictions" / f"{stem}.csv.gz"
                res = Path(root) / "results" / f"{stem}.json"
                if pred.exists():
                    predictions[pred.name] = file_sha256(pred)
                if res.exists():
                    results[res.name] = file_sha256(res)
                if not (pred.exists() and res.exists()):
                    missing.append(stem)
    return {
        "expected": len(PROTOCOLS) * len(SEEDS) * len(METHOD_IDS),
        "predictions": predictions,
        "results": results,
        "incomplete": missing,
    }


def outputs_section(root) -> dict:
    """Contract-4 deliverables: the tables and the figure D includes.

    Indexed here so the chain is unbroken -- ``tables/manifest.json`` ties each
    printed number to its inputs, and this ties that manifest, and the .tex
    files beside it, to a commit and an environment. The figure lives at the
    repository root in every case: its counts come from ``paper/labels.py``
    rather than from a run, so it has no per-root variant.
    """
    tables = {p.name: file_sha256(p)
              for p in sorted((Path(root) / "tables").glob("*"))
              if p.is_file()}
    figures = {p.name: file_sha256(p)
               for p in sorted((REPO_ROOT / "figures").glob("*"))
               if p.is_file()}
    return {"tables": tables, "figures": figures}


def _check(name, problems) -> dict:
    if problems is None:
        return {"check": name, "status": "skipped"}
    return {
        "check": name,
        "status": "pass" if not problems else "fail",
        "problems": problems,
    }


def consistency(root, splits_dir, data, rows) -> list[dict]:
    """Verdicts, gathered from the validators rather than re-derived here."""
    checks = []

    bundles = sorted(p for p in (Path(root) / "probs").glob("*_r?") if p.is_dir())
    if not bundles:
        checks.append(_check("probability bundles", None))
        checks.append(_check("one training recipe across the study", None))
    else:
        problems = []
        for d in bundles:
            problems += validate_probs_bundle(d, splits_dir=splits_dir)
        by_run: dict[str, list[Path]] = {}
        for d in bundles:
            by_run.setdefault(d.name.rsplit("_r", 1)[0], []).append(d)
        for run, dirs in sorted(by_run.items()):
            if len(dirs) == N_FOLDS:
                problems += [f"{run}: {m}" for m in validate_probs_run(dirs, splits_dir)]
        checks.append(_check("probability bundles", problems))
        checks.append(_check("one training recipe across the study",
                             validate_probs_study(bundles)))

    checks.append(_check("splits carry the data checksum", [
        f"{name}: split records {info['data_checksum']}, data give {data['data_checksum']}"
        for name, info in splits_section(splits_dir).items()
        if info["data_checksum"] != data["data_checksum"]
    ]))

    # An empty set of results must report "skipped", not "pass": a vacuous pass
    # is indistinguishable from a verified one in the printed output, and the
    # normal state for most of the study is that these files do not exist yet.
    results = sorted((Path(root) / "results").glob("*.json"))
    checks.append(_check("results still point at the predictions they summarise",
                         _predictions_links(root) if results else None))
    checks.append(_check("results were built against this data", [
        f"{name}: results record {recorded}, data give {data['data_checksum']}"
        for name, recorded in _results_checksums(root)
        if recorded != data["data_checksum"]
    ] if results else None))

    predictions = sorted((Path(root) / "predictions").glob("*.csv.gz"))
    if not predictions:
        checks.append(_check("per-row files are aligned and legal", None))
    else:
        problems = []
        for path in predictions:
            problems += validate_predictions(path, rows=rows, method=_method_of(path))
        checks.append(_check("per-row files are aligned and legal", problems))
    return checks


def _results_checksums(root):
    for path in sorted((Path(root) / "results").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            yield path.name, json.load(f).get("data_checksum")


def _method_of(path) -> str | None:
    """``pdf_group_seed42_M3.csv.gz`` -> ``M3``; None when named otherwise."""
    stem = Path(path).name.split(".")[0].rsplit("_", 1)[-1]
    return stem if stem in METHOD_IDS else None


def _predictions_links(root) -> list[str]:
    """Each results file's ``predictions_sha256`` against the file on disk.

    Written once by ``paper/evaluate.py`` and never checked again. A
    predictions file regenerated or replaced afterwards leaves every table
    derived from it attributable to a file that no longer exists in that form,
    with nothing on the surface to show it.
    """
    problems = []
    for path in sorted((Path(root) / "results").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        named = Path(root) / results.get("predictions_file", "")
        if not named.exists():
            problems.append(f"{path.name}: names {results.get('predictions_file')!r}, which is absent")
            continue
        actual = file_sha256(named)
        if actual != results.get("predictions_sha256"):
            problems.append(
                f"{path.name}: records {results.get('predictions_sha256')} for "
                f"{named.name}, which now hashes to {actual}"
            )
    return problems


def _relative(path) -> str:
    """Repo-relative where possible, absolute otherwise (a scratch --out dir)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT)) or "."
    except ValueError:
        return str(path)


def warnings_for(manifest) -> list[str]:
    """Things a reader of this manifest must not miss.

    Chiefly: a manifest generated against the synthetic fixtures looks exactly
    like one generated against a real run. Anything derived from it is
    fabricated, and that has to be stated where the manifest is read rather
    than inferred from a per-bundle flag.
    """
    notes = []
    bundles = [b for run in manifest["probs"]["runs"].values()
               for b in run["bundles"].values()]
    synthetic = [b for b in bundles if b["synthetic"]]
    if synthetic:
        notes.append(
            f"{len(synthetic)} of {len(bundles)} probability bundles are synthetic: "
            "no number derived from this manifest is a result."
        )
    if manifest["probs"]["missing"]:
        notes.append(f"{len(manifest['probs']['missing'])} probability bundles are absent.")
    if (manifest["git_sha"] or "").endswith("-dirty"):
        notes.append("the working tree had uncommitted code changes; this run is not reproducible from a commit.")
    return notes


def build_manifest(root=REPO_ROOT, splits_dir=None) -> dict:
    root = Path(root).resolve()
    splits_dir = Path(splits_dir) if splits_dir else REPO_ROOT / "splits"
    rows = load_dev()
    data = data_section(rows)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": now_iso(),
        "git_sha": git_sha(),
        "artifact_root": _relative(root),
        "fields": list(FIELDS),
        "methods": list(METHOD_IDS),
        "environment": environment(),
        "data": data,
        "splits": splits_section(splits_dir),
        "probs": probs_section(root / "probs"),
        "decisions": decisions_section(root),
        "outputs": outputs_section(root),
        "consistency": consistency(root, splits_dir, data, rows),
    }
    manifest["warnings"] = warnings_for(manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO_ROOT,
                    help="where probs/, predictions/ and results/ live")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "run_manifest.json")
    args = ap.parse_args()

    manifest = build_manifest(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")

    probs, decisions = manifest["probs"], manifest["decisions"]
    print(f"{args.out}: git {manifest['git_sha']}")
    print(f"  splits      {len(manifest['splits'])}/6")
    print(f"  probs       {probs['present']}/{probs['expected']}")
    print(f"  predictions {len(decisions['predictions'])}/{decisions['expected']}")
    print(f"  results     {len(decisions['results'])}/{decisions['expected']}")
    outputs = manifest["outputs"]
    print(f"  tables      {len(outputs['tables'])}   figures {len(outputs['figures'])}")
    for check in manifest["consistency"]:
        print(f"  [{check['status']:>7}] {check['check']}")
        for problem in check.get("problems", []):
            print(f"            {problem}")

    for note in manifest["warnings"]:
        print(f"  note: {note}")

    failed = [c for c in manifest["consistency"] if c["status"] == "fail"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
