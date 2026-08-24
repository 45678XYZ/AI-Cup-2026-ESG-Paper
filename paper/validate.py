"""Inbound conformance checks for the contract artifacts A receives.

The writers in ``paper/artifacts.py`` already make a *locally* malformed file
hard to produce: they reject arrays whose shape disagrees with the manifest and
arrays that are not probability distributions, at the moment of writing. What
one write call cannot see is everything around it — that a bundle was produced
against the split it names, that the five rotations of a run belong to the same
partition, that the array on disk is still the one whose checksum was recorded,
that a model revision was ever pinned.

Those are exactly the failures the interface contract exists to catch, by its
own test (§0): they do not raise, they produce plausible numbers. A bundle
built against a stale split scores fine. A predictions file whose rows drifted
one position scores slightly worse and says nothing.

    python -m paper.validate probs/pdf_group_seed42_r*
    python -m paper.validate predictions/pdf_group_seed42_M3.csv.gz
    python -m paper.validate --all

Every function returns a list of human-readable problems; empty means clean.
Nothing raises on a bad artifact, so one broken bundle in a batch cannot hide
the state of the rest.
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from paper.artifacts import PREDICTION_COLUMNS, read_predictions
from paper.corpus import CORPORA, probs_globs, splits_dir as corpus_splits_dir
from paper.corpus import DEFAULT as DEFAULT_CORPUS
from paper.corpus import load_rows
from paper.data import (
    REPO_ROOT,
    canonical_row_order,
    data_checksum,
    file_sha256,
    index_by_id,
    load_dev,
)
from paper.structure_loss import LAMBDA_UNSET
from paper.labels import (
    EVAL_FIELDS,
    FIELD_ALIAS,
    FIELDS,
    INVALID_STATE_ID,
    tuple_to_state_id,
)

SPLITS_DIR = REPO_ROOT / "splits"

# Keys without which the bundle cannot be interpreted or joined at all.
REQUIRED_META = (
    "contract_version", "protocol", "seed", "rotation", "split_file",
    "split_fingerprint", "data_checksum", "calibration_ids", "test_ids",
    "model_name", "model_revision", "git_sha", "created_at", "artifacts",
)

# Keys §6.1 requires of every official run so the fit can be reconstructed.
# Not required of bundles flagged ``synthetic``, which reconstruct nothing.
PROVENANCE_META = (
    "train_config_sha256", "checkpoint_rule", "checkpoint_last_k",
    "epochs", "hardware", "started_at", "finished_at",
)

# Keys that must agree across the five rotations of one run. The five test
# partitions are concatenated into a single 2,000-row score, so a rotation
# re-run after ``EPOCHS`` moved or after the revision was re-pinned would put
# two different models into one number. Each such bundle is internally valid
# and passes every per-bundle check; the mixture is only visible across the set.
#
# ``git_sha`` and ``hardware`` are deliberately *not* compared: B may run
# rotations across a pull or on a second GPU without changing the fit, and what
# does define the fit is hashed into ``train_config_sha256``.
RECIPE_META = (
    "model_name", "model_revision", "train_config_sha256",
    "checkpoint_rule", "checkpoint_last_k", "epochs", "structure_lambda",
)

# Keys a bundle may predate. Absent is not "unknown": the 30 bundles of the
# frozen study were written before the structural arm existed, and reading
# their silence as a different recipe would report the control arm as a
# mixture of two. Anything not listed here is compared as-is, so a genuinely
# missing key still surfaces.
RECIPE_DEFAULTS = {"structure_lambda": LAMBDA_UNSET}


def recipe_value(meta, key):
    """One bundle's value for ``key``, with the pre-arm default applied."""
    return meta.get(key, RECIPE_DEFAULTS.get(key))

EXPECTED_ARRAYS = tuple(
    f"{partition}_{field}.npy"
    for partition in ("calibration", "test")
    for field in FIELDS
)

UNPINNED_REVISIONS = (None, "", "main", "latest")


def load_split(protocol, seed, splits_dir=SPLITS_DIR) -> dict:
    with open(Path(splits_dir) / f"{protocol}_seed{seed}.json", encoding="utf-8") as f:
        return json.load(f)


def _split_named_by(predictions_path, splits_dir=SPLITS_DIR) -> dict | None:
    """The split a ``{protocol}_seed{seed}_{method}.csv.gz`` filename points at.

    Returns None rather than raising when the file is named otherwise or the
    manifest is absent: the rotation check is then skipped, while every check
    that does not need a split still runs.
    """
    parts = Path(predictions_path).name.split(".")[0].rsplit("_", 2)
    if len(parts) != 3 or not parts[1].startswith("seed"):
        return None
    try:
        return load_split(parts[0], int(parts[1].removeprefix("seed")), splits_dir)
    except (FileNotFoundError, ValueError):
        return None


# --------------------------------------------------------------------------
# Contract 2: probability bundles
# --------------------------------------------------------------------------

def validate_probs_bundle(bundle_dir, split=None, splits_dir=SPLITS_DIR) -> list[str]:
    """Check one ``probs/{protocol}_seed{seed}_r{k}/`` against its split."""
    bundle_dir = Path(bundle_dir)
    problems: list[str] = []

    def bad(msg):
        problems.append(f"{bundle_dir.name}: {msg}")

    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        bad("meta.json is missing")
        return problems
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        bad(f"meta.json lacks {missing}")
    if not meta.get("synthetic"):
        missing_prov = [k for k in PROVENANCE_META if k not in meta]
        if missing_prov:
            bad(f"meta.json lacks the provenance keys §6.1 requires: {missing_prov}")
        if meta.get("model_revision") in UNPINNED_REVISIONS:
            bad(f"model_revision is not pinned ({meta.get('model_revision')!r})")

    problems += _check_bundle_against_split(bundle_dir, meta, split, splits_dir)
    problems += _check_bundle_arrays(bundle_dir, meta)
    return problems


def _check_bundle_against_split(bundle_dir, meta, split, splits_dir) -> list[str]:
    problems: list[str] = []

    def bad(msg):
        problems.append(f"{bundle_dir.name}: {msg}")

    if split is None:
        try:
            split = load_split(meta["protocol"], meta["seed"], splits_dir)
        except (KeyError, FileNotFoundError):
            bad("cannot locate the split it names; split cross-checks skipped")
            return problems

    if meta.get("data_checksum") != split["data_checksum"]:
        bad("data_checksum differs from the split's — the two are not interoperable")
    if meta.get("split_fingerprint") != split.get("split_fingerprint"):
        bad("split_fingerprint differs from the split's — built against another partition")

    rot = next((r for r in split["rotations"] if r["k"] == meta.get("rotation")), None)
    if rot is None:
        bad(f"the split has no rotation {meta.get('rotation')!r}")
        return problems

    # Order matters as much as membership: the arrays are aligned by position
    # within these lists, so a permutation is a silent row-level shuffle.
    for partition in ("calibration", "test"):
        if meta.get(f"{partition}_ids") != rot[f"{partition}_ids"]:
            bad(f"{partition}_ids do not match the split's, in content or order")
    return problems


def _check_bundle_arrays(bundle_dir, meta) -> list[str]:
    problems: list[str] = []

    def bad(msg):
        problems.append(f"{bundle_dir.name}: {msg}")

    declared = meta.get("artifacts", {})
    on_disk = {p.name for p in bundle_dir.glob("*.npy")}
    if on_disk != set(EXPECTED_ARRAYS):
        bad(f"array set is {sorted(on_disk)}, expected {list(EXPECTED_ARRAYS)}")
    if set(declared) != set(EXPECTED_ARRAYS):
        bad(f"meta.artifacts lists {sorted(declared)}, expected {list(EXPECTED_ARRAYS)}")

    for partition in ("calibration", "test"):
        ids = meta.get(f"{partition}_ids")
        for field in FIELDS:
            name = f"{partition}_{field}.npy"
            path = bundle_dir / name
            if not path.exists():
                continue
            arr = np.load(path)

            if arr.dtype != np.float32:
                bad(f"{name}: dtype is {arr.dtype}, expected float32")
            if ids is not None:
                expected = (len(ids), len(EVAL_FIELDS[field]))
                if arr.shape != expected:
                    bad(f"{name}: shape is {arr.shape}, expected {expected}")

            if not np.isfinite(arr).all():
                bad(f"{name}: contains NaN or Inf")
            elif (arr < 0).any():
                bad(f"{name}: contains negative values, so it is not a probability")
            elif arr.size:
                dev = float(np.abs(arr.sum(axis=1) - 1.0).max())
                if dev > 1e-4:
                    bad(f"{name}: rows do not sum to 1 (max deviation {dev:.2e}) — "
                        "logits or a postprocessed array rather than raw probabilities?")

            recorded = declared.get(name, {}).get("sha256")
            if recorded is not None and file_sha256(path) != recorded:
                bad(f"{name}: sha256 differs from meta.artifacts — modified or truncated")
    return problems


def validate_probs_run(bundle_dirs, splits_dir=SPLITS_DIR) -> list[str]:
    """Check that a set of bundles forms one coherent (protocol, seed) run.

    Invariant 1b of contract §3 exists for a failure no per-bundle check can
    see: five rotations each produced against a different version of the split,
    every one internally consistent, the set as a whole incoherent. The same
    argument applies to the training recipe (``RECIPE_META``), because the five
    rotations are concatenated before anything is scored.
    """
    metas = []
    problems: list[str] = []
    for d in bundle_dirs:
        path = Path(d) / "meta.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                metas.append((Path(d).name, json.load(f)))
    if not metas:
        return ["no bundles with a meta.json were given"]

    for key in ("protocol", "seed", "split_fingerprint", "data_checksum", *RECIPE_META):
        values = {recipe_value(m, key) for _, m in metas}
        if len(values) > 1:
            problems.append(f"bundles disagree on {key}: {sorted(map(str, values))}")

    rotations = sorted(m.get("rotation") for _, m in metas)
    if rotations != list(range(5)):
        problems.append(f"rotations present are {rotations}, expected [0, 1, 2, 3, 4]")
        return problems

    # Every row is Test exactly once, which is what makes the five test
    # partitions concatenable into one 2,000-row score.
    seen: list[str] = []
    for _, m in metas:
        seen += m.get("test_ids", [])
    protocol, seed = metas[0][1].get("protocol"), metas[0][1].get("seed")
    try:
        split = load_split(protocol, seed, splits_dir)
        expected = split["canonical_row_order"]
    except (FileNotFoundError, TypeError):
        expected = canonical_row_order()
    if sorted(seen) != sorted(expected):
        problems.append(
            f"concatenated test_ids cover {len(set(seen))} distinct rows "
            f"({len(seen)} with duplicates), expected {len(expected)} each once"
        )
    return problems


def _run_of(bundle_name) -> str:
    """``pdf_group_seed42_r3`` -> ``pdf_group_seed42``."""
    return str(bundle_name).rsplit("_r", 1)[0]


def validate_probs_study(bundle_dirs) -> list[str]:
    """Check that every run of the study shares one training recipe.

    ``validate_probs_run`` only ever sees one (protocol, seed) at a time, so a
    config edit *between* two runs is invisible to it: each set of five is
    internally consistent and passes. But the study aggregates across runs --
    3-seed mean±std (plan §4.5) and the two-protocol contrast of Table 3 -- and
    both readings rest on one fixed base model throughout (§3.1). A recipe that
    moved between seed 42 and seed 123 turns that std into a mixture of
    pipeline variance and a config change, with nothing on the surface to show
    which is which.

    Runs that are already internally inconsistent are skipped for that key:
    ``validate_probs_run`` reports them more precisely.
    """
    by_run: dict[str, list[dict]] = {}
    for d in bundle_dirs:
        path = Path(d) / "meta.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                by_run.setdefault(_run_of(Path(d).name), []).append(json.load(f))

    problems: list[str] = []
    for key in RECIPE_META:
        holders: dict[str, list[str]] = {}
        for run, metas in by_run.items():
            values = {json.dumps(recipe_value(m, key)) for m in metas}
            if len(values) == 1:
                holders.setdefault(values.pop(), []).append(run)
        if len(holders) > 1:
            detail = "; ".join(
                f"{value} in {sorted(runs)}" for value, runs in sorted(holders.items())
            )
            problems.append(f"the study's runs disagree on {key}: {detail}")
    return problems


# --------------------------------------------------------------------------
# Contract 3: per-row predictions
# --------------------------------------------------------------------------

def validate_predictions(path, rows=None, split=None, method=None,
                         splits_dir=SPLITS_DIR) -> list[str]:
    """Check one ``predictions/{protocol}_seed{seed}_{method}.csv.gz``.

    The load-bearing check is that the ``gold_*`` columns still agree with the
    dataset. They are redundant by design (contract §4.1), which turns them
    into a row-alignment detector: if the predictions drifted against the ids,
    the gold columns drift with them and stop matching.
    """
    path = Path(path)
    problems: list[str] = []

    def bad(msg):
        problems.append(f"{path.name}: {msg}")

    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header != list(PREDICTION_COLUMNS):
        bad(f"columns are {header}, expected {list(PREDICTION_COLUMNS)}")
        return problems

    records = read_predictions(path)
    rows = rows if rows is not None else load_dev()
    by_id = index_by_id(rows)
    order = canonical_row_order(rows)

    ids = [r["id"] for r in records]
    if len(ids) != len(order):
        bad(f"has {len(ids)} rows, expected {len(order)}")
    if len(set(ids)) != len(ids):
        bad(f"{len(ids) - len(set(ids))} duplicate ids")
    unknown = sorted(set(ids) - set(order))
    if unknown:
        bad(f"{len(unknown)} ids are not in the dataset, e.g. {unknown[:3]}")
    absent = sorted(set(order) - set(ids))
    if absent:
        bad(f"{len(absent)} dataset rows are missing, e.g. {absent[:3]}")

    problems += _check_prediction_cells(path, records, by_id, method)

    if split is None:
        split = _split_named_by(path, splits_dir)
    if split is not None:
        rotation_of = {
            row_id: rot["k"] for rot in split["rotations"] for row_id in rot["test_ids"]
        }
        wrong = [r["id"] for r in records if rotation_of.get(r["id"]) != r["rotation"]]
        if wrong:
            bad(f"{len(wrong)} rows name a rotation that is not their test fold, "
                f"e.g. {wrong[:3]}")
    return problems


def _check_prediction_cells(path, records, by_id, method) -> list[str]:
    problems: list[str] = []

    def bad(msg):
        problems.append(f"{path.name}: {msg}")

    gold_drift, bad_label, state_drift, invalid = [], [], [], []
    for r in records:
        row = by_id.get(r["id"])
        gold = tuple(r[f"gold_{FIELD_ALIAS[f]}"] for f in FIELDS)
        pred = tuple(r[f"pred_{FIELD_ALIAS[f]}"] for f in FIELDS)

        if row is not None and gold != tuple(row[f] for f in FIELDS):
            gold_drift.append(r["id"])
        if any(p not in EVAL_FIELDS[f] for f, p in zip(FIELDS, pred)):
            bad_label.append(r["id"])
        if (r["gold_state_id"] != tuple_to_state_id(*gold)
                or r["pred_state_id"] != tuple_to_state_id(*pred)):
            state_drift.append(r["id"])
        if r["pred_state_id"] == INVALID_STATE_ID:
            invalid.append(r["id"])

    if gold_drift:
        bad(f"{len(gold_drift)} rows carry gold labels that disagree with the dataset "
            f"(row misalignment), e.g. {gold_drift[:3]}")
    if bad_label:
        bad(f"{len(bad_label)} rows predict a label outside the frozen enumeration, "
            f"e.g. {bad_label[:3]}")
    if state_drift:
        bad(f"{len(state_drift)} rows have a state_id disagreeing with their own label "
            f"columns, e.g. {state_drift[:3]}")
    if invalid and method not in (None, "M0"):
        bad(f"{method} emitted {len(invalid)} hierarchy-invalid tuples; M1-M6 must "
            f"emit none (contract §4.2 invariant 2)")
    return problems


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _method_of(path) -> str | None:
    stem = Path(path).name.split(".")[0]
    tail = stem.rsplit("_", 1)[-1]
    return tail if tail.startswith("M") and tail[1:].isdigit() else None


def _validate_paths(paths, rows, splits_dir=SPLITS_DIR) -> list[str]:
    problems: list[str] = []
    bundles = [p for p in paths if p.is_dir()]
    predictions = [p for p in paths if p.is_file()]

    for p in bundles:
        problems += validate_probs_bundle(p, splits_dir=splits_dir)
    by_run: dict[str, list[Path]] = {}
    for p in bundles:
        by_run.setdefault(p.name.rsplit("_r", 1)[0], []).append(p)
    for run, dirs in sorted(by_run.items()):
        if len(dirs) > 1:
            problems += [f"{run}: {m}"
                         for m in validate_probs_run(dirs, splits_dir=splits_dir)]
    if len(by_run) > 1:
        problems += validate_probs_study(bundles)

    for p in predictions:
        problems += validate_predictions(p, rows=rows, method=_method_of(p),
                                         splits_dir=splits_dir)
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate contract artifacts.")
    ap.add_argument("paths", nargs="*", type=Path,
                    help="probability bundle directories and/or predictions .csv.gz")
    ap.add_argument("--all", action="store_true",
                    help="validate every bundle and predictions file of --corpus")
    # Which rows and which fold manifests the artifacts are checked against.
    # Without this, English bundles are compared to the Chinese manifests and
    # the Chinese data checksum, and every mismatch is reported as a defect in
    # the bundle rather than as the wrong corpus having been named.
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPORA),
                    help="which corpus these artifacts belong to")
    args = ap.parse_args()

    splits_dir = REPO_ROOT / corpus_splits_dir(args.corpus)
    bundle_glob, predictions_glob = probs_globs(args.corpus)

    paths = list(args.paths)
    if args.all:
        paths += sorted(p for p in REPO_ROOT.glob(bundle_glob) if p.is_dir())
        paths += sorted(REPO_ROOT.glob(predictions_glob))
        if not paths:
            # The normal state of a fresh clone: B has not delivered yet.
            # Nothing to check is not a usage error.
            print(f"nothing to validate: {bundle_glob} and "
                  f"{predictions_glob} match nothing")
            return
    if not paths:
        ap.error("give at least one path, or --all")

    rows = load_rows(args.corpus)
    print(f"corpus: {args.corpus}")
    print(f"data_checksum: {data_checksum(rows)}")
    problems = _validate_paths(paths, rows, splits_dir=splits_dir)

    for p in problems:
        print(f"  FAIL  {p}")
    verdict = "clean" if not problems else f"{len(problems)} problem(s)"
    print(f"\n{len(paths)} artifact(s) checked: {verdict}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
