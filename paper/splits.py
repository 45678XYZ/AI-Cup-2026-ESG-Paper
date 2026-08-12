"""Rotating three-way split generation.

Two protocols, five rotations each:

    Test        = fold k
    Calibration = fold (k + 1) mod 5
    Train       = the remaining three folds

``pdf_group``  folds are groups of source PDFs, so Train/Calibration/Test share
               no report. Estimates document-disjoint generalisation.
``row_strat``  folds are label-stratified paragraphs. Rows are disjoint but a
               PDF may span partitions; every Calibration/Test PDF is required
               to have at least one *other* row in Train, which is what makes
               "seen report, unseen paragraph" a property of the protocol
               rather than merely of the dataset.

Deliberately dependency-free (stdlib only). B must be able to regenerate
splits on the GPU box without installing the study environment, and fold
assignment is too load-bearing to depend on a library's tie-breaking behaviour
staying constant across versions.

Output conforms to interface contract §2.
"""

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from paper.data import REPO_ROOT, canonical_row_order, data_checksum, load_dev
from paper.labels import EVAL_FIELDS, row_to_state_id
from paper.train_config import N_FOLDS, SEEDS

CONTRACT_VERSION = "1.0"
SPLITS_DIR = REPO_ROOT / "splits"

# Stratification key for row_strat. Includes verification_timeline specifically
# so the 34 within_2_years rows are spread across folds rather than clustering.
STRAT_KEYS = ("promise_status", "evidence_status", "verification_timeline")

MAX_COVERAGE_ATTEMPTS = 50


# --------------------------------------------------------------------------
# Fold assignment
# --------------------------------------------------------------------------

def _assign_pdf_folds(rows, rng):
    """Partition PDFs into N_FOLDS groups of roughly equal row count.

    Randomised greedy: PDFs are placed largest-first into one of the two
    least-loaded folds, chosen at random. Largest-first keeps folds balanced
    despite the 4..91 spread in paragraphs per PDF; the random choice between
    the two lightest folds is what makes the partition genuinely seed-dependent
    (a purely deterministic greedy would return the same folds for every seed
    and collapse the seed-to-seed variance the paper reports).

    Note this uses no labels. Rare classes are therefore not balanced across
    folds by construction; their per-fold support is reported instead, and
    absent calibration classes drive the documented bias fallback.
    """
    by_pdf = defaultdict(list)
    for r in rows:
        by_pdf[r["pdf_url"]].append(r["id"])

    pdfs = sorted(by_pdf)
    rng.shuffle(pdfs)
    pdfs.sort(key=lambda p: -len(by_pdf[p]))  # stable: keeps shuffled order within ties

    folds = [[] for _ in range(N_FOLDS)]
    sizes = [0] * N_FOLDS
    for pdf in pdfs:
        lightest = sorted(range(N_FOLDS), key=lambda i: (sizes[i], i))[:2]
        j = rng.choice(lightest)
        folds[j].append(pdf)
        sizes[j] += len(by_pdf[pdf])

    return {pdf: j for j, group in enumerate(folds) for pdf in group}


def _assign_row_folds(rows, rng):
    """Label-stratified row folds.

    Rows are bucketed by STRAT_KEYS, each bucket is shuffled, then dealt
    round-robin into folds. The starting fold rotates per bucket so that small
    strata do not all deposit their single row into fold 0.
    """
    buckets = defaultdict(list)
    for r in rows:
        buckets["|".join(r[k] for k in STRAT_KEYS)].append(r["id"])

    assignment = {}
    for offset, key in enumerate(sorted(buckets)):
        ids = buckets[key]
        rng.shuffle(ids)
        for i, row_id in enumerate(ids):
            assignment[row_id] = (i + offset) % N_FOLDS
    return assignment


# --------------------------------------------------------------------------
# Rotations and their checks
# --------------------------------------------------------------------------

def _rotation_partitions(row_fold, order, k):
    """Return (train_ids, calibration_ids, test_ids) in canonical order."""
    test_fold = k
    calib_fold = (k + 1) % N_FOLDS
    train, calib, test = [], [], []
    for row_id in order:
        f = row_fold[row_id]
        if f == test_fold:
            test.append(row_id)
        elif f == calib_fold:
            calib.append(row_id)
        else:
            train.append(row_id)
    return train, calib, test


def _same_document_coverage(by_id, train_ids, held_ids):
    """Every held-out PDF must have at least one *different* row in Train.

    Returns (satisfied, violating_pdfs, n_pdfs_checked).
    """
    train_pdfs = Counter(by_id[i]["pdf_url"] for i in train_ids)
    held_pdfs = sorted({by_id[i]["pdf_url"] for i in held_ids})
    violating = [p for p in held_pdfs if train_pdfs[p] == 0]
    return not violating, violating, len(held_pdfs)


def _support(by_id, ids):
    out = {}
    for field, labels in EVAL_FIELDS.items():
        counts = Counter(by_id[i][field] for i in ids)
        out[field] = {lab: counts.get(lab, 0) for lab in labels}
    return out


def _tuple_support(by_id, ids):
    counts = Counter(row_to_state_id(by_id[i]) for i in ids)
    return {str(state_id): n for state_id, n in sorted(counts.items())}


def _absent_calibration_classes(by_id, calib_ids, train_ids):
    """Classes present in Train but with no Calibration sample.

    These cannot have a bias estimated (contract §2) and fall back to 0.0.
    Only classes the model could actually predict are listed, so a class absent
    from the whole protocol is not reported as a calibration problem.
    """
    absent = {}
    for field, labels in EVAL_FIELDS.items():
        in_calib = {by_id[i][field] for i in calib_ids}
        in_train = {by_id[i][field] for i in train_ids}
        missing = [lab for lab in labels if lab in in_train and lab not in in_calib]
        if missing:
            absent[field] = missing
    return absent


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_split(protocol, seed, rows=None):
    """Build one split manifest. Raises if same-document coverage is unmet."""
    rows = rows if rows is not None else load_dev()
    by_id = {r["id"]: r for r in rows}
    order = canonical_row_order(rows)

    for attempt in range(MAX_COVERAGE_ATTEMPTS):
        # Seeded from a string so the stream depends on protocol and attempt
        # too; PYTHONHASHSEED cannot perturb it the way hash(tuple) would.
        rng = random.Random(f"{protocol}:{seed}:{attempt}")
        if protocol == "pdf_group":
            pdf_fold = _assign_pdf_folds(rows, rng)
            row_fold = {r["id"]: pdf_fold[r["pdf_url"]] for r in rows}
        elif protocol == "row_strat":
            row_fold = _assign_row_folds(rows, rng)
        else:
            raise ValueError(f"unknown protocol: {protocol}")

        rotations = [
            _build_rotation(by_id, row_fold, order, k, protocol)
            for k in range(N_FOLDS)
        ]
        if all(r["checks"]["same_document_coverage"] is None
               or r["checks"]["same_document_coverage"]["satisfied"]
               for r in rotations):
            break
    else:
        raise RuntimeError(
            f"{protocol} seed={seed}: same-document coverage unmet after "
            f"{MAX_COVERAGE_ATTEMPTS} attempts"
        )

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "protocol": protocol,
        "seed": seed,
        "data_checksum": data_checksum(rows),
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "paper/splits.py",
        "resample_attempts": attempt,
        "canonical_row_order": order,
        "rotations": rotations,
        "fallback_rule": (
            "A class absent from the Calibration partition gets bias 0.0 and is "
            "listed in calibration_absent_classes; the applied fallback is "
            "recorded per rotation in the results manifest."
        ),
    }
    manifest["split_fingerprint"] = split_fingerprint(manifest)
    return manifest


def _build_rotation(by_id, row_fold, order, k, protocol):
    train, calib, test = _rotation_partitions(row_fold, order, k)

    def pdfs_of(ids):
        return sorted({by_id[i]["pdf_url"] for i in ids})

    train_pdfs, calib_pdfs, test_pdfs = pdfs_of(train), pdfs_of(calib), pdfs_of(test)

    if protocol == "row_strat":
        satisfied, violating, n_checked = _same_document_coverage(by_id, train, calib + test)
        coverage = {
            "satisfied": satisfied,
            "n_pdfs_checked": n_checked,
            "violating_pdfs": violating,
        }
    else:
        coverage = None

    return {
        "k": k,
        "train_ids": train,
        "calibration_ids": calib,
        "test_ids": test,
        "train_pdfs": train_pdfs,
        "calibration_pdfs": calib_pdfs,
        "test_pdfs": test_pdfs,
        "support": {
            "train": _support(by_id, train),
            "calibration": _support(by_id, calib),
            "test": _support(by_id, test),
        },
        "tuple_support": {
            "train": _tuple_support(by_id, train),
            "calibration": _tuple_support(by_id, calib),
            "test": _tuple_support(by_id, test),
        },
        "checks": {
            "rows_disjoint": len(set(train) & set(calib)) == 0
                             and len(set(train) & set(test)) == 0
                             and len(set(calib) & set(test)) == 0,
            "rows_cover_all_2000": len(train) + len(calib) + len(test) == len(order),
            "pdfs_disjoint": not (set(train_pdfs) & set(calib_pdfs))
                             and not (set(train_pdfs) & set(test_pdfs))
                             and not (set(calib_pdfs) & set(test_pdfs)),
            "same_document_coverage": coverage,
        },
        "calibration_absent_classes": _absent_calibration_classes(by_id, calib, train),
    }


def split_fingerprint(split) -> str:
    """sha256 over the partition content alone.

    Covers protocol, seed, the canonical row order and every rotation's three
    id lists — everything that determines what a run actually trains and scores
    on. Deliberately excludes ``created_at``, ``git_sha`` and the support
    tables, so regenerating a semantically identical manifest does not
    invalidate bundles already produced against it.

    Probability bundles copy this value, so a set of rotations carries a
    compact record of which partition it belongs to. Comparing it across the
    five bundles of one (protocol, seed) is how a run split across a manifest
    change becomes visible.
    """
    h = hashlib.sha256()
    h.update(f"{split['protocol']}|{split['seed']}".encode())
    for row_id in split["canonical_row_order"]:
        h.update(b"\x1f" + row_id.encode())
    for rot in sorted(split["rotations"], key=lambda r: r["k"]):
        h.update(f"\x1e{rot['k']}".encode())
        for name in ("train_ids", "calibration_ids", "test_ids"):
            h.update(b"\x1d" + name.encode())
            for row_id in rot[name]:
                h.update(b"\x1f" + row_id.encode())
    return "sha256:" + h.hexdigest()


def _git_sha():
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Generate rotating split manifests.")
    ap.add_argument("--protocols", nargs="+", default=["pdf_group", "row_strat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    args = ap.parse_args()

    rows = load_dev()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for protocol in args.protocols:
        for seed in args.seeds:
            split = build_split(protocol, seed, rows)
            path = args.out_dir / f"{protocol}_seed{seed}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(split, f, ensure_ascii=False, indent=1)
            sizes = [len(r["test_ids"]) for r in split["rotations"]]
            print(f"{path.name}: test fold sizes={sizes} attempts={split['resample_attempts']}")


if __name__ == "__main__":
    main()
