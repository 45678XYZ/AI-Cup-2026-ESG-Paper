"""Writers for the contract artifacts, kept free of torch.

The training driver and the synthetic fixture generator both write contract-2
probability bundles. They go through the same function here so a fixture and a
real GPU run cannot drift apart in shape — which matters, because the whole
point of the fixtures is that switching to real probabilities is a path change
and nothing else.

Importable without torch or transformers, so the bundle format stays testable
on any machine.
"""

import csv
import gzip
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from paper.data import REPO_ROOT, file_sha256
from paper.labels import EVAL_FIELDS, FIELD_ALIAS, FIELDS, tuple_to_state_id

CONTRACT_VERSION = "1.0"

PREDICTION_COLUMNS = (
    ["id", "pdf_url", "rotation"]
    + [f"gold_{a}" for a in FIELD_ALIAS.values()]
    + [f"pred_{a}" for a in FIELD_ALIAS.values()]
    + ["gold_state_id", "pred_state_id"]
)


def git_sha(repo_root=REPO_ROOT):
    """Current commit, suffixed ``-dirty`` when the tree has changes.

    Returns None outside a repository or before the first commit; callers
    record that as-is rather than inventing a value.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Contract 2: probability bundles
# --------------------------------------------------------------------------

def _check_distribution(arr, label):
    """Reject anything that is not a row-stochastic array (contract §3).

    Cheap here, and the failure it catches is otherwise invisible: logits, a
    half-applied softmax or an argmaxed one-hot all have the right shape and
    produce a perfectly plausible-looking results table.
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{label}: contains NaN or Inf")
    if (arr < 0).any():
        raise ValueError(f"{label}: contains negative values, so it is not a probability")
    if arr.size:
        dev = float(np.abs(arr.sum(axis=1) - 1.0).max())
        if dev > 1e-4:
            raise ValueError(f"{label}: rows do not sum to 1 (max deviation {dev:.2e})")


def write_probs_bundle(out_dir, split, rotation, probs, extra_meta=None):
    """Write one ``probs/{protocol}_seed{seed}_r{k}/`` directory.

    ``probs`` is ``{"calibration": {field: (N, C) array}, "test": {...}}``.
    Row order must already match the split's id lists — this function checks
    the lengths and the probability range but cannot check the ordering, which
    is why the driver slices its input straight from the manifest rather than
    re-deriving it. ``paper/validate.py`` covers what one call cannot see.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rot = next(r for r in split["rotations"] if r["k"] == rotation)

    artifacts = {}
    for partition in ("calibration", "test"):
        ids = rot[f"{partition}_ids"]
        for field in FIELDS:
            arr = np.asarray(probs[partition][field], dtype=np.float32)
            expected = (len(ids), len(EVAL_FIELDS[field]))
            if arr.shape != expected:
                raise ValueError(
                    f"{partition}/{field}: got {arr.shape}, expected {expected}"
                )
            _check_distribution(arr, f"{partition}/{field}")
            path = out_dir / f"{partition}_{field}.npy"
            np.save(path, arr)
            artifacts[path.name] = {"sha256": file_sha256(path), "shape": list(arr.shape)}

    split_file = split.get("_source_path", f"splits/{split['protocol']}_seed{split['seed']}.json")
    meta = {
        "contract_version": CONTRACT_VERSION,
        "protocol": split["protocol"],
        "seed": split["seed"],
        "rotation": rotation,
        "split_file": str(split_file),
        "data_checksum": split["data_checksum"],
        "split_fingerprint": split["split_fingerprint"],
        "calibration_ids": rot["calibration_ids"],
        "test_ids": rot["test_ids"],
        "git_sha": git_sha(),
        "created_at": now_iso(),
        "artifacts": artifacts,
    }
    meta.update(extra_meta or {})

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return out_dir


# --------------------------------------------------------------------------
# Contract 3: per-row predictions
# --------------------------------------------------------------------------

def prediction_row(row, rotation, pred):
    """Build one predictions-CSV record.

    ``pred`` maps the four field names to predicted class names. ``pdf_url``
    and the gold columns are duplicated from the dataset on purpose: C's
    statistics then need neither the raw data nor the split logic, so there is
    no join left to get wrong.
    """
    out = {"id": row["id"], "pdf_url": row["pdf_url"], "rotation": rotation}
    for field, alias in FIELD_ALIAS.items():
        out[f"gold_{alias}"] = row[field]
        out[f"pred_{alias}"] = pred[field]
    out["gold_state_id"] = tuple_to_state_id(*(row[f] for f in FIELDS))
    out["pred_state_id"] = tuple_to_state_id(*(pred[f] for f in FIELDS))
    return out


def write_predictions(path, records):
    """Write the gzipped per-row predictions CSV (contract §4.1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(records)
    return path


def read_predictions(path):
    """Read a predictions CSV back, keeping ids as strings."""
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["id"] = str(r["id"])
        r["rotation"] = int(float(r["rotation"]))
        r["gold_state_id"] = int(float(r["gold_state_id"]))
        r["pred_state_id"] = int(float(r["pred_state_id"]))
    return rows
