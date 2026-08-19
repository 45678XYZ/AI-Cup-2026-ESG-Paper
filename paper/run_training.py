"""Driver for one or more rotations: split manifest in, contract bundle out.

    python -m paper.run_training --protocol pdf_group --seed 42
    python -m paper.run_training --protocol pdf_group --seed 42 --rotations 0 1

This is the only supported way to produce probabilities for the study. It
exists so that conforming to the contract is not something the operator has to
remember: the driver slices its training rows straight out of the split
manifest, predicts in the manifest's own id order, and writes through the
shared bundle writer, so a run is correct by construction rather than by
discipline.

Run it once per (protocol, seed); the full study is 2 protocols x 3 seeds x 5
rotations = 30 bundles from 30 fits.
"""

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from paper.artifacts import write_probs_bundle
from paper.data import REPO_ROOT, data_checksum, file_sha256, index_by_id, load_dev
from paper.provenance import environment, now_iso
from paper.train_config import (
    CHECKPOINT_LAST_K,
    CHECKPOINT_RULE,
    EPOCHS,
    MODEL_NAME,
    MODEL_REVISION,
    SEEDS,
)
from paper.train_fold import predict_probs, train_rotation
from paper.validate import UNPINNED_REVISIONS

TRAIN_CONFIG_PATH = Path(__file__).resolve().parent / "train_config.py"


def _hardware():
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (cuda:0 of {torch.cuda.device_count()})"
    return "cpu"


def _command() -> str:
    """The invocation, in a form that can actually be re-run.

    ``sys.argv[0]`` under ``python -m`` is the resolved path of the module
    *file*, so joining it yields a command that puts ``paper/`` on sys.path and
    dies importing ``paper.artifacts`` -- the one thing this field exists for.
    ``sys.orig_argv`` keeps the real ``-m paper.run_training`` form.
    """
    return shlex.join(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])


def _runtime_meta():
    # ``environment`` is the same stamp run_manifest records, so a bundle and
    # the manifest that indexes it can be compared field by field. What is
    # local to a fit -- the command, and which GPU CUDA exposed -- is added here.
    return {
        "command": _command(),
        "environment": environment(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def resolve_pinned_snapshot(revision=MODEL_REVISION):
    """Fetch-or-reuse the pinned snapshot once, and prove it is the pinned one.

    Every rotation then loads from the returned directory rather than by Hub
    id: transformers 4.40 otherwise probes for an optional safetensors
    conversion on each load, even when pytorch_model.bin is already cached.

    That path costs the pin, which is why the check below is not decoration.
    ``from_pretrained`` silently ignores ``revision`` for a local directory, so
    once the driver hands over a path nothing downstream enforces anything --
    while meta.json goes on recording MODEL_REVISION as though something did.
    Hugging Face lays snapshots out as ``snapshots/<commit>``, so the directory
    name *is* the resolved commit and can be compared against the pin.
    """
    path = Path(snapshot_download(MODEL_NAME, revision=revision))
    if revision not in UNPINNED_REVISIONS and path.name != revision:
        raise SystemExit(
            f"{MODEL_NAME} resolved to snapshot {path.name}, not the pinned "
            f"{revision}. Refusing to train against unidentified weights."
        )
    return path


def run_rotation(split, rotation, rows, tokenizer, out_root, save_checkpoint=False,
                 model_source=None):
    by_id = index_by_id(rows)
    rot = next(r for r in split["rotations"] if r["k"] == rotation)

    # Slice from the manifest, never re-derive: the id lists are the contract,
    # and predicting in their exact order is what makes the arrays alignable.
    train_data = [by_id[i] for i in rot["train_ids"]]
    partitions = {
        "calibration": [by_id[i] for i in rot["calibration_ids"]],
        "test": [by_id[i] for i in rot["test_ids"]],
    }

    print(
        f"[{split['protocol']} seed={split['seed']} r{rotation}] "
        f"train={len(train_data)} calib={len(partitions['calibration'])} "
        f"test={len(partitions['test'])}",
        flush=True,
    )

    started = now_iso()
    t0 = time.time()
    model, avg_state = train_rotation(
        train_data, tokenizer, split["seed"], model_name=model_source,
        local_files_only=True,
    )
    probs = {name: predict_probs(model, data, tokenizer) for name, data in partitions.items()}
    elapsed = time.time() - t0

    out_dir = Path(out_root) / f"{split['protocol']}_seed{split['seed']}_r{rotation}"
    write_probs_bundle(
        out_dir, split, rotation, probs,
        extra_meta={
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "train_config_sha256": file_sha256(TRAIN_CONFIG_PATH),
            "checkpoint_rule": CHECKPOINT_RULE,
            "checkpoint_last_k": CHECKPOINT_LAST_K,
            "epochs": EPOCHS,
            "hardware": _hardware(),
            "started_at": started,
            "finished_at": now_iso(),
            "train_seconds": round(elapsed, 1),
            "n_train_rows": len(train_data),
            **_runtime_meta(),
        },
    )
    if save_checkpoint:
        torch.save(avg_state, out_dir / "model.pt")

    print(f"  -> {out_dir}  ({elapsed / 60:.1f} min)", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", required=True, choices=["pdf_group", "row_strat"])
    ap.add_argument("--seed", required=True, type=int, choices=list(SEEDS))
    ap.add_argument("--rotations", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "splits")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "probs")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="also write the averaged weights (~1.3GB per rotation)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip rotations whose bundle already has a meta.json")
    ap.add_argument("--allow-unpinned-revision", action="store_true",
                    help="smoke tests only: run without a pinned MODEL_REVISION")
    args = ap.parse_args()

    # An unpinned revision means the run cannot be reproduced against a known
    # set of weights. Failing here rather than after the fits saves the GPU time.
    if MODEL_REVISION in UNPINNED_REVISIONS and not args.allow_unpinned_revision:
        raise SystemExit(
            f"MODEL_REVISION is {MODEL_REVISION!r} in paper/train_config.py, "
            "which does not identify a fixed set of weights.\n"
            f"Set it to the commit `huggingface-cli scan-cache` reports for "
            f"{MODEL_NAME}, or what the Hub's Files tab shows for the branch "
            "you intend, or pass --allow-unpinned-revision for a throwaway "
            "smoke test."
        )

    split_path = (args.splits_dir / f"{args.protocol}_seed{args.seed}.json").resolve()
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    try:
        split["_source_path"] = split_path.relative_to(REPO_ROOT)
    except ValueError:
        split["_source_path"] = split_path

    rows = load_dev()
    if split["data_checksum"] != data_checksum(rows):
        raise SystemExit(f"{split_path} was built against different data; refusing to run.")

    # Downloads on a cold machine and reuses the cache on a warm one. Only
    # OSError is caught: LocalEntryNotFoundError, HTTP failures and a full or
    # unreadable cache all derive from it, while a TypeError from a changed
    # huggingface_hub signature must keep its own traceback rather than be
    # relabelled as a fetch problem right before a 30-fit campaign.
    try:
        model_source = str(resolve_pinned_snapshot())
    except OSError as exc:
        raise SystemExit(
            f"Could not obtain {MODEL_NAME} at revision {MODEL_REVISION}: {exc}\n"
            "Check network access to huggingface.co, or pre-populate the cache "
            f"with `huggingface-cli download {MODEL_NAME} "
            f"--revision {MODEL_REVISION}`."
        ) from exc
    # The snapshot is complete by now, so the per-load Hub probes are pure cost.
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)

    for k in args.rotations:
        out_dir = args.out_dir / f"{args.protocol}_seed{args.seed}_r{k}"
        if args.skip_existing and (out_dir / "meta.json").exists():
            print(f"[r{k}] exists, skipping", flush=True)
            continue
        run_rotation(
            split, k, rows, tokenizer, args.out_dir, args.save_checkpoint,
            model_source=model_source,
        )

    print(f"\nbundles written to {args.out_dir}/{args.protocol}_seed{args.seed}_r*")


if __name__ == "__main__":
    main()
