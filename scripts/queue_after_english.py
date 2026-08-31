"""Finish the English matrix, then launch both multilingual GPU workers.

This is a persistent, resume-safe queue.  It first waits until the English
training processes that were already running have exited. It then fills any
missing English bundles, generates their decisions, verifies the expected 150
bundles, and only then starts French/Japanese/Korean training.
"""

import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def resolve_english_root(multi_root: Path, configured: str | None = None) -> Path:
    """Resolve a distinct English worktree without requiring it to exist."""
    multi_root = Path(multi_root).resolve()
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = multi_root.parent / "AI-Cup-2026-ESG-Paper"
    if candidate.resolve() == multi_root:
        candidate = multi_root.with_name(f"{multi_root.name}-english")
    return candidate


MULTI_ROOT = Path(__file__).resolve().parent.parent
ENGLISH_ROOT = resolve_english_root(
    MULTI_ROOT, configured=os.environ.get("AI_CUP_ENGLISH_ROOT")
)
PYTHON = Path("/home/tom1030507/miniconda3/envs/aicup-esg/bin/python")
STATE = MULTI_ROOT / "local_data" / "multilingual_queue_state.json"

ENGLISH_MODELS = (
    {
        "name": "roberta-large",
        "revision": "722cf37b1afa9454edce342e7895e588b6ff1d59",
        "protocols": ("pdf_group", "row_strat"),
        "worker": "large",
    },
    {
        "name": "microsoft/deberta-v3-large",
        "revision": "64a8c8eab3e352a784c658aef62be1662607476f",
        "protocols": ("pdf_group",),
        "worker": "large",
    },
    {
        "name": "google/electra-large-discriminator",
        "revision": "c13c3df7efadc2162f42588bd28eb4e187d602a5",
        "protocols": ("pdf_group",),
        "worker": "large",
    },
    {
        "name": "roberta-base",
        "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "protocols": ("pdf_group",),
        "worker": "base",
    },
)
LAMBDAS = (0.0, 0.3)
SEEDS = (42, 123, 456)


def write_state(stage: str, **extra) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "updated_at": time.strftime("%FT%T%z"), **extra},
                  f, indent=1)
        f.write("\n")
    os.replace(temporary, STATE)


def active_english_training() -> list[int]:
    active = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "paper.run_training" in cmd and "--corpus mlpromise_en" in cmd:
            active.append(int(proc.name))
    return active


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def english_worker(worker: str, gpu: int) -> None:
    env = dict(os.environ)
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        # DeBERTa/ELECTRA are pre-populated at pinned snapshots before this
        # queue is launched. Offline mode prevents the older English driver
        # from fetching TensorFlow/Flax copies it will never load.
        "HF_HUB_OFFLINE": "1",
    })
    for model in (m for m in ENGLISH_MODELS if m["worker"] == worker):
        for structure_lambda in LAMBDAS:
            for protocol in model["protocols"]:
                for seed in SEEDS:
                    run([
                        str(PYTHON), "-u", "-m", "paper.run_training",
                        "--corpus", "mlpromise_en",
                        "--protocol", protocol,
                        "--seed", str(seed),
                        "--model-name", model["name"],
                        "--model-revision", model["revision"],
                        "--structure-lambda", str(structure_lambda),
                        "--skip-existing",
                    ], ENGLISH_ROOT, env)


def finish_english_decisions() -> None:
    env = dict(os.environ)
    for model in ENGLISH_MODELS:
        slug = model["name"].rsplit("/", 1)[-1].replace("-", "_").replace(".", "_")
        for structure_lambda in LAMBDAS:
            probs = ENGLISH_ROOT / "runs_en" / slug / f"lambda_{structure_lambda:.1f}" / "probs"
            for protocol in model["protocols"]:
                for seed in SEEDS:
                    run([
                        str(PYTHON), "-u", "-m", "paper.run_decisions",
                        "--corpus", "mlpromise_en",
                        "--protocol", protocol,
                        "--seed", str(seed),
                        "--probs-dir", str(probs),
                    ], ENGLISH_ROOT, env)


def english_bundle_count() -> int:
    return len(list((ENGLISH_ROOT / "runs_en").glob("*/*/probs/*/meta.json")))


def validate_english_arms() -> None:
    """The English branch validator predates multi-arm --all grouping."""
    env = dict(os.environ)
    for model in ENGLISH_MODELS:
        slug = model["name"].rsplit("/", 1)[-1].replace("-", "_").replace(".", "_")
        for structure_lambda in LAMBDAS:
            arm = ENGLISH_ROOT / "runs_en" / slug / f"lambda_{structure_lambda:.1f}"
            paths = [
                *sorted(path for path in (arm / "probs").iterdir() if path.is_dir()),
                *sorted((arm / "predictions").glob("*.csv.gz")),
            ]
            run([str(PYTHON), "-m", "paper.validate", "--corpus", "mlpromise_en",
                 *map(str, paths)], ENGLISH_ROOT, env)


def multilingual_worker_commands() -> list[list[str]]:
    """Build workers as modules so the project root remains importable."""
    return [
        [str(PYTHON), "-u", "-m", "scripts.run_multilingual_experiments",
         "--worker", "large", "--gpu", "1"],
        [str(PYTHON), "-u", "-m", "scripts.run_multilingual_experiments",
         "--worker", "base", "--gpu", "0"],
    ]


def main() -> None:
    write_state("waiting_for_active_english")
    quiet_polls = 0
    while quiet_polls < 2:
        active = active_english_training()
        quiet_polls = quiet_polls + 1 if not active else 0
        write_state("waiting_for_active_english", active_pids=active)
        if quiet_polls < 2:
            time.sleep(30)

    write_state("completing_english")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(english_worker, "large", 1),
            pool.submit(english_worker, "base", 0),
        ]
        for future in futures:
            future.result()
    finish_english_decisions()
    count = english_bundle_count()
    if count != 150:
        raise SystemExit(f"English completion produced {count} bundles, expected 150")
    validate_english_arms()
    english_predictions = len(list(
        (ENGLISH_ROOT / "runs_en").glob("*/*/predictions/*.csv.gz")
    ))
    english_results = len(list(
        (ENGLISH_ROOT / "runs_en").glob("*/*/results/*.json")
    ))
    if (english_predictions, english_results) != (210, 210):
        raise SystemExit(
            "English decisions incomplete: "
            f"{english_predictions} predictions, {english_results} results"
        )

    write_state("running_multilingual", english_bundles=count,
                english_predictions=english_predictions,
                english_results=english_results)
    env = dict(os.environ)
    commands = multilingual_worker_commands()
    processes = [subprocess.Popen(command, cwd=MULTI_ROOT, env=env) for command in commands]
    codes = [process.wait() for process in processes]
    if codes != [0, 0]:
        raise SystemExit(f"multilingual workers failed: exit codes {codes}")

    for corpus in ("mlpromise_fr", "mlpromise_ja", "mlpromise_ko"):
        run([str(PYTHON), "-m", "paper.validate", "--all", "--corpus", corpus],
            MULTI_ROOT, env)
        suffix = corpus.rsplit("_", 1)[-1]
        root = MULTI_ROOT / f"runs_{suffix}"
        counts = {
            "bundles": len(list(root.glob("*/*/probs/*/meta.json"))),
            "predictions": len(list(root.glob("*/*/predictions/*.csv.gz"))),
            "results": len(list(root.glob("*/*/results/*.json"))),
        }
        if counts != {"bundles": 150, "predictions": 210, "results": 210}:
            raise SystemExit(f"{corpus} incomplete: {counts}")
    write_state("complete", english_bundles=count,
                multilingual_bundles=450,
                multilingual_predictions=630,
                multilingual_results=630)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_state("failed", error=repr(exc))
        raise
