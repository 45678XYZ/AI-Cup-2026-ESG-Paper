"""Validate and close the queue after split recovery workers finish.

The original two-worker queue can no longer own the large worker after a
recoverable model-specific failure.  This finalizer watches the shared,
resume-safe artifacts instead of trusting either parent process's exit code.
It marks the campaign complete only after every requested artifact exists and
all three corpus validators pass.
"""

import json
import os
import subprocess
import time
from pathlib import Path

from paper.data import REPO_ROOT
from paper.multilingual_config import LANGUAGES
from scripts.queue_after_english import ENGLISH_ROOT, PYTHON, write_state

EXPECTED = {"bundles": 150, "predictions": 210, "results": 210}


def artifact_counts(corpus: str) -> dict[str, int]:
    suffix = corpus.rsplit("_", 1)[-1]
    root = REPO_ROOT / f"runs_{suffix}"
    return {
        "bundles": len(list(root.glob("*/*/probs/*/meta.json"))),
        "predictions": len(list(root.glob("*/*/predictions/*.csv.gz"))),
        "results": len(list(root.glob("*/*/results/*.json"))),
    }


def active_multilingual_workers() -> list[int]:
    active = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "scripts.run_multilingual_experiments" in command:
            active.append(int(proc.name))
    return active


def validate_corpus(corpus: str) -> None:
    subprocess.run(
        [str(PYTHON), "-m", "paper.validate", "--all", "--corpus", corpus],
        cwd=REPO_ROOT,
        env=dict(os.environ),
        check=True,
    )


def english_counts() -> dict[str, int]:
    root = ENGLISH_ROOT / "runs_en"
    return {
        "bundles": len(list(root.glob("*/*/probs/*/meta.json"))),
        "predictions": len(list(root.glob("*/*/predictions/*.csv.gz"))),
        "results": len(list(root.glob("*/*/results/*.json"))),
    }


def main() -> None:
    inactive_polls = 0
    while True:
        counts = {corpus: artifact_counts(corpus) for corpus in LANGUAGES}
        overfull = {
            corpus: values
            for corpus, values in counts.items()
            if any(values[key] > EXPECTED[key] for key in EXPECTED)
        }
        if overfull:
            raise SystemExit(f"artifact counts exceed frozen matrix: {overfull}")

        if all(values == EXPECTED for values in counts.values()):
            if english_counts() != EXPECTED:
                raise SystemExit(
                    f"English artifacts changed before finalization: {english_counts()}"
                )
            for corpus in LANGUAGES:
                validate_corpus(corpus)
            write_state(
                "complete",
                english_bundles=150,
                english_predictions=210,
                english_results=210,
                multilingual_bundles=450,
                multilingual_predictions=630,
                multilingual_results=630,
                per_language=counts,
            )
            return

        active = active_multilingual_workers()
        inactive_polls = inactive_polls + 1 if not active else 0
        if inactive_polls >= 2:
            raise SystemExit(
                "multilingual workers exited before the frozen matrix was complete: "
                + json.dumps(counts, sort_keys=True)
            )
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_state("failed", error=repr(exc), finalizer=True)
        raise
