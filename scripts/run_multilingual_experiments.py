"""Run one GPU's half of the frozen multilingual experiment matrix.

The two supported workers are disjoint: ``large`` owns XLM-R-large/RemBERT
and ``base`` owns XLM-R-base/mBERT.  The script is resume-safe because every
training invocation uses ``--skip-existing``; decision files are deterministic
and may be regenerated.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from paper.corpus import arm_dir
from paper.data import REPO_ROOT
from paper.multilingual_config import LANGUAGES, LAMBDAS, SEEDS, models_for


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", required=True, choices=["large", "base"])
    ap.add_argument("--gpu", required=True,
                    help="physical GPU index exposed as cuda:0")
    ap.add_argument("--languages", nargs="+", choices=LANGUAGES,
                    default=list(LANGUAGES))
    ap.add_argument("--decisions-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    if args.decisions_only and args.train_only:
        ap.error("--decisions-only and --train-only are mutually exclusive")

    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python = sys.executable

    for corpus in args.languages:
        for model in models_for(args.worker):
            for structure_lambda in LAMBDAS:
                arm = Path(arm_dir(corpus, model["name"], structure_lambda))
                probs = REPO_ROOT / arm / "probs"
                if not args.decisions_only:
                    for protocol in model["protocols"]:
                        for seed in SEEDS:
                            run([
                                python, "-u", "-m", "paper.run_training",
                                "--corpus", corpus,
                                "--protocol", protocol,
                                "--seed", str(seed),
                                "--model-name", model["name"],
                                "--model-revision", model["revision"],
                                "--structure-lambda", str(structure_lambda),
                                "--skip-existing",
                            ], env)
                if not args.train_only:
                    for protocol in model["protocols"]:
                        for seed in SEEDS:
                            run([
                                python, "-u", "-m", "paper.run_decisions",
                                "--corpus", corpus,
                                "--protocol", protocol,
                                "--seed", str(seed),
                                "--probs-dir", str(probs),
                            ], env)


if __name__ == "__main__":
    main()
