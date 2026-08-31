"""Provenance stamps carried by every generated artifact.

Kept in its own stdlib-only module because both writers need it and they cannot
share one: ``paper/artifacts.py`` imports numpy, while ``paper/splits.py`` is
deliberately dependency-free so splits can be regenerated on the GPU box
without the study environment.
"""

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from paper.data import REPO_ROOT

# What "dirty" is allowed to mean. git_sha answers "which version of the code
# produced this file", so only the code counts -- writing the artifacts
# themselves must not make the stamp claim the code was modified.
#
# Without the exclusions the marker fires on the generator's own output: probs/
# is version controlled, so B's first bundle would dirty the tree and rotations
# 1-4 would all be stamped -dirty despite no code change, and the same happens
# to every artifact written under contracts/examples/ after the first.
#
# The environment files are in because they pin torch and transformers, and a
# version change there moves the numbers without touching a line of Python.
# analysis/ is in for the same reason paper/ is: it writes four stamped files
# under tables/, so an uncommitted edit there produces a brief whose git_sha
# names code that did not generate it. It was missing until 8/22, and the test
# that exists to catch exactly this listed the watched directories instead of
# deriving them -- so it passed throughout.
# Deliberately out: dataset/ (covered by data_checksum), splits/ (covered by
# split_fingerprint), docs/ and tests/ (cannot change an artifact).
CODE_PATHSPEC = (
    "paper",
    "analysis",
    "contracts",
    "scripts",
    "ntcir19-esg-validity-layer",
    "environment.yml",
    "pyproject.toml",
    ":(exclude)contracts/examples",
)


def git_sha(repo_root=REPO_ROOT):
    """Current commit, suffixed ``-dirty`` when the *code* has uncommitted edits.

    Returns None outside a repository or before the first commit; callers
    record that as-is rather than inventing a value.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *CODE_PATHSPEC], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# Recorded because a version change here moves the numbers without touching a
# line of this repository's code. torch is optional: A's half of the study runs
# without it, and its absence is a fact worth recording rather than an error.
PACKAGES = ("numpy", "scikit-learn", "pandas", "torch", "transformers")


def environment() -> dict:
    """Interpreter and package versions behind a generated artifact.

    Lives beside git_sha rather than in either writer because both need it, and
    a stamp comparable only to itself is not provenance: the bundle written by
    run_training and the manifest written at results freeze must carry the same
    shape, or no one can check that the run and the index agree.
    """
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
