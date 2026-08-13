"""Provenance stamps carried by every generated artifact.

Kept in its own stdlib-only module because both writers need it and they cannot
share one: ``paper/artifacts.py`` imports numpy, while ``paper/splits.py`` is
deliberately dependency-free so splits can be regenerated on the GPU box
without the study environment.
"""

import subprocess
from datetime import datetime, timezone

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
# Deliberately out: dataset/ (covered by data_checksum), splits/ (covered by
# split_fingerprint), docs/ and tests/ (cannot change an artifact).
CODE_PATHSPEC = (
    "paper",
    "contracts",
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
