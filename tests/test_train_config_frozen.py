"""``paper/train_config.py`` is pinned by its own hash, comment lines included.

Every probability bundle records ``train_config_sha256`` as its proof of recipe,
and ``run_manifest.json`` repeats it once per run. Nothing on the read side
checks that value against the file: ``paper/validate.py`` compares bundles
against each other, so they can agree with each other and disagree with the
thing they claim to describe.

The trap is that the pinned content is the whole file, not the constants. A
tidy-up of a stale path inside a comment moves the hash exactly as far as
changing ``EPOCHS`` would, and the fits it invalidates are the ones nobody can
re-run: 30 official bundles on a GPU that is no longer on the critical path.

So the check is stated here, where it costs a hash of one small file.
"""

import json
from pathlib import Path

import pytest

from paper.data import REPO_ROOT, file_sha256

TRAIN_CONFIG = REPO_ROOT / "paper" / "train_config.py"
KEY = "train_config_sha256"
SKIP_PARTS = {".git", ".claude", "contracts"}


def _collect(node, out: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == KEY and isinstance(value, str):
                out.add(value)
            else:
                _collect(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect(item, out)


def _recorded() -> dict[Path, set[str]]:
    """Every ``train_config_sha256`` any versioned artifact records."""
    found: dict[Path, set[str]] = {}
    for path in sorted(REPO_ROOT.rglob("*.json")):
        # Relative, not absolute: a git worktree of this repo lives under
        # .claude/, so matching on the absolute parts skips the entire tree.
        if SKIP_PARTS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if KEY not in text:
            continue
        values: set[str] = set()
        _collect(json.loads(text), values)
        if values:
            found[path] = values
    return found


@pytest.fixture(scope="module")
def recorded():
    found = _recorded()
    if not found:
        pytest.skip("no artifact in this tree records train_config_sha256")
    return found


def test_the_artifacts_agree_on_one_recipe_hash(recorded):
    """A split vote would mean two recipes were used and neither is identified."""
    values = {v for vs in recorded.values() for v in vs}
    assert len(values) == 1, sorted(values)


def test_the_file_still_hashes_to_what_the_artifacts_recorded(recorded):
    """The check nothing else performs: recorded value against the file itself."""
    actual = file_sha256(TRAIN_CONFIG)
    stale = sorted(p.relative_to(REPO_ROOT).as_posix()
                   for p, vs in recorded.items() if actual not in vs)
    assert not stale, (
        f"{TRAIN_CONFIG.relative_to(REPO_ROOT)} now hashes to {actual}, which "
        f"{len(stale)} artifact(s) do not record -- e.g. {stale[:3]}. Every fit "
        "behind those bundles was run against the previous content, so the "
        "provenance chain no longer resolves. Revert the edit; comments are "
        "part of the hash."
    )
