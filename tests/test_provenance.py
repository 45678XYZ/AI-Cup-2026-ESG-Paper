"""What the ``-dirty`` marker is allowed to mean.

``git_sha`` answers one question — which version of the *code* produced this
artifact — and §6.1 leans on the answer. The subtlety is that the marker must
not fire on the generator's own output, or B's second rotation would be stamped
``-dirty`` because the first one landed in a version-controlled directory.

That makes ``CODE_PATHSPEC`` load-bearing and invisible: get it wrong and every
artifact still gets a plausible stamp, just a false one. So the four cases are
asserted here against a real repository rather than reasoned about.
"""

import subprocess

import pytest

from paper.data import REPO_ROOT
from paper.provenance import CODE_PATHSPEC, git_sha


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo shaped like this one, with one commit."""
    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True,
                       capture_output=True, text=True)

    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    for rel in ("paper/labels.py", "analysis/findings.py",
                "contracts/make_examples.py",
                "contracts/examples/results/x.json", "docs/plan.md",
                "tests/test_x.py", "splits/s.json", "environment.yml"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "initial")
    return tmp_path


def _is_dirty(repo):
    sha = git_sha(repo)
    assert sha is not None
    return sha.endswith("-dirty")


def test_a_clean_tree_is_not_dirty(repo):
    assert not _is_dirty(repo)
    assert len(git_sha(repo)) == 40


@pytest.mark.parametrize("path", ["paper/labels.py", "analysis/findings.py",
                                  "contracts/make_examples.py",
                                  "environment.yml"])
def test_editing_code_marks_the_stamp_dirty(repo, path):
    (repo / path).write_text("changed\n", encoding="utf-8")
    assert _is_dirty(repo), f"{path} is code; a change to it must be visible"


@pytest.mark.parametrize("path", [
    "contracts/examples/results/x.json",   # the generator's own output
    "docs/plan.md",                        # cannot change an artifact
    "tests/test_x.py",
    "splits/s.json",                       # covered by split_fingerprint instead
])
def test_editing_generated_or_inert_files_does_not(repo, path):
    (repo / path).write_text("changed\n", encoding="utf-8")
    assert not _is_dirty(repo), f"{path} is not code; it must not fake a code change"


def test_writing_a_probs_bundle_does_not_dirty_the_stamp(repo):
    """B's real case: probs/ is version controlled, so the first bundle would
    otherwise mark rotations 1-4 as produced by modified code."""
    bundle = repo / "probs" / "pdf_group_seed42_r0"
    bundle.mkdir(parents=True)
    (bundle / "meta.json").write_text("{}", encoding="utf-8")
    assert not _is_dirty(repo)


def test_an_untracked_code_file_counts_as_dirty(repo):
    """A new module that was never committed still changes what the code does."""
    (repo / "paper" / "decision.py").write_text("x = 1\n", encoding="utf-8")
    assert _is_dirty(repo)


def test_outside_a_repository_the_stamp_is_none_rather_than_invented(tmp_path):
    assert git_sha(tmp_path) is None


# Directories that hold .py files but cannot change an artifact, so a stamp
# must not fire on them. Anything else with Python in it is watched.
INERT_CODE_DIRS = {"tests"}


def test_the_pathspec_covers_every_directory_holding_study_code():
    """A new top-level code directory would silently stop being watched, and
    every artifact after it would carry a stamp that reads clean.

    Derived from the tree rather than listed. Listing is what let ``analysis``
    -- which writes four provenance-stamped files under ``tables/`` -- go
    unwatched while an assertion named for this exact property passed.
    """
    included = {p for p in CODE_PATHSPEC if not p.startswith(":")}
    code_dirs = {
        d.name for d in REPO_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and any(d.glob("*.py")) and d.name not in INERT_CODE_DIRS
    }
    assert code_dirs, "no code directories found; the derivation is broken"
    assert code_dirs <= included, (
        "these directories hold study code but the dirty marker cannot see "
        f"them: {sorted(code_dirs - included)}")
    assert "environment.yml" in included, "dependency pins move results too"
    assert ":(exclude)contracts/examples" in CODE_PATHSPEC
