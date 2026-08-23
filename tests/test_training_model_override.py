"""The training driver keeps custom backbone identity pinned and auditable.

What is checked here -- that ``snapshot_download`` is asked for the requested
model and revision, and that a snapshot resolving to a *different* revision is
refused -- is pure path logic and needs no torch. The import chain does:
``run_training`` pulls in ``train_fold``, which imports torch at module scope.
Isolating the two functions would mean restructuring the frozen training path,
so the module skips instead, exactly as ``tests/test_training_path.py`` does.

The consequence is that these two assertions run only in the conda environment.
That is where training happens and where an unpinned revision would do damage,
but it does mean a CPU-only suite is not evidence that they still pass.
"""

import pytest

pytest.importorskip("torch")

from paper import run_training


def test_snapshot_override_uses_requested_model_and_revision(monkeypatch, tmp_path):
    revision = "0123456789abcdef"
    snapshot = tmp_path / revision
    snapshot.mkdir()
    seen = {}

    def fake_download(model_name, *, revision):
        seen.update(model_name=model_name, revision=revision)
        return str(snapshot)

    monkeypatch.setattr(run_training, "snapshot_download", fake_download)

    resolved = run_training.resolve_pinned_snapshot("example/base-model", revision)

    assert resolved == snapshot
    assert seen == {
        "model_name": "example/base-model",
        "revision": revision,
    }


def test_snapshot_override_rejects_a_different_resolved_revision(monkeypatch, tmp_path):
    snapshot = tmp_path / "different-revision"
    snapshot.mkdir()
    monkeypatch.setattr(
        run_training, "snapshot_download",
        lambda model_name, *, revision: str(snapshot),
    )

    with pytest.raises(SystemExit) as excinfo:
        run_training.resolve_pinned_snapshot("example/base-model", "expected-revision")

    assert "expected-revision" in str(excinfo.value)
    assert "different-revision" in str(excinfo.value)
