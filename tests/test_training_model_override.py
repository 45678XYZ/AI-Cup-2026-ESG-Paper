"""The training driver keeps custom backbone identity pinned and auditable."""

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

    try:
        run_training.resolve_pinned_snapshot("example/base-model", "expected-revision")
    except SystemExit as exc:
        assert "expected-revision" in str(exc)
        assert "different-revision" in str(exc)
    else:
        raise AssertionError("an incorrectly resolved revision was accepted")
