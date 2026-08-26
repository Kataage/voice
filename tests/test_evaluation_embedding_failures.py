from __future__ import annotations

from pathlib import Path

from personavoice import evaluation


def test_best_effort_embeddings_isolates_invalid_samples(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeWorker:
        def call(self, repo_root, command, payload):
            assert repo_root == Path("repo")
            assert command == "embed"
            audio = str(payload["audio"])
            calls.append((command, audio))
            if audio.endswith("bad.wav"):
                raise RuntimeError("diarization worker returned an invalid response schema for 'embed'")
            return {"embedding": [0.1, 0.2, 0.3]}

    monkeypatch.setattr(evaluation, "worker", lambda _repo_root, name: FakeWorker())

    results, errors = evaluation._best_effort_embeddings(
        Path("repo"),
        [
            {"id": "good", "audio": "good.wav"},
            {"id": "bad", "audio": "bad.wav"},
        ],
    )

    assert results == {"good": {"embedding": [0.1, 0.2, 0.3]}}
    assert "bad" in errors
    assert "invalid response schema" in errors["bad"]
    assert calls == [("embed", "good.wav"), ("embed", "bad.wav")]


def test_best_effort_embeddings_records_empty_embedding(monkeypatch):
    class FakeWorker:
        def call(self, _repo_root, command, payload):
            assert command == "embed"
            assert payload == {"audio": "empty.wav"}
            return {"embedding": []}

    monkeypatch.setattr(evaluation, "worker", lambda _repo_root, _name: FakeWorker())

    results, errors = evaluation._best_effort_embeddings(
        Path("repo"),
        [{"id": "empty", "audio": "empty.wav"}],
    )

    assert results == {}
    assert errors == {"empty": "speaker embedding was empty"}
