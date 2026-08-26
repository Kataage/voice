from __future__ import annotations

from pathlib import Path

import personavoice.evaluation as evaluation


def test_best_effort_embeddings_isolates_invalid_worker_response(monkeypatch):
    class FakeWorker:
        def call(self, repo_root, command, payload):
            assert repo_root == Path("repo")
            assert command == "embed"
            if payload["audio"] == "bad.wav":
                raise RuntimeError("diarization worker returned an invalid response schema for 'embed'")
            return {"embedding": [0.1, 0.2, 0.3]}

    monkeypatch.setattr(evaluation, "worker", lambda _root, name: FakeWorker())

    results, errors = evaluation._best_effort_embeddings(
        Path("repo"),
        [
            {"id": "good", "audio": "good.wav"},
            {"id": "bad", "audio": "bad.wav"},
        ],
        label="evaluation",
    )

    assert results == {"good": {"embedding": [0.1, 0.2, 0.3]}}
    assert errors == {"bad": "evaluation: invalid-response-schema"}


def test_best_effort_embeddings_rejects_empty_embedding_without_aborting(monkeypatch):
    class FakeWorker:
        def call(self, _repo_root, command, payload):
            assert command == "embed"
            assert payload == {"audio": "empty.wav"}
            return {"embedding": []}

    monkeypatch.setattr(evaluation, "worker", lambda _root, _name: FakeWorker())

    results, errors = evaluation._best_effort_embeddings(
        Path("repo"),
        [{"id": "empty", "audio": "empty.wav"}],
        label="identity",
    )

    assert results == {}
    assert errors == {"empty": "identity: empty-embedding"}
