from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading

import pytest

from modelopt.torch.puzzletron.scoring_parent import (
    ensure_scoring_parent,
    load_scoring_parent,
    resolve_scoring_parent,
    write_scoring_parent,
)
from modelopt.torch.puzzletron.stages.depth import _resolve_depth_source


def _checkpoint(path, model_type):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": model_type}) + "\n")
    return path


def _config(tmp_path, *, enabled=True, use_bypassed=True):
    return {
        "experiment": {"dir": str(tmp_path)},
        "bypass": {
            "enabled": enabled,
            "use_nested_bypassed_checkpoint_for_scoring": use_bypassed,
        },
    }


def test_scoring_parent_uses_sorted_teacher_when_bypass_is_disabled(tmp_path):
    sorted_teacher = _checkpoint(tmp_path / "ckpts" / "sorted_teacher", "teacher")

    parent = resolve_scoring_parent(
        _config(tmp_path, enabled=False, use_bypassed=True)
    )

    assert parent.role == "sorted_teacher"
    assert parent.path == sorted_teacher.resolve()
    assert parent.sorted_teacher_fingerprint == parent.fingerprint
    assert parent.bypass_manifest_fingerprint is None


def test_depth_can_use_explicit_teacher_before_sorted_checkpoint_exists(tmp_path):
    teacher = _checkpoint(tmp_path / "ckpts" / "teacher", "teacher")

    source, record = _resolve_depth_source(
        {"depth_importance": {"source_checkpoint_dir": str(teacher)}}
    )

    assert source == teacher.resolve()
    assert record["role"] == "configured_depth_source"
    assert record["path"] == str(teacher.resolve())
    assert record["fingerprint"]


def test_scoring_parent_uses_validated_nested_bypass_checkpoint(tmp_path):
    sorted_teacher = _checkpoint(tmp_path / "ckpts" / "sorted_teacher", "teacher")
    bypassed = _checkpoint(tmp_path / "ckpts" / "elastic_sorted_teacher", "student")
    manifest = tmp_path / "manifests" / "bypass.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "success", "outputs": {"checkpoint": str(bypassed)}}))

    parent = resolve_scoring_parent(_config(tmp_path))

    assert parent.role == "nested_bypassed"
    assert parent.path == bypassed.resolve()
    assert parent.fingerprint != parent.sorted_teacher_fingerprint
    assert parent.bypass_manifest_fingerprint is not None
    assert sorted_teacher.exists()


def test_scoring_parent_fails_closed_when_requested_bypass_is_missing(tmp_path):
    _checkpoint(tmp_path / "ckpts" / "sorted_teacher", "teacher")

    with pytest.raises(FileNotFoundError, match="nested bypass scoring parent"):
        resolve_scoring_parent(_config(tmp_path))


def test_scoring_parent_artifact_rejects_stale_checkpoint_identity(tmp_path):
    bypassed = _checkpoint(tmp_path / "ckpts" / "elastic_sorted_teacher", "student")
    _checkpoint(tmp_path / "ckpts" / "sorted_teacher", "teacher")
    manifest = tmp_path / "manifests" / "bypass.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "success"}))
    artifact = tmp_path / "artifacts" / "scoring_parent.json"
    write_scoring_parent(resolve_scoring_parent(_config(tmp_path)), artifact)
    (bypassed / "config.json").write_text(json.dumps({"model_type": "changed"}))

    with pytest.raises(RuntimeError, match="stale scoring parent"):
        load_scoring_parent(artifact)


def test_checkpoint_identity_ignores_weight_file_mtime(tmp_path):
    from modelopt.torch.puzzletron.distributed_eval.config import checkpoint_identity

    checkpoint = _checkpoint(tmp_path / "checkpoint", "teacher")
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"weights")
    before = checkpoint_identity(checkpoint)
    stat = weight.stat()
    os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    after = checkpoint_identity(checkpoint)

    assert after == before


def test_ensure_scoring_parent_writes_and_reuses_valid_artifact(tmp_path):
    _checkpoint(tmp_path / "ckpts" / "sorted_teacher", "teacher")
    config = _config(tmp_path, enabled=False, use_bypassed=False)

    first = ensure_scoring_parent(config)
    second = ensure_scoring_parent(config)

    assert first == second
    assert (tmp_path / "artifacts" / "scoring_parent.json").is_file()


def test_scoring_parent_artifact_supports_concurrent_atomic_writers(
    tmp_path, monkeypatch
):
    _checkpoint(tmp_path / "ckpts" / "sorted_teacher", "teacher")
    parent = resolve_scoring_parent(
        _config(tmp_path, enabled=False, use_bypassed=False)
    )
    artifact = tmp_path / "artifacts" / "scoring_parent.json"
    writers_ready = threading.Barrier(2)
    original_write_text = Path.write_text

    def synchronized_write(path, *args, **kwargs):
        result = original_write_text(path, *args, **kwargs)
        if path.name.startswith("scoring_parent.json.tmp"):
            writers_ready.wait(timeout=5)
        return result

    monkeypatch.setattr(Path, "write_text", synchronized_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: write_scoring_parent(parent, artifact), range(2)))

    assert results == [artifact, artifact]
    assert load_scoring_parent(artifact) == parent
