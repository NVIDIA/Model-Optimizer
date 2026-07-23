# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelopt.torch.puzzletron.checkpoint_transactions import (
    REALIZATION_MANIFEST,
    invalidate_realization,
    prepare_realization_retry,
    quarantine_incomplete_realization,
    realization_is_complete,
    remove_realization_temp_dir,
)


def test_realization_is_complete_requires_manifest_and_shards(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_text("weights\n")
    assert realization_is_complete(checkpoint) is False

    (checkpoint / REALIZATION_MANIFEST).write_text(
        json.dumps({"status": "complete"}) + "\n"
    )
    assert realization_is_complete(checkpoint) is True


def test_prepare_realization_retry_quarantines_partial_dir(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / REALIZATION_MANIFEST).write_text(
        json.dumps({"status": "running", "solution_identity": "abc"}) + "\n"
    )

    assert prepare_realization_retry(checkpoint) is True
    assert not checkpoint.exists()
    quarantined = next(tmp_path.glob(".checkpoint.realization_quarantine.*"))
    assert quarantined.is_dir()


def test_prepare_realization_retry_skips_complete_identity(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_text("weights\n")
    identity = {"solution_identity": "abc", "config_identity": "cfg", "sorted_teacher_identity": "src"}
    (checkpoint / REALIZATION_MANIFEST).write_text(
        json.dumps({"status": "complete", **identity}) + "\n"
    )

    assert prepare_realization_retry(checkpoint, expected_identity=identity) is False
    assert realization_is_complete(checkpoint)


def test_prepare_realization_retry_rejects_identity_mismatch(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_text("weights\n")
    (checkpoint / REALIZATION_MANIFEST).write_text(
        json.dumps({"status": "complete", "solution_identity": "old"}) + "\n"
    )

    with pytest.raises(FileExistsError):
        prepare_realization_retry(
            checkpoint,
            expected_identity={"solution_identity": "new"},
        )


def test_remove_realization_temp_dir(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    tmp_dir = tmp_path / "checkpoint.puzzletron-tmp"
    tmp_dir.mkdir()
    (tmp_dir / "partial.txt").write_text("partial\n")

    remove_realization_temp_dir(checkpoint)
    assert not tmp_dir.exists()


def test_quarantine_refuses_complete_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_text("weights\n")
    (checkpoint / REALIZATION_MANIFEST).write_text(
        json.dumps({"status": "complete"}) + "\n"
    )

    with pytest.raises(FileExistsError):
        quarantine_incomplete_realization(checkpoint)


def test_prepare_realization_retry_quarantines_corrupt_complete_manifest(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / REALIZATION_MANIFEST).write_text(
        json.dumps({"status": "complete", "solution_identity": "abc"}) + "\n"
    )

    assert prepare_realization_retry(
        checkpoint,
        expected_identity={"solution_identity": "abc"},
    )
    assert not checkpoint.exists()
    assert next(tmp_path.glob(".checkpoint.realization_quarantine.*")).is_dir()


def test_first_load_failure_can_invalidate_complete_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_text("weights\n")
    (checkpoint / REALIZATION_MANIFEST).write_text(
        json.dumps({"status": "complete"}) + "\n"
    )
    tmp_dir = tmp_path / "checkpoint.puzzletron-tmp"
    tmp_dir.mkdir()

    quarantined = invalidate_realization(checkpoint)

    assert quarantined is not None
    assert quarantined.is_dir()
    assert not checkpoint.exists()
    assert not tmp_dir.exists()
