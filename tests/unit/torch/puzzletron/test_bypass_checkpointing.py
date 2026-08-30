# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors.torch import save_file

from modelopt.torch.puzzletron.bypass_distillation import checkpointing
from modelopt.torch.puzzletron.bypass_distillation.checkpointing import (
    save_ranked_state_checkpoint,
    validate_consolidated_hf_checkpoint,
    validate_ranked_state_checkpoint,
)


def _write_checkpoint(path, tensors_by_shard, weight_map):
    path.mkdir(parents=True)
    for shard_name, tensors in tensors_by_shard.items():
        save_file(tensors, path / shard_name)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}) + "\n"
    )


def test_validate_consolidated_hf_checkpoint_rejects_missing_indexed_shard(tmp_path):
    checkpoint = tmp_path / "consolidated"
    _write_checkpoint(
        checkpoint,
        {"model-00001-of-00002.safetensors": {"model.layers.0.weight": torch.ones(1)}},
        {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    )

    with pytest.raises(RuntimeError, match="missing indexed shard"):
        validate_consolidated_hf_checkpoint(
            checkpoint,
            expected_layer_prefixes=("model.layers.0", "model.layers.1"),
        )


def test_validate_consolidated_hf_checkpoint_rejects_missing_decoder_layer(tmp_path):
    checkpoint = tmp_path / "consolidated"
    _write_checkpoint(
        checkpoint,
        {"model.safetensors": {"model.layers.0.weight": torch.ones(1)}},
        {"model.layers.0.weight": "model.safetensors"},
    )

    with pytest.raises(RuntimeError, match="missing expected layer prefix.*model.layers.1"):
        validate_consolidated_hf_checkpoint(
            checkpoint,
            expected_layer_prefixes=("model.layers.0", "model.layers.1"),
        )


def test_validate_consolidated_hf_checkpoint_reports_exact_key_inventory(tmp_path):
    checkpoint = tmp_path / "consolidated"
    _write_checkpoint(
        checkpoint,
        {
            "model-00001-of-00002.safetensors": {"model.layers.0.weight": torch.ones(1)},
            "model-00002-of-00002.safetensors": {"model.layers.1.weight": torch.ones(1)},
        },
        {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    )

    inventory = validate_consolidated_hf_checkpoint(
        checkpoint,
        expected_layer_prefixes=("model.layers.0", "model.layers.1"),
    )

    assert inventory == {
        "checkpoint": str(checkpoint.resolve()),
        "indexed_keys": 2,
        "actual_keys": 2,
        "shards": 2,
        "expected_layer_prefixes": 2,
        "status": "complete",
    }


def test_ranked_state_checkpoint_rejects_missing_rank(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    save_ranked_state_checkpoint(
        checkpoint,
        state_name="rng",
        rank=0,
        state={"value": torch.tensor([1])},
    )

    with pytest.raises(RuntimeError, match=r"rng.*missing.*\[1\]"):
        validate_ranked_state_checkpoint(
            checkpoint,
            state_name="rng",
            expected_ranks=range(2),
        )


def test_ranked_state_checkpoint_is_atomic_and_loadable(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    for rank in range(2):
        save_ranked_state_checkpoint(
            checkpoint,
            state_name="rng",
            rank=rank,
            state={"rank": rank},
        )

    inventory = validate_ranked_state_checkpoint(
        checkpoint,
        state_name="rng",
        expected_ranks=range(2),
    )

    assert inventory == {
        "state_name": "rng",
        "expected_ranks": [0, 1],
        "files": 2,
        "status": "complete",
    }
    assert not list((checkpoint / "rng").glob("*.tmp.*"))


def test_incomplete_checkpoint_is_quarantined_before_retry(tmp_path):
    checkpoint = tmp_path / "epoch_0_step_4"
    checkpoint.mkdir()
    (checkpoint / "partial").write_text("interrupted")

    quarantined = checkpointing.quarantine_incomplete_checkpoint(checkpoint)

    assert not checkpoint.exists()
    assert quarantined is not None
    assert quarantined.parent == tmp_path
    assert quarantined.name.startswith(".epoch_0_step_4.quarantine.")
    assert (quarantined / "partial").read_text() == "interrupted"


def test_completed_checkpoint_is_never_quarantined(tmp_path):
    checkpoint = tmp_path / "epoch_0_step_4"
    checkpoint.mkdir()
    (checkpoint / "saving_completed").touch()

    with pytest.raises(FileExistsError, match="completed checkpoint"):
        checkpointing.quarantine_incomplete_checkpoint(checkpoint)

    assert checkpoint.exists()


def _write_minimal_automodel_checkpoint(path, *, rng_ranks=range(2)):
    path.mkdir()
    (path / "config.yaml").write_text("model: {}\n")
    (path / "losses.json").write_text('{"train_loss": 1.0}\n')
    torch.save({}, path / "grad_scaler.pt")
    torch.save({}, path / "step_scheduler.pt")
    model_dir = path / "model"
    model_dir.mkdir()
    save_file({"model.layers.0.weight": torch.ones(1)}, model_dir / "shard.safetensors")
    optim_dir = path / "optim"
    optim_dir.mkdir()
    (optim_dir / ".metadata").write_bytes(b"metadata")
    (optim_dir / "__0_0.distcp").write_bytes(b"optimizer")
    for rank in rng_ranks:
        save_ranked_state_checkpoint(
            path,
            state_name="rng",
            rank=rank,
            state={"rank": rank},
        )


def test_automodel_checkpoint_validation_covers_resume_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_minimal_automodel_checkpoint(checkpoint)

    inventory = checkpointing.validate_automodel_bypass_checkpoint(
        checkpoint,
        expected_rng_ranks=range(2),
    )

    assert inventory["status"] == "complete"
    assert inventory["model_shards"] == 1
    assert inventory["optimizer_shards"] == 1
    assert inventory["rng"]["files"] == 2


def test_automodel_checkpoint_validation_rejects_missing_optimizer_metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_minimal_automodel_checkpoint(checkpoint)
    (checkpoint / "optim" / ".metadata").unlink()

    with pytest.raises(RuntimeError, match="optimizer metadata"):
        checkpointing.validate_automodel_bypass_checkpoint(
            checkpoint,
            expected_rng_ranks=range(2),
        )
