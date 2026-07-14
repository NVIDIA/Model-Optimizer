from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from examples.puzzletron.verify_cross_model_activation import verify_model_activation


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "scoring": {"eval_samples": 4, "micro_batch_size": 1},
                "pruning": {
                    "activation_passes": [
                        {"name": "attention", "axis_ids": ["kv_groups", "q_heads"]},
                        {"name": "dense_ffn", "axis_ids": ["ffn_intermediate"]},
                    ]
                },
            }
        )
    )


def test_verify_model_activation_reports_nonempty_and_zero_target_passes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    model_root = tmp_path / "model"
    base = model_root / "pruning/pruning_scores/automodel/all_axes"
    _config(config)
    _write_json(base / "activation_passes_manifest.json", {"passes": ["attention"]})
    for name in ("attention", "dense_ffn"):
        _write_json(
            base / name / "args.json",
            {"eval_samples": 4, "micro_batch_size": 1, "eval_iters": 4, "num_nodes": 2},
        )
    torch.save(
        {"model.layers.0.attn": {"score": torch.tensor([3.0, 1.0])}},
        base / "attention/rank_0.pth",
    )
    torch.save(
        {"model.layers.1.attn": {"score": torch.tensor([2.0, 0.5])}},
        base / "attention/rank_1.pth",
    )

    result = verify_model_activation("toy", model_root, config)

    assert result["status"] == "passed"
    assert result["passes"]["attention"]["status"] == "passed"
    assert result["passes"]["attention"]["modules"] == 2
    assert result["passes"]["attention"]["rank_files"] == 2
    assert result["passes"]["attention"]["values"] == 4
    assert result["passes"]["dense_ffn"]["status"] == "zero_targets"


def test_verify_model_activation_rejects_nonfinite_scores(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    model_root = tmp_path / "model"
    base = model_root / "pruning/pruning_scores/automodel/all_axes"
    _config(config)
    _write_json(base / "activation_passes_manifest.json", {"passes": ["attention"]})
    _write_json(
        base / "attention/args.json",
        {"eval_samples": 4, "micro_batch_size": 1, "eval_iters": 4, "num_nodes": 1},
    )
    torch.save(
        {"model.layers.0.attn": {"score": torch.tensor([float("nan")])}},
        base / "attention/rank_0.pth",
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        verify_model_activation("toy", model_root, config)
