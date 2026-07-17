# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint validation through the native AutoModel scoring recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

import modelopt.torch.utils.distributed as dist

__all__ = ["validate_model"]


def _cfg_get(args: DictConfig, key: str, default: Any = None) -> Any:
    try:
        return args.get(key, default)
    except Exception:
        return getattr(args, key, default)


def _standalone_config(args: DictConfig) -> DictConfig:
    automodel = _cfg_get(args, "automodel", None)
    if automodel is None or not _cfg_get(automodel, "parallel", None):
        raise ValueError(
            "AutoModel validation requires the full Hydra config or "
            "args.automodel.parallel; stitched/HF validation has been removed"
        )
    checkpoint = Path(str(_cfg_get(args, "model_name_or_path"))).resolve()
    scoring = OmegaConf.to_container(args, resolve=True)
    scoring["automodel"] = OmegaConf.to_container(automodel, resolve=True)
    return OmegaConf.create(
        {
            "descriptor": _cfg_get(args, "descriptor"),
            "puzzle_dir": str(checkpoint.parent.parent),
            "model": {"force_hf": False},
            "scoring": scoring,
            "realize_model": scoring,
        }
    )


@torch.no_grad()
def validate_model(
    args: DictConfig,
    model=None,
    tokenizer=None,
    target_hidden_states_per_batch=None,
    return_hidden_states: bool = False,
    calculate_full_score_ablations: bool = False,
    val_dataloader=None,
    *,
    hydra_cfg: DictConfig | None = None,
):
    """Validate one realized checkpoint and preserve the legacy metrics return shape.

    Direct preloaded-model and external-hidden-state validation belonged to the deleted stitched
    runtime. Callers must provide a checkpoint; its stage-local ``automodel.parallel``
    mesh is compiled into the native AutoModel configuration at launch time.
    """
    unsupported = []
    if model is not None:
        unsupported.append("preloaded model")
    if tokenizer is not None:
        unsupported.append("preloaded tokenizer")
    if target_hidden_states_per_batch is not None:
        unsupported.append("external hidden-state targets")
    if return_hidden_states:
        unsupported.append("return_hidden_states")
    if calculate_full_score_ablations:
        unsupported.append("calculate_full_score_ablations")
    if val_dataloader is not None:
        unsupported.append("external dataloader")
    if unsupported:
        raise ValueError(
            "AutoModel checkpoint validation does not accept " + ", ".join(unsupported)
        )

    checkpoint = Path(str(_cfg_get(args, "model_name_or_path", None) or ""))
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"validation checkpoint has no config.json: {checkpoint}")
    cfg = hydra_cfg if hydra_cfg is not None else _standalone_config(args)
    validation_args = OmegaConf.create(OmegaConf.to_container(args, resolve=True))
    validation_args.teacher_dir = str(checkpoint)
    output_dir = Path(
        str(
            _cfg_get(
                args,
                "output_dir",
                checkpoint / "validation" / "automodel",
            )
        )
    )

    from ..plugins.automodel.validation import validate_realized_checkpoints_automodel

    validate_realized_checkpoints_automodel(
        cfg,
        validation_args,
        (),
        output_dir,
    )
    dist.barrier()
    result_path = output_dir / "teacher.json"
    losses = json.loads(result_path.read_text()) if result_path.is_file() else None
    return losses, None
