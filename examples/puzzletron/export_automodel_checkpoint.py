#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore a distributed AutoModel checkpoint and export exact HF weights."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _distributed_sizes(recipe: dict) -> tuple[int, int, int]:
    distributed = recipe["distributed"]
    return (
        int(distributed.get("tp_size", 1)),
        int(distributed.get("cp_size", 1)),
        int(distributed.get("pp_size", 1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text())
    tp, cp, pp = _distributed_sizes(recipe)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    model_parallel_size = tp * cp * pp
    if world_size % model_parallel_size:
        raise ValueError(
            f"WORLD_SIZE={world_size} is not divisible by TP*CP*PP={model_parallel_size}"
        )
    recipe["distributed"]["dp_size"] = world_size // model_parallel_size
    recipe["checkpoint"]["model_save_format"] = "safetensors"
    recipe["checkpoint"]["save_consolidated"] = "final"

    export_recipe = args.output_dir / "export_recipe.json"
    if int(os.environ.get("RANK", "0")) == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        export_recipe.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    deadline = time.monotonic() + 120
    while not export_recipe.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {export_recipe}")
        time.sleep(0.1)

    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    from modelopt.torch.puzzletron.distillation.global_kd_recipe import (
        KnowledgeDistillationRecipeForNextTokenPrediction,
        KnowledgeDistillationRecipeForVLM,
        install_pp_checkpoint_state_dict_support,
    )
    from modelopt.torch.puzzletron.plugins.automodel.patch import apply_patch

    apply_patch()
    cfg = parse_args_and_load_config(str(export_recipe), argv=[])
    recipe_cls = (
        KnowledgeDistillationRecipeForVLM
        if recipe.get("recipe") == "KnowledgeDistillationRecipeForVLM"
        else KnowledgeDistillationRecipeForNextTokenPrediction
    )
    if pp > 1:
        install_pp_checkpoint_state_dict_support()
    trainer = recipe_cls(cfg)
    trainer.setup()

    export_step = args.output_dir / "checkpoint"
    if torch.distributed.get_rank() == 0:
        export_step.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()
    trainer.checkpointer.save_model(
        trainer.model_parts,
        str(export_step),
        tokenizer=getattr(trainer, "tokenizer", None),
        is_final_checkpoint=True,
    )
    torch.distributed.barrier()

    consolidated = export_step / "model" / "consolidated"
    if torch.distributed.get_rank() == 0:
        from modelopt.torch.puzzletron.plugins.automodel.local_kd_recipe import (
            _copy_hf_auxiliary_assets,
        )

        _copy_hf_auxiliary_assets(
            Path(recipe["model"]["pretrained_model_name_or_path"]),
            consolidated,
        )
        if not (consolidated / "config.json").is_file():
            raise RuntimeError(f"consolidated export has no config.json: {consolidated}")
        if args.summary is not None:
            summary = json.loads(args.summary.read_text())
            summary["post_kd_checkpoint"] = str(consolidated)
            temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
            temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            temporary.replace(args.summary)
        print(consolidated, flush=True)
    torch.distributed.barrier()
    trainer.checkpointer.close()


if __name__ == "__main__":
    main()
