#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare deterministic subblock-only vLLM samples for one width scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.block_config import maybe_cast_block_configs
from modelopt.torch.puzzletron.candidates import build_candidate_library
from modelopt.torch.puzzletron.distributed_eval.config import checkpoint_identity
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.sampling.sparse import (
    SparseSamplingPolicy,
    sample_subblock_configs,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--max-pairwise-per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = pipeline_config_from_path(args.config)
    scenario_dir = Path(args.scenario_dir)
    teacher = scenario_dir / "ckpts" / "sorted_teacher"
    scenario = json.loads((scenario_dir / "scenario_manifest.json").read_text())
    width = int(scenario["hidden_width"])
    model_cfg = cfg.get("model") or {}
    descriptor = resolve_descriptor_from_pretrained(
        str(teacher),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=model_cfg.get("descriptor_override"),
    ).descriptor
    model_config = load_model_config(
        teacher,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    blocks = list(maybe_cast_block_configs(model_config.block_configs))
    candidates = build_candidate_library(
        blocks,
        search_space=cfg.get("search_space") or {},
        parent_checkpoint_identity=checkpoint_identity(teacher)["fingerprint"],
        include_self=True,
        include_noops=bool(
            (cfg.get("build_replacement_library") or {}).get(
                "include_noops", False
            )
        ),
        hidden_width=width,
    )
    manifest = sample_subblock_configs(
        candidates,
        policy=SparseSamplingPolicy(
            max_pairwise_per_family=args.max_pairwise_per_family,
            seed=args.seed,
        ),
    ).to_dict()
    manifest.update(
        {
            "model_id": Path(cfg["puzzle_dir"]).name,
            "hidden_width": width,
            "scenario_dir": str(scenario_dir.resolve()),
            "parent_checkpoint_identity": checkpoint_identity(teacher),
            "candidate_count": len(candidates),
        }
    )
    output = scenario_dir / "sparse_subblock_samples.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "identity": manifest["identity"],
                "eligible": len(manifest["eligible"]),
                "selected": len(manifest["selected"]),
                "excluded": len(manifest["excluded"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
