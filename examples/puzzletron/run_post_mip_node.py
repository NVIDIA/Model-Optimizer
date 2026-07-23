# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run or aggregate one campaign-configured post-MIP node."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from puzzletron_orchestrator.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    shard_index = (
        int(os.environ.get("PUZZLETRON_GROUP_INDEX", os.environ.get("SLURM_PROCID", "0")))
        if args.shard_index is None
        else args.shard_index
    )
    if args.aggregate:
        from puzzletron_orchestrator.post_mip import aggregate_post_mip_node

        config = load_experiment_config(args.config, overrides=args.override)
        payload = aggregate_post_mip_node(config, args.stage_id)
    else:
        # Candidate execution runs in the full worker environment. Defer these
        # GPU-heavy imports so login-node aggregation remains dependency-light.
        from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
        from modelopt.torch.puzzletron.post_mip import run_post_mip_node_shard

        config = pipeline_config_from_path(args.config, overrides=args.override)
        payload = {
            "result_path": str(
                run_post_mip_node_shard(
                    config,
                    args.stage_id,
                    shard_index=shard_index,
                    shard_count=args.shard_count,
                )
            )
        }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
