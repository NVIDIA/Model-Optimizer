# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run or aggregate one campaign-configured post-MIP node."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from puzzletron_orchestrator.config import load_experiment_config  # noqa: E402


def _register_evaluation_profiles(config: dict) -> None:
    post_mip = config.get("post_mip")
    flows = post_mip.get("flows") if isinstance(post_mip, Mapping) else None
    profiles = set()
    for flow in flows.values() if isinstance(flows, Mapping) else ():
        nodes = flow.get("nodes") if isinstance(flow, Mapping) else None
        for node in nodes.values() if isinstance(nodes, Mapping) else ():
            if not isinstance(node, Mapping) or node.get("type") != "downstream_evaluation":
                continue
            node_config = node.get("config")
            if isinstance(node_config, Mapping):
                profile = node_config.get("profile")
                if isinstance(profile, str):
                    profiles.add(profile)
    if profiles & {
        "qwen35_vlm_realworldqa",
        "qwen35_vlm_e2e_full_eval",
        "qwen35_vlm_short_v1",
    }:
        from examples.puzzletron.evaluation.vlm.post_mip import register_profiles

        register_profiles()


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
        _register_evaluation_profiles(config)
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
