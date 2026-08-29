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

"""Adapt the pinned Qwen 3.5 VLM profile to a post-MIP checkpoint node."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import evaluate

__all__ = ["evaluate_realworldqa_checkpoint", "register_profiles"]

_RUNNER_OVERRIDES = frozenset(
    {
        "dtype",
        "gpu_memory_utilization",
        "limit_mm_per_prompt",
        "max_model_len",
        "topology",
    }
)


def register_profiles() -> None:
    """Install the example-owned profile into the generic post-MIP runner."""

    from modelopt.torch.puzzletron.post_mip.runner import register_downstream_evaluation_profile

    register_downstream_evaluation_profile(
        "qwen35_vlm_realworldqa",
        evaluate_realworldqa_checkpoint,
    )


def evaluate_realworldqa_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the two-sample pinned RealWorldQA profile for one saved checkpoint."""

    settings = dict(settings)
    unexpected = set(settings) - _RUNNER_OVERRIDES - {"batch_size", "timeout_seconds"}
    if unexpected:
        raise ValueError(f"unsupported Qwen 3.5 VLM profile settings: {sorted(unexpected)}")
    output_dir = Path(output_root).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        checkpoint=Path(checkpoint_path).expanduser().absolute(),
        output_dir=output_dir,
        suite="realworldqa-smoke",
        batch_size=int(settings.pop("batch_size", 1)),
        seed=42,
        timeout_seconds=settings.pop("timeout_seconds", None),
        hf_home=Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else None,
        quick_manifest=None,
        mmvu_judge_api_type=None,
        mmvu_judge_model=None,
        allow_judge_calls=False,
        preflight_only=False,
    )
    profile_path = output_dir / "profile.json"

    def write_preflight(report: dict[str, object]) -> None:
        checkpoint.write_generated(
            profile_path,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )

    result = evaluate(
        args,
        settings_overrides=settings,
        preflight_callback=write_preflight,
    )
    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise RuntimeError("pinned RealWorldQA profile returned an invalid run count")
    return {**runs[0], "profile_path": str(profile_path)}
