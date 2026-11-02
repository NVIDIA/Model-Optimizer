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

"""Worker-side setup and assertions for megatron_bridge README scenarios."""

import json
import os
from pathlib import Path

from _test_utils.doc_tests.fixtures.megatron_bridge import (
    prepare_bridge_full_scale,
    verify_bridge_hf_export,
)

__all__ = [
    "bridge_generation",
    "bridge_import",
    "bridge_login",
    "bridge_mlflow",
    "bridge_prune_active",
    "bridge_prune_memory",
    "bridge_prune_vlm",
    "bridge_qad",
    "bridge_real_data",
]


def bridge_login(ctx):
    """Run and verify the bridge-login scenario."""
    assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN for the manual login test"
    ctx.cwd = ctx.tmp
    ctx.env["HF_HOME"] = str(ctx.tmp / "hf-home")
    ctx.run()
    assert (ctx.tmp / "hf-home/token").stat().st_size > 0


def bridge_import(ctx):
    """Run and verify the bridge-import scenario."""

    prepare_bridge_full_scale(ctx)
    output = Path(ctx.env["OUTPUT_ROOT"])
    ctx.run()
    assert (output / "Qwen3-8B-megatron").is_dir()


def bridge_mlflow(ctx):
    """Run and verify the bridge-mlflow scenario."""

    prepare_bridge_full_scale(ctx)
    ctx.run()
    pointer = Path(ctx.env["OUTPUT_ROOT"]) / "Qwen3-8B-NVFP4-megatron/.experiment.json"
    assert json.loads(pointer.read_text())


def bridge_real_data(ctx):
    """Run and verify the bridge-real-data scenario."""

    prepare_bridge_full_scale(ctx)
    ctx.run()
    checkpoints = Path(ctx.env["OUTPUT_ROOT"]) / "qwen3_8b_to_4b_distill/checkpoints"
    assert (checkpoints / "latest_checkpointed_iteration.txt").is_file()
    assert list(checkpoints.glob("iter_*"))


def bridge_qad(ctx):
    """Run and verify the bridge-qad scenario."""
    # Import the optional Megatron dependency only in this worker.
    from _test_utils.torch.megatron.modelopt_state import assert_has_modelopt_state

    prepare_bridge_full_scale(ctx)
    ctx.run()
    checkpoints = Path(ctx.env["OUTPUT_ROOT"]) / "qwen3_8b_nvfp4_qad/checkpoints"
    assert (checkpoints / "latest_checkpointed_iteration.txt").is_file()
    assert list(checkpoints.glob("iter_*"))

    assert_has_modelopt_state(checkpoints)


def bridge_prune_active(ctx):
    """Run and verify the bridge-prune-active scenario."""

    prepare_bridge_full_scale(ctx)
    output = Path(ctx.env["OUTPUT_ROOT"])
    ctx.run()
    verify_bridge_hf_export(output / "Nemotron-3-Nano-30B-A3B-BF16-Pruned-3B-Active")


def bridge_prune_memory(ctx):
    """Run and verify the bridge-prune-memory scenario."""

    prepare_bridge_full_scale(ctx)
    output = Path(ctx.env["OUTPUT_ROOT"])
    ctx.run()
    verify_bridge_hf_export(output / "Qwen3-8B-Pruned-12GB")


def bridge_prune_vlm(ctx):
    """Run and verify the bridge-prune-vlm scenario."""

    prepare_bridge_full_scale(ctx)
    output = Path(ctx.env["OUTPUT_ROOT"])
    ctx.run()
    verify_bridge_hf_export(output / "Qwen3.5-4B-Pruned-3B")


def bridge_generation(ctx):
    """Run and verify the bridge-generation scenario."""
    ctx.cwd = ctx.repo / "examples/megatron_bridge"
    ctx.run()
