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
    prepare_bridge_smoke,
    run_bridge_distill_prerequisite,
    verify_bridge_distilled,
    verify_bridge_hf_export,
    verify_bridge_pruned,
    verify_bridge_quantized,
)

__all__ = [
    "bridge_distill",
    "bridge_distill_export",
    "bridge_distill_help",
    "bridge_generation",
    "bridge_import",
    "bridge_login",
    "bridge_mlflow",
    "bridge_prune_active",
    "bridge_prune_manual",
    "bridge_prune_memory",
    "bridge_prune_minitron_help",
    "bridge_prune_params",
    "bridge_prune_vlm",
    "bridge_ptq_export",
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


def bridge_ptq_export(ctx):
    """Run and verify the bridge-ptq-export scenario."""

    prepare_bridge_smoke(ctx)
    ctx.run()

    verify_bridge_quantized(ctx)


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


def bridge_distill_help(ctx):
    """Run and verify the bridge-distill-help scenario."""
    ctx.cwd = ctx.repo / "examples/megatron_bridge"
    ctx.run()


def bridge_distill(ctx):
    """Run and verify the bridge-distill scenario."""

    prepare_bridge_smoke(ctx)
    ctx.run()
    checkpoint = Path(ctx.env["OUTPUT_ROOT"]) / "test_distill/checkpoints"
    assert (checkpoint / "iter_0000002").is_dir()
    assert (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == "2"


def bridge_distill_export(ctx):
    """Run and verify the bridge-distill-export scenario."""

    prepare_bridge_smoke(ctx)
    run_bridge_distill_prerequisite(ctx)
    ctx.run()
    verify_bridge_distilled(ctx)


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


def bridge_prune_params(ctx):
    """Run and verify the bridge-prune-params scenario."""

    prepare_bridge_full_scale(ctx)
    output = Path(ctx.env["OUTPUT_ROOT"])
    ctx.run()
    verify_bridge_hf_export(output / "Qwen3-8B-Pruned-6B")


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


def bridge_prune_manual(ctx):
    """Run and verify the bridge-prune-manual scenario."""

    prepare_bridge_smoke(ctx)
    ctx.run()

    verify_bridge_pruned(ctx)


def bridge_prune_minitron_help(ctx):
    """Run and verify the bridge-prune-minitron-help scenario."""
    ctx.cwd = ctx.repo / "examples/megatron_bridge"
    ctx.run()


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
