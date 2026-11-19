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

"""Worker-side setup and assertions for llm_qat README scenarios."""

import json
import math
import shutil

__all__ = [
    "llm_qad_trainer",
    "llm_qat_cli_override",
    "llm_qat_copy_config",
    "llm_qat_llama_wrap",
    "llm_qat_python",
    "llm_qat_simple",
    "llm_qat_trainer",
]


def llm_qat_simple(ctx):
    """Run and verify the llm-qat-simple scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace

    workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp, llama=True)
    ctx.cwd = workspace / "examples/llm_qat"
    ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    ctx.run()
    assert (ctx.cwd / "qat_model/modelopt_state.pth").is_file()
    assert (ctx.cwd / "qat_model/model.safetensors").stat().st_size > 0


def llm_qat_python(ctx):
    """Run and verify the llm-qat-python scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import (
        prepare_llm_qat_python_api,
        verify_llm_qat_python_api,
    )

    namespace = prepare_llm_qat_python_api(ctx, quantized=False)
    ctx.run(namespace)
    verify_llm_qat_python_api(namespace["trainer"], namespace["initial_weights"])


def llm_qat_trainer(ctx):
    """Run and verify the llm-qat-trainer scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import (
        prepare_llm_qat_python_api,
        verify_llm_qat_python_api,
    )

    namespace = prepare_llm_qat_python_api(ctx, quantized=True)
    ctx.run(namespace)
    verify_llm_qat_python_api(namespace["trainer"], namespace["initial_weights"])


def llm_qad_trainer(ctx):
    """Run and verify the llm-qad-trainer scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import (
        prepare_llm_qat_python_api,
        verify_llm_qat_python_api,
    )

    namespace = prepare_llm_qat_python_api(ctx, quantized=True)
    ctx.run(namespace)
    verify_llm_qat_python_api(namespace["trainer"], namespace["initial_weights"])


def llm_qat_llama_wrap(ctx):
    """Run and verify the llm-qat-llama-wrap scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_quantized_input

    prepare_llm_qat_quantized_input(ctx, llama=True)
    (ctx.cwd / "llama-quantized").symlink_to(
        ctx.cwd / "qwen3-8b-quantized", target_is_directory=True
    )
    ctx.run()
    assert json.loads((ctx.cwd / "llama-qat/trainer_state.json").read_text())["global_step"] == 2


def llm_qat_copy_config(ctx):
    """Run and verify the llm-qat-copy-config scenario."""
    ctx.cwd = ctx.tmp
    shutil.copytree(ctx.repo / "examples/llm_qat/configs", ctx.tmp / "configs")
    ctx.run()
    assert (ctx.cwd / "configs/accelerate/fsdp2_llama.yaml").read_bytes() == (
        ctx.cwd / "configs/accelerate/fsdp2.yaml"
    ).read_bytes()


def llm_qat_cli_override(ctx):
    """Run and verify the llm-qat-cli-override scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    import torch
    from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_quantized_input

    prepare_llm_qat_quantized_input(ctx)
    ctx.run()
    output = ctx.cwd / "qwen3-8b-qat-override"
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    scheduler = torch.load(
        output / "checkpoint-2/scheduler.pt", map_location="cpu", weights_only=True
    )
    base_rates = scheduler["base_lrs"]
    assert base_rates and all(math.isclose(rate, 5e-5) for rate in base_rates), base_rates
