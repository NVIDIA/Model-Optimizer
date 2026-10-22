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
import os
import shutil
from pathlib import Path

__all__ = [
    "llm_qad_quickstart",
    "llm_qad_trainer",
    "llm_qat_cli_override",
    "llm_qat_copy_config",
    "llm_qat_dataset",
    "llm_qat_evaluation",
    "llm_qat_install",
    "llm_qat_llama_wrap",
    "llm_qat_python",
    "llm_qat_quickstart",
    "llm_qat_simple",
    "llm_qat_trainer",
    "llm_qlora_quickstart",
    "llm_qlora_serve",
]


def llm_qat_install(ctx):
    """Run and verify the llm-qat-install scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat_installation import (
        prepare_llm_qat_installation,
        verify_llm_qat_installation,
    )

    prepare_llm_qat_installation(ctx)
    ctx.run()
    verify_llm_qat_installation(ctx)


def llm_qat_quickstart(ctx):
    """Run and verify the llm-qat-quickstart scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import (
        prepare_llm_qat_workspace,
        verify_llm_qat_workspace,
    )

    workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
    ctx.cwd = workspace
    ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    ctx.run()
    verify_llm_qat_workspace(workspace)


def llm_qad_quickstart(ctx):
    """Run and verify the llm-qad-quickstart scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import (
        prepare_llm_qat_workspace,
        verify_llm_qat_workspace,
    )

    workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
    ctx.cwd = workspace / "examples/llm_qat"
    ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    ctx.run()
    verify_llm_qat_workspace(workspace, variant="qad")


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


def llm_qlora_quickstart(ctx):
    """Run and verify the llm-qlora-quickstart scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace

    workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
    ctx.cwd = workspace / "examples/llm_qat"
    ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    ctx.run()
    trained = ctx.cwd / "qwen3-8b-fp4-qlora"
    exported = ctx.cwd / "qwen3-8b-fp4-qlora-hf"
    state = json.loads((trained / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    losses = [entry["train_loss"] for entry in state["log_history"] if "train_loss" in entry]
    assert losses and all(math.isfinite(value) for value in losses)
    assert (exported / "adapter_config.json").is_file()
    assert (exported / "adapter_model.safetensors").stat().st_size > 0
    assert (exported / "base_model/config.json").is_file()


def llm_qlora_serve(ctx):
    """Run and verify the llm-qlora-serve scenario."""
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat_serving import (
        prepare_llm_qat_adapter_server,
        verify_llm_qat_adapter_server,
    )

    prepare_llm_qat_adapter_server(ctx)
    ctx.run()
    verify_llm_qat_adapter_server(ctx)


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
    from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_quantized_input

    prepare_llm_qat_quantized_input(ctx)
    ctx.run()
    state = json.loads((ctx.cwd / "qwen3-8b-qat-override/trainer_state.json").read_text())
    assert state["global_step"] == 2
    assert any(entry.get("learning_rate") == 5e-05 for entry in state["log_history"])


def llm_qat_dataset(ctx):
    """Run and verify the llm-qat-dataset scenario."""
    # Dataset dependencies are optional for the CPU-only config-copy scenario.
    # Optional ML dependencies are imported only inside the scenario worker.
    from _test_utils.doc_tests.fixtures.llm_qat import prepare_llm_qat_workspace
    from datasets import load_from_disk

    workspace = prepare_llm_qat_workspace(ctx.repo, ctx.tmp)
    ctx.cwd = workspace / "examples/llm_qat"
    ctx.env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    ctx.run()

    caches = list((ctx.cwd / ".dataset_cache/tokenized").glob("*/dataset_dict.json"))
    assert len(caches) == 1
    cached = load_from_disk(str(caches[0].parent))
    assert len(cached["train"]) == 16 and len(cached["eval"]) == 4
    assert {"input_ids", "attention_mask", "labels"} <= set(cached["train"].column_names)


def llm_qat_evaluation(ctx):
    """Run and verify the llm-qat-evaluation scenario."""
    checkpoint = Path(os.environ["MODELOPT_DOC_EVAL_CHECKPOINT"]).resolve()
    assert (checkpoint / "modelopt_state.pth").is_file()
    ctx.cwd = ctx.tmp / "examples/llm_qat"
    ctx.cwd.mkdir(parents=True)
    (ctx.cwd / "qwen3-8b-qat-nvfp4").symlink_to(checkpoint, target_is_directory=True)
    shutil.copytree(ctx.repo / "examples/llm_eval", ctx.cwd.parent / "llm_eval")
    ctx.env["EVAL_OUTPUT"] = str(ctx.tmp / "eval-results")
    ctx.run()
    files = list((ctx.tmp / "eval-results").rglob("results*.json"))
    assert files and json.loads(files[0].read_text())["results"]
