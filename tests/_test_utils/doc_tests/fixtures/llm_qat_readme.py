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
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

__all__ = [
    "llm_qad_trainer",
    "llm_qat_cli_override",
    "llm_qat_copy_config",
    "llm_qat_evaluation",
    "llm_qat_install",
    "llm_qat_llama_wrap",
    "llm_qat_python",
    "llm_qat_simple",
    "llm_qat_trainer",
    "llm_qlora_serve",
]


def llm_qat_install(ctx):
    """Run and verify the llm-qat-install scenario."""
    venv = ctx.tmp / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    ctx.cwd = ctx.tmp
    target = ctx.tmp / "examples/llm_qat"
    target.mkdir(parents=True)
    shutil.copy2(ctx.repo / "examples/llm_qat/requirements.txt", target / "requirements.txt")
    ctx.env["PATH"] = str(venv / "bin") + os.pathsep + ctx.env["PATH"]
    ctx.env.pop("PYTHONPATH", None)
    ctx.env.pop("VIRTUAL_ENV", None)
    ctx.run()
    result = subprocess.run(
        [
            str(ctx.tmp / "venv/bin/python"),
            "-c",
            "import json, modelopt, accelerate, peft; print(json.dumps(modelopt.__file__))",
        ],
        cwd=ctx.cwd,
        env=ctx.env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(json.loads(result.stdout.strip().splitlines()[-1])).is_relative_to(ctx.tmp / "venv")


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


def llm_qlora_serve(ctx):
    """Run and verify the llm-qlora-serve scenario."""
    checkpoint = Path(ctx.env["MODELOPT_DOC_QLORA_CHECKPOINT"]).resolve()
    assert (checkpoint / "adapter_config.json").is_file()
    assert (checkpoint / "base_model/config.json").is_file()
    ctx.cwd = ctx.tmp
    (ctx.cwd / "qwen3-8b-fp4-qlora-hf").symlink_to(checkpoint, target_is_directory=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        ctx.env["PORT"] = str(sock.getsockname()[1])
    ctx.run()
    url = f"http://127.0.0.1:{ctx.env['PORT']}"
    deadline = time.monotonic() + 600
    while True:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                assert response.status == 200
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError("vLLM did not become ready") from None
            time.sleep(1)
    request = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(
            {"model": "adapter", "prompt": "The capital of France is", "max_tokens": 4}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    assert result["choices"] and result["usage"]["completion_tokens"] > 0


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
