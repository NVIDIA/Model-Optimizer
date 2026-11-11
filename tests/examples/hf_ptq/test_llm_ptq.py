# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import dataclasses
import json
import os

import pytest
import torch
import transformers
from _test_utils.examples.hf_ptq_utils import PTQCommand
from _test_utils.examples.models import MIXTRAL_PATH, T5_PATH, TINY_LLAMA_PATH
from _test_utils.examples.run_command import run_example_command
from _test_utils.torch.distributed.utils import get_free_port
from _test_utils.torch.misc import minimum_gpu
from _test_utils.torch.transformers_models import create_tiny_qwen3_moe_dir
from packaging.version import Version
from safetensors.torch import load_file


def _run_sharing_the_gpu(command: PTQCommand, model_path: str):
    # Sharing the GPU, so set a batch size: hf_ptq sizes one to fill the free GPU memory.
    if command.calib_batch_size is None:
        command = dataclasses.replace(command, calib_batch_size=command.calib)
    command.run(model_path)


# Encoder-decoder representative; BART shares the same export path.
@pytest.mark.parametrize("command", [PTQCommand(quant="fp8", min_sm=89)], ids=PTQCommand.param_str)
@pytest.mark.usefixtures("shared_gpu")
def test_ptq_t5(command):
    _run_sharing_the_gpu(command, T5_PATH)


@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="fp8", min_sm=90),
    ],
    ids=PTQCommand.param_str,
)
@pytest.mark.usefixtures("shared_gpu")
def test_ptq_mixtral(command):
    _run_sharing_the_gpu(command, MIXTRAL_PATH)


# Per-format export on tiny models (int8_weight_only, nvfp4_awq_lite, ...) is covered by
# tests/gpu/torch/export/test_unified_hf_export_and_check_safetensors.py.
@pytest.mark.parametrize(
    "command",
    [
        PTQCommand(quant="int8_smoothquant", kv_cache_quant="none"),
        PTQCommand(quant="int4_awq", kv_cache_quant="none"),
        PTQCommand(quant="w4a8_awq_beta", kv_cache_quant="none"),
        # GGML IQ weight-only, recipe-driven: the five formats between 1.56 and 2.56 bits per
        # weight, on the MLP layers with layerwise GPTQ. These encoders require every weight's
        # input dimension to be a multiple of 256; TinyLlama's 2048 and 5632 both are.
        PTQCommand(recipe="general/ptq/iq1_s", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq1_m", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq2_xxs", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq2_xs", kv_cache_quant="none"),
        PTQCommand(recipe="general/ptq/iq2_s", kv_cache_quant="none"),
        # Q8_0 has 32-value blocks and needs no calibration forward pass.
        PTQCommand(recipe="general/ptq/q8_0", kv_cache_quant="none"),
        # fp8 and nvfp4 checkpoints are also deployed with TRT-LLM
        PTQCommand(quant="fp8", min_sm=89),
        PTQCommand(quant="fp8", kv_cache_quant="none", min_sm=89),
        PTQCommand(quant="nvfp4"),
        PTQCommand(quant="mxfp8", min_sm=100),
        # Calibrated KV cache
        PTQCommand(quant="nvfp4_awq_lite", kv_cache_quant="nvfp4"),
        # AutoQuantize recipe; KV via --kv_cache_quant fallback
        PTQCommand(
            recipe="general/auto_quantize/nvfp4_fp8_at_5p4bits",
            kv_cache_quant="nvfp4",
            calib_batch_size=4,
        ),
        # multi_gpu
        PTQCommand(quant="fp8", min_gpu=2, min_sm=89),
        PTQCommand(quant="nvfp4", min_gpu=2, min_sm=100),
    ],
    ids=PTQCommand.param_str,
)
@pytest.mark.usefixtures("shared_gpu")
def test_ptq_llama(command):
    _run_sharing_the_gpu(command, TINY_LLAMA_PATH)


@minimum_gpu(2)
@pytest.mark.usefixtures("exclusive_gpu")
@pytest.mark.parametrize(
    ("launch_args", "launch_env", "min_transformers"),
    [
        # qwen3_moe's base_model_ep_plan ships in transformers 5.7.
        ([], {"MODELOPT_EP_DEGREE": "2"}, "5.7"),
        (["--use_fsdp2"], {}, None),
    ],
    ids=["expert_parallel", "fsdp2"],
)
def test_ptq_qwen3_moe_sharded(tmp_path, launch_args, launch_env, min_transformers):
    """A torchrun launch shards a tiny Qwen3-MoE over two ranks, previews, calibrates and exports.

    The model's experts are sharded over the ranks, or the whole model is with FSDP2. Each rank holds
    and writes only its own share, so every expert of every layer must come out FP8-quantized: a rank
    whose share is lost, or written under the wrong expert index, fails it. The previews generate
    before and after calibration: a sharded preview broke in three ways (ranks finishing at different
    steps hung in a collective, Transformers refused an FSDP2-wrapped class's experts switch, and the
    switch decoded unquantized experts), and only a real launch reaches it.
    """
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("Requires sm89 or higher")
    if min_transformers and Version(transformers.__version__) < Version(min_transformers):
        pytest.skip(f"Requires transformers {min_transformers}")
    num_layers, num_experts = 2, 4
    model_dir = create_tiny_qwen3_moe_dir(
        tmp_path,
        with_tokenizer=True,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_experts=num_experts,
        num_experts_per_tok=2,
        num_hidden_layers=num_layers,
        max_position_embeddings=128,
    )
    # A local calibration set: no dataset download.
    calib = tmp_path / "calib.jsonl"
    calib.write_text(
        "".join(
            json.dumps({"text": f"Sample {i}: the quick brown fox jumps over the lazy dog."}) + "\n"
            for i in range(8)
        )
    )
    export_dir = tmp_path / "export"
    run_example_command(
        [
            "torchrun",
            "--nproc_per_node=2",
            f"--master_port={get_free_port()}",
            "hf_ptq.py",
            "--pyt_ckpt_path",
            str(model_dir),
            "--recipe",
            "general/ptq/fp8_default-kv_fp8_cast",
            "--dataset",
            str(calib),
            "--calib_size",
            "8",
            "--batch_size",
            "2",
            "--calib_seq",
            "64",
            "--no-verbose",
            "--export_path",
            str(export_dir),
            *launch_args,
        ],
        "hf_ptq",
        env={**os.environ, **launch_env},
    )

    config = json.loads((export_dir / "config.json").read_text())
    assert config["quantization_config"]["quant_algo"] == "FP8"
    state_dict = {}
    for shard in export_dir.glob("*.safetensors"):
        state_dict.update(load_file(shard))
    for layer in range(num_layers):
        for expert in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                assert state_dict[f"{name}.weight"].dtype == torch.float8_e4m3fn, name
                assert f"{name}.weight_scale" in state_dict, name
