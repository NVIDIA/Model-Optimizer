# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from functools import partial

import pytest
import torch
import torch.nn.init as init
from _test_utils.import_helper import skip_if_no_megatron
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import initialize_for_megatron

import modelopt.torch.peft as mtpeft
import modelopt.torch.quantization as mtq
from modelopt.torch.peft.lora.layer import LoRAModule
from modelopt.torch.utils.plugins import megatron_prefill

skip_if_no_megatron(apex_or_te_required=True)
pytest.importorskip("transformer_engine")

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear

DEFAULT_LORA_CFG_TEST = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {
            "rank": 32,
            "scale": 1,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

DEFAULT_LORA_CFG_RANDOM_INIT_TEST = {
    "adapter_type": "lora",
    "adapter_name": "random",
    "adapter_cfg": {
        "*": {
            "rank": 32,
            "scale": 1,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

NVFP4_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*output_quantizer": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}


def _gpt_model_provider(tp_size: int, hidden_size=256, vocab_size=64):
    """Build a TE-backed Megatron GPT model."""
    gpt_model = get_mcore_gpt_model(
        tensor_model_parallel_size=tp_size,
        num_layers=2,
        ffn_hidden_size=None,
        num_attention_heads=tp_size,
        activation_func="squared_relu",
        transformer_impl="transformer_engine",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    ).cuda()
    return gpt_model.eval()


def _test_forward_with_one_lora_te(lora_config, rank, size):
    hidden_size = 320
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)

    assert any(
        isinstance(module, (TEColumnParallelLinear, TERowParallelLinear))
        for module in model.modules()
    )

    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    original_output = megatron_prefill(model, prompt_tokens)
    mtpeft.update_model(model, lora_config)
    lora_output = megatron_prefill(model, prompt_tokens)
    assert lora_output.shape == original_output.shape
    if lora_config == DEFAULT_LORA_CFG_RANDOM_INIT_TEST:
        assert not torch.allclose(lora_output, original_output, rtol=1e-5)
    else:
        assert torch.allclose(lora_output, original_output, rtol=1e-5)

    lora_module_count = 0
    lora_with_adapter_count = 0
    for _, module in model.named_modules():
        if isinstance(module, LoRAModule):
            lora_module_count += 1
            for adapter_name in module._lora_adapters:
                assert hasattr(module, f"lora_a_{adapter_name}")
                assert hasattr(module, f"lora_b_{adapter_name}")
            lora_with_adapter_count += 1

    assert lora_module_count > 0
    assert lora_with_adapter_count > 0


def _test_quantize_then_lora_te(lora_config, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)

    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    def forward_func(_):
        _ = megatron_prefill(model, prompt_tokens)

    mtq.quantize(model, NVFP4_DEFAULT_CONFIG, forward_func)
    mtpeft.update_model(model, lora_config)

    for name, module in model.named_modules():
        if isinstance(module, LoRAModule) and "output_layer" not in name:
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            for adapter_name in module._lora_adapters:
                lora_a = module._lora_adapters[adapter_name]["lora_a"]
                lora_b = module._lora_adapters[adapter_name]["lora_b"]
                assert not hasattr(lora_a, "input_quantizer")
                assert not hasattr(lora_b, "weight_quantizer")

    quantized_lora_output = megatron_prefill(model, prompt_tokens)
    mtq.disable_quantizer(model, "*")
    unquantized_lora_output = megatron_prefill(model, prompt_tokens)
    assert not torch.allclose(quantized_lora_output, unquantized_lora_output)


def _test_lora_then_quantize_te(lora_config, rank, size):
    hidden_size = 512
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _gpt_model_provider(tp_size=size, hidden_size=hidden_size)

    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()
    mtpeft.update_model(model, lora_config)
    lora_output = megatron_prefill(model, prompt_tokens)

    def forward_func(_):
        _ = megatron_prefill(model, prompt_tokens)

    mtq.quantize(model, NVFP4_DEFAULT_CONFIG, forward_func)
    quantized_output = megatron_prefill(model, prompt_tokens)

    for name, module in model.named_modules():
        if isinstance(module, LoRAModule) and "output_layer" not in name:
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            for adapter_name in module._lora_adapters:
                lora_a = module._lora_adapters[adapter_name]["lora_a"]
                lora_b = module._lora_adapters[adapter_name]["lora_b"]
                assert hasattr(lora_a, "input_quantizer")
                assert hasattr(lora_b, "weight_quantizer")

    assert not torch.allclose(lora_output, quantized_output)


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_TEST,
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_forward_with_one_lora_te(lora_config):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_forward_with_one_lora_te, lora_config),
        backend="nccl",
    )


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_quantize_then_lora_te(lora_config):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_quantize_then_lora_te, lora_config),
        backend="nccl",
    )


@pytest.mark.parametrize(
    "lora_config",
    [
        DEFAULT_LORA_CFG_RANDOM_INIT_TEST,
    ],
)
def test_lora_then_quantize_te(lora_config):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_lora_then_quantize_te, lora_config),
        backend="nccl",
    )
