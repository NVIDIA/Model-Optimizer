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

"""Smoke tests for the TE-grouped MoE LoRA plugin (OMNIML-4944).

Exercises modelopt/torch/peft/lora/plugins/megatron_moe.py end-to-end on a
small MoE GPT model built with ``moe_grouped_gemm=True`` and the TE
spec -- i.e. the production path used by Nemotron-3 Hybrid. The tests
verify registration, the zero-init forward identity, random-init
divergence from base, ``lora_dtype`` pinning, and gradient flow.
"""

from functools import partial

import pytest
import torch
import torch.nn.init as init
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import initialize_for_megatron

import modelopt.torch.peft as mtpeft
from modelopt.torch.peft.lora.layer import LoRAModule
from modelopt.torch.peft.lora.plugins.megatron_moe import HAVE_TE_GROUPED
from modelopt.torch.utils.plugins import megatron_prefill

pytestmark = pytest.mark.skipif(
    not HAVE_TE_GROUPED,
    reason="Requires Transformer Engine >= 1.9.0.dev0 with TEGroupedLinear support",
)

if HAVE_TE_GROUPED:
    from modelopt.torch.peft.lora.plugins.megatron_moe import (
        _LoRATEGroupedColumnParallelLinear,
        _LoRATEGroupedRowParallelLinear,
    )

NUM_EXPERTS = 4

DEFAULT_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "default",
    "adapter_cfg": {
        "*": {"rank": 8, "scale": 1.0, "enable": True},
        "*output_layer*": {"enable": False},
    },
}

RANDOM_INIT_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "random",
    "adapter_cfg": {
        "*": {
            "rank": 8,
            "scale": 1.0,
            "lora_a_init": init.kaiming_uniform_,
            "lora_b_init": init.kaiming_uniform_,
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}

BF16_SIDECAR_LORA_CFG = {
    "adapter_type": "lora",
    "adapter_name": "bf16_sidecar",
    "adapter_cfg": {
        "*": {
            "rank": 8,
            "scale": 1.0,
            "lora_dtype": "bf16",
            "enable": True,
        },
        "*output_layer*": {"enable": False},
    },
}


def _moe_model_provider(tp_size: int, hidden_size: int = 256):
    """Build a tiny TE-MoE GPT with grouped GEMM (production path)."""
    model = get_mcore_gpt_model(
        tensor_model_parallel_size=tp_size,
        num_layers=2,
        ffn_hidden_size=None,
        num_attention_heads=4,
        activation_func="swiglu",
        transformer_impl="transformer_engine",
        hidden_size=hidden_size,
        vocab_size=64,
        moe_grouped_gemm=True,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=hidden_size * 2,
    ).cuda()
    return model.eval()


def _grouped_lora_modules(model):
    """Yield (name, module) for every TE-grouped LoRA wrapper in the model."""
    for name, module in model.named_modules():
        if isinstance(
            module, (_LoRATEGroupedColumnParallelLinear, _LoRATEGroupedRowParallelLinear)
        ):
            yield name, module


def _test_registration_and_zero_init_identity(rank, size):
    """Default (Kaiming on A / zeros on B) init must not perturb the base output."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    base_output = megatron_prefill(model, prompt_tokens)
    mtpeft.update_model(model, DEFAULT_LORA_CFG)
    lora_output = megatron_prefill(model, prompt_tokens)

    # The plugin must have replaced at least one expert linear per expert layer.
    grouped_lora_modules = list(_grouped_lora_modules(model))
    assert len(grouped_lora_modules) > 0, "no TE-grouped LoRA modules were registered"

    for _, module in grouped_lora_modules:
        # Per-expert stacked factor shapes
        assert "default" in module._lora_adapters
        adapter = module._lora_adapters["default"]
        a_w = adapter["lora_a"].weight
        b_w = adapter["lora_b"].weight
        assert a_w.shape == (module.num_gemms, 8, module.in_features)
        assert b_w.shape == (module.num_gemms, module.out_features, 8)

    # Zero-init on B → LoRA contribution is exactly zero → outputs match base.
    assert lora_output.shape == base_output.shape
    assert torch.allclose(lora_output, base_output, rtol=1e-5, atol=1e-5)

    # Disabling + re-enabling adapters round-trips correctly.
    mtpeft.disable_adapters(model)
    assert torch.allclose(megatron_prefill(model, prompt_tokens), base_output, rtol=1e-5, atol=1e-5)
    mtpeft.enable_adapters(model)
    assert torch.allclose(megatron_prefill(model, prompt_tokens), lora_output, rtol=1e-5, atol=1e-5)


def test_registration_and_zero_init_identity(dist_workers):
    dist_workers.run(_test_registration_and_zero_init_identity)


def _test_random_init_perturbs_output(rank, size):
    """Random init on both A and B must change the model output."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    base_output = megatron_prefill(model, prompt_tokens)
    mtpeft.update_model(model, RANDOM_INIT_LORA_CFG)
    lora_output = megatron_prefill(model, prompt_tokens)

    assert lora_output.shape == base_output.shape
    assert not torch.allclose(lora_output, base_output, rtol=1e-5)


def test_random_init_perturbs_output(dist_workers):
    dist_workers.run(_test_random_init_perturbs_output)


def _test_lora_dtype_pin(rank, size):
    """lora_dtype='bf16' must pin LoRA factor tensors to bfloat16."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    mtpeft.update_model(model, BF16_SIDECAR_LORA_CFG)

    found = False
    for _, module in _grouped_lora_modules(model):
        adapter = module._lora_adapters["bf16_sidecar"]
        assert adapter["lora_a"].weight.dtype == torch.bfloat16
        assert adapter["lora_b"].weight.dtype == torch.bfloat16
        found = True
    assert found, "no TE-grouped LoRA modules with bf16_sidecar adapter"


def test_lora_dtype_pin(dist_workers):
    dist_workers.run(_test_lora_dtype_pin)


def _test_gradient_flow(rank, size):
    """Random-init LoRA factors must receive gradients; base weights must not (default freeze)."""
    initialize_for_megatron(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
    model = _moe_model_provider(tp_size=size)
    prompt_tokens = torch.randint(0, model.vocab_size, (2, model.max_sequence_length)).cuda()

    mtpeft.update_model(model, RANDOM_INIT_LORA_CFG)
    # Intentionally keep the model in the eval mode set by _moe_model_provider.
    # Megatron's MoELayer.forward asserts at moe_layer.py:616 when
    # self.training AND attn_tp_group.size() > 1 AND not sequence_parallel --
    # the test factory hardcodes sequence_parallel=False, and dist_workers
    # runs at tp_size=4 here, so model.train() would raise before our LoRA
    # forward dispatch is even reached. PyTorch autograd is mode-independent,
    # so loss.backward() in eval mode still produces gradients on the LoRA
    # factors and base weights as required by this test.

    batch_size, seq_len = prompt_tokens.shape
    attention_mask = (
        torch.triu(torch.ones((batch_size, seq_len, seq_len), device=prompt_tokens.device), diagonal=1)
        .bool()
        .view(batch_size, 1, seq_len, seq_len)
    )
    output = model(prompt_tokens, position_ids=None, attention_mask=attention_mask)
    output.sum().backward()

    grouped_lora_param_names = []
    for mod_name, _ in _grouped_lora_modules(model):
        grouped_lora_param_names.append(mod_name)
    assert grouped_lora_param_names, "no TE-grouped LoRA modules present"

    saw_grouped_lora_grad = False
    for name, param in model.named_parameters():
        if "lora" in name and any(g in name for g in grouped_lora_param_names):
            assert param.grad is not None, f"lora param {name} has no grad"
            assert torch.any(param.grad != 0), f"lora param {name} grad is all zero"
            saw_grouped_lora_grad = True
        elif "lora" not in name:
            # default freeze_base_model=True → base params should not have grads
            assert param.grad is None, f"base param {name} unexpectedly got a grad"

    assert saw_grouped_lora_grad, "no per-expert stacked LoRA param received a gradient"


def test_gradient_flow(dist_workers):
    dist_workers.run(_test_gradient_flow)
