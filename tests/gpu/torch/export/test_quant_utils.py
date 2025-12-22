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

import pytest
import torch

pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import fuse_prequant_to_linear


def get_tiny_llama(attention_heads=4, key_value_heads=4):
    """Create a tiny Llama model for testing."""
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=attention_heads,
        num_key_value_heads=key_value_heads,
        max_position_embeddings=128,
        vocab_size=256,
    )
    return LlamaForCausalLM(config)


@pytest.mark.parametrize(
    "quant_config",
    [
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_AWQ_LITE_CFG,
    ],
)
@pytest.mark.parametrize(
    "attention_kv_heads_pair",
    [
        (4, 4),  # MHA
        (4, 2),  # GQA
        (4, 1),  # MQA
    ],
)
def test_pattern_fuse_prequant(quant_config, attention_kv_heads_pair):
    """Test pattern_fuse_prequant on modules from a tiny Llama model."""
    model = get_tiny_llama(attention_kv_heads_pair[0], attention_kv_heads_pair[1]).to("cuda")

    # Quantize the model
    dummy_input = torch.randint(0, 256, (1, 16), device="cuda")
    mtq.quantize(model, quant_config, lambda m: m(dummy_input))

    # Run forward pass before fusion
    model.eval()
    with torch.no_grad():
        output_before_fuse = model(dummy_input)

    traget_module_name_list = [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.1.mlp.down_proj",
    ]

    # Apply fusion
    fuse_prequant_to_linear(model, fuse_grouped_heads=True)

    # Check if pre_quant_scale and fused_with_prequant flag are removed correctly
    for target_module_name in traget_module_name_list:
        target_module = model.get_submodule(target_module_name)

        # Verify pre_quant_scale was removed
        assert not hasattr(target_module.input_quantizer, "_pre_quant_scale"), (
            f"{target_module_name}: pre_quant_scale should be removed after fusion"
        )

        # Verify fused_with_prequant flag was set
        assert (
            hasattr(target_module, "fused_with_prequant") and target_module.fused_with_prequant
        ), f"{target_module_name}: fused_with_prequant flag should be set"

    # Verify output is close to the original output
    with torch.no_grad():
        output_after_fuse = model(dummy_input)
    # There will be some small difference due to quantization errors after pre_quant_scale fusion to the weights
    assert torch.allclose(
        output_before_fuse.logits, output_after_fuse.logits, rtol=1e-1, atol=5e-1
    ), "Output should be the same before and after fusion"


@pytest.mark.parametrize(
    "quant_config",
    [
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_AWQ_LITE_CFG,
    ],
)
def test_pattern_fuse_prequant_moe(quant_config):
    """Test pattern_fuse_prequant on Qwen3 MoE sparse MLP."""
    pytest.importorskip("transformers")
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    # Create a tiny Qwen3MoE model for testing
    config = Qwen3MoeConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        vocab_size=256,
        shared_expert_intermediate_size=256,
    )
    model = Qwen3MoeForCausalLM(config).to("cuda")

    # Quantize the model
    dummy_input = torch.randint(0, 256, (1, 16), device="cuda")
    mtq.quantize(model, quant_config, lambda m: m(dummy_input))

    # Collect MoE expert modules to verify (down_proj should be fused)
    moe_down_proj_modules = []
    moe_gate_proj_modules = []
    moe_up_proj_modules = []
    for name, module in model.named_modules():
        if "mlp" in name and "experts" in name:
            if "gate_proj" in name and not any(x in name for x in ["weight", "quantizer"]):
                moe_gate_proj_modules.append((name, module))
            elif "down_proj" in name and not any(x in name for x in ["weight", "quantizer"]):
                moe_down_proj_modules.append((name, module))
            elif "up_proj" in name and not any(x in name for x in ["weight", "quantizer"]):
                moe_up_proj_modules.append((name, module))

    # Verify experts have pre_quant_scale before fusion
    for name, module in moe_gate_proj_modules:
        if hasattr(module, "input_quantizer"):
            assert hasattr(module.input_quantizer, "_pre_quant_scale"), (
                f"{name}: gate_proj should have pre_quant_scale before fusion"
            )

    for name, module in moe_up_proj_modules:
        if hasattr(module, "input_quantizer"):
            assert hasattr(module.input_quantizer, "_pre_quant_scale"), (
                f"{name}: up_proj should have pre_quant_scale before fusion"
            )

    for name, module in moe_down_proj_modules:
        if hasattr(module, "input_quantizer"):
            assert hasattr(module.input_quantizer, "_pre_quant_scale"), (
                f"{name}: down_proj should have pre_quant_scale before fusion"
            )

    # Run forward pass before fusion
    model.eval()
    with torch.no_grad():
        output_before_fuse = model(dummy_input)

    # Apply fusion (fuse_mismatch_dim only needed for GQA/MQA attention, not for MLP)
    fuse_prequant_to_linear(model)

    # Check if down_proj's pre_quant_scale was removed and fused into up_proj
    for name, module in moe_down_proj_modules:
        if hasattr(module, "input_quantizer"):
            # Verify pre_quant_scale was removed from down_proj
            assert not hasattr(module.input_quantizer, "_pre_quant_scale"), (
                f"{name}: down_proj pre_quant_scale should be removed after fusion"
            )
            # Verify fused_with_prequant flag was set
            assert hasattr(module, "fused_with_prequant") and module.fused_with_prequant, (
                f"{name}: down_proj should have fused_with_prequant flag set"
            )

    # Verify output is close to the original output
    with torch.no_grad():
        output_after_fuse = model(dummy_input)

    # There will be some difference due to quantization errors after pre_quant_scale fusion
    assert torch.allclose(
        output_before_fuse.logits, output_after_fuse.logits, rtol=1e-1, atol=5e-1
    ), "Output should be similar before and after Qwen3 MoE fusion"
