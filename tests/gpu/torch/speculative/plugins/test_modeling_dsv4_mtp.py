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

"""DSV4 sparse attention through the shared multi-round MTP training loop."""

from copy import deepcopy

import pytest
import torch
import torch.distributed.fsdp as fsdp

pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel

from modelopt.torch.speculative.config import MTPConfig
from modelopt.torch.speculative.plugins.modeling_dsv4_mtp import DeepSeekV4MTPModule
from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseConfig, FakeBaseModel
from modelopt.torch.speculative.plugins.modeling_mtp import create_mtp_model


def _make_model(mix, dtype):
    torch.manual_seed(7)
    config = DeepseekV4Config(
        num_hidden_layers=1,
        hidden_size=128,
        vocab_size=16,
        moe_intermediate_size=64,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        sliding_window=128,
        max_position_embeddings=1024,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
    )
    initializer = DeepseekV4PreTrainedModel(config)
    config._attn_implementation = "flex_attention"
    config._experts_implementation = "eager"
    mtp = DeepSeekV4MTPModule(config)
    mtp.apply(initializer._init_weights)
    base = FakeBaseModel(
        FakeBaseConfig(num_hidden_layers=1, hidden_size=128, vocab_size=16, dtype=torch.float32)
    )
    model = create_mtp_model(mtp, base, MTPConfig(mtp_ttt_steps=3, mtp_mix_hidden_states=mix))
    return model.to(device="cuda", dtype=dtype).train()


@pytest.mark.parametrize("mix", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flex_matches_eager_training(mix, dtype):
    flex = _make_model(mix, dtype)
    eager = deepcopy(flex)
    eager.eagle_module.config._attn_implementation = "eager"

    # An unaligned length exercises partial tiles; without HSM the KV cache spans rounds.
    length = 257
    mask = torch.ones(2, length, device="cuda", dtype=torch.long)
    mask[1, -3:] = 0
    inputs = {
        "input_ids": torch.arange(length, device="cuda").expand(2, -1) % 16,
        "attention_mask": mask,
        "loss_mask": mask,
        "base_model_outputs": {
            "draft_hidden_states": torch.randn(2, length, 2, 128, device="cuda", dtype=dtype),
            "base_model_hidden_states": torch.randn(2, length, 128, device="cuda", dtype=dtype),
        },
    }
    changed_suffix = deepcopy(inputs)
    changed_suffix["input_ids"][:, -2:] = (inputs["input_ids"][:, -2:] + 1) % 16
    changed_suffix["base_model_outputs"]["draft_hidden_states"][:, -2:] += 1
    losses = []
    for model in (flex, eager):
        rounds = []
        handle = model.eagle_module.register_forward_hook(
            lambda _module, _inputs, outputs: rounds.append(outputs[0].detach().clone())
        )
        torch.manual_seed(17)
        output = model(**inputs)
        expected_rounds = rounds[:]
        output.loss.backward()
        losses.append(output.loss)
        rounds.clear()
        torch.manual_seed(17)
        model(**changed_suffix)
        handle.remove()
        for depth, (actual, expected) in enumerate(zip(rounds, expected_rounds)):
            torch.testing.assert_close(actual[:, depth:-3], expected[:, depth:-3], rtol=0, atol=0)
    torch.testing.assert_close(
        losses[0], losses[1], rtol=1e-2 if dtype == torch.bfloat16 else 1e-5, atol=1e-6
    )
    for name, param in flex.eagle_module.named_parameters():
        if not param.requires_grad:
            continue
        reference = eager.eagle_module.get_parameter(name).grad
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        if dtype == torch.float32:
            torch.testing.assert_close(param.grad, reference, rtol=1e-3, atol=1e-6)
    assert flex.eagle_module.decoder.self_attn.sinks.grad.abs().sum() > 0


def _fsdp_preserves_controls(rank, size):
    model = _make_model(True, torch.float32)
    mtp = model.eagle_module
    mtp.config._attn_implementation = "eager"
    bias = torch.tensor([0.1000, 0.1001], device="cuda")
    with torch.no_grad():
        mtp.decoder.mlp.gate.weight.zero_()
        mtp.decoder.mlp.gate.e_score_correction_bias.copy_(bias)
    policy = fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    implementation = fsdp._fully_shard
    with mtp.fsdp2_context():
        fsdp.fully_shard(mtp, mp_policy=policy)
        fsdp.fully_shard(model, mp_policy=policy)
    assert fsdp._fully_shard is implementation

    controls = [mtp.decoder.attn_hc, mtp.decoder.ffn_hc, mtp.hc_head, mtp.decoder.self_attn]

    def check_precision(module, _inputs):
        assert all(p.dtype == torch.float32 for p in module.parameters(recurse=False))
        if module is mtp.decoder.mlp.gate:
            torch.testing.assert_close(module.e_score_correction_bias, bias, rtol=0, atol=0)

    def check_routing(module, _inputs, output):
        assert (output[2] == 1).all()

    for module in controls:
        module.register_forward_pre_hook(check_precision)
    mtp.decoder.mlp.gate.register_forward_pre_hook(check_precision)
    mtp.decoder.mlp.gate.register_forward_hook(check_routing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(
        input_ids=torch.arange(8, device="cuda").unsqueeze(0),
        attention_mask=torch.ones(1, 8, device="cuda", dtype=torch.long),
        loss_mask=torch.ones(1, 8, device="cuda", dtype=torch.long),
        base_model_outputs={
            "draft_hidden_states": torch.randn(1, 8, 2, 128, device="cuda", dtype=torch.bfloat16),
            "base_model_hidden_states": torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16),
        },
    ).loss
    loss.backward()
    for module in controls:
        for parameter in module.parameters(recurse=False):
            assert parameter.grad is not None and parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad.to_local()).all()
    optimizer.step()
    torch.testing.assert_close(
        mtp.decoder.mlp.gate.e_score_correction_bias.full_tensor(), bias, rtol=0, atol=0
    )


def test_fsdp_preserves_fp32_controls_and_expert_selection(dist_workers):
    dist_workers.run(_fsdp_preserves_controls)
