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


"""K3.1 reference math and complete differentiable text execution."""

import copy

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.distributed.configuration_utils import DistributedConfig

import modelopt.torch.quantization as mtq
from modelopt.torch.models.kimi_k31 import (
    KimiK31Config,
    KimiK31ForCausalLM,
    convert_kimi_k31_state_dict,
)
from modelopt.torch.models.kimi_k31.modeling_kimi_k31 import KimiK31DSAAttention, KimiK31MoE


def _config(**overrides):
    return KimiK31Config(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        routed_expert_hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        first_k_dense_replace=1,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        attn_res_block_size=2,
        linear_attn_config={
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        dsa_config={
            "dsa_layers": [2],
            "index_topk": 2,
            "index_num_heads": 2,
            "index_head_dim": 4,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "index_topk_freq": 1,
        },
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        **overrides,
    )


def _situ(gate, up, config):
    gate, up = gate.float(), up.float()
    beta, linear_beta = config.activation_situ_beta, config.activation_situ_linear_beta
    return (
        beta
        * torch.tanh(gate / beta)
        * gate.sigmoid()
        * (linear_beta * torch.tanh(up / linear_beta))
    )


def test_k31_kda_matches_bounded_delta_rule_and_padding_unpadding():
    torch.manual_seed(1)
    config = _config()
    attention = KimiK31ForCausalLM(config).model.layers[0].self_attn
    x = torch.randn(1, 4, config.hidden_size)
    qkv = []
    for name in ("q", "k", "v"):
        projected = getattr(attention, name + "_proj")(x)
        conv = getattr(attention, name + "_conv1d")
        value = F.conv1d(projected.transpose(1, 2), conv.weight, groups=8, padding=2)[..., :4]
        qkv.append(F.silu(value).transpose(1, 2).view(1, 4, 2, 4))
    query, key, value = qkv
    query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() / 2
    key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    raw = attention.f_b_proj(attention.f_a_proj(x)).view(1, 4, 2, 4)
    decay = (
        -5 * (attention.A_log.exp().view(2, 1) * (raw + attention.dt_bias.view(2, 4))).sigmoid()
    ).exp()
    beta = attention.b_proj(x).sigmoid()
    state = torch.zeros(1, 2, 4, 4)
    values = []
    for index in range(4):
        state = decay[:, index, :, :, None] * state
        correction = value[:, index] - (state * key[:, index, :, :, None]).sum(-2)
        state = state + torch.einsum(
            "bhk,bhv->bhkv", key[:, index], beta[:, index, :, None] * correction
        )
        values.append(torch.einsum("bhk,bhkv->bhv", query[:, index], state))
    expected = torch.stack(values, 1)
    expected = expected * torch.rsqrt(
        expected.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    expected = expected * attention.o_norm.weight * attention.g_proj(x).view(1, 4, 2, 4).sigmoid()
    expected = attention.o_proj(expected.flatten(-2))
    torch.testing.assert_close(attention(x), expected)
    padded = torch.cat([x[:, :1], torch.randn_like(x[:, :1]), x[:, 1:]], dim=1)
    actual = attention(padded, attention_mask=torch.tensor([[1, 0, 1, 1, 1]]))
    torch.testing.assert_close(actual[:, [0, 2, 3, 4]], expected)
    assert torch.count_nonzero(actual[:, 1]) == 0


def test_k31_checkpoint_mapping_restores_split_experts_and_padded_a_log():
    config = _config()
    model = KimiK31ForCausalLM(config)
    checkpoint = {}
    for name, value in model.state_dict().items():
        name = "language_model." + name.replace(".mlp.", ".block_sparse_moe.")
        name = name.replace(".self_attn.decay_gate.", ".self_attn.")
        if name.endswith("experts.gate_up_proj"):
            base = name.removesuffix("gate_up_proj")
            for expert in range(config.num_experts):
                gate, up = value[expert].chunk(2, dim=0)
                checkpoint[base + f"{expert}.w1.weight"] = gate
                checkpoint[base + f"{expert}.w3.weight"] = up
        elif name.endswith("experts.down_proj"):
            base = name.removesuffix("down_proj")
            for expert in range(config.num_experts):
                checkpoint[base + f"{expert}.w2.weight"] = value[expert]
        else:
            checkpoint[name] = F.pad(value, (0, 2)) if name.endswith("A_log") else value
    for malformed in (
        {
            **checkpoint,
            "model.embed_tokens.weight": checkpoint["language_model.model.embed_tokens.weight"],
        },
        {**checkpoint, "language_model.model.layers.0.self_attn.A_log": torch.ones(1)},
        {
            **checkpoint,
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.ones(
                8, 8, dtype=torch.uint8
            ),
        },
    ):
        with pytest.raises(ValueError):
            convert_kimi_k31_state_dict(malformed, config)
    converted = convert_kimi_k31_state_dict(checkpoint, config)
    restored = KimiK31ForCausalLM(KimiK31Config.from_hf_config(config.to_dict()))
    restored.load_state_dict(converted, strict=True)
    tokens = torch.tensor([[1, 3, 2]])
    torch.testing.assert_close(restored(tokens).logits, model(tokens).logits, rtol=0, atol=0)
    hidden = torch.randn(3, config.routed_expert_hidden_size)
    indices = torch.tensor([[0, 3], [1, 2], [3, 0]])
    routing = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    expected = torch.zeros_like(hidden)
    prefix = "language_model.model.layers.1.block_sparse_moe.experts."
    for expert in range(config.num_experts):
        gate = F.linear(hidden, checkpoint[prefix + f"{expert}.w1.weight"])
        up = F.linear(hidden, checkpoint[prefix + f"{expert}.w3.weight"])
        output = F.linear(_situ(gate, up, config), checkpoint[prefix + f"{expert}.w2.weight"])
        expected += output * (routing * (indices == expert)).sum(-1, keepdim=True)
    actual = restored.model.layers[1].mlp.experts(hidden, indices, routing)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_k31_bf16_reload_preserves_precision_children_and_executes_fsdp_boundaries(tmp_path):
    model = KimiK31ForCausalLM(
        _config(
            quantization_config={
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
            }
        )
    )
    assert not hasattr(model.config, "quantization_config")
    precision_parts = {
        "A_log",
        "dt_bias",
        "q_conv1d",
        "k_conv1d",
        "v_conv1d",
        "o_norm",
        "k_norm",
        "gate",
    }
    expected = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if precision_parts.intersection(name.split(".")):
                parameter.add_(0.001234567)
                expected[name] = parameter.detach().clone()
    model.save_pretrained(tmp_path)
    loaded = KimiK31ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert not hasattr(loaded.config, "quantization_config")
    assert (
        loaded.config.checkpoint_quantization_config == model.config.checkpoint_quantization_config
    )
    parameters = dict(loaded.named_parameters())
    for name, value in expected.items():
        assert parameters[name].dtype == torch.float32, name
        torch.testing.assert_close(parameters[name], value, rtol=0, atol=0, msg=name)
    assert loaded.model.embed_tokens.weight.dtype == torch.bfloat16
    precision_scopes = {}
    if hasattr(loaded, "_resolve_parallel_plans"):
        _, _, fsdp_plan = loaded._resolve_parallel_plans(DistributedConfig(tp_size=1, fsdp_size=2))
        assert "model.layers.0.self_attn" not in fsdp_plan
        precision_scopes = {
            name: loaded.get_submodule(name) for name in fsdp_plan if name != "lm_head"
        }
        assert "model.layers.0.self_attn.decay_gate" in precision_scopes
    called = set()
    handles = [
        module.register_forward_pre_hook(lambda module, args, name=name: called.add(name))
        for name, module in precision_scopes.items()
    ]
    try:
        tokens = torch.tensor([[1, 3, 5, 2]])
        output = loaded(tokens, labels=tokens)
        assert torch.isfinite(output.loss)
        output.loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert called == set(precision_scopes)
    assert loaded.model.layers[0].self_attn.q_conv1d.weight.grad is not None


def test_k31_latent_situ_moe_matches_independent_expert_sum_and_gradients():
    torch.manual_seed(14)
    config = _config()
    model = KimiK31ForCausalLM(config)
    block = model.model.layers[1].mlp
    assert isinstance(block, KimiK31MoE)
    reference = copy.deepcopy(block)
    x = torch.randn(2, 3, config.hidden_size, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    flat = y.flatten(0, 1)
    base_indices, base_weights = block.gate(flat.detach())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_indices, amp_weights = block.gate(flat.detach())
    torch.testing.assert_close(amp_indices, base_indices)
    torch.testing.assert_close(amp_weights, base_weights)
    scores = F.linear(flat.float(), reference.gate.weight.float()).sigmoid()
    indices = (scores + reference.gate.e_score_correction_bias).topk(2, dim=-1).indices
    weights = scores.gather(-1, indices)
    weights = weights / weights.sum(-1, keepdim=True)
    latent = reference.routed_expert_down_proj(flat)
    routed = torch.zeros_like(latent)
    for expert in range(config.num_experts):
        gate, up = F.linear(latent, reference.experts.gate_up_proj[expert]).chunk(2, -1)
        out = F.linear(_situ(gate, up, config), reference.experts.down_proj[expert])
        routed = routed + out * ((indices == expert) * weights).sum(-1, keepdim=True)
    routed = reference.routed_expert_up_proj(reference.routed_expert_norm(routed))
    shared = reference.shared_experts
    expected = routed + shared.down_proj(
        _situ(shared.gate_proj(flat), shared.up_proj(flat), config)
    )
    actual = block(x)
    torch.testing.assert_close(actual.flatten(0, 1), expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for (name, parameter), (_, other) in zip(
        block.named_parameters(), reference.named_parameters()
    ):
        if other.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, msg=name)


def test_k31_sparse_indexer_matches_published_nope_scoring_and_gated_mla():
    torch.manual_seed(3)
    config = _config()
    attention = KimiK31ForCausalLM(config).model.layers[1].self_attn
    assert isinstance(attention, KimiK31DSAAttention)
    x = torch.randn(1, 5, config.hidden_size)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
    q_latent = attention.q_a_layernorm(attention.q_a_proj(x))
    indexer = attention.indexer
    iq = indexer.wq_b(q_latent).view(1, 5, 2, 4)
    ik = F.layer_norm(
        indexer.wk(x).float(),
        (4,),
        indexer.k_norm.weight,
        indexer.k_norm.bias,
        config.rms_norm_eps,
    )
    scores = torch.einsum("bshd,btd->bsht", iq.float(), ik.float()).relu()
    gates = indexer.weights_proj(x) * (2**-0.5)
    scores = (scores * gates[..., None]).sum(2) * (4**-0.5)
    expected_indices = scores.masked_fill(~mask[:, 0], -torch.inf).topk(2, -1).indices
    torch.testing.assert_close(indexer(x, q_latent, mask), expected_indices)
    q = attention.q_b_proj(q_latent).view(1, 5, 2, 6).transpose(1, 2)
    kv_latent, kr = attention.kv_a_proj_with_mqa(x).split([4, 2], -1)
    kn, value = (
        attention.kv_b_proj(attention.kv_a_layernorm(kv_latent))
        .view(1, 5, 2, 8)
        .transpose(1, 2)
        .split([4, 4], -1)
    )
    key = torch.cat([kn, kr.unsqueeze(1).expand(1, 2, 5, 2)], -1)
    selected = torch.zeros_like(mask[:, 0]).scatter_(-1, expected_indices, True) & mask[:, 0]
    weights = (
        (q @ key.transpose(-1, -2) * (6**-0.5))
        .masked_fill(~selected[:, None], -torch.inf)
        .softmax(-1)
    )
    output = (weights @ value).transpose(1, 2).flatten(-2)
    expected = attention.o_proj(output * attention.g_proj(x).sigmoid())
    torch.testing.assert_close(attention(x, attention_mask=mask), expected)


@pytest.mark.parametrize("experts_implementation", ["eager", "grouped_mm"])
def test_k31_ptq_preserves_situ_fused_experts_and_full_backward(experts_implementation):
    model = KimiK31ForCausalLM(_config())
    model.set_experts_implementation(experts_implementation)
    quantized = mtq.quantize(model, mtq.INT8_WEIGHT_ONLY_CFG)
    tokens = torch.tensor([[1, 4, 8, 2]])
    loss = quantized(tokens, labels=tokens).loss
    assert torch.isfinite(loss)
    loss.backward()
    experts = quantized.model.layers[1].mlp.experts
    assert experts.down_proj.grad is not None
    assert len(experts.gate_up_proj_weight_quantizers) == 4


@pytest.mark.parametrize("experts_implementation", ["eager", "grouped_mm"])
def test_k31_full_text_causal_forward_backward_and_cache(experts_implementation):
    torch.manual_seed(8)
    model = KimiK31ForCausalLM(_config())
    model.set_experts_implementation(experts_implementation)
    tokens = torch.tensor([[1, 5, 8, 3, 7]])
    output = model(tokens, labels=tokens)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    names = dict(model.named_parameters())
    for name in [
        "model.layers.0.self_attn.f_b_proj.weight",
        "model.layers.0.self_attn.g_proj.weight",
        "model.layers.1.self_attn.kv_b_proj.weight",
        "model.layers.1.self_attn.g_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.routed_expert_down_proj.weight",
        "model.layers.1.mlp.routed_expert_norm.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.output_attn_res_proj.weight",
    ]:
        assert names[name].grad is not None, name
        assert torch.isfinite(names[name].grad).all(), name
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(tokens[:, :3]).logits, output.logits[:, :3])
        # Recurrent cache reads are available only on newer HF releases.
        if hasattr(DynamicCache, "get_recurrent_state"):
            pref = model(tokens[:, :3], use_cache=True)
            rest = model(tokens[:, 3:], use_cache=True, past_key_values=pref.past_key_values)
            torch.testing.assert_close(rest.logits, output.logits[:, 3:], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(model(tokens, logits_to_keep=1).logits, output.logits[:, -1:])
