# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.models.mimo_v2 import MiMoV2Config, MiMoV2ForCausalLM


@pytest.mark.parametrize("swa_kv_heads", [2, 4])
def test_mimo_release_geometry_and_native_forward(swa_kv_heads, tmp_path):
    config = MiMoV2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        swa_num_key_value_heads=swa_kv_heads,
        head_dim=24,
        v_head_dim=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
        layernorm_epsilon=1e-6,
        attention_value_scale=0.707,
        quantization_config={
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "weight_block_size": [128, 128],
        },
        use_cache=False,
    )
    assert not hasattr(config, "quantization_config")
    assert config.checkpoint_quantization_config["store_dtype"] == "mxfp4"
    with pytest.raises(ValueError, match="rotary dimension"):
        MiMoV2Config(**{**config.to_dict(), "head_dim": 16})
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = MiMoV2ForCausalLM(config)
    model.model.layers[1].mlp.gate.e_score_correction_bias.copy_(
        torch.tensor([0.001234567, -0.009876543, 0.003456789, -0.004567891])
    )
    assert model.model.layers[0].self_attn.k_proj.out_features == 2 * 24
    assert model.model.layers[1].self_attn.k_proj.out_features == swa_kv_heads * 24
    assert config.layer_types == ["full_attention", "sliding_attention"]
    assert config.mlp_layer_types == ["dense", "sparse"]
    ep_plan = getattr(model, "ep_plan", model._ep_plan)
    assert all(style != "ep_router" for style in ep_plan.values())
    # Release text headers use fused QKV, attention_sink_bias and per-expert projections.
    source = {}
    for name, value in model.state_dict().items():
        if any(name.endswith("self_attn." + part + "_proj.weight") for part in ("q", "k", "v")):
            prefix = name.rsplit(".", 2)[0] + "."
            if name.endswith("q_proj.weight"):
                source[prefix + "qkv_proj.weight"] = torch.cat(
                    [model.state_dict()[prefix + part + "_proj.weight"] for part in ("q", "k", "v")]
                )
        elif name.endswith(".sinks"):
            source[name.removesuffix("sinks") + "attention_sink_bias"] = value.clone()
        elif name.endswith("experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for expert, combined in enumerate(value):
                for part, weight in zip(("gate", "up"), combined.chunk(2, 0), strict=True):
                    source[prefix + f"{expert}.{part}_proj.weight"] = weight.clone()
        elif name.endswith("experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for expert, weight in enumerate(value):
                source[prefix + f"{expert}.down_proj.weight"] = weight.clone()
        else:
            source[name] = value.clone()
    restored = MiMoV2ForCausalLM(config)
    before = {name: value.clone() for name, value in restored.state_dict().items()}
    key = "model.layers.0.self_attn.qkv_proj.weight"
    with pytest.raises(ValueError, match=r"floating|reconstruct"):
        restored.load_published_state_dict({**source, key: source[key].to(torch.uint8)})
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    restored.load_published_state_dict(source)
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(restored(tokens).logits, model(tokens).logits, rtol=0, atol=0)
    attention = restored.model.layers[1].self_attn
    hidden = torch.randn(1, 3, 32)
    prefix = "model.layers.1.self_attn."
    query, key, value = F.linear(hidden, source[prefix + "qkv_proj.weight"]).split(
        (96, swa_kv_heads * 24, swa_kv_heads * 16), -1
    )
    query = query.view(1, 3, 4, 24).transpose(1, 2)
    key = key.view(1, 3, swa_kv_heads, 24).transpose(1, 2).repeat_interleave(4 // swa_kv_heads, 1)
    value = (
        value.view(1, 3, swa_kv_heads, 16).transpose(1, 2).repeat_interleave(4 // swa_kv_heads, 1)
    )
    angles = torch.arange(3)[:, None] * (10000.0 ** (-torch.arange(0, 8, 2) / 8))
    cos, sin = angles.repeat(1, 2).cos()[None], angles.repeat(1, 2).sin()[None]
    rotated = []
    for projection in (query, key):
        first, second = projection[..., :8].chunk(2, -1)
        half_rotated = torch.cat((-second, first), -1)
        rotated.append(
            torch.cat(
                (
                    projection[..., :8] * cos[:, None] + half_rotated * sin[:, None],
                    projection[..., 8:],
                ),
                -1,
            )
        )
    query, key = rotated
    value = value * 0.707
    mask = torch.zeros(1, 1, 3, 3).masked_fill(
        torch.ones(3, 3, dtype=torch.bool).triu(1), -torch.inf
    )
    scores = query @ key.transpose(-1, -2) / (24**0.5) + mask
    sink = source[prefix + "attention_sink_bias"].view(1, 4, 1, 1).expand(1, 4, 3, 1)
    probabilities = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
    expected = F.linear(
        (probabilities @ value).transpose(1, 2).flatten(2), source[prefix + "o_proj.weight"]
    )
    actual = attention(hidden, (cos, sin), attention_mask=mask)[0]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    result = model(torch.tensor([[1, 2, 3, 4]]), logits_to_keep=1)
    assert result.logits.shape == (1, 1, 32)
    result.logits.square().mean().backward()
    gradient = model.model.layers[1].mlp.experts.gate_up_proj.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    native = MiMoV2ForCausalLM._from_config(config, dtype=torch.bfloat16)
    bf16_source = {
        name: value if name.endswith("e_score_correction_bias") else value.bfloat16()
        for name, value in source.items()
    }
    native.load_published_state_dict(bf16_source)
    native.save_pretrained(tmp_path)
    reloaded = MiMoV2ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert not hasattr(reloaded.config, "quantization_config")
    assert reloaded.config.checkpoint_quantization_config == config.checkpoint_quantization_config
    bias = reloaded.model.layers[1].mlp.gate.e_score_correction_bias
    assert bias.dtype == torch.float32
    torch.testing.assert_close(
        bias, source["model.layers.1.mlp.gate.e_score_correction_bias"], rtol=0, atol=0
    )
    torch.testing.assert_close(reloaded(tokens).logits, native(tokens).logits, rtol=0, atol=0)

    experts = copy.deepcopy(model.model.layers[1].mlp.experts).to(torch.bfloat16)
    torch.manual_seed(13)
    hidden = torch.randn(8, 32).bfloat16()
    indices = torch.arange(4).expand(8, -1)
    routing = torch.rand(8, 4).softmax(-1)
    outputs = []
    for expert in range(4):
        gate, up = F.linear(hidden, experts.gate_up_proj[expert]).chunk(2, -1)
        outputs.append(F.linear(F.silu(gate) * up, experts.down_proj[expert]))
    expected = (torch.stack(outputs, 1).float() * routing[..., None]).sum(1).bfloat16()
    torch.testing.assert_close(experts(hidden, indices, routing), expected, rtol=0, atol=0)
