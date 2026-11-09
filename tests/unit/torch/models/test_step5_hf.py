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


"""Step-5 text math, sparse region selection, and differentiable causal execution."""

import copy

import pytest
import torch
import torch.nn.functional as F

import modelopt.torch.quantization as mtq
from modelopt.torch.models.step5 import Step5Config, Step5ForCausalLM
from modelopt.torch.models.step5.modeling_step5 import (
    Step5RMSNorm,
    Step5SparseIndexer,
    clamped_swiglu,
)


def _config(**overrides):
    return Step5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=8,
        layer_types=["sliding_attention", "full_attention"],
        partial_rotary_factors=[1.0, 0.5],
        rope_theta=[10000.0, 10000000.0],
        rope_scaling={
            "rope_type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": 64,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
        sliding_window=3,
        moe_layer_list=[1],
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=8,
        share_expert_dim=8,
        sparse_config={
            "proxy_dim": 8,
            "sparse_indexer_rope_dim": 4,
            "sparse_indexer_num_heads": 4,
            "num_provider_groups": 2,
            "topk": 1,
            "region_block_size": 2,
        },
        **overrides,
    )


def test_step5_fp32_norm_router_and_clamped_swiglu():
    norm = Step5RMSNorm(4)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.2, -0.2, 0.03, -0.07]))
    x = torch.tensor([[12.1, -0.31, 0.023, -7.1]], requires_grad=True)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
    expected = (expected * (1 + norm.weight)).to(torch.bfloat16)
    actual = norm(x, output_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gate = torch.tensor([-12, -0.5, 2.25, 12], dtype=torch.bfloat16, requires_grad=True)
    up = torch.tensor([12, -12, 0.7, -2.3], dtype=torch.bfloat16, requires_grad=True)
    result = clamped_swiglu(gate, up, 2.0)
    reference = (F.silu(gate.float()).clamp(max=2) * up.float().clamp(-2, 2)).to(gate.dtype)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    result.float().sum().backward()
    assert torch.isfinite(gate.grad).all()
    assert up.grad[0] == 0
    router = Step5ForCausalLM(_config()).model.layers[1].moe.gate
    routed = torch.randn(3, 16, dtype=torch.bfloat16, requires_grad=True)
    expected = F.linear(routed.float(), router.weight.float())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = router(routed)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert torch.isfinite(routed.grad).all()


@pytest.mark.parametrize("tied", [False, True])
def test_step5_csa_compression_signed_scores_and_causal_regions(tied):
    torch.manual_seed(13)
    config = _config()
    config.sparse_config.sparse_indexer_use_rope = False
    indexer = Step5SparseIndexer(config, 1)
    with torch.no_grad():
        indexer.q_norm.weight.normal_(std=0.1)
        if tied:
            indexer.w_proj.weight.zero_()
    hidden = torch.randn(1, 8, 16)
    positions = torch.arange(8).unsqueeze(0)
    valid = torch.ones_like(positions, dtype=torch.bool)
    selected = indexer(hidden, positions, valid)
    query = F.linear(hidden[:, -1], indexer.q_proj.weight).view(1, 4, 8)
    query = query * torch.rsqrt(query.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    query = (query * (indexer.q_norm.weight + 1)).to(torch.float8_e4m3fn).float()
    keys = F.linear(hidden, indexer.k_proj.weight)
    mean = keys.mean(-1, keepdim=True)
    variance = keys.square().mean(-1, keepdim=True) - mean.square()
    keys = (keys - mean) * torch.rsqrt(variance + config.rms_norm_eps)
    keys = keys * indexer.k_norm.weight + indexer.k_norm.bias
    z = F.linear(hidden, indexer.z_proj.weight)
    summaries = (
        torch.stack(
            [
                (z[:, 2 * r : 2 * r + 2].softmax(1) * keys[:, 2 * r : 2 * r + 2]).sum(1)
                for r in range(3)
            ],
            1,
        )
        .to(torch.bfloat16)
        .to(torch.float8_e4m3fn)
        .float()
    )
    weights = F.linear(hidden[:, -1], indexer.w_proj.weight).view(1, 2, 2) * 2**-0.5
    scores = torch.stack(
        [
            sum(
                (query[:, group * 2 + head, None] * summaries).sum(-1).relu()
                * weights[:, group, head, None]
                for head in range(2)
            )
            for group in range(2)
        ],
        1,
    )
    best = scores.argmax(-1)
    expected = torch.stack(
        (2 * best, 2 * best + 1, torch.full_like(best, 6), torch.full_like(best, 7)), -1
    )
    torch.testing.assert_close(selected[:, :, -1], expected)
    assert not selected.requires_grad
    assert not any(p.requires_grad for p in indexer.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_step5_text_forward_backward_cache_and_packed_boundaries(dtype, tmp_path):
    torch.manual_seed(5)
    model = Step5ForCausalLM(
        _config(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    ).to(dtype=dtype)
    assert not hasattr(model.config, "quantization_config")
    with torch.no_grad():
        model.model.layers[1].self_attn.q_norm.weight.fill_(0.031337)
    inputs = torch.tensor([[1, 3, 4, 6, 7, 9, 11, 2]])
    output = model(inputs, labels=inputs, output_hidden_states=True, use_cache=False)
    assert output.logits.shape == (1, 8, 32)
    assert output.hidden_states[1].dtype == torch.float32
    output.loss.backward()
    assert model.model.layers[1].moe.experts.gate_up_proj.grad.abs().sum() > 0
    assert model.model.layers[1].self_attn.qkv_proj.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        full = model(inputs).logits
        prefix = model(inputs[:, :4], use_cache=True)
        suffix = model(inputs[:, 4:], past_key_values=prefix.past_key_values, use_cache=True)
        torch.testing.assert_close(suffix.logits, full[:, 4:], atol=2e-3, rtol=2e-2)
        positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        packed = model(inputs, position_ids=positions).logits
        first = model(inputs[:, :4]).logits
        second = model(inputs[:, 4:]).logits
        torch.testing.assert_close(packed, torch.cat((first, second), 1), atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(model(inputs, logits_to_keep=2).logits, full[:, -2:])
        model.save_pretrained(tmp_path)
        restored = Step5ForCausalLM.from_pretrained(tmp_path, dtype=dtype)
        assert not hasattr(restored.config, "quantization_config")
        assert (
            restored.config.checkpoint_quantization_config
            == model.config.checkpoint_quantization_config
        )
        for name, value in model.state_dict().items():
            if value.dtype == torch.float32:
                loaded = restored.state_dict()[name]
                assert loaded.dtype == torch.float32
                torch.testing.assert_close(loaded, value, atol=0, rtol=0)
        torch.testing.assert_close(restored(inputs).logits, full, atol=0, rtol=0)
        grouped = copy.deepcopy(model)
        grouped.set_experts_implementation("grouped_mm")
        torch.testing.assert_close(grouped(inputs).logits, full, atol=2e-3, rtol=2e-2)
        generated = model.generate(inputs[:, :2], max_new_tokens=2, use_cache=True, do_sample=False)
        greedy = inputs[:, :2]
        for _ in range(2):
            greedy = torch.cat((greedy, model(greedy).logits[:, -1:].argmax(-1)), -1)
        torch.testing.assert_close(generated, greedy)
        left_padded = torch.cat((torch.zeros(1, 1, dtype=inputs.dtype), inputs), -1)
        mask = torch.cat((torch.zeros(1, 1, dtype=inputs.dtype), torch.ones_like(inputs)), -1)
        torch.testing.assert_close(
            model(left_padded, attention_mask=mask).logits[:, 1:], full, atol=1e-6, rtol=1e-4
        )
        padded = model(inputs, attention_mask=torch.zeros_like(inputs), labels=inputs)
        assert padded.loss == 0
        assert torch.isfinite(padded.logits).all()


def test_step5_loads_published_bf16_tensor_layout():
    model = Step5ForCausalLM(_config())
    source = {}
    for name, value in model.state_dict().items():
        value = value.clone()
        if name.endswith("self_attn.qkv_proj.weight"):
            for part, tensor in zip(("q", "k", "v"), value.split((32, 16, 16), 0)):
                source[name.replace("qkv_proj", part + "_proj")] = tensor
        elif name.endswith("moe.experts.gate_up_proj"):
            for part, tensor in zip(("gate", "up"), value.chunk(2, 1)):
                source[name.replace("experts.gate_up_proj", part + "_proj.weight")] = tensor
        elif name.endswith("moe.experts.down_proj"):
            source[name.replace("experts.down_proj", "down_proj.weight")] = value
        elif ".self_attn.indexer." in name:
            name = name.replace(".indexer.ssmax_s", ".ssmax_s")
            for part in ("q", "k", "z", "w"):
                name = name.replace(".indexer." + part + "_proj", ".sparse_indexer_" + part)
            for part in ("q", "k"):
                name = name.replace(
                    ".indexer." + part + "_norm", ".sparse_indexer_" + part + "_norm"
                )
            source[name] = value
        else:
            source[name] = value
    restored = Step5ForCausalLM(_config())
    before = {name: value.clone() for name, value in restored.state_dict().items()}
    key = "model.layers.0.self_attn.q_proj.weight"
    with pytest.raises(ValueError, match=r"BF16|floating"):
        restored.load_published_state_dict({**source, key: source[key].to(torch.uint8)})
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    restored.load_published_state_dict(source)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    torch.testing.assert_close(restored(inputs).logits, model(inputs).logits, rtol=0, atol=0)


def test_step5_cache_mutations_match_fresh_causal_outputs():
    torch.manual_seed(17)
    model = Step5ForCausalLM(_config()).eval()
    prefix = torch.tensor([[1, 2, 3, 4], [0, 6, 7, 8]])
    valid = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    with torch.no_grad():
        cache = model(prefix, attention_mask=valid, use_cache=True).past_key_values
        cache.crop(-1)
        prefix, valid = prefix[:, :-1], valid[:, :-1]
        cache.batch_repeat_interleave(2)
        prefix, valid = prefix.repeat_interleave(2, 0), valid.repeat_interleave(2, 0)
        indices = torch.tensor([3, 0, 2])
        cache.batch_select_indices(indices)
        prefix, valid = prefix[indices], valid[indices]
        indices = torch.tensor([2, 0, 1])
        cache.reorder_cache(indices)
        prefix, valid = prefix[indices], valid[indices]
        suffix = torch.tensor([[9], [10], [11]])
        valid = torch.cat((valid, torch.ones_like(suffix)), -1)
        expected = model(torch.cat((prefix, suffix), -1), attention_mask=valid).logits[:, -1:]
        actual = model(suffix, attention_mask=valid, past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        cache.crop(-cache.get_seq_length())
        fresh = torch.tensor([[3, 5], [5, 7], [7, 9]])
        actual = model(fresh, past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(actual, model(fresh).logits, atol=0, rtol=0)
        cache.reset()
        fresh = torch.tensor([[3, 5, 7]])
        actual = model(fresh, past_key_values=cache, use_cache=True).logits
        torch.testing.assert_close(actual, model(fresh).logits, atol=0, rtol=0)


def test_step5_fused_experts_ptq_keeps_full_model_differentiable():
    model = Step5ForCausalLM(_config())
    student = mtq.quantize(model, mtq.INT8_WEIGHT_ONLY_CFG)
    experts = student.model.layers[1].moe.experts
    assert hasattr(experts, "gate_up_proj_weight_quantizers")
    if hasattr(student, "activation_checkpointing_enable"):
        student.activation_checkpointing_enable(
            activation_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    inputs = torch.tensor([[1, 3, 4, 6, 7, 9, 11, 2]])
    output = student(inputs, labels=inputs)
    output.loss.backward()
    assert experts.gate_up_proj.grad.abs().sum() > 0
    assert torch.isfinite(output.logits).all()
