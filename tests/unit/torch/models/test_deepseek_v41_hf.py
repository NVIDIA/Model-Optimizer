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


"""V4.1 MoE checkpoint reconstruction and differentiable reference execution."""

import copy
import json

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from transformers.distributed.configuration_utils import DistributedConfig

# The complete V4.1 port uses native dispatch APIs introduced after HF 5.18.
expert_parallel = pytest.importorskip("transformers.distributed.expert_parallel")
parallel_utils = pytest.importorskip("transformers.distributed.parallel_utils")
EpDispatchExpertsParallel = expert_parallel.EpDispatchExpertsParallel
ALL_PARALLEL_STYLES = parallel_utils.ALL_PARALLEL_STYLES

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter
from modelopt.torch.models.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41MoE,
    DeepseekV41MoEConfig,
)
from modelopt.torch.models.deepseek_v41._hf_moe import V41FusedExperts


def test_v41_autocast_preserves_required_fp32_math():
    torch.manual_seed(53)
    model = DeepseekV41ForCausalLM._from_config(_tiny_hf_config(2), dtype=torch.bfloat16).eval()
    tokens = torch.tensor([[3, 5, 7, 11]])
    with torch.no_grad():
        expected = model(tokens).logits
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model(tokens).logits
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _reference(module, hidden, image_mask):
    x = hidden.reshape(-1, hidden.shape[-1])
    scores = F.linear(x.float(), module.gate.weight.float()) / module.config.gate_temp
    if module.config.scoring_func == "softmax":
        scores = scores.softmax(-1)
    elif module.config.scoring_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    bias = torch.where(image_mask.reshape(-1, 1), module.gate.bias_vl, module.gate.bias)
    indices = (scores + bias).topk(module.config.num_experts_per_tok, dim=-1).indices
    routing = scores.gather(-1, indices)
    if module.config.norm_topk_prob and routing.shape[-1] > 1:
        routing = routing / (routing.sum(-1, keepdim=True) + 1e-20)
    routing = routing * module.config.routed_scaling_factor
    limit = module.config.swiglu_limit

    def expert_forward(expert, weights):
        gate, up = F.linear(x, expert.w1.weight).float(), F.linear(x, expert.w3.weight).float()
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        value = F.silu(gate) * up
        return F.linear((value * weights).to(x.dtype), expert.w2.weight)

    output = expert_forward(module.shared_experts, 1).float()
    for index, expert in enumerate(module.experts):
        weights = ((indices == index) * routing).sum(-1, keepdim=True)
        output = output + expert_forward(expert, weights).float()
    return output.to(hidden.dtype).reshape_as(hidden)


@pytest.mark.parametrize("scoring_func", ["sqrtsoftplus", "softmax", "sigmoid"])
def test_v41_moe_matches_reference_outputs_and_gradients(scoring_func):
    torch.manual_seed(13)
    config = DeepseekV41MoEConfig(
        hidden_size=32,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        scoring_func=scoring_func,
        gate_temp=0.7,
        swiglu_limit=0.2,
        vision_routing_bias=True,
    )
    module = DeepseekV41MoE(config)
    with torch.no_grad():
        module.gate.bias.copy_(torch.tensor([0.1, -0.3, 0.2, 0.4]))
        module.gate.bias_vl.copy_(module.gate.bias.flip(0))
    other = copy.deepcopy(module)
    x = torch.randn(2, 3, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mask = torch.tensor([[True, False, True], [False, False, True]])
    actual, expected = module(x, image_mask=mask), _reference(other, y, mask)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-7, rtol=1e-5)
    for (name, value), (_, ref) in zip(module.named_parameters(), other.named_parameters()):
        if ref.grad is not None:
            torch.testing.assert_close(value.grad, ref.grad, atol=2e-7, rtol=1e-5, msg=name)
    assert module.gate.bias.grad is None
    torch.testing.assert_close(module(x.detach(), image_mask=mask), actual.detach())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v41_moe_reconstructs_mixed_checkpoint_weights(dtype):
    config = DeepseekV41MoEConfig(
        hidden_size=32, moe_intermediate_size=32, n_routed_experts=2, num_experts_per_tok=1
    )
    module = DeepseekV41MoE(config, dtype=dtype)
    state = {
        "layers.7.ffn." + key: value.detach().clone() for key, value in module.state_dict().items()
    }
    prefix = "layers.7.ffn."
    fp8 = torch.linspace(-3, 3, 1024).reshape(32, 32).to(torch.float8_e4m3fn)
    state[prefix + "shared_experts.w1.weight"] = fp8
    state[prefix + "shared_experts.w1.weight_scale_inv"] = torch.tensor(
        [[124]], dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    state[prefix + "experts.0.w3.weight"] = torch.full((32, 16), 0xA3, dtype=torch.uint8)
    state[prefix + "experts.0.w3.weight_scale_inv"] = torch.full(
        (32, 1), 128, dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    restored = DeepseekV41MoE.from_checkpoint(config, state, prefix=prefix, dtype=dtype)
    torch.testing.assert_close(restored.shared_experts.w1.weight, (fp8.float() * 0.125).to(dtype))
    expected_fp4 = torch.tensor([3.0, -2.0], dtype=dtype).repeat(32, 16)
    torch.testing.assert_close(restored.experts[0].w3.weight, expected_fp4)
    x = torch.linspace(-1, 1, 64, dtype=dtype).reshape(2, 32)
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(x.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    )
    qdq = ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(dtype)
    expert = restored.shared_experts
    hidden = F.silu(F.linear(qdq, expert.w1.weight).float().clamp(max=config.swiglu_limit))
    hidden = hidden * expert.w3(x).float().clamp(-config.swiglu_limit, config.swiglu_limit)
    torch.testing.assert_close(expert(x), expert.w2(hidden.to(dtype)))
    config_roundtrip = DeepseekV41MoEConfig.from_hf_config(
        json.loads(json.dumps(restored.config.to_dict()))
    )
    reloaded = DeepseekV41MoE(config_roundtrip, dtype=dtype)
    reloaded.load_state_dict(restored.state_dict())
    torch.testing.assert_close(reloaded(x), restored(x), rtol=0, atol=0)
    malformed = {**state, prefix + "experts.0.w3.weight_scale_inv": torch.ones(1, 1)}
    with pytest.raises(ValueError, match="scale"):
        DeepseekV41MoE.from_checkpoint(config, malformed, prefix=prefix)


@pytest.mark.parametrize("mapping_batch", [False, True])
def test_v41_standalone_moe_ptq_bypass_preserves_teacher(mapping_batch):
    torch.manual_seed(41)
    config = DeepseekV41MoEConfig(
        hidden_size=32, moe_intermediate_size=32, n_routed_experts=2, num_experts_per_tok=2
    )
    teacher = DeepseekV41MoE(config).eval().requires_grad_(False)
    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_WEIGHT_ONLY_CFG)
    inputs = torch.randn(1, 8, 32)
    batch = (
        {"hidden_states": inputs, "image_mask": torch.zeros(1, 8, dtype=torch.bool)}
        if mapping_batch
        else inputs
    )
    expected = teacher(inputs).detach().clone()
    result = mtd.local_distill(
        student,
        teacher,
        [batch],
        {"units": [""], "updates": 1, "global_batch_size": 1},
        adapter=TransformersLocalDistillationAdapter(),
    )
    report = result.metrics["history"][0]["units"][""]
    assert report["valid_tokens"] == 8
    assert report["gradient_norm"] > 0
    torch.testing.assert_close(teacher(inputs), expected, rtol=0, atol=0)


def _tiny_hf_config(ratio=2, **kwargs):
    values = {
        "vocab_size": 31,
        "hidden_size": 32,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "o_lora_rank": 16,
        "o_groups": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 2,
        "compress_ratios": [ratio, ratio],
        "kv_source_layer_ids": [0] if ratio else [],
        "index_source_layer_ids": [0] if ratio else [],
        "index_n_heads": 2,
        "index_head_dim": 32,
        "index_topk": 2,
        "candidate_source_layer_id": 0 if ratio else -1,
        "candidate_topk_blocks": 1,
        "candidate_block_size": 2,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "sliding_window": 3,
        "engram_layer_ids": [1],
        "engram_num_embeddings": [18],
        "engram_max_ngram_size": 2,
        "engram_vocab_size": 7,
        "engram_n_heads": 2,
        "engram_head_dim": 32,
        "engram_compressed_vocab_size": 31,
        "engram_token_map": list(range(31)),
        "num_nextn_predict_layers": 0,
        "dspark_block_size": 0,
        "use_cache": False,
    }
    values.update(kwargs)
    return DeepseekV41Config(**values)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_v41_hf_two_layer_text_is_causal_differentiable_and_cache_equivalent(ratio):
    torch.manual_seed(5)
    model = DeepseekV41ForCausalLM(_tiny_hf_config(ratio)).eval()
    ids = torch.tensor([[3, 7, 11, 5, 8, 4]])
    output = model(ids, labels=ids)
    assert output.logits.shape == (1, 6, 31)
    assert torch.isfinite(output.loss)
    changed = ids.clone()
    changed[:, 4:] = 9
    torch.testing.assert_close(model(changed).logits[:, :4], output.logits[:, :4], rtol=0, atol=0)
    output.loss.backward()
    for path in [
        "model.embed.weight",
        "model.layers.0.attn.wq_b.weight",
        "model.layers.0.hc_attn_fn",
        "model.layers.0.ffn.experts.gate_up_proj",
        "model.layers.1.engram.wkv.weight",
    ]:
        grad = model.get_parameter(path).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0, path
    prefill = model(ids[:, :3], use_cache=True)
    decoded = model(ids[:, 3:], past_key_values=prefill.past_key_values, use_cache=True)
    torch.testing.assert_close(decoded.logits, output.logits[:, 3:], rtol=0, atol=0)
    assert decoded.past_key_values.get_seq_length() == 6
    decoded.past_key_values.crop(-2)
    assert decoded.past_key_values.get_seq_length() == 4
    torch.testing.assert_close(decoded.past_key_values.input_ids, ids[:, :4])
    resumed = model(ids[:, 4:], past_key_values=decoded.past_key_values, use_cache=True)
    torch.testing.assert_close(resumed.logits, output.logits[:, 4:], rtol=0, atol=0)
    with pytest.raises(ValueError, match="negative"):
        resumed.past_key_values.crop(1)
    assert resumed.past_key_values.get_seq_length() == 6
    torch.testing.assert_close(model(ids, logits_to_keep=2).logits, output.logits[:, -2:])
    assert model.config.base_model_ep_plan["layers.*.ffn.experts"] == "ep_dispatch_experts"
    if ratio == 2:
        generated = model.generate(ids, max_new_tokens=2, do_sample=False, use_cache=True)
        greedy = torch.cat((ids, output.logits[:, -1].argmax(-1, keepdim=True)), -1)
        greedy = torch.cat((greedy, model(greedy).logits[:, -1].argmax(-1, keepdim=True)), -1)
        torch.testing.assert_close(generated, greedy)
    with pytest.raises(ValueError, match="Unsupported"):
        model(ids, pixel_values=torch.ones(1))


def test_v41_hf_fused_moe_preserves_published_router_and_predown_weighting():
    torch.manual_seed(9)
    config = _tiny_hf_config(0, swiglu_limit=0.1)
    model = DeepseekV41ForCausalLM(config)
    fused = model.model.layers[0].ffn
    reference = DeepseekV41MoE(config)
    with torch.no_grad():
        reference.gate.load_state_dict(fused.gate.state_dict())
        reference.shared_experts.load_state_dict(fused.shared_experts.state_dict())
        for i, expert in enumerate(reference.experts):
            expert.w1.weight.copy_(fused.experts.gate_up_proj[i, :32])
            expert.w3.weight.copy_(fused.experts.gate_up_proj[i, 32:])
            expert.w2.weight.copy_(fused.experts.down_proj[i])
    x = torch.randn(2, 3, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = fused(x), reference(y)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("quantized", [False, True])
def test_v41_hf_published_checkpoint_fusion_and_native_roundtrip(tmp_path, quantized):
    torch.manual_seed(17)
    model = DeepseekV41ForCausalLM(_tiny_hf_config(2)).eval()
    published = {}
    for name, tensor in model.state_dict().items():
        if name == "model.engram_hash.token_map":
            continue
        if name.endswith("experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for index in range(tensor.shape[0]):
                published[prefix + str(index) + ".w1.weight"] = tensor[index, :32].clone()
                published[prefix + str(index) + ".w3.weight"] = tensor[index, 32:].clone()
        elif name.endswith("experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for index in range(tensor.shape[0]):
                published[prefix + str(index) + ".w2.weight"] = tensor[index].clone()
        else:
            published[name.replace("lm_head.weight", "model.head.weight")] = tensor.clone()
    if quantized:
        for projection in ("w1", "w3"):
            key = f"model.layers.0.ffn.experts.0.{projection}.weight"
            published[key] = torch.full((32, 16), 0x21, dtype=torch.int8)
            published[key.removesuffix("weight") + "weight_scale_inv"] = torch.full((32, 1), 0.125)
        with torch.no_grad():
            model.model.layers[0].ffn.experts.gate_up_proj[0].copy_(
                torch.tensor([0.0625, 0.125]).repeat(64, 16)
            )
        model.model.layers[0].ffn.experts.quantized_gate_up = frozenset({0})
    restored = DeepseekV41ForCausalLM.from_checkpoint(
        model.config, published, dtype=torch.float32
    ).eval()
    ids = torch.tensor([[3, 5, 7, 11]])
    torch.testing.assert_close(restored(ids).logits, model(ids).logits, rtol=0, atol=0)
    restored.save_pretrained(tmp_path)
    loaded = DeepseekV41ForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(loaded(ids).logits, restored(ids).logits, rtol=0, atol=0)
    unsupported = dict(published)
    prefix = "model.layers.0.ffn.experts."
    for name in list(unsupported):
        if name.startswith(prefix):
            del unsupported[name]
    unsupported[prefix + "gate_up_proj"] = model.model.layers[0].ffn.experts.gate_up_proj.to(
        torch.float8_e4m3fn
    )
    unsupported[prefix + "down_proj"] = model.model.layers[0].ffn.experts.down_proj
    with pytest.raises(ValueError, match=r"floating|fused"):
        DeepseekV41ForCausalLM.from_checkpoint(model.config, unsupported, dtype=torch.float32)
    with pytest.raises(ValueError, match="auxiliary"):
        DeepseekV41ForCausalLM.from_checkpoint(
            model.config, {**published, "model.vision.weight": torch.ones(1)}
        )


@pytest.mark.parametrize("experts_impl", ["eager", "grouped_mm"])
def test_v41_hf_grouped_experts_and_full_model_bypass_preserve_teacher(experts_impl):
    torch.manual_seed(19)
    teacher = DeepseekV41ForCausalLM(_tiny_hf_config(2)).eval().requires_grad_(False)
    ids = torch.tensor([[3, 5, 7, 11]])
    expected = teacher(ids).logits.clone()
    teacher.set_experts_implementation("grouped_mm")
    torch.testing.assert_close(teacher(ids).logits, expected, rtol=1e-5, atol=1e-7)
    teacher.set_experts_implementation(experts_impl)
    student = mtq.quantize(copy.deepcopy(teacher), mtq.INT8_WEIGHT_ONLY_CFG)
    result = mtd.local_distill(
        student,
        teacher,
        [{"input_ids": ids}],
        {"units": ["model.layers.0.ffn"], "updates": 1, "global_batch_size": 1},
        adapter=TransformersLocalDistillationAdapter(),
    )
    metrics = result.metrics["history"][0]["units"]["model.layers.0.ffn"]
    assert metrics["valid_tokens"] == 4 and metrics["gradient_norm"] > 0
    torch.testing.assert_close(teacher(ids).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("with_labels", [False, True])
def test_v41_hf_forward_respects_native_fsdp_module_boundaries(with_labels):
    model = DeepseekV41ForCausalLM._from_config(_tiny_hf_config(2), dtype=torch.bfloat16).eval()
    events, head_inputs = [], []
    decoder = model.model.layers[1]
    handles = [
        decoder.register_forward_pre_hook(lambda *_: events.append("decoder.enter")),
        decoder.register_forward_hook(lambda *_: events.append("decoder.exit")),
        decoder.engram.register_forward_pre_hook(lambda *_: events.append("engram")),
    ]
    compressor = model.model.layers[0].attn.compressor
    handles.extend(
        [
            compressor.wkv.register_forward_pre_hook(lambda *_: events.append("compressor.wkv")),
            compressor.wgate.register_forward_pre_hook(
                lambda *_: events.append("compressor.wgate")
            ),
            model.model.layers[0].attn.wo_a.register_forward_pre_hook(
                lambda *_: events.append("attention.wo_a")
            ),
        ]
    )

    def record_head(module, inputs):
        events.append("lm_head")
        head_inputs.append(inputs[0].detach())

    handles.append(model.lm_head.register_forward_pre_hook(record_head))
    ids = torch.tensor([[3, 5, 7, 11]])
    output = model(ids, labels=ids if with_labels else None)
    for handle in handles:
        handle.remove()
    assert events.index("decoder.enter") < events.index("engram") < events.index("decoder.exit")
    assert "compressor.wkv" in events and "compressor.wgate" in events
    assert "attention.wo_a" in events
    assert events.count("lm_head") == 1
    assert output.logits.dtype == torch.float32
    torch.testing.assert_close(
        output.logits, F.linear(head_inputs[0].float(), model.lm_head.weight.float())
    )
    _, _, fsdp_plan = model._resolve_parallel_plans(DistributedConfig(fsdp_size=4, ep_size=2))
    assert {
        "model.embed",
        "model.layers.0",
        "model.layers.1",
        "model.norm",
        "lm_head",
    } <= fsdp_plan.keys()


@pytest.mark.parametrize("ratio", [0, 2])
def test_v41_hf_native_bf16_fsdp_ownership_groups_have_uniform_parameter_dtype(ratio, tmp_path):
    model = DeepseekV41ForCausalLM._from_config(_tiny_hf_config(ratio), dtype=torch.bfloat16)
    models = [model]
    if ratio == 2:
        model.save_pretrained(tmp_path)
        models.append(DeepseekV41ForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16))
    for model in models:
        assert model.model.layers[0].hc_attn_fn.dtype == torch.float32
        assert model.model.layers[0].attn.attn_sink.dtype == torch.float32
        assert model.model.layers[0].attn.wq_a.weight.dtype == torch.bfloat16
        assert model.model.layers[0].ffn.experts.gate_up_proj.dtype == torch.bfloat16
        if ratio == 2:
            assert model.model.layers[0].attn.compressor.wkv.weight.dtype == torch.float32
        _, ep_plan, fsdp_plan = model._resolve_parallel_plans(
            DistributedConfig(fsdp_size=4, ep_size=2)
        )
        boundaries = set(fsdp_plan) | {
            name for name, style in ep_plan.items() if style == "ep_dispatch_experts"
        }
        ownership = {}
        for name, parameter in model.named_parameters():
            parent = max(
                (path for path in boundaries if name.startswith(path + ".")), key=len, default=""
            )
            ownership.setdefault(parent, set()).add(parameter.dtype)
        assert all(len(dtypes) == 1 for dtypes in ownership.values()), ownership


def _v41_weighted_dispatch_job(rank, world_size, *, prepare_only=False):
    if prepare_only:
        return
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
    for dtype in (torch.float32, torch.bfloat16):
        for implementation in ("eager", "grouped_mm"):
            for empty_owner in (False, True):
                torch.manual_seed(37)
                config = _tiny_hf_config(0, n_routed_experts=4, swiglu_limit=0.35)
                reference = V41FusedExperts(config).to(dtype)
                with torch.no_grad():
                    reference.gate_up_proj.normal_(std=0.25)
                    reference.down_proj.normal_(std=0.25)
                local = copy.deepcopy(reference)
                local.config._experts_implementation = implementation
                hidden = torch.randn(world_size * 3, 32, dtype=dtype, requires_grad=True)
                weights = (
                    torch.linspace(0.13173, 1.21119, world_size * 6).reshape(-1, 2).requires_grad_()
                )
                indices = torch.tensor([[0, 1], [2, 3], [0, 3], [1, 2], [0, 2], [1, 3]])
                if empty_owner:
                    indices = torch.tensor([[0, 1]]).expand(world_size * 3, -1)
                expected = torch.zeros(hidden.shape, dtype=torch.float32)
                for expert in range(4):
                    gate = (
                        F.linear(hidden, reference.gate_up_proj[expert, :32])
                        .float()
                        .clamp(max=0.35)
                    )
                    up = (
                        F.linear(hidden, reference.gate_up_proj[expert, 32:])
                        .float()
                        .clamp(-0.35, 0.35)
                    )
                    routing = (weights * (indices == expert)).sum(-1, keepdim=True)
                    value = (F.silu(gate) * up * routing).to(hidden.dtype)
                    expected = expected + F.linear(value, reference.down_proj[expert]).float()
                upstream = torch.randn_like(expected)
                (expected * upstream).sum().backward()
                for name in ("gate_up_proj", "down_proj"):
                    ALL_PARALLEL_STYLES["grouped_gemm"].shard_param(local, name, mesh)
                with pytest.raises(RuntimeError, match="model-owned"):
                    local(
                        hidden.detach()[:1], torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1)
                    )
                strategy = getattr(local, "_modelopt_ep_dispatch", EpDispatchExpertsParallel())
                strategy.install_forward(local, ep_mesh=mesh)
                rows = slice(rank * 3, (rank + 1) * 3)
                inputs = hidden.detach()[rows].clone().requires_grad_()
                routing = weights.detach()[rows].clone().requires_grad_()
                actual = local(inputs, indices[rows], routing)
                assert actual.dtype == torch.float32
                torch.testing.assert_close(actual, expected.detach()[rows], rtol=1e-5, atol=2e-6)
                (actual * upstream[rows]).sum().backward()
                grad_rtol, grad_atol = (1e-5, 2e-6) if dtype == torch.float32 else (0.03, 0.004)
                # Separate BF16 gate/up projections round their input gradients before adding;
                # the fused projection sums them before one BF16 cast.
                error = None
                try:
                    torch.testing.assert_close(
                        inputs.grad, hidden.grad[rows], rtol=grad_rtol, atol=grad_atol
                    )
                    torch.testing.assert_close(
                        routing.grad, weights.grad[rows], rtol=1e-5, atol=2e-6
                    )
                    owned = slice(rank * 2, (rank + 1) * 2)
                    for name in ("gate_up_proj", "down_proj"):
                        actual_grad = getattr(local, name).grad.to_local()
                        expected_grad = getattr(reference, name).grad[owned]
                        torch.testing.assert_close(
                            actual_grad, expected_grad, rtol=grad_rtol, atol=grad_atol
                        )
                except AssertionError as exc:
                    error = str(exc)
                errors = [None] * world_size
                torch.distributed.all_gather_object(errors, error)
                assert not any(errors), errors


@pytest.fixture
def v41_dispatch_workers(cpu_dist_workers):
    cpu_dist_workers.run(_v41_weighted_dispatch_job, prepare_only=True)
    return cpu_dist_workers


def test_v41_native_ep_dispatch_preserves_predown_weights_and_fp32_output(v41_dispatch_workers):
    v41_dispatch_workers.run(_v41_weighted_dispatch_job)
