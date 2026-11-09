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


"""Chronological CP layouts, scoped SDPA and explicit full-context fallbacks."""

import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter
from modelopt.torch.distill.plugins.transformers.context_parallel import CPSequenceLayout
from modelopt.torch.models.deepseek_v41 import DeepseekV41MoE, DeepseekV41MoEConfig


@pytest.fixture
def cp_mesh():
    size = int(os.environ.get("WORLD_SIZE", 1))
    if size not in (2, 4) or not torch.cuda.is_available():
        pytest.skip("Launch with torchrun on two or four GPU ranks.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    return init_device_mesh("cuda", (size,), mesh_dim_names=("cp",))


@pytest.mark.timeout(120)
def test_cp_layout_full_context_preserves_chronology_and_autograd(cp_mesh):
    torch.manual_seed(53)
    full = torch.randn(2, 13, 8, device="cuda", requires_grad=True)
    layout = CPSequenceLayout(cp_mesh, 13)
    local = layout.shard(full.detach(), 1).requires_grad_()
    torch.testing.assert_close(layout.gather(local, 1), full.detach(), rtol=0, atol=0)
    actual = layout.full_context(
        lambda value: value.cumsum(1),
        (local,),
        {},
        input_dims={("args", 0): 1},
        output_dims={(): 1},
    )
    expected = full.cumsum(1)
    valid = layout.shard(torch.ones(2, 13, dtype=torch.bool, device="cuda"), 1)
    torch.testing.assert_close(actual[valid], layout.shard(expected.detach(), 1)[valid])
    actual[valid].square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(local.grad, layout.shard(full.grad, 1), rtol=1e-5, atol=1e-5)


@pytest.mark.timeout(120)
def test_cp_sdpa_style_matches_serial_forward_and_backward(cp_mesh):
    torch.manual_seed(59)
    layout = CPSequenceLayout(cp_mesh, 13)
    reference = [
        torch.randn(1, 2, 13, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for _ in range(3)
    ]
    local = [layout.shard(value.detach(), 2).requires_grad_() for value in reference]
    original = F.scaled_dot_product_attention
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = original(*reference, is_causal=True)
        valid = layout.shard(torch.ones(1, 13, dtype=torch.bool, device="cuda"), 1)
        with layout.attention():
            actual = F.scaled_dot_product_attention(*local, is_causal=True)
            difference = actual - layout.shard(expected.detach(), 2)
            torch.testing.assert_close(
                difference.transpose(1, 2)[valid],
                torch.zeros_like(difference.transpose(1, 2)[valid]),
                rtol=0,
                atol=0.015625,
            )
            (actual.float().square() * valid[:, None, :, None]).sum().backward()
        expected.float().square().sum().backward()
    assert F.scaled_dot_product_attention is original
    for actual_input, reference_input in zip(local, reference, strict=True):
        torch.testing.assert_close(
            actual_input.grad, layout.shard(reference_input.grad, 2), rtol=0.04, atol=0.03125
        )


@pytest.mark.parametrize("kind", ["recurrent", "moe"])
@pytest.mark.timeout(120)
def test_cp_fallback_and_pointwise_units_preserve_valid_token_gradients(cp_mesh, kind):
    torch.manual_seed(61)
    layout = CPSequenceLayout(cp_mesh, 13)
    if kind == "recurrent":
        module = nn.GRU(8, 8, batch_first=True).cuda()
    else:
        module = DeepseekV41MoE(
            DeepseekV41MoEConfig(
                hidden_size=8, moe_intermediate_size=16, n_routed_experts=4, num_experts_per_tok=2
            )
        ).cuda()
    reference = copy.deepcopy(module)
    inputs = torch.randn(2, 13, 8, device="cuda")
    valid = torch.ones(2, 13, device="cuda", dtype=torch.bool)
    valid[:, 10:] = False
    target = torch.randn_like(inputs)
    expected = reference(inputs)
    expected = expected[0] if isinstance(expected, tuple) else expected
    if kind == "recurrent":
        batch = layout.replicated_batch({"hidden_states": inputs, "valid_mask": valid})
        actual = module(batch["hidden_states"])[0]
        mask, local_target = batch["valid_mask"], target
    else:
        actual = module(layout.shard(inputs, 1))
        mask, local_target = layout.shard(valid, 1), layout.shard(target, 1)
    ((actual - local_target).square()[mask].sum() / (valid.sum() * 8)).backward()
    ((expected - target).square()[valid].mean()).backward()
    count = mask.sum()
    dist.all_reduce(count, group=cp_mesh.get_group())
    assert count == valid.sum()
    for parameter, expected_parameter in zip(
        module.parameters(), reference.parameters(), strict=True
    ):
        if expected_parameter.grad is not None:
            assert parameter.grad is not None
            dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
            torch.testing.assert_close(
                parameter.grad, expected_parameter.grad, rtol=2e-4, atol=2e-5
            )


@pytest.mark.parametrize(
    "kind",
    ["packed_qwen35_decoder", "v41_tensor_moe", "unmasked_hc_decoder", "static_block_attention"],
)
@pytest.mark.timeout(180)
def test_cp_adapter_two_updates_match_serial_reference(cp_mesh, kind):
    torch.manual_seed(67)
    if kind == "v41_tensor_moe":
        teacher = DeepseekV41MoE(
            DeepseekV41MoEConfig(
                hidden_size=32, moe_intermediate_size=64, n_routed_experts=4, num_experts_per_tok=2
            )
        )
        batch = torch.randn(1, 13, 32, device="cuda")
        selection = {"units": [""]}
    elif kind == "packed_qwen35_decoder":
        config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            attn_implementation="eager",
            use_cache=False,
        )
        teacher = Qwen3_5ForCausalLM(config)
        documents = torch.tensor([[1] * 5 + [2] * 7 + [0] * 4], device="cuda")
        batch = {
            "input_ids": torch.randint(1, 64, (1, 16), device="cuda"),
            "_packed_seq_ids": documents,
            "valid_mask": documents > 0,
            "position_ids": torch.tensor(
                [list(range(5)) + list(range(7)) + [0] * 4], device="cuda"
            ),
        }
        selection = {"unit_types": ("decoder",)}
    elif kind == "static_block_attention":
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            attn_implementation="sdpa",
            use_cache=False,
        )
        config._experts_implementation = "eager"
        teacher = Qwen3MoeForCausalLM(config)
        batch = {
            "input_ids": torch.randint(1, 64, (1, 16), device="cuda"),
            "attention_mask": torch.ones(1, 16, device="cuda", dtype=torch.bool),
            "position_ids": torch.arange(16, device="cuda")[None],
        }
        selection = {"unit_types": ("attention",)}
    else:
        config = DeepseekV4Config(
            vocab_size=64,
            hidden_size=32,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            q_lora_rank=16,
            o_groups=2,
            o_lora_rank=8,
            partial_rotary_factor=0.5,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            layer_types=["sliding_attention"],
            mlp_layer_types=["moe"],
            index_n_heads=2,
            index_head_dim=16,
            index_topk=2,
            hc_mult=2,
            hc_sinkhorn_iters=3,
            sliding_window=8,
            max_position_embeddings=32,
            use_cache=False,
            attn_implementation="eager",
        )
        config._experts_implementation = "eager"
        teacher = DeepseekV4ForCausalLM(config)
        # Absence of masks is deliberate: HC outputs are [B,S,hc,H], but S owns tokens.
        batch = {"input_ids": torch.randint(1, 64, (1, 13), device="cuda")}
        selection = {"unit_types": ("decoder",)}
    teacher = teacher.cuda().float().eval().requires_grad_(False)
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer*", "cfg": {"num_bits": 4, "axis": 0}},
    ]
    forward_loop = None
    if kind == "static_block_attention":
        quant_cfg[1]["quantizer_name"] = "*self_attn*weight_quantizer*"
        quant_cfg.append(
            {
                "quantizer_name": "*self_attn*input_quantizer",
                "cfg": {"num_bits": 8, "block_sizes": {-1: 16}},
            }
        )

        def forward_loop(model):
            return model(**batch, use_cache=False)

    student = mtq.quantize(
        copy.deepcopy(teacher),
        {"quant_cfg": quant_cfg, "algorithm": "max"},
        forward_loop=forward_loop,
    )
    if kind == "static_block_attention":
        calibrated = [
            module
            for name, module in student.named_modules()
            if name.endswith("input_quantizer") and module.is_enabled
        ]
        assert calibrated and all(module._amax.numel() > 2 for module in calibrated)
    reference, reference_teacher = copy.deepcopy(student), copy.deepcopy(teacher)
    config = {
        **selection,
        "updates": 2,
        "global_batch_size": 1,
        "micro_batch_size": 1,
        "warmup_fraction": 0,
        "scheduler": "constant",
    }
    serial_mesh = DeviceMesh("cuda", [dist.get_rank()], mesh_dim_names=("dp",), _init_backend=False)
    reference_result = mtd.local_distill(
        reference,
        reference_teacher,
        [batch, batch],
        config,
        adapter=TransformersLocalDistillationAdapter(serial_mesh),
    )
    result = mtd.local_distill(
        student,
        teacher,
        [batch, batch],
        config,
        adapter=TransformersLocalDistillationAdapter(cp_mesh),
    )
    if kind == "static_block_attention":
        assert result.metrics["execution"]["context_parallel_modes"] == ["replicated_full_context"]
    expected_tokens = (
        12 if kind == "packed_qwen35_decoder" else 16 if kind == "static_block_attention" else 13
    )
    for actual, expected in zip(
        result.metrics["history"], reference_result.metrics["history"], strict=True
    ):
        for path in result.metrics["units"]:
            row, ref = actual["units"][path], expected["units"][path]
            assert row["valid_tokens"] == ref["valid_tokens"] == expected_tokens
            torch.testing.assert_close(row["mse"], ref["mse"], rtol=3e-4, atol=1e-8)
    actual_state, expected_state = student.state_dict(), reference.state_dict()
    for name, actual in actual_state.items():
        if isinstance(actual, torch.Tensor) and actual.is_floating_point():
            torch.testing.assert_close(
                actual, expected_state[name], rtol=0.003, atol=3e-6, msg=name
            )
