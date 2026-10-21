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

"""Megatron tensor-parallel LoRA gradients and distributed checkpoint restoration."""

from functools import partial

import pytest
import torch
from _test_utils.torch.megatron.models import MegatronModel, get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import initialize_for_megatron, sharded_state_dict_test_helper
from torch.nn import functional as F

import modelopt.torch.quantization as mtq


def _test_quant_lora(checkpoint_dir, rank, size):
    initialize_for_megatron(tensor_model_parallel_size=size)
    teacher = MegatronModel(tp_size=size).cuda()
    student = MegatronModel(tp_size=size).cuda()
    student.load_state_dict(teacher.state_dict())
    inputs = student.get_dummy_input(seed=123).cuda()
    with torch.no_grad():
        targets = teacher(inputs).softmax(dim=-1)
    mtq.quantize(student, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))
    baseline = student(inputs).detach()
    weights = {name: p.detach().clone() for name, p in student.named_parameters()}
    mtq.enable_quant_lora(student, {"rank": 4})
    torch.testing.assert_close(student(inputs), baseline, rtol=0, atol=0)
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        loss = F.kl_div(student(inputs).log_softmax(dim=-1), targets, reduction="batchmean")
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in (student.fc1.lora_A, student.fc2.lora_B):
            replicas = [torch.empty_like(parameter.grad) for _ in range(size)]
            torch.distributed.all_gather(replicas, parameter.grad)
            for replica in replicas:
                torch.testing.assert_close(replica, parameter.grad)
        optimizer.step()
    assert student.fc1.lora_A.grad.count_nonzero() > 0
    for name, value in weights.items():
        torch.testing.assert_close(dict(student.named_parameters())[name], value, rtol=0, atol=0)
    restored = MegatronModel(tp_size=size).cuda()
    sharded_state_dict_test_helper(checkpoint_dir, student, restored, lambda m: m(inputs))
    assert all(p.requires_grad == ("lora_" in name) for name, p in restored.named_parameters())
    expected = restored(inputs).detach()
    mtq.merge_quant_lora(restored)
    torch.testing.assert_close(restored(inputs), expected)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_quant_lora_distributed_checkpoint(tmp_path, request, tp_size):
    workers = request.getfixturevalue(f"dist_workers_size_{tp_size}")
    workers.run(partial(_test_quant_lora, str(tmp_path)))


def _test_hybrid_iq2_lora_gradients(recompute, rank, size):
    initialize_for_megatron(tensor_model_parallel_size=1)
    model = get_mcore_hybrid_model(
        num_layers=2,
        hybrid_layer_pattern="EE",
        hidden_size=256,
        num_attention_heads=4,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=None,
        num_moe_experts=2,
        moe_router_topk=2,
        transformer_impl="transformer_engine",
        gradient_accumulation_fusion=False,
        hidden_dropout=0,
        attention_dropout=0,
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
    ).cuda()
    targets = [
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1",
        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2",
    ]
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                *[
                    {
                        "quantizer_name": name + ".weight_quantizer",
                        "cfg": {
                            "num_bits": "iq2_xxs",
                            "block_sizes": {-1: 256},
                            "backend": "ggml",
                        },
                        "enable": True,
                    }
                    for name in targets
                ],
            ],
            "algorithm": None,
        },
    )
    backbone = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    mtq.enable_quant_lora(model, {"rank": 4, "target_modules": targets})
    adapters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert len(adapters) == 16
    optimizer = torch.optim.SGD(adapters.values(), lr=0.1)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], device="cuda"),
        "position_ids": torch.arange(4, device="cuda").unsqueeze(0),
        "attention_mask": None,
    }
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs).float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in adapters.items():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if step or name.endswith("lora_B"):
                assert parameter.grad.count_nonzero(), name
        optimizer.step()
    for name, expected in backbone.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "recompute",
    [False, True],
)
def test_hybrid_iq2_lora_gradients(dist_workers_size_1, recompute):
    dist_workers_size_1.run(partial(_test_hybrid_iq2_lora_gradients, recompute))
