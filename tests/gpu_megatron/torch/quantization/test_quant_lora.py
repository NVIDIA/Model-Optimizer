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
from _test_utils.torch.megatron.models import MegatronModel
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
