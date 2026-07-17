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

from collections import defaultdict

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard

from modelopt.torch.puzzletron.anymodel.models.nemotron_h.nemotron_h_model_descriptor import (
    NemotronHModelDescriptor,
)
from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
)
from modelopt.torch.puzzletron.bypass_distillation.subblock_boundaries import (
    install_teacher_subblock_capture_hooks,
    replay_subblock,
    resolve_subblock_boundaries,
    selected_subblock_kinds,
)


class _ToyLayer(nn.Module):
    def __init__(self, attention_scale: float, ffn_scale: float):
        super().__init__()
        self.mixer = nn.Linear(2, 2, bias=False)
        self.ffn = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.mixer.weight.copy_(torch.eye(2) * attention_scale)
            self.ffn.weight.copy_(torch.eye(2) * ffn_scale)

    def forward(self, hidden_states):
        mixed = self.mixer(hidden_states)
        return self.ffn(hidden_states + mixed)


class _ToyDescriptor:
    @classmethod
    def local_kd_subblock_module_paths(cls, block_config, *, layer_idx):
        del layer_idx
        return {
            (subblock.kind, subblock.name): ("mixer" if subblock.kind == "attention" else "ffn")
            for subblock in block_config.subblock_configs
        }


def _block():
    return BlockConfig(
        subblock_configs=(
            AttentionConfig(name="mixer", num_query_heads=2, num_kv_heads=1),
            FFNConfig(name="feed_forward", intermediate_size=4),
        )
    )


def test_each_student_subblock_replays_the_exact_teacher_boundary_input():
    teacher = _ToyLayer(attention_scale=2.0, ffn_scale=3.0)
    student = _ToyLayer(attention_scale=0.5, ffn_scale=0.25)
    teacher_boundaries = resolve_subblock_boundaries({0: teacher}, _ToyDescriptor, [_block()])
    student_boundaries = resolve_subblock_boundaries({0: student}, _ToyDescriptor, [_block()])
    records = defaultdict(list)
    handles = install_teacher_subblock_capture_hooks(
        teacher_boundaries,
        records,
        capture_enabled=lambda: True,
    )
    hidden_states = torch.tensor([[[1.0, -2.0]]])
    teacher(hidden_states)

    observed_inputs = {}
    student_handles = []
    for key, boundary in student_boundaries.items():

        def _observe(module, args, kwargs, *, boundary_key=key):
            observed_inputs[boundary_key] = args[0].detach().clone()

        student_handles.append(
            boundary.module.register_forward_pre_hook(
                _observe,
                with_kwargs=True,
            )
        )
        output = replay_subblock(boundary, records[key][0])
        assert output.requires_grad

    assert torch.equal(
        observed_inputs[(0, "attention", "mixer")],
        hidden_states,
    )
    expected_ffn_input = hidden_states + teacher.mixer(hidden_states)
    assert torch.equal(
        observed_inputs[(0, "ffn", "feed_forward")],
        expected_ffn_input,
    )
    assert all(
        not record.target.requires_grad for entries in records.values() for record in entries
    )

    for handle in reversed(student_handles + handles):
        handle.remove()


def test_missing_descriptor_boundary_is_a_capability_error():
    class _UnsupportedDescriptor:
        pass

    with pytest.raises(NotImplementedError, match="subblock bypass"):
        resolve_subblock_boundaries({0: _ToyLayer(1.0, 1.0)}, _UnsupportedDescriptor, [_block()])


def test_no_op_subblocks_do_not_create_replay_boundaries():
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(name="attention", num_query_heads=2, num_kv_heads=1),
            FFNConfig(name="ffn", no_op=True),
        )
    )

    boundaries = resolve_subblock_boundaries({0: _ToyLayer(1.0, 1.0)}, _ToyDescriptor, [block])

    assert set(boundaries) == {(0, "attention", "attention")}


def test_nemotron_subblock_boundary_resolves_to_native_mixer():
    layer = nn.Module()
    layer.mixer = nn.Identity()
    block = BlockConfig(
        subblock_configs=(
            MambaConfig(name="mamba", num_heads=2, head_dim=8),
            FFNConfig(name="ffn", no_op=True),
        )
    )

    boundaries = resolve_subblock_boundaries(
        {0: layer}, NemotronHModelDescriptor, [block]
    )

    assert set(boundaries) == {(0, "mamba", "mamba")}
    assert boundaries[(0, "mamba", "mamba")].module is layer.mixer


def test_selected_subblock_kinds_keeps_attention_and_mamba_distinct():
    assert selected_subblock_kinds(["subblock_attention"]) == frozenset({"attention"})
    assert selected_subblock_kinds(["subblock_mamba"]) == frozenset({"mamba"})
    assert selected_subblock_kinds(["subblock_ffn"]) == frozenset({"ffn", "moe"})
    assert selected_subblock_kinds("entire_block") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke test")
def test_isolated_subblock_replay_optimizes_for_two_cuda_steps():
    device = torch.device("cuda")
    teacher = _ToyLayer(2.0, 3.0).to(device)
    student = _ToyLayer(0.5, 0.25).to(device)
    teacher_boundaries = resolve_subblock_boundaries({0: teacher}, _ToyDescriptor, [_block()])
    student_boundaries = resolve_subblock_boundaries({0: student}, _ToyDescriptor, [_block()])
    records = defaultdict(list)
    handles = install_teacher_subblock_capture_hooks(
        teacher_boundaries, records, capture_enabled=lambda: True
    )
    teacher(torch.tensor([[[1.0, -2.0]]], device=device))
    optimizer = torch.optim.SGD(student.parameters(), lr=0.001)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(
            torch.nn.functional.mse_loss(
                replay_subblock(boundary, records[key][0]), records[key][0].target
            )
            for key, boundary in student_boundaries.items()
        )
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()

    assert losses[1] < losses[0]
    for handle in reversed(handles):
        handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke test")
def test_isolated_subblock_replay_runs_through_fsdp2_owner_hooks(tmp_path):
    if dist.is_initialized():
        pytest.skip("requires ownership of the default process group")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'fsdp2_init'}",
        rank=0,
        world_size=1,
    )
    try:
        device = torch.device("cuda")
        teacher = _ToyLayer(2.0, 3.0).to(device)
        student = _ToyLayer(0.5, 0.25).to(device)
        fully_shard(student)
        teacher_boundaries = resolve_subblock_boundaries({0: teacher}, _ToyDescriptor, [_block()])
        student_boundaries = resolve_subblock_boundaries({0: student}, _ToyDescriptor, [_block()])
        records = defaultdict(list)
        handles = install_teacher_subblock_capture_hooks(
            teacher_boundaries, records, capture_enabled=lambda: True
        )
        teacher(torch.tensor([[[1.0, -2.0]]], device=device))

        loss = sum(
            torch.nn.functional.mse_loss(
                replay_subblock(boundary, records[key][0]), records[key][0].target
            )
            for key, boundary in student_boundaries.items()
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert any(parameter.grad is not None for parameter in student.parameters())
        for handle in reversed(handles):
            handle.remove()
    finally:
        dist.destroy_process_group()
