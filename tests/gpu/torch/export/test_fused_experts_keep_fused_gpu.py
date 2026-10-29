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

"""The keep-fused fold of sharded fused experts on every visible GPU, over NCCL."""

import copy
import re
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import modelopt.torch.quantization as mtq
from modelopt.torch.export.moe_utils import (
    _dtensor_dim0_offset,
    _export_fused_experts,
    _export_fused_experts_keep_fused,
    split_fused_experts_state_dict,
)

HIDDEN_DIM = 32
INTERMEDIATE_DIM = 16
PREFIX = "model.layers.0.mlp.experts"


class _FusedExperts(nn.Module):
    """Fused gate+up experts in the transformers 5.x layout (Qwen3-MoE / Mixtral)."""

    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * INTERMEDIATE_DIM, HIDDEN_DIM) * 0.02
        )
        self.down_proj = nn.Parameter(torch.randn(num_experts, HIDDEN_DIM, INTERMEDIATE_DIM) * 0.02)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0]:
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(
                2, dim=-1
            )
            current = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            out.index_add_(0, token_idx, current * top_k_weights[token_idx, top_k_pos, None])
        return out


def _quantized_experts(quant_cfg, num_experts):
    """Fused experts calibrated on one token each, moved to this rank's GPU to be folded.

    Every rank builds and calibrates the same seeded module, so all hold identical quantizers.
    """
    torch.manual_seed(0)
    model = nn.Module()
    model.experts = _FusedExperts(num_experts)
    hidden = torch.randn(num_experts, HIDDEN_DIM)
    tokens = (torch.arange(num_experts).unsqueeze(1), torch.ones(num_experts, 1))
    mtq.quantize(model, quant_cfg, lambda m: m.experts(hidden, *tokens))
    return model.experts.cuda()


def _state(experts):
    return {f"{PREFIX}.{key}": value for key, value in experts.state_dict().items()}


def _raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()


def _whole(value):
    """A sharded value gathered whole, one-byte values moved as bytes."""
    if not isinstance(value, DTensor):
        return value
    if value.element_size() != 1:
        return value.full_tensor()
    raw = DTensor.from_local(
        value.to_local().view(torch.uint8),
        value.device_mesh,
        value.placements,
        run_check=False,
        shape=value.shape,
        stride=value.stride(),
    )
    return raw.full_tensor().view(value.dtype)


def _sharded(experts, mesh):
    experts = copy.deepcopy(experts)
    for proj in ("gate_up_proj", "down_proj"):
        weight = distribute_tensor(getattr(experts, proj).data, mesh, [Shard(0)])
        setattr(experts, proj, nn.Parameter(weight, requires_grad=False))
    return experts


def _fold_job(rank, world_size, backend):
    # Every collective below runs on this group, which gives up after a minute.
    group = dist.new_group(backend=backend, timeout=timedelta(seconds=60))
    try:
        # A mesh compares equal to any other over the same ranks and names, whatever its group, so
        # DTensor's caches could hand a later job a mesh over this group once it is destroyed.
        mesh = DeviceMesh.from_group(group, "cuda", mesh_dim_names=(f"ep{group.group_name}",))
        # Every rank keeps every expert's quantizers, as under FSDP2. Two experts a rank, then one
        # expert fewer: the last rank holds one, and the folds must keep the global expert count.
        cases = [(cfg, 2 * world_size) for cfg in (mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG)]
        cases.append((mtq.NVFP4_DEFAULT_CFG, 2 * world_size - 1))
        for quant_cfg, num_experts in cases:
            experts = _quantized_experts(quant_cfg, num_experts)
            reference = copy.deepcopy(experts)
            _export_fused_experts(reference, torch.bfloat16)
            plain = copy.deepcopy(experts)
            _export_fused_experts_keep_fused(plain, torch.bfloat16)
            plain = _state(plain)
            sharded = _sharded(experts, mesh)

            _export_fused_experts_keep_fused(sharded, torch.bfloat16)

            fused = _state(sharded)
            assert set(fused) == set(plain)
            offset = _dtensor_dim0_offset(fused[f"{PREFIX}.gate_up_proj"])
            for key, value in fused.items():
                assert value.device.type == "cuda", key
                if key.endswith("_input_scale"):  # one scalar shared by every expert
                    assert not isinstance(value, DTensor), key
                    assert torch.equal(_raw_bytes(value), _raw_bytes(plain[key])), key
                    continue
                assert isinstance(value, DTensor) and value.placements == (Shard(0),), key
                assert value.shape == plain[key].shape, key
                rows = plain[key][offset : offset + value.to_local().shape[0]]
                assert torch.equal(_raw_bytes(value.to_local()), _raw_bytes(rows)), key
            # Gathered whole, the ranks' folds split into exactly what the per-expert export writes.
            split = split_fused_experts_state_dict({k: _whole(v) for k, v in fused.items()})
            want = _state(reference)
            assert set(split) == set(want)
            for key, value in want.items():
                assert split[key].dtype == value.dtype and split[key].shape == value.shape, key
                assert torch.equal(_raw_bytes(split[key]), _raw_bytes(value)), key

        # A layout that leaves a rank no expert is refused on every rank.
        lone = _sharded(_quantized_experts(mtq.FP8_DEFAULT_CFG, world_size - 1), mesh)
        with pytest.raises(NotImplementedError, match="leaves a rank none"):
            _export_fused_experts_keep_fused(lone, torch.bfloat16)

        # Under expert parallelism a rank keeps only its own experts' quantizers, so the ranks gather
        # each projection's quantizer states over the group before they fold it.
        experts = _quantized_experts(mtq.NVFP4_DEFAULT_CFG, 2 * world_size)
        plain = copy.deepcopy(experts)
        _export_fused_experts_keep_fused(plain, torch.bfloat16)
        plain = _state(plain)
        mine = slice(2 * rank, 2 * rank + 2)

        def keeping_own_quantizers():
            sharded = _sharded(experts, mesh)
            for proj in ("gate_up_proj", "down_proj"):
                name = f"{proj}_weight_quantizers"
                setattr(sharded, name, nn.ModuleList(list(getattr(sharded, name))[mine]))
            return sharded

        sharded = keeping_own_quantizers()
        _export_fused_experts_keep_fused(sharded, torch.bfloat16)
        for key, value in _state(sharded).items():
            local = value.to_local() if isinstance(value, DTensor) else value
            want = plain[key][mine] if isinstance(value, DTensor) else plain[key]
            assert torch.equal(_raw_bytes(local), _raw_bytes(want)), key

        # The last rank disabling one or both of its experts' quantizers is refused on every rank,
        # naming the global experts; every expert disabled leaves the projection as it is on all.
        last = 2 * world_size - 1
        for disabled in ([last], [last - 1, last], list(range(2 * world_size))):
            sharded = keeping_own_quantizers()
            for expert, quantizer in enumerate(sharded.down_proj_weight_quantizers, mine.start):
                if expert in disabled:
                    quantizer.disable()
            if len(disabled) < 2 * world_size:
                match = re.escape(f"quantizers {disabled} of {2 * world_size}")
                with pytest.raises(NotImplementedError, match=match):
                    _export_fused_experts_keep_fused(sharded, torch.bfloat16)
            else:
                _export_fused_experts_keep_fused(sharded, torch.bfloat16)
                assert hasattr(sharded, "gate_up_proj_weight_scale")
                assert not hasattr(sharded, "down_proj_weight_scale")
    finally:
        dist.destroy_process_group(group)


@pytest.mark.parametrize("backend", [None, "nccl"], ids=["objects-over-gloo", "objects-over-nccl"])
def test_ranks_fold_their_own_experts_over_nccl(need_2_gpus, dist_workers, backend):
    """Each rank folds the experts it holds on its GPU, at their global offsets, and the folds,
    gathered over NCCL, split into the per-expert export, an uneven shard included. Under expert
    parallelism the ranks decide on each projection together: in the export's ``cpu:gloo,cuda:nccl``
    group their quantizer states go over Gloo, in a group of NCCL alone over NCCL."""
    dist_workers.run(_fold_job, backend=backend)
