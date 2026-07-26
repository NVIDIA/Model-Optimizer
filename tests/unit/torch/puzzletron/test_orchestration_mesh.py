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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for orchestration mesh packing and validation."""

import pytest

from modelopt.torch.puzzletron.orchestration.mesh import (
    ParallelMesh,
    extract_stage_mesh,
    gpus_per_instance,
    normalize_vllm_topology,
    pack_gpu_allocation,
    vllm_topology_to_mesh,
)


def test_gpus_per_instance_uses_automodel_world_size_without_ep_multiplier():
    mesh = ParallelMesh(pp=2, dp_replicate=2, dp_shard=4, cp=1, tp=1, ep=2)
    assert gpus_per_instance(mesh) == 16


def test_ep_overlay_requires_dp_shard_divisibility():
    with pytest.raises(ValueError, match="dp_shard must be divisible by ep"):
        gpus_per_instance(ParallelMesh(dp_shard=3, ep=2))


def test_pack_sixteen_one_gpu_instances_onto_two_eight_gpu_nodes():
    allocation = pack_gpu_allocation(
        mesh=ParallelMesh(),
        instances=16,
        gpus_per_node=8,
    )
    assert allocation.gpus_per_instance == 1
    assert allocation.total_gpus == 16
    assert allocation.nodes == 2
    assert allocation.instances_per_node == 8
    assert allocation.exclusive is True


def test_pack_sixteen_gpu_single_mesh_spans_two_nodes_non_exclusively():
    mesh = ParallelMesh(pp=2, dp_replicate=2, dp_shard=4, ep=1)
    allocation = pack_gpu_allocation(mesh=mesh, instances=1, gpus_per_node=8)
    assert allocation.gpus_per_instance == 16
    assert allocation.nodes == 2
    assert allocation.exclusive is False


def test_extract_vllm_stats_mesh_from_runtime_topology():
    config = {
        "vllm_stats": {
            "runtime_stats": {
                "topology": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "prefill_context_parallel_size": 1,
                    "gpu_group_size": 1,
                }
            }
        }
    }
    mesh = extract_stage_mesh(config, "vllm_stats")
    assert mesh.tp == 1
    assert gpus_per_instance(mesh) == 1


def test_extract_aiperf_mesh_uses_topology_dimensions_without_gpu_group_double_count():
    config = {
        "aiperf": {
            "topology": {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 2,
                "prefill_context_parallel_size": 2,
                "decode_context_parallel_size": 1,
                "data_parallel_size": 4,
                "enable_expert_parallel": True,
                "gpu_group_size": 32,
            }
        }
    }

    mesh = extract_stage_mesh(config, "aiperf")

    assert mesh == ParallelMesh(tp=2, pp=2, cp=2, dp_shard=1, dp_replicate=4, ep=1)
    assert gpus_per_instance(mesh) == 32


def test_vllm_expert_parallel_is_tp_times_dp_without_extra_allocation_axis():
    topology = normalize_vllm_topology(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 4,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": True,
            "gpu_group_size": 16,
        }
    )

    assert topology["effective_ep"] == 8
    assert topology["gpu_count"] == 16
    assert vllm_topology_to_mesh(topology) == ParallelMesh(
        tp=2,
        pp=2,
        cp=1,
        dp_replicate=4,
        dp_shard=1,
        ep=1,
    )


def test_vllm_legacy_expert_parallel_rejects_an_independent_size():
    with pytest.raises(
        ValueError,
        match=r"expert_parallel_size=4.*expected 1 or TP \* DP=8",
    ):
        normalize_vllm_topology(
            {
                "tensor_parallel_size": 2,
                "data_parallel_size": 4,
                "expert_parallel_size": 4,
            }
        )


def test_vllm_legacy_effective_expert_parallel_enables_boolean_mode():
    topology = normalize_vllm_topology(
        {
            "tensor_parallel_size": 2,
            "data_parallel_size": 4,
            "expert_parallel_size": 8,
            "gpu_group_size": 8,
        }
    )

    assert topology["enable_expert_parallel"] is True
    assert topology["effective_ep"] == 8


def test_vllm_disabled_expert_parallel_keeps_effective_ep_one():
    topology = normalize_vllm_topology(
        {
            "tensor_parallel_size": 2,
            "data_parallel_size": 4,
            "enable_expert_parallel": False,
            "gpu_group_size": 8,
        }
    )

    assert topology["enable_expert_parallel"] is False
    assert topology["effective_ep"] == 1
