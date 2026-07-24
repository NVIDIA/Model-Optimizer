# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for orchestration mesh packing and validation."""

import pytest

from modelopt.torch.puzzletron.orchestration.mesh import (
    ParallelMesh,
    extract_stage_mesh,
    gpus_per_instance,
    pack_gpu_allocation,
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


def test_pack_thirty_two_gpu_single_mesh_is_exclusive_on_four_nodes():
    mesh = ParallelMesh(pp=2, dp_replicate=2, dp_shard=4, ep=1)
    allocation = pack_gpu_allocation(mesh=mesh, instances=1, gpus_per_node=8)
    assert allocation.gpus_per_instance == 32
    assert allocation.nodes == 4
    assert allocation.exclusive is True


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
                "expert_parallel_size": 4,
                "gpu_group_size": 32,
            }
        }
    }

    mesh = extract_stage_mesh(config, "aiperf")

    assert mesh == ParallelMesh(tp=2, pp=2, cp=2, dp_shard=4, dp_replicate=1, ep=4)
    assert gpus_per_instance(mesh) == 32
