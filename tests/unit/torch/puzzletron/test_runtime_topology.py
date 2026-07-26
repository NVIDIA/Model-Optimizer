# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelopt.torch.puzzletron.plugins.automodel.launch import _observed_num_nodes
from modelopt.torch.puzzletron.stages.pipeline import _runtime_split
from modelopt.torch.puzzletron.subblock_stats.topology import RuntimeTopology


def test_runtime_split_infers_direct_multi_node_torchrun(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("GROUP_RANK", "1")

    assert _runtime_split({"_runtime": {"num_nodes": 1, "node_index": 0}}) == (2, 1)
    assert _observed_num_nodes(1) == 2


def test_runtime_split_preserves_explicit_larger_campaign(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    assert _runtime_split({"_runtime": {"num_nodes": 4, "node_index": 2}}) == (4, 2)
    assert _observed_num_nodes(4) == 4


def test_runtime_topology_includes_data_and_expert_parallelism():
    topology = RuntimeTopology.from_config(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": True,
            "gpu_group_size": 8,
        }
    )

    assert topology.world_size == 8
    assert topology.to_dict()["data_parallel_size"] == 2
    assert topology.to_dict()["enable_expert_parallel"] is True
