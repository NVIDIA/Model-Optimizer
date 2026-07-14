# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from modelopt.torch.puzzletron.plugins.automodel.launch import _observed_num_nodes
from modelopt.torch.puzzletron.stages.pipeline import _runtime_split


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
