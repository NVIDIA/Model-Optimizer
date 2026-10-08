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

"""CPU tests for streaming checkpoint shard assembly."""

import pytest
import torch
from safetensors import safe_open

from modelopt.torch.export.unified_export_hf_streaming import (
    _StreamingShardWriter,
    name_shards_and_write_index,
)


def test_name_shards_and_write_index_rejects_duplicate_keys(tmp_path):
    """Different rank writers must not hide one another's tensors in the final index."""
    closed = []
    for rank in range(2):
        writer = _StreamingShardWriter(tmp_path, max_shard_size=1, part_tag=f"r{rank}_")
        writer.add("head.weight", torch.full((2, 2), float(rank)))
        closed.append(writer.close())
    with pytest.raises(ValueError, match=r"collision.*head\.weight"):
        name_shards_and_write_index(tmp_path, closed)
    assert not (tmp_path / "model.safetensors.index.json").exists()
    for rank, (names, _, _) in enumerate(closed):
        with safe_open(str(tmp_path / names[0]), framework="pt") as f:
            torch.testing.assert_close(f.get_tensor("head.weight"), torch.full((2, 2), float(rank)))
