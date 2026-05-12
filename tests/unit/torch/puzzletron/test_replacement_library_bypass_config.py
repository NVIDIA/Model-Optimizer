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

"""Tests for bypass checkpoint metadata consumed by replacement-library extraction."""

import json
from pathlib import Path

import pytest

from modelopt.torch.puzzletron.replacement_library.build_replacement_library import (
    _infer_subblocks_to_extract,
)


@pytest.mark.parametrize(
    ("keys_to_learn", "expected_subblocks"),
    [
        ("entire_block", ["block"]),
        ("subblock_ffn", ["ffn"]),
        ("subblock_attention", ["attention"]),
        ("subblock_mamba", ["attention"]),
        (["subblock_attention", "subblock_ffn"], ["attention", "ffn"]),
    ],
)
def test_infer_subblocks_to_extract_accepts_bypass_keys(
    tmp_path: Path,
    keys_to_learn,
    expected_subblocks,
):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "bypass_config.json").write_text(json.dumps({"keys_to_learn": keys_to_learn}))

    assert _infer_subblocks_to_extract(checkpoint_dir, []) == expected_subblocks


@pytest.mark.parametrize("keys_to_learn", ["mlp", "attn", ["mlp", "attn"]])
def test_infer_subblocks_to_extract_rejects_legacy_keys(tmp_path: Path, keys_to_learn):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "bypass_config.json").write_text(json.dumps({"keys_to_learn": keys_to_learn}))

    with pytest.raises(ValueError, match="keys_to_learn"):
        _infer_subblocks_to_extract(checkpoint_dir, [])
