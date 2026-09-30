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

"""Tests for multimodal synthetic-shard preparation helpers."""

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "examples/speculative_decoding/recipes/prepare_multimodal_synthetic_shards.py"
)
_spec = importlib.util.spec_from_file_location("prepare_multimodal_synthetic_shards", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
prepare_multimodal_synthetic_shards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_multimodal_synthetic_shards)


def test_media_path_within_root_rejects_path_traversal_and_absolute_paths(tmp_path):
    """PAI metadata cannot point preparation outside its media root."""

    media_root = tmp_path / "media"
    media_path = media_root / "videos" / "clip.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.touch()
    outside_path = tmp_path / "outside.mp4"
    outside_path.touch()

    assert (
        prepare_multimodal_synthetic_shards.media_path_within_root(media_root, "videos/clip.mp4")
        == media_path
    )
    assert (
        prepare_multimodal_synthetic_shards.media_path_within_root(media_root, "../outside.mp4")
        is None
    )
    assert (
        prepare_multimodal_synthetic_shards.media_path_within_root(media_root, str(outside_path))
        is None
    )
