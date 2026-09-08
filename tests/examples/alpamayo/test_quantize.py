# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CPU-only unit tests for pure helpers in examples/alpamayo/quantize.py."""

import sys
from pathlib import Path

import pytest

# quantize.py imports the gated ``alpamayo_r1`` package (and transformers) at module
# load and monkeypatches at import time, so guard collection on those being installed.
pytest.importorskip("alpamayo_r1")
pytest.importorskip("transformers")

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "alpamayo"))

import quantize


class TestReadClipIdsFromParquet:
    def test_dedup_preserves_first_occurrence_order(self, tmp_path):
        path = tmp_path / "clips.parquet"
        pd.DataFrame({"key": ["a", "b", "a", "c", "b"]}).to_parquet(path)

        assert quantize.read_clip_ids_from_parquet(path) == ["a", "b", "c"]

    def test_missing_key_column_raises(self, tmp_path):
        path = tmp_path / "clips.parquet"
        pd.DataFrame({"id": ["a", "b"]}).to_parquet(path)

        with pytest.raises(KeyError):
            quantize.read_clip_ids_from_parquet(path)


class TestCreateMessage:
    def test_roles_and_structure(self):
        messages = quantize.create_message(torch.zeros(3, 3, 8, 8))

        assert [m["role"] for m in messages] == ["system", "user", "assistant"]

    def test_one_image_entry_per_frame_plus_trailing_text(self):
        num_frames = 3
        messages = quantize.create_message(torch.zeros(num_frames, 3, 8, 8))

        user_content = messages[1]["content"]
        image_entries = [c for c in user_content if c["type"] == "image"]
        text_entries = [c for c in user_content if c["type"] == "text"]

        assert len(image_entries) == num_frames
        assert len(text_entries) == 1

    def test_history_trajectory_tokens(self):
        messages = quantize.create_message(torch.zeros(1, 3, 8, 8))

        user_text = next(c["text"] for c in messages[1]["content"] if c["type"] == "text")
        assert user_text.count("<|traj_history|>") == 48
        assert "<|traj_history_start|>" in user_text
        assert "<|traj_history_end|>" in user_text

    def test_non_4d_frames_raises(self):
        with pytest.raises(AssertionError):
            quantize.create_message(torch.zeros(3, 8, 8))
