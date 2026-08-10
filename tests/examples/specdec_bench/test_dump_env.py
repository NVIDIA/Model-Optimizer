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

"""Tests for the derived provenance fields dump_env writes.

configuration.json is the only record of how a benchmark ran once the job is
gone, and downstream tooling reads it verbatim. These fields exist because the
raw argparse namespace is ambiguous about what the engine actually did.
"""

import argparse
import json

import pytest
from specdec_bench.utils import dump_env


def _args(**kw):
    """An argparse namespace with the fields dump_env reaches for."""
    base = {
        "engine": "VLLM",
        "model_dir": None,
        "draft_length": 3,
        "block_size": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _dump(tmp_path, **kw):
    dump_env(_args(**kw), str(tmp_path))
    with open(tmp_path / "configuration.json") as f:
        return json.load(f)


class TestNumSpeculativeTokens:
    def test_draft_length_path(self, tmp_path):
        """Algorithms that take --draft_length record it directly."""
        assert _dump(tmp_path, draft_length=7)["num_speculative_tokens"] == 7

    def test_block_size_path(self, tmp_path):
        """DFLASH takes --block_size and leaves --draft_length at its default,
        so reading draft_length would report 3 for a block_size=8 run."""
        cfg = _dump(tmp_path, draft_length=3, block_size=8)
        assert cfg["num_speculative_tokens"] == 7
        # The raw flags stay for provenance; only the derived field is
        # authoritative.
        assert cfg["draft_length"] == 3
        assert cfg["block_size"] == 8

    def test_block_size_wins_over_draft_length(self, tmp_path):
        assert _dump(tmp_path, draft_length=3, block_size=4)["num_speculative_tokens"] == 3

    def test_missing_attrs_do_not_raise(self, tmp_path):
        """dump_env is called from harnesses that build their own namespace."""
        dump_env(argparse.Namespace(engine="VLLM", model_dir=None), str(tmp_path))
        with open(tmp_path / "configuration.json") as f:
            assert json.load(f)["num_speculative_tokens"] is None


class TestDraftHuggingfaceModelId:
    def test_absent_by_default(self, tmp_path):
        assert _dump(tmp_path)["draft_huggingface_model_id"] is None

    def test_read_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRAFT_HUGGINGFACE_MODEL_ID", "org/Drafter-A")
        assert _dump(tmp_path)["draft_huggingface_model_id"] == "org/Drafter-A"

    @pytest.mark.parametrize("value", ["", None])
    def test_blank_normalizes_to_none(self, tmp_path, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("DRAFT_HUGGINGFACE_MODEL_ID", raising=False)
        else:
            monkeypatch.setenv("DRAFT_HUGGINGFACE_MODEL_ID", value)
        assert _dump(tmp_path)["draft_huggingface_model_id"] is None
