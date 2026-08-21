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
        "speculative_algorithm": "EAGLE3",
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
    @pytest.mark.parametrize("algorithm", ["EAGLE3", "EAGLE", "MTP", "DRAFT_TARGET"])
    def test_draft_length_path(self, tmp_path, algorithm):
        """Algorithms configured by --draft_length record it directly."""
        cfg = _dump(tmp_path, speculative_algorithm=algorithm, draft_length=7)
        assert cfg["num_speculative_tokens"] == 7

    def test_block_size_path(self, tmp_path):
        """DFLASH takes --block_size and leaves --draft_length at its default,
        so reading draft_length would report 3 for a block_size=8 run.

        The recorded value matches what the engine receives: models/vllm.py
        passes block_size through as num_speculative_tokens unchanged."""
        cfg = _dump(tmp_path, speculative_algorithm="DFLASH", draft_length=3, block_size=8)
        assert cfg["num_speculative_tokens"] == 8
        # The raw flags stay for provenance; only the derived field is
        # authoritative.
        assert cfg["draft_length"] == 3
        assert cfg["block_size"] == 8

    def test_none_algorithm_records_zero(self, tmp_path):
        """--speculative_algorithm NONE speculates nothing, but --draft_length
        still defaults to 3. Baselines are the rows most often joined against
        specdec rows, so a phantom width there is worse than no width."""
        cfg = _dump(tmp_path, speculative_algorithm="NONE", draft_length=3)
        assert cfg["num_speculative_tokens"] == 0

    def test_block_size_ignored_for_non_dflash(self, tmp_path):
        """Only the DFLASH branches consume block_size as the width. Passing it
        alongside EAGLE3 must not override the flag the engine actually read."""
        cfg = _dump(tmp_path, speculative_algorithm="EAGLE3", draft_length=5, block_size=8)
        assert cfg["num_speculative_tokens"] == 5

    def test_missing_attrs_do_not_raise(self, tmp_path):
        """dump_env is called from harnesses that build their own namespace."""
        dump_env(argparse.Namespace(engine="VLLM", model_dir=None), str(tmp_path))
        with open(tmp_path / "configuration.json") as f:
            assert json.load(f)["num_speculative_tokens"] is None


class TestDraftHuggingfaceModelId:
    def test_absent_by_default(self, tmp_path, monkeypatch):
        # Cleared explicitly: exporting this var is the documented way to run a
        # benchmark, so reading the ambient environment would fail for anyone
        # who does.
        monkeypatch.delenv("DRAFT_HUGGINGFACE_MODEL_ID", raising=False)
        assert _dump(tmp_path)["draft_huggingface_model_id"] is None

    def test_read_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DRAFT_HUGGINGFACE_MODEL_ID", "org/Drafter-A")
        assert _dump(tmp_path)["draft_huggingface_model_id"] == "org/Drafter-A"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_normalizes_to_none(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("DRAFT_HUGGINGFACE_MODEL_ID", value)
        assert _dump(tmp_path)["draft_huggingface_model_id"] is None


class TestHubModelIdValidation:
    """These ids are copied verbatim into a published configuration.json, so a
    malformed environment value is dropped rather than recorded."""

    @pytest.mark.parametrize(
        "value",
        [
            "https://user:hf_secret@huggingface.co/org/model",  # embedded credential
            "org/model?token=hf_secret",
            "org/model#hf_secret",
            "not-an-id",  # no org
            "org/model/extra",  # too many segments
            "org / model",  # whitespace
            # Non-canonical repo names: harmless, but validate_repo_id rejects
            # them, so they should not be recorded as if they resolved.
            "org/foo..bar",
            "org/foo--bar",
            "org/foo.git",
        ],
    )
    @pytest.mark.parametrize("env_var", ["DRAFT_HUGGINGFACE_MODEL_ID", "HUGGINGFACE_MODEL_ID"])
    def test_malformed_is_dropped_with_warning(self, tmp_path, monkeypatch, env_var, value):
        monkeypatch.setenv(env_var, value)
        with pytest.warns(UserWarning, match=env_var):
            cfg = _dump(tmp_path)
        assert cfg[env_var.lower()] is None

    @pytest.mark.parametrize(
        "value", ["org/model", "nvidia/Kimi-K3-DSpark", "some-org/some.model-v1_2"]
    )
    def test_wellformed_ids_pass_through(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("DRAFT_HUGGINGFACE_MODEL_ID", value)
        assert _dump(tmp_path)["draft_huggingface_model_id"] == value
