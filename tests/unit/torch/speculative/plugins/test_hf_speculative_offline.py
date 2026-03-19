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

"""Unit tests for offline speculative decoding PTQ support."""

import argparse
import importlib.util
import os

# ---------------------------------------------------------------------------
# Load eagle_utils from examples/ via importlib (not a package, so no import).
# eagle_utils has a top-level `from scripts.ar_validate import validate_ar` that
# only resolves when run from examples/speculative_decoding/. We stub it out here.
# ---------------------------------------------------------------------------
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.eagle.default_config import default_eagle_config

_mock_scripts = types.ModuleType("scripts")
_mock_ar = types.ModuleType("scripts.ar_validate")
_mock_ar.validate_ar = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("scripts", _mock_scripts)
sys.modules.setdefault("scripts.ar_validate", _mock_ar)

_EAGLE_UTILS_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../..",
    "examples/speculative_decoding/eagle_utils.py",
)
_spec = importlib.util.spec_from_file_location("eagle_utils", _EAGLE_UTILS_PATH)
assert _spec is not None and _spec.loader is not None
_eagle_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eagle_utils)
make_eagle_supervised_data_module = _eagle_utils.make_eagle_supervised_data_module


# ---------------------------------------------------------------------------
# sample_size truncation tests
# ---------------------------------------------------------------------------


def _make_data_args(sample_size, tmp_path, n_files=5):
    """Create a temp dir with n_files dummy .pt files and an argparse.Namespace."""
    for i in range(n_files):
        torch.save({}, tmp_path / f"sample_{i}.pt")
    return argparse.Namespace(
        vlm_processor=None,
        vlm_img_dir=None,
        offline_data_path=str(tmp_path),
        lazy_preprocess=True,
        sample_size=sample_size,
    )


def test_sample_size_positive_truncates(tmp_path):
    """sample_size > 0 should truncate the dataset to that many samples."""
    data_args = _make_data_args(sample_size=3, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_eagle_supervised_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 3


def test_sample_size_minus_one_uses_all(tmp_path):
    """sample_size=-1 should use all samples."""
    data_args = _make_data_args(sample_size=-1, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_eagle_supervised_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 5


def test_sample_size_zero_uses_all(tmp_path):
    """sample_size=0 (non-positive) should use all samples."""
    data_args = _make_data_args(sample_size=0, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_eagle_supervised_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 5


def test_sample_size_larger_than_dataset_uses_all(tmp_path):
    """sample_size > number of files should use all samples without error."""
    data_args = _make_data_args(sample_size=100, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_eagle_supervised_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 5


def test_sample_size_no_pt_files_raises(tmp_path):
    """Empty directory should raise ValueError."""
    data_args = argparse.Namespace(
        vlm_processor=None,
        vlm_img_dir=None,
        offline_data_path=str(tmp_path),
        lazy_preprocess=True,
        sample_size=-1,
    )
    tokenizer = MagicMock()
    with pytest.raises(ValueError, match="No .pt files found"):
        make_eagle_supervised_data_module(tokenizer, data_args, train_len=8)


# ---------------------------------------------------------------------------
# offline_specdec_input propagation through export path
# ---------------------------------------------------------------------------

TINY_EAGLE_ARCH_CFG = {
    "num_hidden_layers": 1,
    "intermediate_size": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 2,
    "use_last_layernorm": True,
    "use_aux_hidden_state": False,
    "eagle_aux_hidden_state_layer_ids": [],
}

TINY_EAGLE_MODE_CFG = {
    "eagle_architecture_config": {**default_eagle_config, **TINY_EAGLE_ARCH_CFG},
}


@pytest.fixture
def eagle_model():
    model = get_tiny_llama(num_hidden_layers=4)
    mtsp.convert(model, mode=[("eagle", TINY_EAGLE_MODE_CFG)])
    return model


def test_export_offline_specdec_input_propagated(eagle_model, tmp_path):
    """offline_specdec_input should be forwarded through export_speculative_decoding."""
    from modelopt.torch.export import export_speculative_decoding

    offline_input = {"input_ids": torch.ones(1, 4, dtype=torch.long)}
    captured = {}

    mock_exporter = MagicMock()

    def capture_export(export_dir, dtype=None, offline_specdec_input=None):
        captured["offline_specdec_input"] = offline_specdec_input

    mock_exporter.export.side_effect = capture_export

    with patch.object(eagle_model, "get_exporter", return_value=mock_exporter):
        export_speculative_decoding(
            eagle_model, export_dir=tmp_path, offline_specdec_input=offline_input
        )

    assert captured.get("offline_specdec_input") is offline_input


def test_export_offline_specdec_input_none_by_default(eagle_model, tmp_path):
    """When offline_specdec_input is not provided, it defaults to None."""
    from modelopt.torch.export import export_speculative_decoding

    captured = {}

    mock_exporter = MagicMock()

    def capture_export(export_dir, dtype=None, offline_specdec_input=None):
        captured["offline_specdec_input"] = offline_specdec_input

    mock_exporter.export.side_effect = capture_export

    with patch.object(eagle_model, "get_exporter", return_value=mock_exporter):
        export_speculative_decoding(eagle_model, export_dir=tmp_path)

    assert captured.get("offline_specdec_input") is None
