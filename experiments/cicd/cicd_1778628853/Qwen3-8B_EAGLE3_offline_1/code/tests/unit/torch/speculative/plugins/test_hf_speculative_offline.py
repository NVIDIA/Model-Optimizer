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
from unittest.mock import MagicMock

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.eagle.default_config import default_eagle_config
from modelopt.torch.speculative.eagle.utils import (
    EagleOfflineDataCollator,
    OfflineSupervisedDataset,
)

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
make_speculative_data_module = _eagle_utils.make_speculative_data_module


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
    module = make_speculative_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 3


def test_sample_size_minus_one_uses_all(tmp_path):
    """sample_size=-1 should use all samples."""
    data_args = _make_data_args(sample_size=-1, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_speculative_data_module(tokenizer, data_args, train_len=8)
    assert len(module["train_dataset"]) == 5


def test_sample_size_zero_raises(tmp_path):
    """sample_size=0 should raise ValueError."""
    data_args = _make_data_args(sample_size=0, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    with pytest.raises(ValueError, match="sample_size must be -1"):
        make_speculative_data_module(tokenizer, data_args, train_len=8)


def test_sample_size_larger_than_dataset_uses_all(tmp_path):
    """sample_size > number of files should use all samples without error."""
    data_args = _make_data_args(sample_size=100, tmp_path=tmp_path, n_files=5)
    tokenizer = MagicMock()
    module = make_speculative_data_module(tokenizer, data_args, train_len=8)
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
        make_speculative_data_module(tokenizer, data_args, train_len=8)


# ---------------------------------------------------------------------------
# get_dummy_inputs() for export forward pass
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


def test_get_dummy_inputs_online(eagle_model):
    """Online EAGLE model returns input_ids only (no base_model_outputs)."""
    eagle_model.eagle_offline = False
    dummy = eagle_model.get_dummy_inputs()
    assert "input_ids" in dummy
    assert "base_model_outputs" not in dummy


def test_get_dummy_inputs_offline(eagle_model):
    """Offline EAGLE model returns input_ids and base_model_outputs with correct shapes."""
    eagle_model.eagle_offline = True
    dummy = eagle_model.get_dummy_inputs()
    assert "input_ids" in dummy
    assert "base_model_outputs" in dummy
    hidden_size = eagle_model.config.hidden_size
    assert dummy["base_model_outputs"]["base_model_hidden_states"].shape[-1] == hidden_size
    assert dummy["base_model_outputs"]["base_model_input_embeds"].shape[-1] == hidden_size


# ---------------------------------------------------------------------------
# OfflineSupervisedDataset tests
# ---------------------------------------------------------------------------

SEQ_LEN = 16
HIDDEN_SIZE = 8


def _make_offline_pt(path, seq_len=SEQ_LEN, hidden_size=HIDDEN_SIZE):
    """Write a realistic .pt file matching the format expected by OfflineSupervisedDataset."""
    data = {
        "input_ids": torch.randint(0, 100, (seq_len,)),
        "hidden_states": torch.randn(seq_len, hidden_size),
        "aux_hidden_states": torch.randn(seq_len, hidden_size),
        "base_model_input_embeds": torch.randn(seq_len, hidden_size),
    }
    torch.save(data, path)
    return data


def test_offline_dataset_len_and_getitem(tmp_path):
    """OfflineSupervisedDataset should load .pt files and return proper keys."""
    n = 3
    files = []
    for i in range(n):
        p = tmp_path / f"sample_{i}.pt"
        _make_offline_pt(p)
        files.append(str(p))

    ds = OfflineSupervisedDataset(files)
    assert len(ds) == n

    item = ds[0]
    assert set(item.keys()) == {
        "input_ids",
        "base_model_hidden_states",
        "aux_hidden_states",
        "attention_mask",
        "loss_mask",
        "labels",
    }
    assert item["input_ids"].shape == (SEQ_LEN,)
    assert item["attention_mask"].shape == (SEQ_LEN,)
    assert item["labels"].shape == (SEQ_LEN,)


def test_offline_dataset_labels_shift(tmp_path):
    """Labels should be input_ids shifted left by 1."""
    p = tmp_path / "sample.pt"
    orig = _make_offline_pt(p)
    ds = OfflineSupervisedDataset([str(p)])
    item = ds[0]
    # labels[:-1] should equal input_ids[1:]
    assert torch.equal(item["labels"][:-1], orig["input_ids"][1:])


# ---------------------------------------------------------------------------
# EagleOfflineDataCollator tests
# ---------------------------------------------------------------------------


def test_collator_truncates(tmp_path):
    """Collator should truncate sequences longer than train_len."""
    train_len = 8
    p = tmp_path / "sample.pt"
    _make_offline_pt(p, seq_len=SEQ_LEN)  # SEQ_LEN > train_len
    ds = OfflineSupervisedDataset([str(p)])
    collator = EagleOfflineDataCollator(train_len=train_len)
    batch = collator([ds[0]])
    assert batch["input_ids"].shape == (1, train_len)
    assert batch["base_model_outputs"]["base_model_hidden_states"].shape[1] == train_len


def test_collator_pads(tmp_path):
    """Collator should pad sequences shorter than train_len."""
    train_len = 32
    p = tmp_path / "sample.pt"
    _make_offline_pt(p, seq_len=SEQ_LEN)  # SEQ_LEN < train_len
    ds = OfflineSupervisedDataset([str(p)])
    collator = EagleOfflineDataCollator(train_len=train_len)
    batch = collator([ds[0]])
    assert batch["input_ids"].shape == (1, train_len)
    # Padded region should be zeros
    assert (batch["input_ids"][0, SEQ_LEN:] == 0).all()


def test_collator_batches_multiple(tmp_path):
    """Collator should stack multiple samples into a batch."""
    train_len = SEQ_LEN
    files = []
    for i in range(4):
        p = tmp_path / f"sample_{i}.pt"
        _make_offline_pt(p)
        files.append(str(p))
    ds = OfflineSupervisedDataset(files)
    collator = EagleOfflineDataCollator(train_len=train_len)
    batch = collator([ds[i] for i in range(4)])
    assert batch["input_ids"].shape == (4, train_len)
    assert batch["base_model_outputs"]["base_model_hidden_states"].shape == (
        4,
        train_len,
        HIDDEN_SIZE,
    )
