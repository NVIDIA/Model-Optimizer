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
"""Unit tests for ``examples/llm_ptq/example_utils.py``."""

from types import SimpleNamespace

import pytest
import torch
from _test_utils.examples.llm_ptq_example_utils import example_utils
from safetensors.torch import save_file


class _FakeModel:
    """Minimal stand-in for an HF causal-LM. ``load_mtp_weights`` touches only
    ``model.config``, ``model.state_dict()`` and ``model.load_state_dict()``.
    """

    def __init__(self, config, state_dict_keys):
        self.config = config
        self._sd = {k: torch.zeros(1) for k in state_dict_keys}
        self.loaded = {}

    def state_dict(self):
        return dict(self._sd)

    def load_state_dict(self, state_dict, strict=True):
        self.loaded.update(state_dict)
        self._sd.update(state_dict)


@pytest.mark.parametrize(
    ("num_nextn", "num_hidden", "expected"),
    [
        (0, 80, []),
        (1, 78, ["model.layers.78"]),
        (3, 80, ["model.layers.80", "model.layers.81", "model.layers.82"]),
    ],
)
def test_get_inlined_mtp_prefixes_returns_expected_prefixes(num_nextn, num_hidden, expected):
    """Pure config -> prefix list. Documents the inlined-MTP detection contract."""
    cfg = SimpleNamespace(num_nextn_predict_layers=num_nextn, num_hidden_layers=num_hidden)
    assert example_utils.get_inlined_mtp_prefixes(cfg) == expected


def test_get_inlined_mtp_prefixes_missing_field_returns_empty():
    """Configs without num_nextn_predict_layers (non-MTP architectures) yield []."""
    cfg = SimpleNamespace(num_hidden_layers=32)
    assert example_utils.get_inlined_mtp_prefixes(cfg) == []


def test_get_inlined_mtp_prefixes_none_field_returns_empty():
    """``num_nextn_predict_layers=None`` must not crash with ``int(None)``."""
    cfg = SimpleNamespace(num_hidden_layers=32, num_nextn_predict_layers=None)
    assert example_utils.get_inlined_mtp_prefixes(cfg) == []


def _write_safetensors(path, tensors):
    save_file(tensors, str(path), metadata={"format": "pt"})


def test_load_mtp_weights_inlined_orphaned(tmp_path):
    """GLM-5.1 case: HF model class doesn't instantiate MTP, so inlined MTP
    tensors at ``model.layers.{N}.*`` are returned as orphans for
    ``extra_state_dict``."""
    main_keys = ["model.embed_tokens.weight", "model.layers.0.x.weight"]
    mtp_keys = ["model.layers.4.eh_proj.weight", "model.layers.4.enorm.weight"]
    _write_safetensors(
        tmp_path / "model.safetensors",
        {k: torch.zeros(2, 2) for k in main_keys + mtp_keys},
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=1)
    model = _FakeModel(cfg, state_dict_keys=main_keys)
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert prefixes == ["model.layers.4"]
    assert set(orphans) == set(mtp_keys)
    assert model.loaded == {}  # nothing matched the (MTP-less) state_dict


def test_load_mtp_weights_inlined_in_state_dict(tmp_path):
    """DeepSeek-V3 case: HF *does* instantiate the inlined layers, so the
    matching keys are loaded into the model and the orphan dict is empty."""
    main_keys = ["model.embed_tokens.weight"]
    mtp_keys = ["model.layers.4.eh_proj.weight", "model.layers.4.enorm.weight"]
    _write_safetensors(
        tmp_path / "model.safetensors",
        {k: torch.ones(2, 2) for k in main_keys + mtp_keys},
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=1)
    model = _FakeModel(cfg, state_dict_keys=main_keys + mtp_keys)
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert prefixes == ["model.layers.4"]
    assert orphans == {}
    assert set(model.loaded) == set(mtp_keys)


def test_load_mtp_weights_separate_standalone_file(tmp_path):
    """Legacy case: a standalone ``mtp.safetensors`` sits alongside the main
    checkpoint with no shard index. The unified loader must still discover and
    load the standalone file."""
    _write_safetensors(
        tmp_path / "model.safetensors", {"model.embed_tokens.weight": torch.zeros(2, 2)}
    )
    _write_safetensors(
        tmp_path / "mtp.safetensors",
        {
            "mtp.fc.weight": torch.zeros(2, 2),
            "mtp.layers.0.q_proj.weight": torch.zeros(2, 2),
        },
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=0)
    model = _FakeModel(cfg, state_dict_keys=["model.embed_tokens.weight"])
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert set(prefixes) == {"mtp", "mtp.layers.0"}
    assert set(orphans) == {"mtp.fc.weight", "mtp.layers.0.q_proj.weight"}


def test_load_mtp_weights_no_mtp_returns_empty(tmp_path):
    """Plain (non-MTP) checkpoint must return ``([], {})``."""
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": torch.zeros(2, 2),
            "model.layers.0.x.weight": torch.zeros(2, 2),
        },
    )
    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=0)
    model = _FakeModel(cfg, state_dict_keys=["model.embed_tokens.weight"])
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))
    assert prefixes == []
    assert orphans == {}
