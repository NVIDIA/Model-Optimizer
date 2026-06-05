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
"""Unit tests for ``examples/llm_ptq/example_utils``.

Covers:

* ``load_mtp_weights`` — one test per supported on-disk MTP convention
  (inlined-orphaned, inlined-in-state-dict, separate-file-standalone,
  separate-file-indexed) plus a negative case.
* the YAML-driven ``QUANT_CFG_CHOICES`` / ``KV_QUANT_CFG_CHOICES`` preset
  discovery — every preset under the model/KV dirs loads, aliases resolve, and
  the KV ``none`` sentinel does not collide with a discovered preset. These are
  cheap smoke tests guarding the eager import-time load: a single malformed
  preset YAML would otherwise break ``import example_utils`` (and every llm_ptq
  script) before the user reaches CLI validation.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from _test_utils.examples.llm_ptq_example_utils import example_utils
from safetensors.torch import save_file

from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT


class _FakeModel:
    """Stub exposing only the surface ``load_mtp_weights`` touches."""

    def __init__(self, config, state_dict_keys):
        self.config = config
        self._sd = {k: torch.zeros(1) for k in state_dict_keys}
        self.loaded = {}

    def state_dict(self):
        return dict(self._sd)

    def load_state_dict(self, state_dict, strict=True):
        self.loaded.update(state_dict)
        self._sd.update(state_dict)


def _write_safetensors(path, tensors):
    save_file(tensors, str(path), metadata={"format": "pt"})


def test_load_mtp_weights_inlined_orphaned(tmp_path):
    # GLM-5.1: HF builds only num_hidden decoders → MTP keys orphaned.
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
    # DeepSeek-V3 via trust_remote_code: MTP slots exist → keys loaded, no orphans.
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
    # GLM-4.7: standalone mtp.safetensors with no shard index.
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


def test_load_mtp_weights_separate_indexed_shard(tmp_path):
    # Qwen3-Next: mtp.* keys in a dedicated indexed tail shard (filename has no "mtp").
    main_shard = "model-00001-of-00002.safetensors"
    mtp_shard = "model-00002-of-00002.safetensors"
    _write_safetensors(tmp_path / main_shard, {"model.embed_tokens.weight": torch.zeros(2, 2)})
    mtp_tensors = {
        "mtp.fc.weight": torch.zeros(2, 2),
        "mtp.norm.weight": torch.zeros(2),
        "mtp.layers.0.input_layernorm.weight": torch.zeros(2),
        "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
    }
    _write_safetensors(tmp_path / mtp_shard, mtp_tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": main_shard,
                    **dict.fromkeys(mtp_tensors, mtp_shard),
                }
            }
        )
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=0)
    model = _FakeModel(cfg, state_dict_keys=["model.embed_tokens.weight"])
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert set(prefixes) == {"mtp", "mtp.layers.0"}
    assert set(orphans) == set(mtp_tensors)


def test_load_mtp_weights_no_mtp_returns_empty(tmp_path):
    # Also pins the ``num_nextn_predict_layers=None`` regression: some configs
    # set the field explicitly to None, which must not crash ``int(None)``.
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": torch.zeros(2, 2),
            "model.layers.0.x.weight": torch.zeros(2, 2),
        },
    )
    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=None)
    model = _FakeModel(cfg, state_dict_keys=["model.embed_tokens.weight"])
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))
    assert prefixes == []
    assert orphans == {}


# ---------------------------------------------------------------------------
# Preset discovery: QUANT_CFG_CHOICES / KV_QUANT_CFG_CHOICES
# ---------------------------------------------------------------------------


def _yaml_basenames(subdir: str) -> set[str]:
    return {
        entry.name.rsplit(".", 1)[0]
        for entry in BUILTIN_CONFIG_ROOT.joinpath(subdir).iterdir()
        if entry.name.endswith((".yaml", ".yml"))
    }


@pytest.mark.parametrize(
    ("choices", "preset_dir"),
    [
        (example_utils.QUANT_CFG_CHOICES, example_utils._QFORMAT_PRESET_DIR),
        (example_utils.KV_QUANT_CFG_CHOICES, example_utils._KV_QFORMAT_PRESET_DIR),
    ],
    ids=["model", "kv"],
)
def test_every_discovered_preset_loads(choices, preset_dir):
    # Configs are loaded eagerly at import, so a malformed preset would already have
    # raised before this test runs. Assert discovery is non-empty, covers every YAML
    # on disk, and that each resolved entry is a usable quant_cfg dict.
    basenames = _yaml_basenames(preset_dir)
    assert basenames, f"no preset YAMLs discovered under {preset_dir}"
    assert basenames <= set(choices), "a preset YAML is missing from the discovered choices"
    for name, cfg in choices.items():
        assert isinstance(cfg, dict), f"{name} did not resolve to a dict"
        assert "quant_cfg" in cfg, f"{name} is missing the 'quant_cfg' key"


def test_aliases_resolve_to_their_canonical_preset():
    for alias, target in example_utils._QFORMAT_ALIASES.items():
        assert alias in example_utils.QUANT_CFG_CHOICES, f"alias {alias!r} not exposed"
        assert target in example_utils.QUANT_CFG_CHOICES, f"alias target {target!r} missing"
        assert example_utils.QUANT_CFG_CHOICES[alias] == example_utils.QUANT_CFG_CHOICES[target]


def test_kv_none_sentinel_is_not_a_discovered_preset():
    # The runtime branches on ``args.kv_cache_qformat != _KV_NONE``; a real preset
    # named "none" would make that branch ambiguous.
    assert example_utils._KV_NONE not in example_utils.KV_QUANT_CFG_CHOICES
