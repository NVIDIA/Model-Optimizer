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
"""End-to-end unit tests for ``examples/hf_ptq/example_utils.load_mtp_weights``.

One test per supported on-disk MTP convention (inlined-orphaned, inlined-in-state-dict,
separate-file-standalone, separate-file-indexed) plus a negative case.
"""

import json
from types import SimpleNamespace

import torch
from _test_utils.examples.llm_ptq_example_utils import example_utils
from safetensors.torch import save_file


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


def test_keys_to_prefixes_drops_model_top_level():
    # nvbug 6108133: inlined keys like "model.layers.92.X" must NOT emit "model"
    # as a top-level prefix (would become "model*" excluding the whole backbone).
    out = example_utils._keys_to_prefixes(
        ["model.layers.92.eh_proj.weight", "mtp.fc.weight", "mtp.layers.0.q_proj.weight"]
    )
    assert "model" not in out
    assert out == {"mtp", "mtp.layers.0", "model.layers.92"}


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


# ---------- get_original_hf_quant_method -------------------------------------
# get_model uses this to detect native MXFP4 checkpoints (e.g. openai/gpt-oss-*) and load
# them dequantized to BF16 GptOssExperts (so ModelOpt can quantize/export the experts).


def test_get_original_hf_quant_method_mxfp4_dict():
    # gpt-oss layout: quantization_config is a plain dict carrying quant_method.
    cfg = SimpleNamespace(
        quantization_config={"quant_method": "mxfp4", "modules_to_not_convert": []}
    )
    assert example_utils.get_original_hf_quant_method(cfg) == "mxfp4"


def test_get_original_hf_quant_method_object():
    # Some configs expose quantization_config as an object with a quant_method attribute.
    cfg = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="fp8"))
    assert example_utils.get_original_hf_quant_method(cfg) == "fp8"


def test_get_original_hf_quant_method_nested_text_config():
    # Multi-modal models nest the quantization_config under text_config.
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(quantization_config={"quant_method": "mxfp4"})
    )
    assert example_utils.get_original_hf_quant_method(cfg) == "mxfp4"


def test_get_original_hf_quant_method_none_for_unquantized():
    assert example_utils.get_original_hf_quant_method(SimpleNamespace()) is None
    assert (
        example_utils.get_original_hf_quant_method(SimpleNamespace(quantization_config=None))
        is None
    )
