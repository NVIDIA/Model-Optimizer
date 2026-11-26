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
"""Unit coverage for the image checkpoint's NVFP4 scale inventory."""

import json

import pytest
import torch
from _test_utils.image_deploy_utils import _check_static_nvfp4
from safetensors.torch import save_file

_QUANT_CONFIG = {
    "quantization_config": {
        "quant_algo": "NVFP4",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {"dynamic": False},
                "input_activations": {"dynamic": False},
            }
        },
    }
}

_WEIGHT = "transformer_blocks.0.attn.to_q.weight"


def _make_component(snapshot, files, index=None):
    """Write config.json plus the given safetensors files under snapshot/transformer."""
    component = snapshot / "transformer"
    component.mkdir(parents=True)
    (component / "config.json").write_text(json.dumps(_QUANT_CONFIG), encoding="utf-8")
    for name, tensors in files.items():
        save_file(tensors, component / name)
    if index is not None:
        (component / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index}), encoding="utf-8"
        )
    return component


def _valid_tensors(**scale_overrides):
    scales = {
        "weight_scale": torch.ones(4, 2),
        "weight_scale_2": torch.ones(1),
        "input_scale": torch.ones(1),
    }
    scales.update(scale_overrides)
    return {
        _WEIGHT: torch.zeros(4, 32, dtype=torch.uint8),
        **{f"{_WEIGHT[:-7]}.{name}": value for name, value in scales.items()},
    }


def test_static_nvfp4_accepts_valid_scales(tmp_path):
    component = _make_component(tmp_path / "snapshot", {"model.safetensors": _valid_tensors()})
    _check_static_nvfp4(component.parent, tmp_path)
    evidence = json.loads((tmp_path / "static-activation-scales.json").read_text())
    assert evidence["packed_modules"] == [_WEIGHT[: -len(".weight")]]
    assert not any(evidence["missing_scales"].values())
    assert not evidence["invalid_scales"]


@pytest.mark.parametrize(
    "bad_scale",
    [
        {"weight_scale": torch.full((4, 2), torch.nan)},
        {"weight_scale": torch.zeros(4, 2, 3)},  # wrong rank for a 2D weight
        {"weight_scale_2": torch.ones(4)},  # global scale must be a single scalar
        {"input_scale": torch.zeros(1)},  # nonpositive
    ],
    ids=["nan", "misranked", "non-scalar-global", "zero"],
)
def test_static_nvfp4_rejects_invalid_scales(tmp_path, bad_scale):
    component = _make_component(
        tmp_path / "snapshot", {"model.safetensors": _valid_tensors(**bad_scale)}
    )
    with pytest.raises(AssertionError, match="invalid scales"):
        _check_static_nvfp4(component.parent, tmp_path)


def test_static_nvfp4_ignores_unreferenced_shards(tmp_path):
    active = {_WEIGHT: torch.zeros(4, 32, dtype=torch.uint8)}
    scales_only = {name: value for name, value in _valid_tensors().items() if name != _WEIGHT}
    component = _make_component(
        tmp_path / "snapshot",
        {
            "model-00001-of-00002.safetensors": active,
            "model-00002-of-00002.safetensors": scales_only,
        },
        index={_WEIGHT: "model-00001-of-00002.safetensors"},
    )
    with pytest.raises(AssertionError, match="missing scales"):
        _check_static_nvfp4(component.parent, tmp_path)
    evidence = json.loads((tmp_path / "static-activation-scales.json").read_text())
    assert evidence["unreferenced_shards"] == ["model-00002-of-00002.safetensors"]
