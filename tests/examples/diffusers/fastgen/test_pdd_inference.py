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

"""Diffusers serialization seam for a trained Qwen-Image PDD projection."""

from __future__ import annotations

import pathlib
import sys

import torch
from _test_utils.torch.diffusers_models import get_tiny_qwen_image_transformer
from diffusers import QwenImageTransformer2DModel

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

from pdd.inference_qwen_image import _load_config, _parse_blocks

from modelopt.torch.fastgen import PDDConfig, PDDOutputProjection
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    convert_qwen_image_to_pdd,
    restore_qwen_image_pdd_projection,
)


def test_widened_diffusers_projection_restores_pdd_fusion_metadata(tmp_path) -> None:
    config = PDDConfig(
        grid_size=4,
        block_size_min=1,
        block_size_max=4,
        inference_blocks=[2, 2],
    )
    transformer = get_tiny_qwen_image_transformer(num_layers=1)
    base_out_channels = transformer.out_channels
    projection = convert_qwen_image_to_pdd(transformer, config)
    base_projection_features = projection.base_out_features
    transformer.register_to_config(out_channels=base_out_channels * config.grid_size)
    expected = {name: value.detach().clone() for name, value in projection.state_dict().items()}
    transformer.save_pretrained(tmp_path)

    restored = QwenImageTransformer2DModel.from_pretrained(tmp_path)
    assert not isinstance(restored.proj_out, PDDOutputProjection)
    restored_weight = restored.proj_out.weight
    restored_bias = restored.proj_out.bias
    adopted = restore_qwen_image_pdd_projection(restored, config)

    assert restored.proj_out is adopted
    assert adopted.weight is restored_weight
    assert adopted.bias is restored_bias
    assert adopted.grid_size == config.grid_size
    assert adopted.base_out_features == base_projection_features
    for name, value in adopted.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_inference_block_override_is_validated(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "pdd:\n"
        "  grid_size: 8\n"
        "  block_size_min: 1\n"
        "  block_size_max: 8\n"
        "  inference_blocks: [4, 4]\n"
    )

    blocks = _parse_blocks("2, 2,4")
    assert blocks == [2, 2, 4]
    assert _load_config(config_path, blocks).inference_blocks == tuple(blocks)
