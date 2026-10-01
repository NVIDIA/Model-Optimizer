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

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("onnx")
pytest.importorskip("onnx_graphsurgeon")

from _test_utils.torch.diffusers_models import get_tiny_flux, get_tiny_unet
from diffusers import SD3Transformer2DModel

from examples.diffusers.quantization.onnx_utils.export import (
    _create_trt_dynamic_shapes,
    generate_dummy_kwargs_and_dynamic_axes_and_shapes,
)


def _backbone(model_name):
    if model_name in ("flux-schnell", "flux-dev"):
        return get_tiny_flux(guidance_embeds=model_name == "flux-dev")
    if model_name == "sd3-medium":
        return SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=8,
            caption_projection_dim=16,
            pooled_projection_dim=8,
            pos_embed_max_size=24,
        )
    return get_tiny_unet(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        block_out_channels=(8,),
        cross_attention_dim=8,
        addition_embed_type="text_time",
        addition_time_embed_dim=2,
        projection_class_embeddings_input_dim=20,
    )


@pytest.mark.parametrize("model_name", ["sdxl-1.0", "sd3-medium", "flux-schnell", "flux-dev"])
def test_smaller_export_inputs_run_backbone_and_match_trt_profiles(model_name):
    backbone = _backbone(model_name).eval()
    flux = model_name.startswith("flux")
    kwargs, _, shapes = generate_dummy_kwargs_and_dynamic_axes_and_shapes(
        model_name,
        backbone,
        height=64,
        width=96,
        vae_scale_factor=4,
        max_sequence_length=64 if model_name != "sdxl-1.0" else None,
        opt_batch_size=2 if flux else 4,
    )
    input_name = "sample" if model_name == "sdxl-1.0" else "hidden_states"
    expected = (1, 96, 4) if flux else (2, 4, 16, 24)
    assert tuple(kwargs[input_name].shape) == expected
    if model_name != "sdxl-1.0":
        assert kwargs["encoder_hidden_states"].shape[1] == (64 if flux else 141)
    with torch.no_grad():
        output = backbone(**kwargs)
    assert output[0].shape == kwargs[input_name].shape
    profiles = _create_trt_dynamic_shapes(shapes)
    assert tuple(profiles["minShapes"][input_name]) == expected
    assert profiles["maxShapes"][input_name][0] == (2 if flux else 4)
    if flux:
        for name in ("timestep", "guidance"):
            if name in kwargs:
                assert list(kwargs[name].shape) == profiles["minShapes"][name] == [1]
                assert profiles["optShapes"][name] == profiles["maxShapes"][name] == [2]
        opt_kwargs = {
            name: value.expand(profiles["optShapes"][name]) if name in shapes else value
            for name, value in kwargs.items()
        }
        with torch.no_grad():
            output = backbone(**opt_kwargs)
        assert list(output[0].shape) == profiles["optShapes"][input_name]


@pytest.mark.parametrize("model_name", ["sdxl-1.0", "sd3-medium", "flux-schnell", "flux-dev"])
def test_export_input_defaults_remain_compatible(model_name):
    kwargs, _, shapes = generate_dummy_kwargs_and_dynamic_axes_and_shapes(
        model_name, _backbone(model_name)
    )
    input_name = "sample" if model_name == "sdxl-1.0" else "hidden_states"
    flux = model_name.startswith("flux")
    expected = (1, 4096, 4) if flux else (2, 4, 8, 8)
    assert tuple(kwargs[input_name].shape) == expected
    assert shapes[input_name]["opt"][0] == (1 if flux else 16)
    if model_name != "sdxl-1.0":
        assert kwargs["encoder_hidden_states"].shape[1] == (512 if flux else 333)
    if flux:
        for name in ("timestep", "guidance"):
            if name in kwargs:
                assert list(kwargs[name].shape) == [1]
                assert shapes[name] == {"min": [1], "opt": [1]}
