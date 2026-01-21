# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import UNet2DConditionModel

try:
    from diffusers.models.transformers import DiTTransformer2DModel, FluxTransformer2DModel
except Exception:  # pragma: no cover - optional diffusers models
    DiTTransformer2DModel = None
    FluxTransformer2DModel = None

import modelopt.torch.opt as mto


def get_tiny_unet(**config_kwargs) -> UNet2DConditionModel:
    """Create a tiny UNet2DConditionModel for testing."""
    kwargs = {
        "sample_size": 8,
        "in_channels": 2,
        "out_channels": 2,
        "down_block_types": ("DownBlock2D",),
        "up_block_types": ("UpBlock2D",),
        "block_out_channels": (2,),
        "layers_per_block": 1,
        "cross_attention_dim": 2,
        "attention_head_dim": 1,
        "norm_num_groups": 1,
        "mid_block_type": None,
    }
    kwargs.update(**config_kwargs)
    tiny_unet = UNet2DConditionModel(**kwargs)

    return tiny_unet


def get_tiny_dit(**config_kwargs):
    """Create a tiny DiTTransformer2DModel for testing."""
    if DiTTransformer2DModel is None:
        pytest.skip("DiTTransformer2DModel is not available in this diffusers version.")

    kwargs = {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 2,
        "out_channels": 2,
        "num_layers": 1,
        "norm_num_groups": 1,
        "sample_size": 8,
        "patch_size": 2,
        "num_embeds_ada_norm": 10,
    }
    kwargs.update(**config_kwargs)
    return DiTTransformer2DModel(**kwargs)


def get_tiny_flux(**config_kwargs):
    """Create a tiny FluxTransformer2DModel for testing."""
    if FluxTransformer2DModel is None:
        pytest.skip("FluxTransformer2DModel is not available in this diffusers version.")

    kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 8,
        "pooled_projection_dim": 8,
        "guidance_embeds": False,
        "axes_dims_rope": (2, 2, 4),
    }
    kwargs.update(**config_kwargs)
    return FluxTransformer2DModel(**kwargs)


def create_tiny_unet_dir(tmp_path: Path, **config_kwargs) -> Path:
    """Create and save a tiny UNet model to a directory."""
    tiny_unet = get_tiny_unet(**config_kwargs)
    tiny_unet.save_pretrained(tmp_path / "tiny_unet")
    return tmp_path / "tiny_unet"


def get_unet_dummy_inputs(model: UNet2DConditionModel, batch_size: int = 1):
    """Generate dummy inputs for testing UNet models."""
    latents = torch.randn(
        batch_size, model.config.in_channels, model.config.sample_size, model.config.sample_size
    )
    timestep = torch.tensor([0])
    encoder_hidden_states = torch.randn(batch_size, 1, model.config.cross_attention_dim)

    return {"sample": latents, "timestep": timestep, "encoder_hidden_states": encoder_hidden_states}


def df_output_tester(model_ref, model_test):
    """Test if two diffusers models produce the same output."""
    inputs = get_unet_dummy_inputs(model_ref)
    model_ref.eval()
    model_test.eval()

    with torch.no_grad():
        output_ref = model_ref(**inputs).sample
        output_test = model_test(**inputs).sample

    assert torch.allclose(output_ref, output_test)


def df_modelopt_state_and_output_tester(model_ref, model_test):
    """Test if two diffusers models have the same modelopt state and outputs."""
    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)
    assert model_ref_state == model_test_state

    df_output_tester(model_ref, model_test)
