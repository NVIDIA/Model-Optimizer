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

import pytest
import torch
from _test_utils.torch.diffusers_models import (
    create_tiny_unet_dir,
    df_modelopt_state_and_output_tester,
    get_tiny_unet,
    get_unet_dummy_inputs,
)
from _test_utils.torch.opt.utils import apply_mode_with_sampling
from diffusers import UNet2DConditionModel

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("model_cls", [UNet2DConditionModel])
@pytest.mark.parametrize("use_subfolder", [False, True], ids=["direct", "subfolder"])
def test_unet_save_restore(tmp_path, model_cls, use_subfolder):
    tiny_unet_dir = create_tiny_unet_dir(tmp_path)
    model_ref = model_cls.from_pretrained(tiny_unet_dir)
    # TODO: Add calibrate, compress mode to the test
    model_ref = apply_mode_with_sampling(
        model_ref, ["sparse_magnitude", "export_sparse", "quantize"]
    )
    component_dir = tmp_path / "checkpoint" / "unet"
    model_ref.save_pretrained(component_dir)

    load_path = component_dir.parent if use_subfolder else component_dir
    load_kwargs = {"subfolder": "unet"} if use_subfolder else {}
    model_test = model_cls.from_pretrained(load_path, local_files_only=True, **load_kwargs)
    df_modelopt_state_and_output_tester(model_ref, model_test)


@pytest.mark.parametrize("save_modelopt_state", [True, False])
@torch.no_grad()
def test_plain_save_replaces_quantized_checkpoint(tmp_path, save_modelopt_state):
    quantized = get_tiny_unet().eval()
    inputs = get_unet_dummy_inputs(quantized)
    mtq.quantize(quantized, mtq.FP8_DEFAULT_CFG, lambda m: m(**inputs))
    quantized.save_pretrained(tmp_path)
    assert (tmp_path / "modelopt_state.pth").is_file()

    plain = get_tiny_unet().eval()
    original_state = (tmp_path / "modelopt_state.pth").read_bytes()
    with pytest.raises(ValueError):
        plain.save_pretrained(tmp_path, max_shard_size="invalid")
    assert (tmp_path / "modelopt_state.pth").read_bytes() == original_state

    plain.save_pretrained(tmp_path / "fresh")
    reference = UNet2DConditionModel.from_pretrained(tmp_path / "fresh", local_files_only=True)
    plain.save_pretrained(tmp_path, save_modelopt_state=save_modelopt_state)
    restored = UNet2DConditionModel.from_pretrained(tmp_path, local_files_only=True)

    assert not any(isinstance(module, TensorQuantizer) for module in restored.modules())
    assert not (tmp_path / "modelopt_state.pth").exists()
    torch.testing.assert_close(restored.state_dict(), plain.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        restored(**inputs).sample, reference(**inputs).sample, rtol=0, atol=0
    )
