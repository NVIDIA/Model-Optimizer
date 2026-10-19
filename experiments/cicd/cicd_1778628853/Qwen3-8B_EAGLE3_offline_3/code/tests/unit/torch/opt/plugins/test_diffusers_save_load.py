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
from _test_utils.torch.diffusers_models import (
    create_tiny_unet_dir,
    df_modelopt_state_and_output_tester,
)
from _test_utils.torch.opt.utils import apply_mode_with_sampling
from diffusers import UNet2DConditionModel


@pytest.mark.parametrize("model_cls", [UNet2DConditionModel])
def test_unet_save_restore(tmp_path, model_cls):
    tiny_unet_dir = create_tiny_unet_dir(tmp_path)
    model_ref = model_cls.from_pretrained(tiny_unet_dir)
    # TODO: Add calibrate, compress mode to the test
    model_ref = apply_mode_with_sampling(
        model_ref, ["sparse_magnitude", "export_sparse", "quantize"]
    )
    model_ref.save_pretrained(tiny_unet_dir / "modelopt_model")

    model_test = model_cls.from_pretrained(tiny_unet_dir / "modelopt_model")
    df_modelopt_state_and_output_tester(model_ref, model_test)
