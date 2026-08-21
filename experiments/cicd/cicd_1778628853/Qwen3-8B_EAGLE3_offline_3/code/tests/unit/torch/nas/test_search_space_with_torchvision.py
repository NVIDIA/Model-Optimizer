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

import copy

import pytest
import torch
from _test_utils.torch.vision_models import get_tiny_mobilenet_and_input, get_tiny_resnet_and_input
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from modelopt.torch.nas.search_space import generate_search_space
from modelopt.torch.utils import random


@pytest.mark.parametrize(
    "get_model_and_input",
    [
        get_tiny_resnet_and_input,
        get_tiny_mobilenet_and_input,
        lambda: (MaskRCNNPredictor(3, 32, 10), (torch.randn(2, 3, 32, 32),), {}),
    ],
)
def test_forward(get_model_and_input) -> None:
    model, args, kwargs = get_model_and_input()

    # set a manual seed
    torch.manual_seed(1)

    # construct search space
    search_space = generate_search_space(model)

    # check that search space is non-trivial
    assert search_space.is_configurable()

    # check that max subnet is the same as original
    assert search_space.sample(max) == search_space.sample(random.original)

    # set deterministic seed
    random._set_deterministic_seed()

    for sample_func in [min, random.choice, max]:
        # sample configurations and do a forward pass
        search_space.sample(sample_func)
        out_ss = search_space.model(*args, **kwargs)

        # export configuration and do a forward pass
        subnet = copy.deepcopy(search_space).export()
        out_sub = subnet(*args, **kwargs)

        # compare outputs
        assert torch.allclose(out_sub, out_ss), "output did not match"

    def _getsubattr(module, target):
        module_path, _, attr_name = target.rpartition(".")
        submodule = module.get_submodule(module_path)
        return getattr(submodule, attr_name)

    # check that params/buffer in max subnet match the dynamic ones
    search_space.sample(max)
    model = search_space.model
    for name, param in model.named_parameters():
        assert param is _getsubattr(model, name)
    for name, buffer in model.named_buffers():
        assert buffer is _getsubattr(model, name)

    # check that params in min subnet do *not always* match the dynamic ones
    search_space.sample(min)
    any_params_different = any(
        param is _getsubattr(model, name) for name, param in model.named_parameters()
    )
    assert any_params_different, "Some parameters should be different!"
