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

"""A collection of common models for testing."""

from collections.abc import Callable
from typing import Any

import timm
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, ResNet


def process_model_and_inputs(
    model: nn.Module,
    args: tuple[torch.Tensor],
    kwargs: dict,
    on_gpu: bool = False,
):
    model.eval()

    # move to GPU if desired
    model = model.cuda() if on_gpu else model.cpu()
    if on_gpu:
        from modelopt.torch.utils import torch_to

        args = torch_to(args, "cuda")
        kwargs = torch_to(kwargs, "cuda")

    return model, args, kwargs


def get_tiny_resnet_and_input(on_gpu: bool = False):
    model = ResNet(block=BasicBlock, layers=[2, 1, 0, 0], num_classes=10)
    args = (torch.randn(1, 3, 56, 56),)
    return process_model_and_inputs(model, args, {}, on_gpu)


def get_tiny_mobilenet_and_input(on_gpu: bool = False):
    model = torchvision.models.mobilenet_v3_small(progress=False, num_classes=10)
    args = (torch.randn(1, 3, 56, 56),)
    return process_model_and_inputs(model, args, {}, on_gpu)


class TinyMobileNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = get_tiny_mobilenet_and_input()[0].features

    def forward(self, x):
        # only run forward on features, not classification head (no dropout and more stable)
        return self.backbone(x)


def _create_timm_fn(name):
    def get_model_and_input(on_gpu: bool = False):
        model = timm.create_model(name)
        return process_model_and_inputs(model, (torch.randn(1, 3, 224, 224),), {}, on_gpu)

    return get_model_and_input


def _create_torchvision_fn(name):
    def get_model_and_input(on_gpu: bool = False):
        if name == "tiny_resnet":
            model, args, kwargs = get_tiny_resnet_and_input()
        elif name == "tiny_mobilenet":
            model, args, kwargs = get_tiny_mobilenet_and_input()
        else:
            raise ValueError(f"Unknown model: {name}")
        return process_model_and_inputs(model, args, kwargs, on_gpu)

    return get_model_and_input


def _create_torchvision_segmentation_fn(name):
    def get_model_and_input(on_gpu: bool = False):
        model = getattr(torchvision.models.segmentation, name)(progress=False)
        return process_model_and_inputs(model, (torch.randn(1, 3, 32, 32),), {}, on_gpu)

    return get_model_and_input


def _create_unet_fn(name):
    def get_model_and_input(on_gpu: bool = False):
        # skip_validation to avoid `HTTP Error 403: rate limit exceeded
        model = torch.hub.load("milesial/Pytorch-UNet", name, skip_validation=True)
        return process_model_and_inputs(model, (torch.randn(1, 3, 40, 40),), {}, on_gpu)

    return get_model_and_input


MODELS = {
    "timm": (
        [
            # "hrnet_w18_small",
            # "cspresnet50",
            # "dpn68",
            "vit_tiny_patch16_224",
            # "vovnet39a",
            # "dm_nfnet_f0",
            "efficientnet_b0",
        ],
        _create_timm_fn,
    ),
    "torchvision": (
        [
            "tiny_resnet",
            "tiny_mobilenet",
        ],
        _create_torchvision_fn,
    ),
    "torchvision_segmentation": (
        [
            # "deeplabv3_resnet50",
            "fcn_resnet50",
            # "lraspp_mobilenet_v3_large",
        ],
        _create_torchvision_segmentation_fn,
    ),
    "unet": (
        ["unet_carvana"],
        _create_unet_fn,
    ),
}


def get_vision_models() -> dict[str, Callable[[bool], tuple[nn.Module, Any, Any]]]:
    """Returns a dict of benchmark model name to a function that returns the model, args, kwargs."""
    return {f"{repo}/{name}": fn(name) for repo, (names, fn) in MODELS.items() for name in names}
