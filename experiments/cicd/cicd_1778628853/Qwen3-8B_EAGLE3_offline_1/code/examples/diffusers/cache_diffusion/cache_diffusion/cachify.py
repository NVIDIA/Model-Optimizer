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

import fnmatch
from contextlib import contextmanager

from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.unets.unet_3d_blocks import (
    CrossAttnDownBlockSpatioTemporal,
    CrossAttnUpBlockSpatioTemporal,
    DownBlockSpatioTemporal,
    UNetMidBlockSpatioTemporal,
    UpBlockSpatioTemporal,
)
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

from .module import CachedModule
from .utils import replace_module

CACHED_PIPE = {
    UNet2DConditionModel: (
        DownBlock2D,
        CrossAttnDownBlock2D,
        UNetMidBlock2DCrossAttn,
        CrossAttnUpBlock2D,
        UpBlock2D,
    ),
    PixArtTransformer2DModel: (BasicTransformerBlock),
    UNetSpatioTemporalConditionModel: (
        CrossAttnDownBlockSpatioTemporal,
        DownBlockSpatioTemporal,
        UpBlockSpatioTemporal,
        CrossAttnUpBlockSpatioTemporal,
        UNetMidBlockSpatioTemporal,
    ),
    SD3Transformer2DModel: (JointTransformerBlock),
}


def _apply_to_modules(model, action, modules=None, config_list=None):
    if hasattr(model, "use_trt_infer") and model.use_trt_infer:
        for key, module in model.engines.items():
            if isinstance(module, CachedModule):
                action(module)
            elif config_list:
                for config in config_list:
                    if _pass(key, config["wildcard_or_filter_func"]):
                        model.engines[key] = CachedModule(module, config["select_cache_step_func"])
    else:
        for name, module in model.named_modules():
            if isinstance(module, CachedModule):
                action(module)
            elif modules and config_list:
                for config in config_list:
                    if _pass(name, config["wildcard_or_filter_func"]) and isinstance(
                        module, modules
                    ):
                        replace_module(
                            model,
                            name,
                            CachedModule(module, config["select_cache_step_func"]),
                        )


def cachify(model, config_list, modules):
    def cache_action(module):
        pass  # No action needed, caching is handled in the loop itself

    _apply_to_modules(model, cache_action, modules, config_list)


def disable(pipe):
    model = get_model(pipe)
    _apply_to_modules(model, lambda module: module.disable_cache())


def enable(pipe):
    model = get_model(pipe)
    _apply_to_modules(model, lambda module: module.enable_cache())


def reset_status(pipe):
    model = get_model(pipe)
    _apply_to_modules(model, lambda module: setattr(module, "cur_step", 0))


def _pass(name, wildcard_or_filter_func):
    if isinstance(wildcard_or_filter_func, str):
        return fnmatch.fnmatch(name, wildcard_or_filter_func)
    elif callable(wildcard_or_filter_func):
        return wildcard_or_filter_func(name)
    else:
        raise NotImplementedError(f"Unsupported type {type(wildcard_or_filter_func)}")


def get_model(pipe):
    if hasattr(pipe, "unet"):
        return pipe.unet
    elif hasattr(pipe, "transformer"):
        return pipe.transformer
    else:
        raise KeyError


@contextmanager
def infer(pipe):
    try:
        yield pipe
    finally:
        reset_status(pipe)


def prepare(pipe, config_list):
    model = get_model(pipe)
    assert model.__class__ in CACHED_PIPE, f"{model.__class__} is not supported!"
    cachify(model, config_list, CACHED_PIPE[model.__class__])
