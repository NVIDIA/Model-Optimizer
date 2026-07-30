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

"""Quantization support for timm modules."""

import torch.nn as nn
from timm.models.resnet import BasicBlock, Bottleneck, ResNet

from ..algorithms import AutoQuantizeGradientSearcher, AutoQuantizeKLDivSearcher
from ..nn import QuantModule, QuantModuleRegistry, TensorQuantizer
from ..nn.modules.quant_conv import _QuantConv2d
from .custom import CUSTOM_MODEL_PLUGINS, CUSTOM_POST_CONVERSION_PLUGINS


# Forward overrides give marker subclasses distinct registry entries without changing computation.
class _ResNetInputConv2d(nn.Conv2d):
    def forward(self, input):
        return super().forward(input)


class _ResNetOutputConv2d(nn.Conv2d):
    def forward(self, input):
        return super().forward(input)


class _ResNetProjectionOutputConv2d(_ResNetOutputConv2d):
    def forward(self, input):
        return super().forward(input)


class _ResNetFinalOutputConv2d(_ResNetOutputConv2d):
    def forward(self, input):
        return super().forward(input)


class _ResNetFinalProjectionOutputConv2d(_ResNetProjectionOutputConv2d):
    def forward(self, input):
        return super().forward(input)


class _ResNetStemConv2d(nn.Conv2d):
    def forward(self, input):
        return super().forward(input)


class _ResNetBasicBlock(BasicBlock):
    def forward(self, input):
        return super().forward(input)


class _ResNetBottleneck(Bottleneck):
    def forward(self, input):
        return super().forward(input)


def is_resnet_quantization_supported(model):
    """Return whether the model uses a supported timm ResNet block implementation."""
    if not isinstance(model, ResNet):
        return False
    blocks = [module for module in model.modules() if isinstance(module, (BasicBlock, Bottleneck))]
    supported_types = (BasicBlock, Bottleneck, _ResNetBasicBlock, _ResNetBottleneck)
    return bool(blocks) and all(
        type(block) in supported_types or getattr(block, "original_cls", None) in supported_types
        for block in blocks
    )


@QuantModuleRegistry.register(
    {
        _ResNetBasicBlock: "timm.ResNetBasicBlock",
        _ResNetBottleneck: "timm.ResNetBottleneck",
    }
)
class _QuantResNetBlock(QuantModule):
    def _setup(self):
        pass

    def forward(self, input):
        output_conv = self.conv3 if isinstance(self, Bottleneck) else self.conv2
        return super().forward(output_conv.block_input_activation_quantizer(input))


def _register_disabled_quantizer(module, name):
    quantizer = TensorQuantizer()
    quantizer.disable()
    module._register_temp_attribute(name, quantizer)


@QuantModuleRegistry.register({_ResNetOutputConv2d: "timm.ResNetOutputConv2d"})
class _QuantResNetOutputConv2d(_QuantConv2d):
    _auto_quantize_quantizer_attrs = (
        "input_quantizer",
        "weight_quantizer",
        "output_quantizer",
        "block_input_activation_quantizer",
    )

    def _setup(self):
        super()._setup()
        _register_disabled_quantizer(self, "block_input_activation_quantizer")


@QuantModuleRegistry.register({_ResNetProjectionOutputConv2d: "timm.ResNetProjectionOutputConv2d"})
class _QuantResNetProjectionOutputConv2d(_QuantResNetOutputConv2d):
    _auto_quantize_quantizer_attrs = (
        *_QuantResNetOutputConv2d._auto_quantize_quantizer_attrs,
        "residual_quantizer",
    )

    def _setup(self):
        super()._setup()
        _register_disabled_quantizer(self, "residual_quantizer")


@QuantModuleRegistry.register({_ResNetFinalOutputConv2d: "timm.ResNetFinalOutputConv2d"})
class _QuantResNetFinalOutputConv2d(_QuantResNetOutputConv2d):
    _auto_quantize_quantizer_attrs = (
        *_QuantResNetOutputConv2d._auto_quantize_quantizer_attrs,
        "model_output_activation_quantizer",
    )

    def _setup(self):
        super()._setup()
        _register_disabled_quantizer(self, "model_output_activation_quantizer")


@QuantModuleRegistry.register(
    {_ResNetFinalProjectionOutputConv2d: "timm.ResNetFinalProjectionOutputConv2d"}
)
class _QuantResNetFinalProjectionOutputConv2d(_QuantResNetProjectionOutputConv2d):
    _auto_quantize_quantizer_attrs = (
        *_QuantResNetProjectionOutputConv2d._auto_quantize_quantizer_attrs,
        "model_output_activation_quantizer",
    )

    def _setup(self):
        super()._setup()
        _register_disabled_quantizer(self, "model_output_activation_quantizer")


QuantModuleRegistry.register({_ResNetInputConv2d: "timm.ResNetInputConv2d"})(_QuantConv2d)


@QuantModuleRegistry.register({_ResNetStemConv2d: "timm.ResNetStemConv2d"})
class _QuantResNetStemConv2d(_QuantConv2d):
    _auto_quantize_disabled = True


def _mark_resnet_convs(model):
    for resnet in (module for module in model.modules() if isinstance(module, ResNet)):
        if not is_resnet_quantization_supported(resnet):
            continue
        stem_conv = next(
            (module for module in resnet.conv1.modules() if type(module) is nn.Conv2d), None
        )
        if stem_conv is not None:
            stem_conv.__class__ = _ResNetStemConv2d
        blocks = [module for module in resnet.modules() if type(module) in (BasicBlock, Bottleneck)]
        for index, block in enumerate(blocks):
            is_bottleneck = isinstance(block, Bottleneck)
            if type(block.conv1) is nn.Conv2d:
                block.conv1.__class__ = _ResNetInputConv2d
            if block.downsample is not None:
                downsample_ops = list(block.downsample.children()) or [block.downsample]
                for module in downsample_ops:
                    if type(module) is nn.Identity:
                        continue
                    if type(module) is nn.Conv2d:
                        module.__class__ = _ResNetInputConv2d
                    break
            output_conv = block.conv3 if is_bottleneck else block.conv2
            if type(output_conv) is nn.Conv2d:
                is_projection = block.downsample is not None
                is_final = index == len(blocks) - 1
                if is_final:
                    output_conv.__class__ = (
                        _ResNetFinalProjectionOutputConv2d
                        if is_projection
                        else _ResNetFinalOutputConv2d
                    )
                else:
                    output_conv.__class__ = (
                        _ResNetProjectionOutputConv2d if is_projection else _ResNetOutputConv2d
                    )
            block.__class__ = _ResNetBottleneck if is_bottleneck else _ResNetBasicBlock


CUSTOM_MODEL_PLUGINS.add(_mark_resnet_convs)


def _register_resnet_quantizer_hooks(model):
    for resnet in (module for module in model.modules() if isinstance(module, ResNet)):
        blocks = [
            module for module in resnet.modules() if isinstance(module, (BasicBlock, Bottleneck))
        ]
        for block in blocks:
            if block.downsample is None:
                continue
            output_conv = block.conv3 if isinstance(block, Bottleneck) else block.conv2
            if not hasattr(output_conv, "residual_quantizer"):
                continue
            handle = block.downsample.register_forward_hook(
                lambda _module, _inputs, output, conv=output_conv: conv.residual_quantizer(output)
            )
            output_conv._register_temp_attribute(
                "_residual_quantizer_hook",
                handle,
                del_hook=lambda module, name: getattr(module, name).remove(),
            )

        if not blocks:
            continue
        final_output_conv = (
            blocks[-1].conv3 if isinstance(blocks[-1], Bottleneck) else blocks[-1].conv2
        )
        if not hasattr(final_output_conv, "model_output_activation_quantizer"):
            continue
        handle = resnet.global_pool.register_forward_pre_hook(
            lambda _module, inputs, conv=final_output_conv: (
                conv.model_output_activation_quantizer(inputs[0]),
                *inputs[1:],
            )
        )
        final_output_conv._register_temp_attribute(
            "_model_output_quantizer_hook",
            handle,
            del_hook=lambda module, name: getattr(module, name).remove(),
        )


CUSTOM_POST_CONVERSION_PLUGINS.add(_register_resnet_quantizer_hooks)


def _resnet_block_context(model, name):
    parts = name.split(".")
    for index in range(len(parts) - 1, -1, -1):
        block_name = ".".join(parts[:index])
        block = model.get_submodule(block_name)
        if not isinstance(block, (BasicBlock, Bottleneck)):
            continue
        relative_name = ".".join(parts[index:])
        if relative_name not in ("conv1", "conv2", "conv3") and not relative_name.startswith(
            "downsample."
        ):
            return None
        output_conv = block.conv3 if isinstance(block, Bottleneck) else block.conv2
        if not hasattr(output_conv, "block_input_activation_quantizer"):
            return None
        return block_name, block, output_conv
    return None


def _resnet_block_group(model, name):
    context = _resnet_block_context(model, name)
    return context[0] if context is not None else None


def _resnet_block_score(model, name):
    context = _resnet_block_context(model, name)
    if context is None:
        return None
    block_name, _, output_conv = context
    if not hasattr(output_conv, "model_output_activation_quantizer"):
        return block_name
    parts = block_name.split(".")
    for index in range(len(parts), -1, -1):
        resnet_name = ".".join(parts[:index])
        if isinstance(model.get_submodule(resnet_name), ResNet):
            return ".".join(filter(None, (resnet_name, "global_pool")))
    return block_name


AutoQuantizeGradientSearcher.quant_grouping_rules.append(_resnet_block_group)
AutoQuantizeGradientSearcher.score_module_rules.append(_resnet_block_score)  # type: ignore[arg-type]
AutoQuantizeKLDivSearcher.score_module_rules.append(_resnet_block_score)
