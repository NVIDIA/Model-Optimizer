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

"""Quantizer ownership for AutoModel's native transposed expert projections."""

import inspect
from contextlib import contextmanager
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from ..nn import (
    GroupedQuantizer,
    QuantModule,
    QuantModuleRegistry,
    SequentialQuantizer,
    TensorQuantizer,
)
from ..utils.layerwise_calib import LayerActivationCollector
from .custom import CUSTOM_MODEL_PLUGINS

__all__ = []

_PROJECTIONS = ("gate_and_up_projs", "down_projs")


def _local_experts(weight):
    if not isinstance(weight, DTensor):
        return weight, 0
    shape, offset = compute_local_shape_and_global_offset(
        weight.shape, weight.device_mesh, weight.placements
    )
    if tuple(shape[1:]) != tuple(weight.shape[1:]):
        raise ValueError("Unshard the owning AutoModel FSDP unit before accessing expert matrices.")
    return weight.to_local(), offset[0]


def _quantized_experts(name, module, weight):
    local, offset = _local_experts(weight)
    quantizers = getattr(module, name + "_weight_quantizers")
    from ..repair_formats import grouped_native_repair

    active_quantizers = list(quantizers[offset : offset + local.shape[0]])
    grouped = grouped_native_repair(active_quantizers, local.dtype)
    if grouped is None:
        matrices = [
            quantizer(matrix.transpose(-1, -2)).transpose(-1, -2)
            for quantizer, matrix in zip(active_quantizers, local.unbind(0), strict=True)
        ]
        result = torch.stack(matrices)
    else:
        result = grouped.transpose(-1, -2).contiguous()
    if isinstance(weight, DTensor):
        return DTensor.from_local(
            result,
            weight.device_mesh,
            weight.placements,
            run_check=False,
            shape=weight.shape,
            stride=weight.stride(),
        )
    return result


class _QuantAutoModelExperts(QuantModule):
    """Preserve AutoModel's routing, dispatch, combine and native expert kernels."""

    def _setup(self):
        for name in _PROJECTIONS:
            weight = self._parameters[name]
            setattr(
                self,
                name + "_weight_quantizers",
                nn.ModuleList([TensorQuantizer() for _ in range(weight.shape[0])]),
            )
            self._register_dynamic_attribute(name, partial(_quantized_experts, name))
        self.gate_and_up_projs_input_quantizer = TensorQuantizer()
        self.down_projs_input_quantizer = TensorQuantizer()
        self._register_temp_attribute("_current_expert_idx", None)
        self._register_temp_attribute("_grouped_offs", None)

    @property
    def num_experts(self):
        return _local_experts(self._parameters[_PROJECTIONS[0]])[0].shape[0]

    def _local_experts(self, weight):
        return _local_experts(weight)[0]

    def iter_expert_projections(self):
        for name in _PROJECTIONS:
            local, offset = _local_experts(self._parameters[name])
            quantizers = getattr(self, name + "_weight_quantizers")[
                offset : offset + local.shape[0]
            ]
            yield (
                name,
                local.transpose(-1, -2),
                quantizers,
                getattr(self, name + "_input_quantizer"),
            )

    def iter_weights_for_calibration(self):
        for name in _PROJECTIONS:
            local, offset = _local_experts(self._parameters[name])
            quantizers = getattr(self, name + "_weight_quantizers")
            for index, matrix in enumerate(local.unbind(0)):
                yield matrix.transpose(-1, -2), quantizers[index + offset]

    def iter_weight_quantizers_for_calibration(self):
        for name in _PROJECTIONS:
            weight = self._parameters[name]
            if isinstance(weight, DTensor):
                shape, offset = compute_local_shape_and_global_offset(
                    weight.shape, weight.device_mesh, weight.placements
                )
                start, count = offset[0], shape[0]
            else:
                start, count = 0, weight.shape[0]
            yield from getattr(self, name + "_weight_quantizers")[start : start + count]

    def forward(self, x, *args, **kwargs):
        # Calibration imports this plugin while registering quantizers; defer
        # the collector type until forward to avoid a circular module import.
        from ..model_calib import _WeightInputHook

        input_quantizer = self.gate_and_up_projs_input_quantizer
        capture = any(
            isinstance(hook, _WeightInputHook)
            for quantizer in (input_quantizer, self.down_projs_input_quantizer)
            for hooks in (quantizer._forward_pre_hooks, quantizer._forward_hooks)
            for hook in hooks.values()
        )
        rows = []
        cursor = 0
        captured_x = None
        loop = None
        had_loop_override = "_forward_loop" in self.__dict__
        if capture:
            if getattr(self, "use_torch_mm", True) or not hasattr(self, "_forward_loop"):
                raise ValueError(
                    "AutoModel expert-input calibration currently requires the native eager expert backend."
                )
            loop = self._forward_loop
            signature = inspect.signature(loop)

            def capture_loop(x, weights, indices, token_mask, *projections, **options):
                nonlocal captured_x, rows, cursor
                # Native EP gathers before this boundary. Capture the physical
                # GEMM input, including tokens routed from other EP ranks.
                captured_x, rows, cursor = x, [], 0
                arguments = signature.bind(
                    x, weights, indices, token_mask, *projections, **options
                ).arguments
                start, end = arguments["experts_start_idx"], arguments["experts_end_idx"]
                for index in range(start, end):
                    tokens = torch.where((indices == index) & token_mask.unsqueeze(-1))[0]
                    if tokens.numel():
                        rows.append((index - start, tokens))
                return loop(x, weights, indices, token_mask, *projections, **options)

            object.__setattr__(self, "_forward_loop", capture_loop)
        self._current_expert_idx = None
        name = (
            "expert_activation_grouped"
            if hasattr(self, "expert_activation_grouped")
            else "expert_activation"
        )
        activation = getattr(self, name)

        def quantized_activation(*args, **kwargs):
            nonlocal cursor
            if capture and cursor < len(rows):
                self._current_expert_idx, tokens = rows[cursor]
                assert captured_x is not None
                routed = captured_x.index_select(0, tokens)
                # Already quantized before EP dispatch. Notify only actual
                # collectors, without quantizing twice or replaying user hooks.
                for hook in tuple(input_quantizer._forward_pre_hooks.values()):
                    if isinstance(hook, _WeightInputHook):
                        hook(input_quantizer, (routed,))
                for hook in tuple(input_quantizer._forward_hooks.values()):
                    if isinstance(hook, _WeightInputHook):
                        hook(input_quantizer, (routed,), routed)
                cursor += 1
            else:
                self._current_expert_idx = None
            return self.down_projs_input_quantizer(activation(*args, **kwargs))

        object.__setattr__(self, name, quantized_activation)
        try:
            return super().forward(input_quantizer(x), *args, **kwargs)
        finally:
            object.__setattr__(self, name, activation)
            if loop is not None:
                if had_loop_override:
                    object.__setattr__(self, "_forward_loop", loop)
                else:
                    object.__delattr__(self, "_forward_loop")
            self._current_expert_idx = None


class _QuantAutoModelGDN(QuantModule):
    """Honor convolution quantizers around native Qwen's fused GDN kernels."""

    def _setup(self):
        pass  # Existing projections own their quantization parameters.

    def iter_weights_for_calibration(self):
        return iter(())

    def iter_weight_quantizers_for_calibration(self):
        return iter(())

    def validate_local_repair(self, config=None):
        output = getattr(self.conv1d, "output_quantizer", None)
        if output is not None and (
            output.is_enabled or output.pre_quant_scale is not None or output.rotate_is_enabled
        ):
            raise ValueError(
                "Fused GDN convolution requires an unfused adapter for output quantization."
            )
        mesh = getattr(self, "_cp_mesh", None)
        activation = getattr(self.conv1d, "input_quantizer", None)
        if mesh is not None and mesh.size() > 1 and activation is not None:
            if activation.is_enabled and activation._dynamic:
                raise ValueError(
                    "Dynamic GDN convolution activation quantization needs a CP-aware threshold adapter."
                )

    def forward(self, *args, **kwargs):
        if not hasattr(self.conv1d, "weight_quantizer"):
            return super().forward(*args, **kwargs)
        self.validate_local_repair()
        previous = self.conv1d._enable_weight_quantization
        self.conv1d._enable_weight_quantization = True
        handle = None
        quantizer = self.conv1d.input_quantizer
        mesh = getattr(self, "_cp_mesh", None)
        functional = getattr(self, "causal_conv1d_fn", None) is not None or (
            mesh is not None and mesh.size() > 1
        )
        if functional and (
            quantizer.is_enabled
            or quantizer.pre_quant_scale is not None
            or quantizer.rotate_is_enabled
        ):
            # The projection is already unpadded for packed input. CP gathers
            # these same calibrated values before its convolution halo exchange.
            def quantize_conv_input(module, inputs, output):
                # Causal-conv's packed seq_idx requires channel-last storage.
                # Quantization may allocate a channels-first contiguous tensor.
                return quantizer(output.transpose(-1, -2)).transpose(-1, -2).contiguous()

            handle = self.in_proj_qkv.register_forward_hook(quantize_conv_input)
        try:
            return super().forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
            self.conv1d._enable_weight_quantization = previous


class _QuantAutoModelMamba(QuantModule):
    """Keep native Mamba kernels while honoring directly consumed projections."""

    def _setup(self):
        pass  # Quantization parameters belong to the existing child modules.

    def iter_weights_for_calibration(self):
        return iter(())

    def iter_weight_quantizers_for_calibration(self):
        return iter(())

    def validate_local_repair(self, config=None):
        output = getattr(self.conv1d, "output_quantizer", None)
        if output is not None and (
            output.is_enabled or output.pre_quant_scale is not None or output.rotate_is_enabled
        ):
            raise ValueError(
                "Native Mamba's fused convolution cannot expose an output quantizer; "
                "use its input quantizer or an unfused convolution adapter."
            )

    def forward(self, *args, **kwargs):
        if not hasattr(self.conv1d, "weight_quantizer"):
            return super().forward(*args, **kwargs)
        from mamba_ssm.ops.triton import ssd_combined

        from ..utils.core_utils import replace_function

        self.validate_local_repair()
        # The native fused kernel reads conv1d.weight directly, outside the
        # QuantConv forward scope. Preserve an enclosing scope when nesting.
        previous = self.conv1d._enable_weight_quantization
        self.conv1d._enable_weight_quantization = True
        handle = None
        quantizer = self.conv1d.input_quantizer
        if (
            quantizer.is_enabled
            or quantizer.pre_quant_scale is not None
            or quantizer.rotate_is_enabled
        ):

            def quantize_conv_input(module, inputs, output):
                gate, convolution, dt = output.split(
                    [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                )
                convolution = quantizer(convolution.transpose(-1, -2)).transpose(-1, -2)
                return torch.cat((gate, convolution, dt), dim=-1)

            handle = self.in_proj.register_forward_hook(quantize_conv_input)
        fused = ssd_combined.mamba_split_conv1d_scan_combined

        def with_projection(*inputs, **options):
            if options.get("outproj_weight") is None or not hasattr(
                self.out_proj, "weight_quantizer"
            ):
                return fused(*inputs, **options)
            # Keep convolution/SSD/gated normalization fused, then call the
            # actual quantized projection so activation quantizers and PTQ
            # collection hooks observe exactly the tensor they will deploy.
            options = {**options, "outproj_weight": None, "outproj_bias": None}
            output = fused(*inputs, **options)
            if isinstance(output, tuple):
                return (self.out_proj(output[0]), *output[1:])
            return self.out_proj(output)

        try:
            with replace_function(
                ssd_combined,
                "mamba_split_conv1d_scan_combined",
                with_projection,
                og_func_cache_name="_modelopt_mamba_projection",
            ):
                return super().forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
            self.conv1d._enable_weight_quantization = previous


def _register_automodel_experts(model):
    for module in model.modules():
        cls = type(module)
        if cls in QuantModuleRegistry:
            continue
        for base in cls.__mro__:
            if (
                base.__module__ == "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn"
                and base.__name__ == "CPAwareGatedDeltaNet"
            ):
                if base not in QuantModuleRegistry:
                    QuantModuleRegistry.register({base: "automodel.CPAwareGatedDeltaNet"})(
                        _QuantAutoModelGDN
                    )
                if cls not in QuantModuleRegistry:
                    QuantModuleRegistry.register_subclass(cls, "automodel." + cls.__name__)
                break
            if (
                base.__module__ == "nemo_automodel.components.models.nemotron_v3.layers"
                and base.__name__ == "NemotronV3Mamba2Mixer"
            ):
                if base not in QuantModuleRegistry:
                    QuantModuleRegistry.register({base: "automodel.NemotronV3Mamba2Mixer"})(
                        _QuantAutoModelMamba
                    )
                if cls not in QuantModuleRegistry:
                    QuantModuleRegistry.register_subclass(cls, "automodel." + cls.__name__)
                break
            if base.__module__ != "nemo_automodel.components.moe.experts" or base.__name__ not in (
                "GroupedExperts",
                "GroupedExpertsDeepEP",
            ):
                continue
            if base not in QuantModuleRegistry:
                QuantModuleRegistry.register({base: "automodel." + base.__name__})(
                    _QuantAutoModelExperts
                )
            if cls not in QuantModuleRegistry:
                QuantModuleRegistry.register_subclass(cls, "automodel." + cls.__name__)
            break


CUSTOM_MODEL_PLUGINS.add(_register_automodel_experts)


@contextmanager
def _quantizers_detached(model):
    """Keep quantizers out of native structural discovery and weight sharding."""
    removed = []
    for module in tuple(model.modules()):
        if not isinstance(module, QuantModule):
            continue
        for name, child in tuple(module._modules.items()):
            if isinstance(child, (TensorQuantizer, SequentialQuantizer, GroupedQuantizer)) or (
                isinstance(child, nn.ModuleList)
                and len(child) > 0
                and all(isinstance(q, TensorQuantizer) for q in child)
            ):
                removed.append((module, name, child))
                del module._modules[name]
    try:
        yield
    finally:
        for module, name, child in removed:
            module.add_module(name, child)


class _DecoderLayerView:
    """Index AutoModel's named container without registering a second owner."""

    def __init__(self, container):
        self.container = container
        self.keys = tuple(container.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.container[self.keys[index]]

    def __setitem__(self, index, layer):
        self.container[self.keys[index]] = layer


def _is_native_automodel(model):
    return any(
        cls.__module__.startswith("nemo_automodel.components.models.")
        for cls in type(model).__mro__
    )


def _native_decoder_layers(model):
    from nemo_automodel.components.distributed.parallelizer import get_model_layer_groups

    with _quantizers_detached(model):
        groups = get_model_layer_groups(model)
    layers = groups.get("language")
    if layers is None and len(groups) == 1:
        layers = next(iter(groups.values()))
    if not layers:
        return None
    # Use the same model-owned discovery as FSDP. The collector swaps skip
    # placeholders in place, so returning a copied ModuleList is incorrect.
    for module in model.modules():
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            children = list(module.values()) if isinstance(module, nn.ModuleDict) else list(module)
            if len(children) == len(layers) and all(a is b for a, b in zip(children, layers)):
                return _DecoderLayerView(module) if isinstance(module, nn.ModuleDict) else module
    return None


def _requires_prefix_replay(model):
    return (
        _is_native_automodel(model)
        and getattr(model.config, "model_type", None) == "mistral_large4"
    )


LayerActivationCollector.register_decoder_layer_support(
    _is_native_automodel,
    _native_decoder_layers,
    requires_prefix_replay=_requires_prefix_replay,
)
