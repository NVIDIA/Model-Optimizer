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

"""Base class for quantization modules."""

import contextlib
import warnings
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls
from modelopt.torch.utils.distributed import ParallelState

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from ...utils import is_torch_export_mode
from .tensor_quantizer import SequentialQuantizer, TensorQuantizer

__all__ = [
    "QuantInputBase",
    "QuantLinearConvBase",
    "QuantModule",
    "QuantModuleRegistry",
]


@dataclass
class _FoldWeightState:
    """State needed to undo one ``_fold_weight_quantizer`` call."""

    quantizer: TensorQuantizer
    weights: list[torch.Tensor]
    original_weights: list[torch.Tensor]
    weight_keys: set[tuple]
    disabled: bool
    rotate: Any
    enable_pre_quant_scale: bool
    input_dtype: torch.dtype | None

    @classmethod
    def capture(cls, quantizer: TensorQuantizer) -> "_FoldWeightState":
        return cls(
            quantizer=quantizer,
            weights=[],
            original_weights=[],
            weight_keys=set(),
            disabled=quantizer._disabled,
            rotate=quantizer._rotate,
            enable_pre_quant_scale=quantizer._enable_pre_quant_scale,
            input_dtype=quantizer._input_dtype,
        )

    def add_weights(self, weights: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Snapshot and return weight views not already recorded for this quantizer."""
        new_weights = []
        for weight in weights:
            key = _tensor_view_key(weight)
            if key in self.weight_keys:
                continue
            self.weight_keys.add(key)
            self.weights.append(weight)
            self.original_weights.append(weight.detach().clone())
            new_weights.append(weight)
        return tuple(new_weights)

    def prepare_quantizer(self):
        """Restore the quantizer settings needed to fold another associated weight."""
        self.quantizer._disabled = self.disabled
        self.quantizer._rotate = self.rotate
        self.quantizer._enable_pre_quant_scale = self.enable_pre_quant_scale
        self.quantizer._input_dtype = self.input_dtype

    @torch.no_grad()
    def restore(self):
        """Restore weights and quantizer runtime state in place."""
        for weight, original_weight in zip(self.weights, self.original_weights):
            weight.data.copy_(original_weight)
        self.quantizer._disabled = self.disabled
        self.quantizer._rotate = self.rotate
        self.quantizer._enable_pre_quant_scale = self.enable_pre_quant_scale
        self.quantizer._input_dtype = self.input_dtype


@dataclass
class _TemporaryFoldWeightContext:
    """State shared by module-specific folds in one temporary transaction."""

    states: list[_FoldWeightState]
    states_by_quantizer: dict[TensorQuantizer, _FoldWeightState]
    shared_parameter_storages: set[tuple]
    blocked_quantizers: set[TensorQuantizer]


_temporary_fold_weight_context: ContextVar[_TemporaryFoldWeightContext | None] = ContextVar(
    "temporary_fold_weight_context", default=None
)


def _tensor_storage_key(tensor: torch.Tensor) -> tuple | None:
    """Return a device-qualified storage key when the tensor exposes local storage."""
    if tensor.is_meta or tensor.numel() == 0:
        return None
    try:
        return ("storage", tensor.device, tensor.untyped_storage().data_ptr())
    except (RuntimeError, NotImplementedError):
        try:
            local = tensor.to_local()
            return ("storage", local.device, local.untyped_storage().data_ptr())
        except (AttributeError, RuntimeError, NotImplementedError):
            return ("object", id(tensor))


def _tensor_view_key(tensor: torch.Tensor) -> tuple:
    """Identify an exact tensor view while allowing distinct slices of one fused weight."""
    return (
        _tensor_storage_key(tensor),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _shared_parameter_storages(model: nn.Module) -> set[tuple]:
    """Find storage referenced by more than one parameter attribute in the model."""
    storage_counts: dict[tuple, int] = {}
    for module in model.modules():
        for parameter in module._parameters.values():
            if parameter is None or (key := _tensor_storage_key(parameter)) is None:
                continue
            storage_counts[key] = storage_counts.get(key, 0) + 1
    return {key for key, count in storage_counts.items() if count > 1}


@contextlib.contextmanager
def _record_fold_weight_states(model: nn.Module, states: list[_FoldWeightState]):
    """Record fold undo state without changing the public ``fold_weight`` contract."""
    context = _TemporaryFoldWeightContext(states, {}, _shared_parameter_storages(model), set())
    token = _temporary_fold_weight_context.set(context)
    try:
        yield
    finally:
        _temporary_fold_weight_context.reset(token)


def _is_temporary_weight_fold() -> bool:
    """Return whether folding is running inside ``temporarily_fold_weights``."""
    return _temporary_fold_weight_context.get() is not None


class QuantModule(DynamicModule):
    """A base class for quantized modules.

    In addition, the class also provides ``parallel_state`` attribute that can be used to access
    the parallel state of the module.
    """

    _parallel_state: ParallelState

    @classmethod
    @torch.no_grad()
    def convert(cls, module: nn.Module, **setup_kwargs: Any) -> "QuantModule":
        """Convert the module to a dynamic module."""
        module = super().convert(module, **setup_kwargs)

        # setup parallel state now that the module is converted
        if module.parallel_state is None:
            module._initialize_parallel_state()

        return module

    @property
    def parallel_state(self) -> ParallelState | None:
        """Return the parallel state of the quant module."""
        return getattr(self, "_parallel_state", None)

    @parallel_state.setter
    def parallel_state(self, parallel_state: ParallelState):
        """Set the parallel state of the dynamic module."""
        assert isinstance(parallel_state, ParallelState), (
            "parallel_state must be a ParallelState object!"
        )
        self._parallel_state = parallel_state

    def _initialize_parallel_state(self):
        """Initialize the parallel state of the dynamic module.

        This method is called only if the `QuantModule` does not have a `parallel_state` attribute
        after `_setup` is called.
        """
        if torch.distributed.is_initialized():
            warnings.warn(
                f"Distributed training is initialized but no parallel_state is set for {type(self)}. "
                "Using default parallel_state which has data_parallel_group set to the default process group and "
                "tensor_parallel_group is unspecified. "
                "If you are using tensor parallelism for this module, you should set the parallel_state "
                "in its `_setup` method."
            )

        self.parallel_state = ParallelState(data_parallel_group=None)

    def modelopt_post_restore(self, prefix: str = ""):
        """Post-restore to correctly configure the TensorQuantizer states.

        TensorQuantizer states are restored to their shape before saving. Now we need to further configure them.
            1. For non-sharded modules this simply involves moving the TensorQuantizer states to the right device.
                This applies for regular Pytorch models and HuggingFace models.
            2. For sharded modules the restored states of TensorQuantizer could be incorrect. This is because
                parallelism such as TP might have been changed between saving and resoring. So we need to re-calculate
                the state shapes. Hence such modules should override this and implement their own logic.
        """
        # Get a parameter or buffer that does not belong to a TensorQuantizer
        non_tq_param_or_buffer = None
        for name, param_or_buffer in self.state_dict().items():
            parent = self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self
            if not isinstance(parent, TensorQuantizer):
                non_tq_param_or_buffer = param_or_buffer
                break

        if non_tq_param_or_buffer is None:
            warnings.warn(
                f"Could not identify the device for TensorQuantizer states of {prefix}. "
                "Please move the model to the right device now. This can be done by calling "
                "`model.to(device)`."
            )
            return

        # Move the TensorQuantizer states to the right device (dtype should have been restored).
        for module in self.modules():
            if isinstance(module, TensorQuantizer):
                module.to(non_tq_param_or_buffer.device)

    def iter_weights_for_calibration(self):
        """Yield ``(weight, weight_quantizer)`` pairs for weight-only calibration."""
        from modelopt.torch.quantization.utils import quantizer_attr_names, weight_attr_names

        for weight_name in weight_attr_names(self):
            weight_quantizer = getattr(self, quantizer_attr_names(weight_name).weight_quantizer)
            yield getattr(self, weight_name), weight_quantizer

    @staticmethod
    @torch.no_grad()
    def _fold_weight_quantizer(
        quantizer: TensorQuantizer,
        weights: Iterable[torch.Tensor],
        keep_attrs: bool = False,
    ):
        """Fold ``quantizer`` into each weight view in place, then disable and clean it once.

        ``weights`` is an iterable so a single quantizer shared across views (e.g. one
        per-tensor quantizer over all experts of a fused MoE weight) can be folded view by
        view while disabling and dropping its calibration attrs exactly once.
        """
        weights = tuple(weights)
        context = _temporary_fold_weight_context.get()
        if context is not None:
            if quantizer in context.blocked_quantizers:
                return
            if any(
                _tensor_storage_key(weight) in context.shared_parameter_storages
                for weight in weights
            ):
                context.blocked_quantizers.add(quantizer)
                if state := context.states_by_quantizer.get(quantizer):
                    state.restore()
                return
            if not quantizer.fake_quant:
                return
            state = context.states_by_quantizer.get(quantizer)
            if state is None:
                if not quantizer.is_enabled:
                    return
                state = _FoldWeightState.capture(quantizer)
                context.states.append(state)
                context.states_by_quantizer[quantizer] = state
            weights = state.add_weights(weights)
            if not weights:
                return
            state.prepare_quantizer()

        for weight in weights:
            weight.data.copy_(quantizer(weight.float().contiguous()).to(weight.dtype))
        quantizer.disable()
        quantizer.disable_rotate()
        if hasattr(quantizer, "_pre_quant_scale"):
            if keep_attrs:
                # The scale is already baked into the folded weight.
                quantizer._enable_pre_quant_scale = False
            else:
                delattr(quantizer, "_pre_quant_scale")
        if not keep_attrs and hasattr(quantizer, "_amax"):
            delattr(quantizer, "_amax")

    def fold_weight(self, keep_attrs: bool = False):
        """Bake each fake-quant weight quantizer into its weight for faster eval.

        Every fake-quant weight quantizer is folded regardless of its enabled state. The folded
        transform is baked into the stored weight and then disabled, so subsequent forwards use
        the stored weight directly. Calibration buffers (``_pre_quant_scale``, ``_amax``) are
        dropped unless ``keep_attrs``. A retained pre-quant scale remains stored but is made
        inactive because it has already been applied to the folded weight.
        """
        # Handle all attributes that end with _weight_quantizer
        for name in dir(self):
            attr = getattr(self, name)
            if (
                name.endswith("weight_quantizer")
                and isinstance(attr, TensorQuantizer)
                and attr.fake_quant
            ):
                # Get the corresponding weight name by removing _weight_quantizer suffix
                weight_name = name[:-10]

                assert hasattr(self, weight_name), (
                    f"{name} doesn't have a corresponding {weight_name} in {self.__class__.__name__}"
                )
                weight = getattr(self, weight_name)
                if isinstance(weight, torch.Tensor):
                    self._fold_weight_quantizer(attr, (weight,), keep_attrs)


QuantModuleRegistry = _DMRegistryCls("Quant", QuantModule)


class QuantInputBase(QuantModule):
    """Base class for modules where the input is quantized."""

    input_quantizer: TensorQuantizer
    output_quantizer: TensorQuantizer
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def forward(self, input, *args, **kwargs):
        """Quantize the input before calling the original forward method."""
        input = self.input_quantizer(input)
        # Check MR: https://github.com/NVIDIA/Model-Optimizer/pull/824
        if hasattr(self, "_forward_pre_dm"):
            pre_fwd = getattr(self, "_forward_pre_dm")

            def _is_forward_in_mro(bound_or_func) -> bool:
                # If this is a bound method, compare its underlying function to any `forward`
                # implementation in the current MRO. If it matches, it's not an external monkey-patch.
                if hasattr(bound_or_func, "__func__"):
                    fn = bound_or_func.__func__
                    for cls in type(self).mro():
                        if cls.__dict__.get("forward") is fn:
                            return True
                return False

            if pre_fwd is getattr(self, "forward") or _is_forward_in_mro(pre_fwd):
                output = super().forward(input, *args, **kwargs)
            else:
                output = pre_fwd(input, *args, **kwargs)
        else:
            output = super().forward(input, *args, **kwargs)
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    def _setup(self):
        """Patch the module's forward method to quantize the input."""
        self._register_temp_attribute(
            "input_quantizer", TensorQuantizer(self.default_quant_desc_input)
        )
        self._register_temp_attribute(
            "output_quantizer", TensorQuantizer(self.default_quant_desc_output)
        )
        self.output_quantizer.disable()


class QuantLinearConvBase(QuantInputBase):
    """Base class for quantized linear modules.

    Quantized linear modules are modules where both the input and the weight are quantized.
    """

    weight_quantizer: TensorQuantizer | SequentialQuantizer
    _enable_weight_quantization: bool
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @contextlib.contextmanager
    def quantize_weight(self):
        """Context in which `self.weight` is quantized."""
        self._enable_weight_quantization = True
        try:
            yield
        finally:
            self._enable_weight_quantization = False

    @staticmethod
    def _get_quantized_weight(module: "QuantLinearConvBase", weight: torch.Tensor) -> torch.Tensor:
        if module._enable_weight_quantization or is_torch_export_mode():
            return module.weight_quantizer(weight)
        return weight

    def forward(self, input, *args, **kwargs):
        """Quantize the input and the weight before calling the original forward method."""
        # self.quntize_weight() setting attributes is not allowed for torch.export.
        if is_torch_export_mode():
            return super().forward(input, *args, **kwargs)

        with self.quantize_weight():
            return super().forward(input, *args, **kwargs)

    def _setup(self):
        super()._setup()
        self._register_temp_attribute(
            "weight_quantizer", TensorQuantizer(self.default_quant_desc_weight)
        )
        self._register_temp_attribute("_enable_weight_quantization", False)
        self._register_dynamic_attribute("weight", self._get_quantized_weight)


class _LegacyQuantInputBaseMixin:
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantInputBase
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_input = quant_desc_input or self.default_quant_desc_input
        super().__init__(*args, **kwargs)
        QuantModuleRegistry.convert(self)


class _LegacyQuantLinearConvBaseMixin(_LegacyQuantInputBaseMixin):
    """A mixin to support legacy quantized modules which needs to have an __init__ method."""

    _quantized_cls = QuantLinearConvBase
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, *args, quant_desc_input=None, quant_desc_weight=None, **kwargs):
        """Initialize the module with its original __init__ and patch its forward."""
        self.default_quant_desc_weight = quant_desc_weight or self.default_quant_desc_weight
        super().__init__(*args, quant_desc_input=quant_desc_input, **kwargs)
