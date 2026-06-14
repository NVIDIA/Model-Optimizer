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

"""Support quantization for megatron linear layers."""

import contextlib
import types
from typing import Any

import megatron.core.parallel_state as mcore_parallel
import megatron.core.tensor_parallel.layers as megatron_parallel
import megatron.core.transformer.mlp as megatron_mlp
import megatron.core.transformer.moe.experts as megatron_moe
import torch
from megatron.core.parallel_state import get_data_parallel_group
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import get_tensor_model_parallel_group_if_none

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.plugins.megatron import (
    _MegatronMLP,
    ensure_metadata_has_dp_cp_group,
    register_modelopt_extra_state_callbacks,
)
from modelopt.torch.utils import warn_rank_0
from modelopt.torch.utils.distributed import ParallelState

from .. import tensor_quant
from ..conversion import maybe_promote_nvfp4_static_quantizer
from ..nn import QuantModule, QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from ..nn.modules.quant_linear import RealQuantLinear
from ..qtensor import QTensorWrapper
from ..utils import is_torch_export_mode, sync_moe_expert_amax
from .custom import CUSTOM_MODEL_PLUGINS, _ParallelLinear

try:
    from megatron.core.ssm.mamba_mixer import MambaMixer
except ImportError:
    MambaMixer = None

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    from .transformer_engine import _QuantTEGroupedLinear, _QuantTELayerNormLinear, _QuantTELinear

    HAS_TE = True
except ImportError:
    HAS_TE = False


__all__ = []


def _check_nvfp4_static_tp_supported(model: torch.nn.Module) -> None:
    """Raise if using NVFP4-static weight quantization with TP>1.

    Static-block _amax is shard-local but sharded_state_dict treats it as replicated.
    """
    offending = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantModule):
            continue
        parallel_state = getattr(module, "parallel_state", None)
        if parallel_state is None:
            continue
        tp_group = getattr(parallel_state, "tensor_parallel_group", None)
        if tp_group is None or not tp_group.is_initialized() or tp_group.world_size() <= 1:
            continue
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if weight_quantizer is None:
            continue
        leaves = (
            list(weight_quantizer)
            if isinstance(weight_quantizer, SequentialQuantizer)
            else [weight_quantizer]
        )
        if any(leaf.is_nvfp4_static for leaf in leaves):
            offending.append((name, tp_group.world_size()))
    if offending:
        raise NotImplementedError(
            "Static-block NVFP4 weight quantization (e.g. MSE) is not supported with TP > 1. Please re-run with TP=1. "
            f"Offending modules (showing first 5 of {len(offending)}): {offending[:5]}"
        )


def real_quant_module_get_extra_state(self) -> dict:
    """Populating real_quantizer_state and q_tensor_state."""
    extra_state = {}

    if isinstance(self, RealQuantLinear) and isinstance(self.weight, QTensorWrapper):
        real_quantizer_state = self.weight_quantizer.get_modelopt_state()
        q_tensor_state = self.weight.get_state()
    elif isinstance(self, RealQuantLinear):
        real_quantizer_state = self.weight_quantizer.get_modelopt_state()
        q_tensor_state = {}
    else:
        real_quantizer_state = None
        q_tensor_state = None

    extra_state["modelopt_real_quantizer_state"] = real_quantizer_state
    extra_state["modelopt_q_tensor_state"] = q_tensor_state

    return extra_state


def quant_module_get_extra_state(self) -> dict:
    """Populating the extra_state when state_dict() is called.

    quantizer_state, real_quantizer_state, and q_tensor_state are usually stored
    with in the modelopt_state metadata where the keys are the full module name. The issue
    is that MCore model's full module name can change
    if pipeline-parallelism (PP) and expert-parallelism (EP)
    are changing. Alternatively, we store quantizer_state in
    QuantModule's extra_state with QuantModule.get_extra_state()
    which avoids the need to store the full module name.
    """
    extra_state = {}

    quantizer_state = {}
    for name, module in self.named_modules():
        if isinstance(module, TensorQuantizer):
            quantizer_state[name] = module.get_modelopt_state()

    extra_state["modelopt_quantizer_state"] = quantizer_state

    # Handle real_quantizer_state and q_tensor_state
    extra_state.update(real_quant_module_get_extra_state(self))

    return extra_state


def real_quant_module_set_extra_state(self, state: Any):
    """Restore q_tensor_state when load_state_dict() is called.

    We skip restoring real_quantizer_state (if exists), since it is the same as
    the weight_quantizer fake quantizer_state.

    Finally, q_tensor_state is restored if meta device initialization is used. During
    meta-device initialization, real_quantize is not called.
    QTensorWrapper should replace the original weight parameter. Due to TP, we also need
    to adjust q_tensor_data_shape and its metadata shape attribute to use the local weight shape.

    When not using meta device initialization, real_quantize is called during compress mode
    restore where the QTensor will be recomputed based on the local weights. Hence we don't
    need to restore q_tensor_state.

    Note:
        The entire restore process can happen on meta device and be materialized later
        with to_empty(). However, to_empty() will reassign the parameter and the
        QTensorWrapper will be removed. We patch RealQuantLinear._apply to preserve
        QTensorWrapper when to_empty() is applied.
    """
    q_tensor_state = state.get("modelopt_q_tensor_state", None)

    if q_tensor_state:
        q_tensor_metadata = q_tensor_state["metadata"]
        q_tensor_metadata["shape"] = self.weight.shape
        q_tensor_data_dtype = q_tensor_state["quantized_data.dtype"]
        q_tensor_shape = self.weight.shape

        # If q_tensor_data_type is uint8, then it is compressed format of 2 elements.
        if q_tensor_data_dtype == torch.uint8:
            q_tensor_shape = list(q_tensor_shape)
            q_tensor_shape[-1] = q_tensor_shape[-1] // 2
            q_tensor_shape = torch.Size(q_tensor_shape)

        self._parameters["weight"] = QTensorWrapper(
            qtensor=torch.empty(
                q_tensor_shape,  # Use the local shape directly (TP-aware)
                dtype=q_tensor_data_dtype,
                device=self.weight.device,
            ),
            metadata=q_tensor_metadata,
        )


def quant_module_set_extra_state(self, state: Any):
    """Restore quantizer_state when load_state_dict() is called.

    With quantizer_state stored in extra_state (MCore `torch-dist`),
    set_extra_state() is used to perform the functionality
    conversion.restore_quantizer_state().
    load_state_dict() is called twice during MCore resume.
    The state_dict only contains the extra_state in the first time.
    set_extra_state() is trigger by the end of the load_state_dict()
    where QuantModule.modelopt_post_restore() will reinitialize
    amax and scalars to the correct shape.
    The 2nd load_state_dict() is loading all states including amax and
    scalars. We disable QuantModule.modelopt_post_restore() to avoid
    reinitialization since set_extra_state() is called at the end.

    We first restore all fake quantizer_state. Per QuantModule can have
    weight_quantizer, input_quantizer, and output_quantizer.

    Once all quantizer_state are resumed, modelopt_post_restore() is called
    to adjust the shape of all buffers (amax, pre_qunat_scale, _scale, ...) since
    the local shape can be different from the shape in the state due to change
    in tensor parallelism (TP).
    """
    if state is None or not self.allow_post_restore:
        return

    quantizer_state = state.get("modelopt_quantizer_state", None)

    if quantizer_state is not None:
        for name, module in self.named_modules():
            if isinstance(module, TensorQuantizer):
                quantizer_substate = quantizer_state[name]
                maybe_promote_nvfp4_static_quantizer(module, quantizer_substate)
                module.set_from_modelopt_state(quantizer_substate, properties_only=False)
        self.modelopt_post_restore()

    # Handle real_quantizer_state and q_tensor_state
    real_quant_module_set_extra_state(self, state)

    self.allow_post_restore = False


if MambaMixer is not None:

    def _save_mamba_extra_state(self, prefix, sharded_offsets, sharded_state_dict):
        with contextlib.suppress(RuntimeError):
            extra_state = self.get_extra_state()
            if extra_state is not None:
                sharded_state_dict.update(
                    **make_sharded_tensors_for_checkpoint(
                        {"_extra_state": extra_state}, prefix, {}, sharded_offsets
                    )
                )
        return sharded_state_dict

    def _save_mamba_conv1d_quantizer_amax(self, prefix, sharded_offsets, sharded_state_dict):
        quantizer = getattr(self, "conv1d_weight_weight_quantizer", None)
        if quantizer is None:
            return sharded_state_dict

        # Direct Mamba conv1d stores its quantizer as a ModelOpt temp attribute.
        # MCore's native MambaMixer sharded state saves conv1d_weight/bias but
        # does not know that this extra quantizer amax must round-trip for export.
        quantizer_state_dict = {
            k: v
            for k, v in quantizer.state_dict(
                prefix="conv1d_weight_weight_quantizer.", keep_vars=True
            ).items()
            if k.endswith(("_amax", "_global_amax"))
        }
        if "conv1d_weight_weight_quantizer._amax" not in quantizer_state_dict:
            calibrator = getattr(quantizer, "_calibrator", None)
            if calibrator is not None:
                calib_amax = calibrator.compute_amax()
                if calib_amax is not None:
                    quantizer.amax = calib_amax.detach()
                    quantizer_state_dict["conv1d_weight_weight_quantizer._amax"] = quantizer.amax
        if (
            "conv1d_weight_weight_quantizer._amax" not in quantizer_state_dict
            and quantizer.is_enabled
        ):
            # After checkpoint reload, the direct temp quantizer may no longer
            # have calibrator statistics. For weight max calibration, the source
            # is the conv1d weight itself, so recompute the same scalar before
            # saving/exporting rather than dropping the required amax.
            quantizer.amax = quantizer._get_amax(self.conv1d_weight.detach().float()).detach()
            quantizer_state_dict["conv1d_weight_weight_quantizer._amax"] = quantizer.amax

        for key, value in quantizer_state_dict.items():
            if value.numel() != 1:
                raise AssertionError(
                    f"Only scalar direct Mamba conv1d quantizer amax is supported, got "
                    f"{key} with shape {tuple(value.shape)}."
                )

        sharded_state_dict.update(
            **make_sharded_tensors_for_checkpoint(quantizer_state_dict, prefix, {}, sharded_offsets)
        )
        return sharded_state_dict

    @QuantModuleRegistry.register({MambaMixer: "megatron_MambaMixer"})
    class _QuantMambaMixer(QuantModule):
        """Quantize new Megatron Mamba direct conv1d parameters."""

        default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL

        @contextlib.contextmanager
        def quantize_conv1d_weight(self):
            """Context in which ``self.conv1d_weight`` is quantized."""
            if not hasattr(self, "conv1d_weight_weight_quantizer"):
                yield
                return

            self._enable_conv1d_weight_quantization = True
            try:
                yield
            finally:
                self._enable_conv1d_weight_quantization = False

        @staticmethod
        def _get_quantized_conv1d_weight(
            module: "_QuantMambaMixer", weight: torch.Tensor
        ) -> torch.Tensor:
            if module._enable_conv1d_weight_quantization or is_torch_export_mode():
                return module.conv1d_weight_weight_quantizer(weight)
            return weight

        def forward(self, *args, **kwargs):
            """Quantize the direct conv1d weight before calling MambaMixer.forward."""
            if is_torch_export_mode() or not hasattr(self, "conv1d_weight_weight_quantizer"):
                return super().forward(*args, **kwargs)

            with self.quantize_conv1d_weight():
                return super().forward(*args, **kwargs)

        def iter_weights_for_calibration(self):
            """Yield direct conv1d weights for weight-only max calibration."""
            seen_quantizers = set()
            for weight, weight_quantizer in super().iter_weights_for_calibration():
                seen_quantizers.add(id(weight_quantizer))
                yield weight, weight_quantizer

            weight_quantizer = getattr(self, "conv1d_weight_weight_quantizer", None)
            if weight_quantizer is not None and id(weight_quantizer) not in seen_quantizers:
                yield self.conv1d_weight, weight_quantizer

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
            sharded_state_dict = _save_mamba_conv1d_quantizer_amax(
                self, prefix, sharded_offsets, sharded_state_dict
            )
            return _save_mamba_extra_state(self, prefix, sharded_offsets, sharded_state_dict)

        def fold_weight(self, keep_attrs: bool = False):
            # NVFP4 export still needs the calibrated scalar amax after the direct
            # conv1d weight is folded.
            super().fold_weight(keep_attrs=True)
            weight_quantizer = getattr(self, "conv1d_weight_weight_quantizer", None)
            if (
                weight_quantizer is not None
                and weight_quantizer.fake_quant
                and weight_quantizer.is_enabled
            ):
                self.conv1d_weight.data.copy_(
                    weight_quantizer(self.conv1d_weight.float()).to(self.conv1d_weight.dtype)
                )
                weight_quantizer.disable()

        def _setup(self):
            if not hasattr(self, "conv1d_weight"):
                return

            if not hasattr(self, "parallel_state") or self.parallel_state is None:
                data_parallel_group = None
                try:
                    data_parallel_group = get_data_parallel_group(with_context_parallel=True)
                except AssertionError:
                    warn_rank_0(
                        "Context parallel group is not initialized, using data parallel group"
                    )
                    data_parallel_group = get_data_parallel_group()
                self.parallel_state = ParallelState(
                    data_parallel_group,
                    mcore_parallel.get_tensor_model_parallel_group(),
                )

            self._register_temp_attribute(
                # Generic weight discovery maps parameter ``<name>`` to
                # ``<name>_weight_quantizer``.
                "conv1d_weight_weight_quantizer",
                TensorQuantizer(self.default_quant_desc_weight),
            )
            self._register_temp_attribute("_enable_conv1d_weight_quantization", False)
            self._register_dynamic_attribute("conv1d_weight", self._get_quantized_conv1d_weight)


def _create_incompatible_method(method_name: str):
    """Create a method that raises an error for incompatible flash decode methods."""

    def _incompatible_method(self, *args, **kwargs):
        raise NotImplementedError(
            f"{method_name} is not compatible with ModelOpt KV cache quantization. "
            f"KV cache quantization requires core_attention to be called. "
            f"Please raise an issue at https://github.com/NVIDIA/Model-Optimizer if you need this feature."
        )

    return _incompatible_method


def megatron_replace_quant_module_hook(model: torch.nn.Module):
    """Configure Megatron-Core model quantization support.

    This callback is called before the QuantModule replacement to reuse the current
    custom callback infra. However, it is meant to target each QuantModule.
    Since the callback is called when megatron is installed, we do a type check on
    MegatronModule first. For each MegatronModule,
    1. We change TransformerConfig to enable heterogenous distributed checkpointing.
    2. We enable all sub- QuantModule to store quantizer_state as extra_state by
       typing-matching the QuantModuleRegistry.
    3. For Attention modules, we configure them to use core_attention path for KV cache quantization.
    """

    def _configure_attention_for_kv_cache_quant(module: Attention):
        """Configure Attention module for KV cache quantization compatibility."""
        # Disable flash_decode if enabled - it bypasses core_attention (only called during inference)
        if getattr(module.config, "flash_decode", False):
            warn_rank_0(
                "flash_decode=True is incompatible with ModelOpt KV cache quantization. "
                "Setting flash_decode=False. Flash decode bypasses core_attention during decode phase."
            )
            module.config.flash_decode = False

        # Set dtype and device for core_attention (needed for modelopt_post_restore)
        assert hasattr(module, "core_attention"), "Attention module must have core_attention"
        param = next(iter(module.parameters()), None)
        if param is not None:
            module.core_attention.dtype = param.dtype
            module.core_attention.device = param.device

        # Patch flash_decode and flash_decode_and_prefill to raise errors
        module.flash_decode = types.MethodType(_create_incompatible_method("flash_decode"), module)
        module.flash_decode_and_prefill = types.MethodType(
            _create_incompatible_method("flash_decode_and_prefill"), module
        )

    def _register_extra_state_callbacks(model: torch.nn.Module):
        for name, module in model.named_modules():
            if type(module) in QuantModuleRegistry:
                # Skip output_layer w/o enabled weight_quantizer
                if name.endswith("output_layer") and not getattr(
                    getattr(module, "weight_quantizer", None), "is_enabled", False
                ):
                    continue
                register_modelopt_extra_state_callbacks(
                    module,
                    quant_module_get_extra_state,
                    quant_module_set_extra_state,
                )

            # Configure Attention modules for KV cache quantization
            if isinstance(module, Attention):
                _configure_attention_for_kv_cache_quant(module)

    for name, module in model.named_modules():
        if isinstance(module, MegatronModule):
            if "vision_model" not in name:
                # We only enable hetereogenous_dist_checkpoint for language model, vision model is not quantized
                module.config.hetereogenous_dist_checkpoint = True
            _register_extra_state_callbacks(module)


CUSTOM_MODEL_PLUGINS.add(megatron_replace_quant_module_hook)


class _MegatronParallelLinear(_ParallelLinear):
    _functionals_to_replace = [
        (megatron_parallel, "linear_with_grad_accumulation_and_async_allreduce"),
        (megatron_parallel, "linear_with_frozen_weight"),
    ]

    def _setup(self):
        if not hasattr(self, "parallel_state") or self.parallel_state is None:
            data_parallel_group = None
            try:
                data_parallel_group = get_data_parallel_group(with_context_parallel=True)
            except AssertionError:
                warn_rank_0("Context parallel group is not initialized, using data parallel group")
                data_parallel_group = get_data_parallel_group()
            self.parallel_state = ParallelState(
                data_parallel_group,
                mcore_parallel.get_tensor_model_parallel_group(),
            )

        if getattr(self, "gradient_accumulation_fusion", False):
            warn_rank_0(
                "gradient_accumulation_fusion is not supported with ModelOpt quantization. "
                "Setting gradient_accumulation_fusion to False."
            )
            self.gradient_accumulation_fusion = False

        super()._setup()

    def _process_quantizer_amax(self, k, v, quantizer_state_dict):
        if v.ndim == 4:
            quantizer_state_dict[k] = v.squeeze(1).squeeze(-1)
        else:
            quantizer_state_dict[k] = (
                v.view(self.weight.shape[0], -1) if v.numel() > 1 else v.view(-1)
            )

    def _process_activation_quantizer_pre_quant_scale(self, k, v, quantizer_state_dict):
        quantizer_state_dict[k] = v

    def _get_shard_axis_dict(self, state_dict):
        raise NotImplementedError

    def _parameter_to_keep_in_quantizer_state_dict(self, key):
        """Determine whether a parameter should be kept in the quantizer_state_dict.

        Used to include additional quantization parameters (e.g., _scale for real quant)
        beyond the default amax and pre_quant_scale tensors.

        Note: When adding parameters here, update _get_shard_axis_dict accordingly for sharding.
        """
        return False

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # Ensure metadata has dp_cp_group to avoid None subscript errors
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        # Only allow output_layer quantization when embeddings and output_weights are untied
        # When embedding and output_layer are sharing weights, PP>1 will have
        #    output_layer.input_quantizer._amax but TP-only does not. This lead to
        #    state_dict mismatch.
        if prefix.endswith("output_layer."):
            try:
                from megatron.training import get_args as _mlm_get_args

                _untied = bool(
                    getattr(_mlm_get_args(), "untie_embeddings_and_output_weights", False)
                )
            except Exception as e:
                warn_rank_0(f"Failed to get Megatron arg untie_embeddings_and_output_weights: {e}")
                _untied = False
            if not _untied:
                return super().sharded_state_dict(prefix, sharded_offsets, metadata)

        quantizer_state_dict = {}
        for k, v in self.state_dict(prefix="", keep_vars=True).items():
            if "_quantizer" in k and "_amax" in k:
                self._process_quantizer_amax(k, v, quantizer_state_dict)
            elif k == "input_quantizer._pre_quant_scale":
                self._process_activation_quantizer_pre_quant_scale(k, v, quantizer_state_dict)
            elif self._parameter_to_keep_in_quantizer_state_dict(k):
                quantizer_state_dict[k] = v
            elif "quantizer" in k:
                warn_rank_0(
                    f"Quantizer state {k} is not supported for sharded_state_dict. "
                    "Please use regular state_dict."
                )
        sharded_axis_dict = self._get_shard_axis_dict(quantizer_state_dict)
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        sharded_state_dict.update(
            **make_sharded_tensors_for_checkpoint(
                quantizer_state_dict, prefix, sharded_axis_dict, sharded_offsets
            )
        )
        return sharded_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for k in list(state_dict.keys()):
            if not any(qt + "_quantizer" in k for qt in ["weight", "input", "output"]):
                continue
            name = k.split(prefix)[-1] if prefix else k
            state_dict[k] = state_dict[k].view_as(self.state_dict()[name])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@QuantModuleRegistry.register(
    {megatron_parallel.ColumnParallelLinear: "megatron_ColumnParallelLinear"}
)
class _MegatronColumnParallelLinear(_MegatronParallelLinear):
    _is_column_parallel = True

    def _get_shard_axis_dict(self, state_dict):
        """Getting the sharded axis for amax and pre_quant_scale.

        By default, ColumnParallelLinear shards the output dimension (dim=0). However,
        depending the quantization algorithm, not all amax or pre_quant_scale need
        to be sharded.

        We check the quantizer.axis to decide whether an amax needs to be sharded.
        Except for dynamic block quantization (NVFP4, axis: None) or per-tensor (FP8,
        axis: None), the rest of algorithms all need to be sharded

        Prequant scaling is applied per-input-channel; hence no sharding is required.
        """
        shard_axis_dict = {}
        for k in state_dict:
            # Static NVFP4 _global_amax is a replicated scalar; only per-block _amax shards.
            if k.endswith("_global_amax"):
                continue
            if "weight_quantizer." in k:
                weight_quantizer_axis = self.get_submodule(k.rsplit(".", 1)[0]).axis
                if weight_quantizer_axis is not None:
                    shard_axis_dict[k] = 0
        return shard_axis_dict


@QuantModuleRegistry.register({megatron_parallel.RowParallelLinear: "megatron_RowParallelLinear"})
class _MegatronRowParallelLinear(_MegatronParallelLinear):
    _is_row_parallel = True

    def _get_shard_axis_dict(self, state_dict):
        """Getting the sharded axis for amax and pre_quant_scale.

        By default, RowParallelLinear shards the input dimension (dim=1). However,
        depending the quantization algorithm, not all amax or pre_quant_scale need
        to be shard.

        We check the quantizer.axis to decide whether an amax needs to be sharded.
        Only static block quantization needs to be sharded and its axis is either (0,) or (0, 2).
        The first case is used in AWQ the later case is used in blocked 2D quantization.
        Dynamic block quantization (NVFP4 axis:None), per-tensor (FP8, axis: None)
        and per-channel (INT8_SQ or FP8_PER_CHANNEL, axis: 1) do not require input sharding.

        Prequant scaling is applied per-input-channel; hence it is always sharded.
        """
        shard_axis_dict = {}
        for k in state_dict:
            # Static NVFP4 _global_amax is a replicated scalar; only per-block _amax shards.
            if k.endswith("_global_amax"):
                continue
            if "weight_quantizer." in k:
                weight_quantizer_axis = None
                if isinstance(self.weight_quantizer, TensorQuantizer):
                    weight_quantizer_axis = self.weight_quantizer.axis
                elif "weight_quantizer.0." in k:
                    weight_quantizer_axis = self.weight_quantizer[0].axis
                elif "weight_quantizer.1." in k:
                    weight_quantizer_axis = self.weight_quantizer[1].axis
                if isinstance(weight_quantizer_axis, tuple):
                    shard_axis_dict[k] = 1
            if k == "input_quantizer._pre_quant_scale":
                shard_axis_dict[k] = 0
        return shard_axis_dict


@QuantModuleRegistry.register({megatron_mlp.MLP: "megatron_MegatronMLP"})
class _QuantMegatronMLP(_MegatronMLP):
    """Module to support special handling of `linear_fc1` in `sharded_state_dict()` of MCore `MLP`."""

    _modelopt_state_keys = [
        r"weight_quantizer\.(\d+\.)*_amax$",
        r"weight_quantizer\.(\d+\.)*_scale$",
    ]


class _RealQuantMegatronParallelLinear(RealQuantLinear):
    allow_real_quant_gemm = True
    _scale_tensor_shard_axis = None

    def _parameter_to_keep_in_quantizer_state_dict(self, key):
        return any(k in key for k in self.list_of_scale_tensors)

    def _get_shard_axis_dict(self, state_dict):
        shard_axis_dict = super()._get_shard_axis_dict(state_dict)
        for k in state_dict:
            if (
                any(k.endswith(suffix) for suffix in self.list_of_scale_tensors)
                and state_dict[k].dim() > 1
            ):
                assert self._scale_tensor_shard_axis is not None, (
                    "scale_tensor_shard_axis is not set, please set it in the subclass"
                )
                shard_axis_dict[k] = self._scale_tensor_shard_axis
        return shard_axis_dict

    def modelopt_post_restore(self, prefix: str = ""):
        """Post restore to correctly configure the realquant scales.

        ModelOpt restores the TensorQuantizer states such as `_amax` and `_pre_quant_scale` to their
        shape before saving. However this is not enough for MCore/distributed frameworks since the tensor parallelism
        could change between saving and restoring. If the tensor parallelism changes, the shape of the quantizer
        states also changes. So we need to re-calculate the quantizer states.

        Note:
            During real quantization, weight_quantizer._fake_quant is set to False which trigger the real quant
            forward path and lead to error. We enable the weight_quantizer fake_quant forward path while recompute
            the correct shape.
        """
        self.weight_quantizer._fake_quant = True
        super().modelopt_post_restore(prefix=prefix)
        self.weight_quantizer._fake_quant = False

        if hasattr(self.weight_quantizer, "_scale"):
            # Recompute all real quantization buffer shapes
            self.weight_quantizer._real_quantize(self.weight)

    def _forward_impl(self, input, *args, **kwargs):
        """Use real quant gemm if available.

        Here the forward is patched such that real quant gemm can be called if available. Both conditions
        below must be satisfied (static and dynamic check based on input args) to use the kernel.
        Otherwise, we fallback.

        Note:
            RealQuantLinear.forward() is doing the same check inside and will fall back to use the super
            class forward(). This is not desired since _forward_impl introduces much more args and kwargs
            while the original forward only takes 1 positional argument. We must above the fallback path
            in RealQuantLinear.forward().
        """
        if (
            self._should_run_real_quant_gemm
            and input.numel() > 1
            and self.has_real_quant_gemm_impl(input, *args, **kwargs)
        ):
            allreduce_dgrad = kwargs.get("allreduce_dgrad", False)
            tp_group = kwargs.get("tp_group")
            sequence_parallel = kwargs.get("sequence_parallel", False)

            tp_group = get_tensor_model_parallel_group_if_none(tp_group)

            if sequence_parallel:
                input = gather_from_sequence_parallel_region(
                    input, tensor_parallel_output_grad=True, group=tp_group
                )
            else:
                input = input

            return RealQuantLinear.forward(
                self,
                input,
                allreduce_dgrad=allreduce_dgrad,
                tp_group=tp_group,
            )
        else:
            return super()._forward_impl(input, *args, **kwargs)


class _RealQuantMegatronColumnParallelLinear(
    _RealQuantMegatronParallelLinear, _MegatronColumnParallelLinear
):
    _scale_tensor_shard_axis = 0

    def forward(self, input, *args, **kwargs):
        return _MegatronColumnParallelLinear.forward(self, input, *args, **kwargs)


class _RealQuantMegatronRowParallelLinear(
    _RealQuantMegatronParallelLinear, _MegatronRowParallelLinear
):
    _scale_tensor_shard_axis = 1

    def forward(self, input, *args, **kwargs):
        return _MegatronRowParallelLinear.forward(self, input, *args, **kwargs)


@QuantModuleRegistry.register({megatron_moe.SequentialMLP: "megatron_moe_SequentialMLP"})
class _MegatronSequentialMLP(DynamicModule):
    def _setup(self):
        if (
            self.config.expert_model_parallel_size > 1
            and self.config.tensor_model_parallel_size > 1
        ):
            raise ValueError(
                "TP+EP is not supported by QuantSequentialMLP. Set either TP or EP to 1!"
            )

        if not hasattr(self, "parallel_state") or self.parallel_state is None:
            self.parallel_state = ParallelState(
                mcore_parallel.get_expert_data_parallel_group(),
                tensor_parallel_group=mcore_parallel.get_expert_tensor_parallel_group(),
                expert_model_parallel_group=mcore_parallel.get_expert_model_parallel_group(),
            )

        # Initialize parallel state for submodules local_experts.*.linear_fc1 and local_experts.*.linear_fc2
        for expert in self.local_experts:
            expert.linear_fc1.parallel_state = self.parallel_state
            expert.linear_fc2.parallel_state = self.parallel_state

    def layer_sync_moe_local_experts_amax(self, sync_weight_amax=False):
        """Sync quantizer amax across local experts in a SequentialMLP.

        Always syncs input quantizer amax across experts. Optionally syncs weight
        quantizer amax as well, which matches TEGroupedMLP behavior where all
        experts are fused into a single GEMM with one quantizer per linear layer.

        Args:
            sync_weight_amax: If True, also sync weight quantizer amax across experts.

        This function operates on a single rank and does not require distributed sync.
        Distributed amax sync across EP and ETP (for RowParallel) happens in
        model_calib.max_calibrate(). This function should be called before the
        distributed sync to ensure the amax values are synchronized across the layer first.

        Note:
            Because there are logic which calls collective communication based on whether amax is not None,
            we need to guarantee that all experts must have amax. Otherwise, there will be deadlock
            when synchronizing over EP since some ranks may have amax None and not calling the collective
            communication.
        """
        sync_moe_expert_amax(self.local_experts, sync_weight_amax=sync_weight_amax)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Override the default to enable singleton_local_shards.

        Note:
            singleton_local_shards must be added to the metadata; otherwise, all experts
            amax are packed to gather and currently the TP replica_id for linear_fc1
            is incorrect. This limits TP=ETP=1 when EP>1. Otherwise, there will be
            sharded_state_dict access error.
        """
        if metadata is None:
            metadata = {}
        metadata["singleton_local_shards"] = True
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        return sharded_state_dict


if HAS_TE:

    @QuantModuleRegistry.register({TERowParallelLinear: "te_mcore_RowParallelLinear"})
    class _QuantTEMCoreRowParallelLinear(_QuantTELinear, _MegatronRowParallelLinear):
        pass

    @QuantModuleRegistry.register({TEColumnParallelLinear: "te_mcore_ColumnParallelLinear"})
    class _QuantTEMCoreColumnParallelLinear(_QuantTELinear, _MegatronColumnParallelLinear):
        pass

    @QuantModuleRegistry.register({TELinear: "te_mcore_Linear"})
    class _QuantTEMCoreLinear(_QuantTELinear):
        pass

    @QuantModuleRegistry.register(
        {TELayerNormColumnParallelLinear: "te_mcore_LayerNormColumnParallelLinear"}
    )
    class _QuantTELayerNormColumnParallelLinear(
        _QuantTELayerNormLinear, _MegatronColumnParallelLinear
    ):
        pass

    # Quantized subclasses to support TEGroupedMLP quantization
    class _QuantMegatronTEGroupedLinear(_QuantTEGroupedLinear, _MegatronParallelLinear):
        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            # _sharded_state_dict_grouped adds _extra_state{gemm_idx} for gemm_idx:[1, num_gemms] in
            # sharded_state_dict which is same as _extra_state. The _extra_state{gemm_idx} is used for
            # TE Fp8 checkpoint, we need to remove the _extra_state{gemm_idx} for gemm_idx:[1, num_gemms]
            # for modelopt checkpoint restore
            filtered_state_dict = {
                k: v
                for k, v in state_dict.items()
                if not any(k.endswith(f"_extra_state{num}") for num in range(1, self.num_gemms))
            }
            return super()._load_from_state_dict(filtered_state_dict, prefix, *args, **kwargs)

        def _process_quantizer_amax(self, k, v, quantizer_state_dict):
            assert v.numel() == 1, "TEGroupedLinear only supports per-tensor quantization"
            quantizer_state_dict[k] = v.view(-1)

    @QuantModuleRegistry.register(
        {TEColumnParallelGroupedLinear: "megatron_TEColumnParallelGroupedLinear"}
    )
    class _MegatronTEGroupedColumnParallelLinear(
        _QuantMegatronTEGroupedLinear, _MegatronColumnParallelLinear
    ):
        pass

    @QuantModuleRegistry.register(
        {TERowParallelGroupedLinear: "megatron_TERowParallelGroupedLinear"}
    )
    class _MegatronTEGroupedRowParallelLinear(
        _QuantMegatronTEGroupedLinear, _MegatronRowParallelLinear
    ):
        pass

    @QuantModuleRegistry.register({megatron_moe.TEGroupedMLP: "megatron_moe_TEGroupedMLP"})
    class _MegatronTEGroupedMLP(_MegatronMLP):
        def _setup(self):
            if not hasattr(self, "parallel_state") or self.parallel_state is None:
                self.parallel_state = ParallelState(
                    mcore_parallel.get_expert_data_parallel_group(),
                    tensor_parallel_group=mcore_parallel.get_expert_tensor_parallel_group(),
                    expert_model_parallel_group=mcore_parallel.get_expert_model_parallel_group(),
                )
            # initialize parallel state for submodules linear_fc1 and linear_fc2
            self.linear_fc1.parallel_state = self.parallel_state
            self.linear_fc2.parallel_state = self.parallel_state

    @QuantModuleRegistry.register({TEDotProductAttention: "TEDotProductAttention"})
    class _QuantTEDotProductAttention(QuantModule):
        """Quantized version of TEDotProductAttention for Megatron models with KV cache quantization.

        This class adds KV cache quantization support to Transformer Engine's TEDotProductAttention
        module used in Megatron-Core models. It introduces three quantizers (q_bmm_quantizer,
        k_bmm_quantizer, v_bmm_quantizer) that quantize the query, key, and value tensors after
        RoPE has been applied.
        """

        def _setup(self):
            """Initialize quantizers for Q, K, V tensors."""
            self.q_bmm_quantizer = TensorQuantizer()
            self.k_bmm_quantizer = TensorQuantizer()
            self.v_bmm_quantizer = TensorQuantizer()

            # Set parallel_state for distributed sync of BMM quantizers
            try:
                data_parallel_group = get_data_parallel_group(with_context_parallel=True)
            except AssertionError:
                data_parallel_group = get_data_parallel_group()
            self.parallel_state = ParallelState(
                data_parallel_group,
                mcore_parallel.get_tensor_model_parallel_group(),
            )

        def forward(self, query, key, value, *args, **kwargs):
            """Apply post-RoPE quantization to KV cache."""
            # Quantize Q, K, V
            query = self.q_bmm_quantizer(query)
            key = self.k_bmm_quantizer(key)
            value = self.v_bmm_quantizer(value)
            return super().forward(query, key, value, *args, **kwargs)

        def modelopt_post_restore(self, name=""):
            """Restore quantizer states after model loading."""
            for tq in [self.q_bmm_quantizer, self.k_bmm_quantizer, self.v_bmm_quantizer]:
                # TODO: Add support for non-scalar states such as
                # Affine KVCache  bias vector which is per head per channel
                if not all(v.numel() == 1 for v in tq.state_dict().values()):
                    raise NotImplementedError(
                        "Only scalar states are supported for KV Cache/BMM Quantizers"
                    )
            # dtype and device should have been set in `megatron_replace_quant_module_hook`
            # via `_configure_attention_for_kv_cache_quant`
            assert hasattr(self, "device") and hasattr(self, "dtype")
            self.to(device=self.device, dtype=self.dtype)

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            # Currently we do not need sharded_state_dict for TEDotProductAttention since the amax are scalar values.
            # However we would need this in future to support non-scalar states such as
            # Affine KVCache Quant bias vector.
            state_dict = self.state_dict(prefix="", keep_vars=True)
            return make_sharded_tensors_for_checkpoint(state_dict, prefix, {}, sharded_offsets)
