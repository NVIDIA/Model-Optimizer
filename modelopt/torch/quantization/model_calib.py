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

"""Calibration utilities."""

import math
import warnings
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.opt.searcher import ForwardLoop
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.distributed import DistributedProcessGroup, ParallelState
from modelopt.torch.utils.network import (
    bind_forward_method,
    get_module_device,
    unpatch_forward_method,
)

from .calib import MseCalibrator
from .conversion import create_and_replace_svdquant_linear_on_the_fly, set_quantizer_by_cfg_context
from .nn import QuantModule, SequentialQuantizer, TensorQuantizer
from .utils import (
    disable_calib,
    enable_fake_quant,
    enable_quant,
    enable_weight_access_and_writeback,
    is_quantized_column_parallel_linear,
    is_quantized_linear,
    is_quantized_row_parallel_linear,
    quantizer_attr_names,
    weight_attr_names,
)

__all__ = ["awq", "max_calibrate", "smoothquant", "svdquant"]


def weight_only_quantize(model: nn.Module):
    """Just quantize the weights of the model."""
    seen_modules = set()
    for name, module in model.named_modules():
        if module in seen_modules:
            continue
        for weight_name in weight_attr_names(module):
            with enable_weight_access_and_writeback(module, model):
                weight_quantizer = getattr(
                    module, quantizer_attr_names(weight_name).weight_quantizer
                )
                weight_quantizer(getattr(module, weight_name))
        seen_modules.add(module)


@torch.no_grad()
def max_calibrate(model: nn.Module, forward_loop: ForwardLoop | None = None, distributed_sync=True):
    """Calibrate the model using max.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`MaxCalibConfig <modelopt.torch.quantization.config.MaxCalibConfig>` for
    details on the remaining arguments.
    """
    enable_stats_collection(model)
    if forward_loop is None:
        weight_only_quantize(model)
    else:
        forward_loop(model)
    finish_stats_collection(model)

    # amax sync for local experts in a SequentialMLP
    for name, module in model.named_modules():
        if hasattr(module, "sync_moe_local_experts_amax"):
            module.sync_moe_local_experts_amax()

    if not distributed_sync:
        return

    def sync_quantizer_amax_across_dp_ep(quantizer, parallel_state, device):
        """Synchronize the amax across all ranks in the data parallel and expert parallel groups."""
        if isinstance(quantizer, SequentialQuantizer):
            for _q in quantizer:
                sync_quantizer_amax_across_dp_ep(_q, parallel_state, device)
            return
        quantizer.sync_amax_across_distributed_group(parallel_state.data_parallel_group, device)
        quantizer.sync_amax_across_distributed_group(
            parallel_state.expert_model_parallel_group, device
        )
        # TODO: create sync_bias_across_distributed_group

    # Step 1:Sync amax across data parallelism
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            for child in module.children():
                if not isinstance(child, (TensorQuantizer, SequentialQuantizer)):
                    continue
                sync_quantizer_amax_across_dp_ep(
                    child, module.parallel_state, get_module_device(module)
                )

    # TP sync:
    # Objective: the quantization parameters when TP = 8 then changed to TP=4 then back to TP=8 should be the same

    # ColumnParallel: X @ [A_1, A_2] (weights split along Cout)
    #   activations:  TPG should have the same amax if axis in [None, -1]
    #   weights:      TPG should have the same amax if axis in [None, -1] (note: we dont use -1 axis for weights)

    # RowParallel:    [X_1, X_2] @  [A_1
    #                                A_2] (weights split along Cin)
    #   activations:  TPG should have the same amax if axis in [None]
    #   weights:      TPG should have the same amax if axis in [None, 0]

    def sync_quantizer_amax_across_tp(
        quantizer: TensorQuantizer | SequentialQuantizer,
        linear_name: str,
        quantizer_type: str,
        axes_for_sync: list,
        parallel_state: ParallelState,
    ):
        # Syncing amax across TP for sequential quantizer
        if isinstance(quantizer, SequentialQuantizer):
            for _q in quantizer:
                # Syncing amax across TP for sequential quantizer
                sync_quantizer_amax_across_tp(
                    _q, linear_name, quantizer_type, axes_for_sync, parallel_state
                )
            return
        # sync is not needed for block quantization
        if quantizer.block_sizes is not None:
            if hasattr(quantizer, "_padding"):
                warnings.warn(
                    f"Found block-quantized padded {quantizer_type} for {linear_name}, amax will"
                    " not be synced correctly."
                )
            # Skip amax sync for INT4 / W4A8 block quantization
            # Sync amax for NVFP4 (dynamic per-block, static per-tensor quantized scale)
            if getattr(quantizer.block_sizes, "type", None) == "dynamic":
                return

        if quantizer.axis in axes_for_sync and quantizer.amax is not None:
            quantizer.sync_amax_across_distributed_group(parallel_state.tensor_parallel_group)

    # Step 2: Sync amax across relevant parallelism (such as TP / EP)
    for name, module in model.named_modules():
        if getattr(module, "_parallel_state", None) is None:
            continue

        if is_quantized_column_parallel_linear(module):
            sync_quantizer_amax_across_tp(
                module.input_quantizer,
                name,
                "input_quantizer",
                axes_for_sync=[None, -1],
                parallel_state=module.parallel_state,
            )

            sync_quantizer_amax_across_tp(
                module.weight_quantizer,
                name,
                "weight_quantizer",
                axes_for_sync=[None, -1],
                parallel_state=module.parallel_state,
            )

        if is_quantized_row_parallel_linear(module):
            sync_quantizer_amax_across_tp(
                module.input_quantizer,
                name,
                "input_quantizer",
                axes_for_sync=[None],
                parallel_state=module.parallel_state,
            )

            sync_quantizer_amax_across_tp(
                module.weight_quantizer,
                name,
                "weight_quantizer",
                axes_for_sync=[None, 0],
                parallel_state=module.parallel_state,
            )

        # KV Cache Quantization
        if hasattr(module, "k_bmm_quantizer") and hasattr(module, "v_bmm_quantizer"):
            # We only support KVCache quantization with scalar per-tensor states for now (NVFP4 & FP8 KV cache)
            # So we should sync amax across DP and TP for these quantizers (DP is already synced from above)
            for quantizer in [module.k_bmm_quantizer, module.v_bmm_quantizer]:
                if isinstance(quantizer, TensorQuantizer) and quantizer.amax is not None:
                    quantizer.sync_amax_across_distributed_group(
                        module.parallel_state.tensor_parallel_group
                    )


@torch.no_grad()
def mse_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync=True,
    num_steps: int = 10,
    start_multiplier: float = 0.25,
    stop_multiplier: float = 4.0,
):
    """Calibrate the model using MSE-based amax search.

    This calibration method first uses max calibration to get initial amax values,
    then searches for better amax values by minimizing the MSE between original
    and quantized tensors.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.
        distributed_sync: Whether to sync amax across distributed processes.
        num_steps: Number of amax candidates to try (default: 10).
        start_multiplier: Starting multiplier for amax search (default: 0.25).
        stop_multiplier: Ending multiplier for amax search (default: 4.0).

    See :class:`MseCalibConfig <modelopt.torch.quantization.config.MseCalibConfig>` for
    details on the remaining arguments.
    """
    # Step 1: First get initial amax using max calibration
    max_calibrate(model, forward_loop, distributed_sync)

    # Step 2: Replace calibrators with MseCalibrator for enabled quantizers
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module._disabled:
            # Static block quantization is not supported by MseCalibrator
            if module.is_static_block_quant:
                raise ValueError(
                    f"MSE calibration does not support static block quantization. "
                    f"Found static block quantization at {name}."
                )
            if module._calibrator is not None and not module._dynamic and hasattr(module, "_amax"):
                # Get the initial amax from max calibration
                initial_amax = module._amax.clone().detach()

                def quant_func(x, amax, quantizer=module):
                    original_amax = quantizer._amax.clone() if hasattr(quantizer, "_amax") else None
                    quantizer._amax = amax

                    with (
                        enable_quant(quantizer),
                        disable_calib(quantizer),
                        enable_fake_quant(quantizer),
                    ):
                        xq = quantizer(x)

                    if original_amax is not None:
                        quantizer._amax = original_amax
                    else:
                        delattr(quantizer, "_amax")

                    return xq

                # Create MSE calibrator with quant_func
                module._calibrator = MseCalibrator(
                    amax=initial_amax,
                    axis=module._calibrator._axis,
                    num_steps=num_steps,
                    start_multiplier=start_multiplier,
                    stop_multiplier=stop_multiplier,
                    quant_func=quant_func,
                )

    # Step 3: Collect data with MSE calibrators
    enable_stats_collection(model)
    if forward_loop is None:
        weight_only_quantize(model)
    else:
        forward_loop(model)

    # Step 4: Compute optimal amax and load it
    finish_stats_collection(model, method="mse")

    # TODO: Sync amax across distributed processes


def enable_stats_collection(model: nn.Module):
    """Enable stats collection for all quantizers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module._disabled:
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def finish_stats_collection(model: nn.Module, method: str | None = None):
    """Finish stats collection for all quantizers in the model."""
    for _, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or module._disabled:
            continue

        cal = getattr(module, "_calibrator", None)
        if cal and not getattr(module, "_dynamic", False):
            if method in {"mse", "entropy"}:
                if cal.compute_amax(method) is not None:
                    if method == "entropy":
                        module.load_calib_amax("entropy")
                    else:
                        module.load_calib_amax()
            elif cal.compute_amax() is not None:
                # Max calibrator
                module.load_calib_amax()

        if module.bias_calibrator is not None and module.bias_type == "static":
            module.load_calib_bias()

        module.enable_quant()
        module.disable_calib()


@torch.no_grad()
def disable_pre_quant_scale_and_resmooth(linear: nn.Module, delete_pre_quant_scale: bool = False):
    """Disable pre_quant_scale and resmooth the quantized linear weights."""
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer._enable_pre_quant_scale, (
        "pre_quant_scale should be enabled first!"
    )
    assert hasattr(linear.input_quantizer, "_pre_quant_scale"), (
        "pre_quant_scale should be available"
    )

    pre_quant_scale = linear.input_quantizer._pre_quant_scale.to(torch.float32)

    linear.weight.copy_(
        (linear.weight * pre_quant_scale.squeeze()[None, :]).to(linear.weight.dtype)
    )
    linear.weight_quantizer.reset_amax()
    max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))

    # Lets not delete the _pre_quant_scale, it might useful later; Instead we will disable it
    linear.input_quantizer._enable_pre_quant_scale = False

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        device, dtype = linear.weight.device, linear.weight.dtype
        linear.input_quantizer.amax = linear.input_quantizer._amax_for_smoothing.amax().to(
            device=device, dtype=dtype
        )

    if delete_pre_quant_scale:
        delattr(linear.input_quantizer, "_pre_quant_scale")
        linear.input_quantizer._enable_pre_quant_scale = False


# A global variable used during auto_quantize to avoid folding pre_quant_scale to weights
_ENABLE_FOLDING_PQS_TO_WEIGHTS = True


@torch.no_grad()
def _apply_weight_pre_quant_scale(linear, pre_quant_scale):
    if _ENABLE_FOLDING_PQS_TO_WEIGHTS:
        linear.weight.data.copy_(
            (linear.weight * pre_quant_scale.to(linear.weight.device).squeeze()[None, :]).to(
                linear.weight.dtype
            )
        )
    else:
        linear.weight_quantizer._enable_pre_quant_scale = True
        linear.weight_quantizer.pre_quant_scale = pre_quant_scale.squeeze()[None, :].to(
            linear.weight.dtype
        )

    linear.weight_quantizer.reset_amax()
    max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))


@torch.no_grad()
def apply_pre_quant_scale_and_smooth(
    linear: nn.Module, pre_quant_scale: torch.Tensor | None = None
):
    """Apply pre_quant_scale and smooth the quantized linear weights.

    If pre_quant_scale is not provided, the existing pre_quant_scale of input_quantizer will be used.
    """
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer.pre_quant_scale is None, "pre_quant_scale should be None first!"

    if pre_quant_scale is None:
        pre_quant_scale = (
            linear.input_quantizer._pre_quant_scale
            if hasattr(linear.input_quantizer, "_pre_quant_scale")
            else None
        )

    assert pre_quant_scale is not None, "pre_quant_scale should be provided or already set"

    assert torch.all(pre_quant_scale > 0), "pre_quant_scale should be positive"

    # pre_quant_scale should be in fp32 for the scaling math to be numerically safe
    pre_quant_scale = pre_quant_scale.to(torch.float32)

    linear.input_quantizer._enable_pre_quant_scale = True
    linear.input_quantizer.pre_quant_scale = pre_quant_scale.to(linear.weight.dtype)

    inv_scale = 1.0 / pre_quant_scale
    _apply_weight_pre_quant_scale(linear, inv_scale)

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        device, dtype = linear.weight.device, linear.weight.dtype
        _amax_for_smoothing = linear.input_quantizer._amax_for_smoothing.to(
            device=device, dtype=dtype
        )
        linear.input_quantizer.amax = (
            (_amax_for_smoothing * pre_quant_scale.to(device)).amax().to(dtype)
        )

        if is_quantized_column_parallel_linear(linear) or is_quantized_row_parallel_linear(linear):
            linear.input_quantizer.sync_amax_across_distributed_group(
                linear.parallel_state.tensor_parallel_group
            )


@torch.no_grad()
def smoothquant(model: nn.Module, forward_loop: ForwardLoop | None = None, alpha=1.0):
    """Smooth-Quant variant with per-channel weight scaling.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`SmoothQuantCalibConfig <modelopt.torch.quantization.config.SmoothQuantCalibConfig>` for
    details on the remaining arguments.
    """
    # distributed synchronization
    # max_calibrate performs amax sync for data parallel

    # Column parallel:
    # activations:  TPG should have the same pre_quant_scale
    #               This is achieved by syncing act_amax and weight_scale across TPG which is used to
    #               compute pre_quant_scale
    # weights:      no-op

    # Row parallel:
    # activations:  TPG should have same activation amax
    # weights:      TPG should have the same weight amax

    assert forward_loop is not None, "forward_loop must be provided for smoothquant"
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.input_quantizer.is_enabled
            and module.input_quantizer.axis is None
        ):
            module.input_quantizer.axis = -1

    max_calibrate(model, forward_loop)

    def postprocess(module):
        # It is important to keep scaling math in fp32 to be numerically safe
        act_amax = module.input_quantizer.amax.float()
        weight_scale = module.weight.abs().amax(dim=0, keepdim=True)
        device, dtype = module.weight.device, module.weight.dtype

        parallel_group = module.parallel_state.tensor_parallel_group
        if is_quantized_column_parallel_linear(module) and parallel_group.is_initialized():
            dist.all_reduce(act_amax, op=dist.ReduceOp.MAX, group=parallel_group.group)
            dist.all_reduce(weight_scale, op=dist.ReduceOp.MAX, group=parallel_group.group)

        scale_a = (weight_scale.pow(1 - alpha) / act_amax.pow(alpha)).squeeze()

        # Now that activation per-channel amax have been collected, use per-tensor quantization for activation
        # TODO: make this a buffer after we support only heterogeneous checkpointing for MCore
        module.input_quantizer._amax_for_smoothing = act_amax.cpu()
        module.input_quantizer.reset_amax()
        module.input_quantizer.axis = None
        module.input_quantizer.amax = act_amax.amax().to(dtype=dtype, device=device)

        # Some channel could have 0 amax which causes scale_a to overflow. Explicitly mask them out here
        epsilon = 1.0 / (1 << 31)
        if scale_a.min() <= epsilon:
            zero_mask = act_amax <= epsilon
            scale_a[zero_mask] = 1
        scale_a = scale_a.clamp(min=1e-4, max=1e4)
        apply_pre_quant_scale_and_smooth(module, scale_a)

    smoothed_modules = 0
    for name, module in model.named_modules():
        if is_quantized_linear(module):
            if not hasattr(module.input_quantizer, "_amax"):
                warnings.warn(f"{name} is not calibrated, skip smoothing")
                continue
            if module.input_quantizer.num_bits != 8 or module.weight_quantizer.num_bits != 8:
                warnings.warn(f"Only int8 smoothing is supported, skip {name}")
                continue
            if module.input_quantizer.axis != -1:
                warnings.warn(f"Only per-channel smoothing is supported, skip {name}")
                continue

            assert module.input_quantizer._amax.numel() > 1, (
                f"Error: {name} has only one channel to smooth"
            )

            with enable_weight_access_and_writeback(module, model):
                postprocess(module)

            smoothed_modules += 1
    print_rank_0(f"Smoothed {smoothed_modules} modules")


def awq(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    algorithm: str = "awq_lite",
    **kwargs,
):
    """Apply AWQ to the model.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`AWQFullCalibConfig <modelopt.torch.quantization.config.AWQFullCalibConfig>` for
    details on the remaining arguments.
    """
    with SequentialQuantizer.convert_to_single_quantizer(model):
        if algorithm in ["awq_full", "awq_lite"]:
            awq_lite(model, forward_loop, **kwargs)

        if algorithm in ["awq_full", "awq_clip"]:
            awq_clip(model, forward_loop, **kwargs)

    # Special handling for SequentialQuantizer
    # Pre-compute name_to_module dict to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in model.named_modules():
        if is_quantized_linear(module) and isinstance(module.weight_quantizer, SequentialQuantizer):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                max_calibrate(module, lambda linear: linear.weight_quantizer(module.weight))


@torch.no_grad()
def awq_lite(
    model: nn.Module,
    forward_loop: ForwardLoop,
    alpha_step: float = 0.1,
    debug: bool = False,
    **kwargs,
):
    """Lite version of AWQ.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`AWQLiteCalibConfig <modelopt.torch.quantization.config.AWQLiteCalibConfig>` for
    details on the remaining arguments.
    """
    if forward_loop is None:
        warnings.warn("forward_loop must be provided for awq_lite; skipping awq_lite")
        return

    class AWQLiteHelper:
        cache_mode: bool = False

        def __init__(self, module, name):
            self.name = name
            self.act_scale = 0.0
            self.num_cache_steps = 0
            self.num_search_steps = 0
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)
            self.weight_scale = get_weight_scale(module.weight, self.block_size)
            self.loss = {
                k.item(): torch.zeros((), device=module.weight.device, dtype=torch.float32)
                for k in torch.arange(0, 1.0 + alpha_step, alpha_step)
            }
            self.best_scale = None
            self.best_alpha = None
            self.is_input_quantized = module.input_quantizer.is_enabled
            self.num_tokens = 0
            self.module = module
            self.is_enabled = True

        def setup(self):
            module = self.module
            bind_forward_method(module, forward, "_forward_no_awq")
            if module.input_quantizer.is_enabled:
                module.input_quantizer.disable()
                if module.input_quantizer.axis not in [None, -1]:
                    self.is_enabled = False
                    return
                module.input_quantizer.axis = -1

        def cleanup(self):
            module = self.module
            if hasattr(module, "_if_calib"):
                delattr(module, "_if_calib")
            unpatch_forward_method(module, "_forward_no_awq")

    def get_weight_scale(weight, block_size=None):
        org_shape = weight.shape
        slice_after_padding = None
        if block_size:
            if org_shape[-1] % block_size != 0:
                slice_after_padding = slice(org_shape[-1])
                weight = F.pad(weight, (0, block_size - org_shape[-1] % block_size), "constant", 0)
                org_shape = weight.shape
            weight = weight.contiguous().view(-1, block_size)
        weight_abs = weight.abs()  # Cache to avoid redundant computation
        weight_abs_amax = weight_abs.amax(dim=1, keepdim=True)
        scale = weight_abs / (weight_abs_amax + torch.finfo(weight.dtype).tiny)
        scale = scale.view(org_shape)
        if slice_after_padding is not None:
            scale = scale[..., slice_after_padding]
        scale = scale.mean(0).to(torch.float32)
        return scale

    def get_act_scale(x):
        return x.abs().contiguous().view(-1, x.shape[-1]).mean(0).to(torch.float32)

    def get_scale(x_max, w_max, alpha, tensor_parallel_group=None):
        scales = (
            (
                x_max.pow(alpha)
                / (w_max.to(x_max.device).pow(1 - alpha) + torch.finfo(torch.float32).tiny)
            )
            .clamp(min=1e-4, max=1e4)
            .view(-1)
        )
        scales = (scales / (scales.max() * scales.min()).sqrt()).view(-1)
        if tensor_parallel_group and tensor_parallel_group.is_initialized():
            dist.all_reduce(scales, op=dist.ReduceOp.SUM, group=tensor_parallel_group.group)
            scales /= tensor_parallel_group.world_size()
        return scales

    def update_loss(self, out, out_actual, alpha):
        out_actual = out_actual[0] if isinstance(out_actual, tuple) else out_actual
        out = out[0] if isinstance(out, tuple) else out
        loss = (out - out_actual).float().pow(2).mean()
        self.awq_lite.loss[alpha] += loss.to(self.awq_lite.loss[alpha].device)

    def update_best_params(self):
        if not self.awq_lite.is_enabled:
            return
        self.awq_lite.loss.update({k: float(v) for k, v in self.awq_lite.loss.items()})
        self.awq_lite.best_alpha = min(self.awq_lite.loss, key=self.awq_lite.loss.get)
        self.awq_lite.best_scale = get_scale(
            self.awq_lite.act_scale,
            self.awq_lite.weight_scale,
            self.awq_lite.best_alpha,
            (
                self.parallel_state.tensor_parallel_group
                if is_quantized_column_parallel_linear(self)
                else None
            ),
        )

    def forward(self, input, *args, **kwargs):
        # Collect actual output without quantization
        self.weight_quantizer.disable()
        if hasattr(self.input_quantizer, "_pre_quant_scale"):
            delattr(self.input_quantizer, "_pre_quant_scale")
        if hasattr(self.weight_quantizer, "_pre_quant_scale"):
            delattr(self.weight_quantizer, "_pre_quant_scale")
        out_actual = self._forward_no_awq(input, *args, **kwargs)
        self.weight_quantizer.enable()

        if input.numel() == 0 or not self.awq_lite.is_enabled:
            # For MoEs, some experts might see 0 tokens
            return out_actual

        if AWQLiteHelper.cache_mode:
            # Get local tensor from Dtensor
            input = input.to_local() if hasattr(input, "to_local") else input

            self.awq_lite.act_scale += get_act_scale(self.input_quantizer(input))
            self.awq_lite.num_cache_steps += 1
            self.awq_lite.num_tokens += input.numel() / input.shape[-1]
            if self.awq_lite.is_input_quantized:
                with set_quantizer_by_cfg_context(self.input_quantizer, {"*": {"enable": True}}):
                    max_calibrate(self.input_quantizer, lambda quantizer: quantizer(input), False)
            return out_actual

        for alpha in self.awq_lite.loss:
            awq_scale = get_scale(
                self.awq_lite.act_scale,
                self.awq_lite.weight_scale,
                alpha,
                (
                    self.parallel_state.tensor_parallel_group
                    if is_quantized_column_parallel_linear(self)
                    else None
                ),
            )
            self.input_quantizer.pre_quant_scale = (1 / awq_scale).to(self.weight.dtype)
            self.weight_quantizer.pre_quant_scale = awq_scale.to(self.weight.dtype)
            out = self._forward_no_awq(input, *args, **kwargs)
            update_loss(self, out, out_actual, alpha)

        self.awq_lite.num_search_steps += 1

        # Now forward the actual output without any quantization
        return out_actual

    # Pre-compute name_to_module dict ONCE to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in name_to_module.items():
        if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            with enable_weight_access_and_writeback(module, model, name_to_module):
                module.awq_lite = AWQLiteHelper(module, name)
            module.awq_lite.setup()

    # Collect activation scale values
    AWQLiteHelper.cache_mode = True
    print_rank_0("awq_lite: Caching activation statistics...")

    # Lets enable stats collection
    # This will collect amax for input_quantizers and KV quantizers during the caching mode forward pass
    enable_stats_collection(model)
    forward_loop(model)

    # Call max_calibrate to load the amax values collected during the caching mode forward pass
    # This will also perform distributed amax sync for input_quantizers
    max_calibrate(model, lambda model: None)

    def sync_act_scale_across_dp(module, data_parallel_group):
        """Sync activation scale across Data Parallel (DP)."""
        if data_parallel_group.is_initialized():
            dist.all_reduce(
                module.awq_lite.act_scale, op=dist.ReduceOp.AVG, group=data_parallel_group.group
            )

    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and hasattr(module, "awq_lite")
            and module.awq_lite.num_cache_steps > 0
        ):
            # Hack: MoEs forward all tokens through all experts if _if_calib is True
            module._if_calib = True
            module.awq_lite.act_scale = module.awq_lite.act_scale / module.awq_lite.num_cache_steps

            has_nan_local = torch.any(torch.isnan(module.awq_lite.act_scale)) or torch.any(
                torch.isnan(module.awq_lite.weight_scale)
            )
            has_nan = DistributedProcessGroup.get_dist_syncd_obj(
                has_nan_local, module.parallel_state.data_parallel_group, lambda objs: any(objs)
            )

            if has_nan:
                module.awq_lite.is_enabled = False
            else:
                sync_act_scale_across_dp(
                    module,
                    module.parallel_state.data_parallel_group,
                )

    AWQLiteHelper.cache_mode = False
    print_rank_0("awq_lite: Searching parameters...")
    with torch.no_grad():
        forward_loop(model)

    def postprocess(module, name):
        update_best_params(module)
        if hasattr(module.weight_quantizer, "_pre_quant_scale"):
            delattr(module.weight_quantizer, "_pre_quant_scale")
        if hasattr(module.input_quantizer, "_pre_quant_scale"):
            delattr(module.input_quantizer, "_pre_quant_scale")
        if module.awq_lite.is_input_quantized:
            if module.input_quantizer.amax is not None:
                act_amax = module.input_quantizer.amax
                # TODO: make this a buffer after we support only heterogeneous checkpointing for MCore
                module.input_quantizer._amax_for_smoothing = act_amax.cpu()
                module.input_quantizer.reset_amax()
                module.input_quantizer.axis = None
                module.input_quantizer.amax = act_amax.amax()
                module.input_quantizer.enable()
            # for dynamic quantization, there is no amax, so we just enable the quantizer
            else:
                module.input_quantizer.enable()

        if module.awq_lite.is_enabled:
            apply_pre_quant_scale_and_smooth(module, 1.0 / module.awq_lite.best_scale)
        else:
            warnings.warn(f"awq_lite: Disabling for {name}, quantizing with max calibration.")
            max_calibrate(module, lambda module: module.weight_quantizer(module.weight))

    for name, module in model.named_modules():
        if hasattr(module, "awq_lite"):
            if module.awq_lite.num_cache_steps == 0:
                module.awq_lite.is_enabled = False
            elif module.awq_lite.num_search_steps == 0:
                module.awq_lite.is_enabled = False
                warnings.warn(
                    "awq_lite: Calling `forward_loop(model)` the second time did not forward data through the"
                    f" {name}. Please provide a valid `forward_loop` function that can be used to"
                    " forward data through the model many times."
                )
            with enable_weight_access_and_writeback(module, model, name_to_module):
                postprocess(module, name)

            module.awq_lite.cleanup()
            if not debug:
                delattr(module, "awq_lite")


@torch.no_grad()
def awq_clip(
    model: nn.Module,
    forward_loop: ForwardLoop,
    max_co_batch_size: int = 1024,
    max_tokens_per_batch: int = 64,
    min_clip_ratio: float = 0.5,
    shrink_step: float = 0.05,
    debug: bool = False,
    **kwargs,
):
    """AWQ-Clip variant.

    Args:
        model: Model to calibrate.
        forward_loop: A callable that runs the forward pass of the model.

    See :class:`AWQClipCalibConfig <modelopt.torch.quantization.config.AWQClipCalibConfig>` for
    details on the remaining arguments.
    """
    assert forward_loop is not None, "forward_loop must be provided for awq_clip"

    class AWQClipHelper:
        def __init__(self, module):
            self.num_tokens = 0
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)

            # Cache the original amax
            module.weight_quantizer.reset_amax()
            enable_stats_collection(module.weight_quantizer)
            module.weight_quantizer(module.weight)
            finish_stats_collection(module.weight_quantizer)
            self.w_amax = module.weight_quantizer.amax.clone()

            co, ci = module.weight.shape
            clip_ratios = [
                round(float(k), 2) for k in torch.arange(min_clip_ratio, 1.0, shrink_step)
            ] + [1.0]
            if self.is_per_tensor_clip(module):
                self.loss = {k: torch.tensor(0.0, device=module.weight.device) for k in clip_ratios}
            else:
                self.loss = {
                    k: torch.zeros(
                        (co, math.ceil(ci / self.block_size)),
                        device=module.weight.device,
                    )
                    for k in clip_ratios
                }
            self.best_clip_val = None
            self.best_loss = None

            self.is_input_quantized = module.input_quantizer.is_enabled
            module.weight_quantizer.disable()

        def is_per_tensor_clip(self, module):
            quantizer = module.weight_quantizer
            is_dynamic_w_per_tensor = (
                hasattr(quantizer, "block_sizes")
                and quantizer.block_sizes.get("type", None) == "dynamic"
                and quantizer.axis is None
            )
            is_per_tensor = quantizer.axis is None and quantizer.block_sizes is None
            return is_dynamic_w_per_tensor or is_per_tensor

    def update_best_params(self):
        self.awq_clip.best_loss = torch.ones_like(self.awq_clip.w_amax) * float("inf")
        self.awq_clip.best_clip_val = torch.zeros_like(self.awq_clip.w_amax)

        for shrink, loss in self.awq_clip.loss.items():
            loss = loss.view_as(self.awq_clip.w_amax)
            indices = loss < self.awq_clip.best_loss
            self.awq_clip.best_loss = torch.where(indices, loss, self.awq_clip.best_loss)
            self.awq_clip.best_clip_val = torch.where(
                indices, self.awq_clip.w_amax * shrink, self.awq_clip.best_clip_val
            )

    def _clip_search(self, inputs, co_bsz=256, max_tokens=16):
        weight = self.weight
        self.weight_quantizer.enable()

        if self.awq_clip.is_per_tensor_clip(self):
            # In NVFP4, only the per-tensor amax is clipped
            out_actual = inputs @ self.weight.T
            original_amax = self.weight_quantizer.amax.clone()
            self.awq_clip.num_tokens += inputs.shape[0]
            for shrink in self.awq_clip.loss:
                self.weight_quantizer.amax = original_amax * shrink
                out = inputs @ self.weight_quantizer(self.weight).T
                loss = (out - out_actual).float().pow(2).mean()
                self.awq_clip.loss[shrink] += loss
        else:
            # weight  [co, ci] -> [co, 1, n_block, block_size]
            # inputs  [..., ci] -> [1, max_tokens, n_block, block_size]

            inputs = inputs.view(-1, inputs.shape[-1])  # _, ci
            # Select max_tokens from the total input tokens of count batch * n_token
            inputs = inputs[0 :: max(1, inputs.shape[0] // max_tokens)]  # max_tokens, ci
            self.awq_clip.num_tokens += inputs.shape[0]

            block_size = self.awq_clip.block_size
            co, ci = weight.shape
            if ci % block_size != 0:
                weight = F.pad(weight, (0, block_size - ci % block_size), "constant", 0)
                inputs = F.pad(inputs, (0, block_size - ci % block_size), "constant", 0)
                ci = weight.shape[-1]

            weight = weight.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size

            # 1, max_tokens, n_block, block_size
            inputs = inputs.reshape(1, inputs.shape[0], -1, block_size)

            for co_batch in range(math.ceil(co / co_bsz)):
                w = weight[co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)]

                org_out = (inputs * w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                for shrink in self.awq_clip.loss:
                    self.weight_quantizer.amax = self.awq_clip.w_amax * shrink
                    quantized_clipped_weight = self.weight_quantizer(self.weight)
                    cur_w = quantized_clipped_weight[
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ]
                    if cur_w.shape[-1] % block_size != 0:
                        cur_w = F.pad(
                            cur_w,
                            (0, block_size - cur_w.shape[-1] % block_size),
                            "constant",
                            0,
                        )
                    cur_w = cur_w.reshape(w.shape)
                    cur_out = (inputs * cur_w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                    # co_bsz, n_block
                    loss = (cur_out - org_out).float().pow(2).mean(dim=1)

                    parallel_group = self.parallel_state.data_parallel_group
                    if parallel_group.is_initialized():
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=parallel_group.group)
                        loss /= parallel_group.world_size()

                    del cur_out, cur_w
                    self.awq_clip.loss[shrink][
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ] += loss
                del org_out

    def forward(name, self, input, *args, **kwargs):
        # input shape : (..., cin)
        # weight shape : (cout, cin)
        if self.awq_clip.is_input_quantized:
            self.input_quantizer.enable()
            max_calibrate(self.input_quantizer, lambda input_quantizer: input_quantizer(input))
            self.input_quantizer.disable()
        try:
            _clip_search(
                self,
                self.input_quantizer(input),
                max_co_batch_size,
                max_tokens_per_batch,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError(
                    f"Clip search on {name} failed due to CUDA out of memory, try reducing"
                    " max_co_batch_size"
                ) from e
            raise RuntimeError(e)

        # Disable quantization
        self.weight_quantizer.disable()
        return self._forward_no_awq(input, *args, **kwargs)

    # Pre-compute name_to_module dict to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.weight_quantizer.is_enabled
            and module.weight_quantizer.block_sizes is not None
        ):
            bind_forward_method(module, partial(forward, name), "_forward_no_awq")
            with enable_weight_access_and_writeback(module, model, name_to_module):
                module.awq_clip = AWQClipHelper(module)

    print_rank_0("awq_clip: Estimating parameters...")
    # Lets enable stats collection
    # This will collect amax for input_quantizers and KV quantizers during the caching mode forward pass
    enable_stats_collection(model)
    forward_loop(model)
    # Call max_calibrate to load the amax values collected during the caching mode forward pass
    # This will also perform distributed amax sync for input_quantizers
    max_calibrate(model, lambda model: None)

    def postprocess(module):
        update_best_params(module)

        # Load the best clip value (amax)
        module.weight_quantizer.amax = module.awq_clip.best_clip_val
        module.weight_quantizer.enable()
        if module.awq_clip.is_input_quantized:
            module.input_quantizer.enable()

    for name, module in model.named_modules():
        if is_quantized_linear(module) and hasattr(module, "awq_clip"):
            if module.awq_clip.num_tokens > 0:
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    postprocess(module)

            if not debug:
                delattr(module, "awq_clip")

            unpatch_forward_method(module, "_forward_no_awq")


def _get_awq_quantizer_block_size(tensor: torch.Tensor, quantizer: TensorQuantizer):
    if quantizer.block_sizes is None:
        return None
    if -1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[-1]
    elif 1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[1]
    else:
        raise ValueError("AWQ requires block quantization along -1 axis")
    return blocksize


@torch.no_grad()
def svdquant(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    lowrank: int = 32,
    **kwargs,
):
    """Lite version of SVDQuant.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`SVDQuantConfig <modelopt.torch.quantization.config.SVDQuantConfig>` for
    details on the remaining arguments.
    """

    def postprocess(module, name):
        print_rank_0(f"SVD {name}")
        weight = module.weight.data
        original_device = weight.device
        original_dtype = weight.dtype
        weight_f64 = weight.to(dtype=torch.float64, device=original_device)
        u, s, vt = torch.linalg.svd(weight_f64, full_matrices=False)
        if u.shape[1] < lowrank or vt.shape[0] < lowrank:
            warnings.warn(
                "The low-rank dimensions do not match the layer dimensions. "
                "Please verify your configuration and model settings. "
                f"SVD will be skipped for this layer {name}."
            )
            return
        us = u[:, :lowrank] * s[:lowrank]
        vt = vt[:lowrank]
        module.weight_quantizer.svdquant_lora_a = vt.to(
            dtype=original_dtype, device=original_device
        )
        module.weight_quantizer.svdquant_lora_b = us.to(
            dtype=original_dtype, device=original_device
        )
        module.weight.data.sub_(
            module.weight_quantizer.svdquant_lora_b @ module.weight_quantizer.svdquant_lora_a
        )
        module.weight_quantizer.reset_amax()
        module.input_quantizer.reset_amax()

    create_and_replace_svdquant_linear_on_the_fly(model=model)
    awq(model, forward_loop, "awq_lite", **kwargs)

    for name, module in model.named_modules():
        if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            with enable_weight_access_and_writeback(module, model):
                postprocess(module, name)
    max_calibrate(model, forward_loop)
