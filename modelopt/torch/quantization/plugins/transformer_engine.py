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

"""Support quantization for Transformer Engine layers."""

import copy
import inspect
import os
import warnings

import torch
import transformer_engine as te
import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear
import transformer_engine.pytorch.module.linear as te_linear
from packaging.version import Version

from modelopt.torch.quantization.utils import replace_function

import modelopt.torch.kernels.quantization.gemm as _triton_kernels
from ..nn import QuantModuleRegistry, SequentialQuantizer
from .custom import _ParallelLinear

_TE_VERSION = Version(te.__version__)


def _per_expert_weight_quantizer_enabled() -> bool:
    """Opt-in MODELOPT_TEGROUPED_PER_EXPERT_QUANTIZER=1: per-gemm weight_quantizer in TEGroupedLinear."""
    return os.environ.get("MODELOPT_TEGROUPED_PER_EXPERT_QUANTIZER", "0") == "1"


def _assert_te_fp8_enabled():
    """Check if Transformer Engine FP8 autocast is enabled and raise error if so."""
    try:
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        if FP8GlobalStateManager.is_fp8_enabled():
            raise RuntimeError(
                "Transformer Engine FP8 training (fp8_autocast) is enabled, which conflicts with "
                "ModelOpt quantization. Please disable TE FP8 autocast when using ModelOpt "
                "quantization, or use ModelOpt's FP8 quantization instead."
            )
    except ImportError:
        pass  # Older TE versions may not have this API


@QuantModuleRegistry.register({te.pytorch.Linear: "te_Linear"})
class _QuantTELinear(_ParallelLinear):
    @property
    def _functionals_to_replace(self):
        return (
            [(te_linear._Linear, "apply")]
            if torch.is_grad_enabled()
            else [(te_linear._Linear, "forward")]
        )

    @_functionals_to_replace.setter
    def _functionals_to_replace(self, value):
        self._functionals_to_replace = value

    def _setup(self):
        super()._setup()
        if getattr(self, "fuse_wgrad_accumulation", False):
            warnings.warn(
                "fuse_wgrad_accumulation is not supported with ModelOpt quantization. "
                "Setting fuse_wgrad_accumulation to False."
            )
            self.fuse_wgrad_accumulation = False

    @staticmethod
    def te_quantized_linear_fn(package, func_name, self, *args, **kwargs):
        """Quantized version specifically for TE with weight first, then input."""
        _assert_te_fp8_enabled()
        # Locate `weight` and `inp` by parameter name in the un-patched `_Linear.forward`
        # signature — robust to TE versions that insert positional args between them
        # (e.g. `weight_fp8` in TE 1.x, `weight_workspace` in TE 2.15).
        # NOTE: we're called from inside `replace_function`'s context, so
        # `_Linear.forward` may currently point at the `functools.partial` wrapper
        # (whose signature collapses to `*args, **kwargs`). The original is cached at
        # `_Linear._forward` while the patch is active (when `_apply` is patched
        # instead, `_forward` is absent and `forward` is itself the original).
        # `_forward` path receives a leading None (placeholder ctx); `_apply` does not.
        orig_forward = getattr(te_linear._Linear, "_forward", te_linear._Linear.forward)
        names = list(inspect.signature(orig_forward).parameters)
        ctx_offset = 0 if func_name == "_forward" else 1
        weight_pos = names.index("weight") - ctx_offset
        inp_pos = names.index("inp") - ctx_offset
        new_args = list(args)
        new_args[weight_pos] = self.weight_quantizer(args[weight_pos])
        new_args[inp_pos] = self.input_quantizer(args[inp_pos])
        output = getattr(package, func_name)(*new_args, **kwargs)
        # TE 2.15+ returns `(out, new_weight_workspace)`; TE <= 2.14 returns just `out`.
        # Only the activation tensor participates in output quantization.
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    # Override the quantized linear function
    _quantized_linear_fn = te_quantized_linear_fn


class _GroupedAxis0FakeQuantFn(torch.autograd.Function):
    """Triton-backed per-expert fake-quant adapter for the N-modules path.

    Forward: single-launch Triton kernel over N expert weights (tensor-of-pointers,
    no stack memcopy). Backward honors modelopt's default `pass_through_bwd=True`
    — gradient flows back unchanged with zero kernel work. When False, the
    clip-aware Triton STE backward kernel runs. See
    `modelopt/torch/kernels/quantization/gemm/grouped_axis0_fakequant.py`.
    """

    @staticmethod
    def forward(ctx, amax_vec, num_bits, narrow_range, pass_through_bwd, *weights):
        outputs = _triton_kernels.grouped_axis0_fakequant(
            list(weights), amax_vec, num_bits=num_bits, narrow_range=narrow_range
        )
        ctx.pass_through_bwd = pass_through_bwd
        if not pass_through_bwd:
            ctx.save_for_backward(amax_vec, *weights)
        ctx.num_weights = len(weights)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.pass_through_bwd:
            return (None, None, None, None, *grad_outputs)
        saved = ctx.saved_tensors
        amax_vec, weights = saved[0], list(saved[1:])
        grad_inputs = _triton_kernels.grouped_axis0_fakequant_backward(
            weights, list(grad_outputs), amax_vec
        )
        return (None, None, None, None, *grad_inputs)


# Register the public te.pytorch.GroupedLinear class
@QuantModuleRegistry.register({te_grouped_linear.GroupedLinear: "te_GroupedLinear"})
class _QuantTEGroupedLinear(_ParallelLinear):
    @property
    def _functionals_to_replace(self):
        return (
            [(te_grouped_linear._GroupedLinear, "apply")]
            if torch.is_grad_enabled()
            else [(te_grouped_linear._GroupedLinear, "forward")]
        )

    @_functionals_to_replace.setter
    def _functionals_to_replace(self, value):
        self._functionals_to_replace = value

    def _setup(self):
        if getattr(self, "fuse_wgrad_accumulation", False):
            warnings.warn(
                "fuse_wgrad_accumulation is not supported with ModelOpt quantization. "
                "Setting fuse_wgrad_accumulation to False."
            )
            self.fuse_wgrad_accumulation = False

        # GroupedMLP stores the weights as weight0, weight1, etc. To run setup in order to
        # initialize the quantizer states, self.weight is used to extract shape, dtype etc. Assigning
        # self.weight0 to self.weight to run the quantizer states initialization.
        assert not hasattr(self, "weight"), "self.weight should not exist for TEGroupedLinear"
        self.weight = self.weight0
        # Memorize the original weight.dtype for modelopt_post_restore given that
        # the dtype can change later.
        super()._setup()
        # Remove self.weight after setup.
        delattr(self, "weight")

        self._per_expert_weight_quantizer = _per_expert_weight_quantizer_enabled()
        if self._per_expert_weight_quantizer:
            for i in range(self.num_gemms):
                self.add_module(f"weight_quantizer_{i}", TensorQuantizer())

    def modelopt_post_restore(self, prefix: str = ""):
        # GroupedMLP stores the weights as weight0, weight1, etc. To run post_restore in order to
        # initialize the quantizer states, self.weight is used to extract shape, dtype etc. Assigning
        # self.weight0 to self.weight to run the quantizer states initialization.
        assert not hasattr(self, "weight"), "self.weight should not exist for TEGroupedLinear"
        self.weight = self.weight0
        super().modelopt_post_restore(prefix=prefix)
        # Remove self.weight after post_restore.
        delattr(self, "weight")

        # Base post_restore only re-calibrates self.weight_quantizer; the per-expert
        # weight_quantizer_{i} also need re-calibration so a TP/EP change between save
        # and restore produces correctly shaped per-channel _amax. Mirror base behavior:
        # only re-calibrate quantizers whose loaded state had _amax (skip unused ones).
        if getattr(self, "_per_expert_weight_quantizer", False):
            from modelopt.torch.quantization.model_calib import max_calibrate

            for i in range(self.num_gemms):
                weight_i = getattr(self, f"weight{i}", None)
                if weight_i is None:
                    continue
                wq_i = self._get_weight_quantizer(i)
                q = wq_i[0] if isinstance(wq_i, SequentialQuantizer) else wq_i
                if not hasattr(q, "_amax"):
                    continue
                wq_i.reset_amax()
                max_calibrate(wq_i, lambda wq, w=weight_i: wq(w), distributed_sync=False)
            # Re-calibration just changed every per-expert _amax. Drop the cache
            # so the next forward rebuilds from the fresh values.
            self._invalidate_per_expert_amax_cache()

    def _get_weight_quantizer(self, gemm_idx: int):
        if getattr(self, "_per_expert_weight_quantizer", False):
            return getattr(self, f"weight_quantizer_{gemm_idx}")
        return self.weight_quantizer

    def _gather_per_expert_amax(self) -> torch.Tensor:
        """Stack N per-expert weight_quantizer_i._amax scalars into a [N] fp32 vector.

        Matches the amax-input contract of `grouped_axis0_fakequant` — one
        entry per expert, indexed by gemm_idx.

        Cached lazily: the per-expert _amax scalars don't change outside
        calibration, and the gate (_can_use_triton_per_expert_path) only
        admits this path when `q._if_calib` is False on every quantizer —
        so once the cache is populated it stays valid for the lifetime of
        the layer's calibrated state. The cache is invalidated explicitly
        via _invalidate_per_expert_amax_cache (called from modelopt_post_restore)
        in case checkpoint reload changes the amax values.

        Eliminates the O(N)-Python-overhead-per-forward walk over N submodules
        observed in OMNIML-5072 AC3's microbench (the gap to Btriton5 grew
        with N — 1.59x at N=32, 2.18x at N=128 — symptomatic of per-forward
        scaling that disappears once the gathered tensor is reused).
        """
        cached = getattr(self, "_per_expert_amax_cache", None)
        if cached is not None:
            return cached
        amaxes = []
        for i in range(self.num_gemms):
            q = self._get_weight_quantizer(i)
            amaxes.append(q._amax.to(torch.float32).reshape(()))
        stacked = torch.stack(amaxes).contiguous()
        self._per_expert_amax_cache = stacked
        return stacked

    def _invalidate_per_expert_amax_cache(self) -> None:
        """Drop the cached _gather_per_expert_amax result.

        Called automatically from modelopt_post_restore (where dist-ckpt load
        may have changed per-expert _amax buffers). Also callable by user code
        after explicit re-calibration that mutates _amax outside the normal
        calibration flow.
        """
        if hasattr(self, "_per_expert_amax_cache"):
            self._per_expert_amax_cache = None

    def _can_use_triton_per_expert_path(self, num_gemms: int) -> bool:
        """Soft-gate the Triton dispatch on availability + ready-to-quantize state."""
        if not getattr(self, "_per_expert_weight_quantizer", False):
            return False
        if not _triton_kernels.IS_AVAILABLE:
            return False
        if not hasattr(_triton_kernels, "grouped_axis0_fakequant"):
            return False
        for i in range(num_gemms):
            q = self._get_weight_quantizer(i)
            # SequentialQuantizer (multi-stage) not supported on the Triton
            # path; fall back to the cuda_ext per-quantizer loop.
            if isinstance(q, SequentialQuantizer):
                return False
            if not hasattr(q, "_amax"):
                return False
            # During calibration each quantizer still needs the cuda_ext path
            # so its _amax gets updated; skip Triton until calib finishes.
            if getattr(q, "_if_calib", False):
                return False
        return True

    def iter_weights_for_calibration(self):
        """Yield ``(weight_i, weight_quantizer)`` for each of the ``num_gemms`` grouped weights."""
        for i in range(self.num_gemms):
            weight_i = getattr(self, f"weight{i}", None)
            if weight_i is not None:
                yield weight_i, self._get_weight_quantizer(i)

    @staticmethod
    def te_grouped_quantized_linear_fn(package, func_name, self, *args):
        _assert_te_fp8_enabled()
        # Locate `inp` and the m_splits-bearing arg by parameter name. The second
        # slot was renamed from `m_splits` (TE < 2.10) to `non_tensor_args` (TE
        # 2.10+, where m_splits is now at non_tensor_args[0]). `*weights_and_biases`
        # is always the trailing variadic — 2 * num_gemms tensors (weights, then biases).
        # See `te_quantized_linear_fn` for why we look up `_forward` here.
        # `_forward` path receives a leading None (placeholder ctx); `_apply` does not.
        orig_forward = getattr(
            te_grouped_linear._GroupedLinear,
            "_forward",
            te_grouped_linear._GroupedLinear.forward,
        )
        sig_params = list(inspect.signature(orig_forward).parameters)
        ctx_offset = 0 if func_name == "_forward" else 1
        inp_pos = sig_params.index("inp") - ctx_offset
        if "non_tensor_args" in sig_params:
            num_gemms = len(args[sig_params.index("non_tensor_args") - ctx_offset][0])
        else:
            num_gemms = len(args[sig_params.index("m_splits") - ctx_offset])
        weights_start = len(args) - 2 * num_gemms

        new_args = list(args)
        new_args[inp_pos] = self.input_quantizer(args[inp_pos])
        if self._can_use_triton_per_expert_path(num_gemms):
            # Single-launch Triton fakequant for the N expert weights.
            # Replaces the per-expert cuda_ext loop (N kernel launches -> 1).
            # All per-expert quantizers share the same config (each is a
            # copy.deepcopy of the base weight_quantizer in _setup), so
            # num_bits/narrow_range/pass_through_bwd read from quantizer 0
            # apply to the whole layer.
            weights = list(args[weights_start : weights_start + num_gemms])
            amax_vec = self._gather_per_expert_amax()
            q0 = self._get_weight_quantizer(0)
            pass_through_bwd = getattr(q0, "_pass_through_bwd", True)
            qweights = _GroupedAxis0FakeQuantFn.apply(
                amax_vec, q0.num_bits, q0.narrow_range, pass_through_bwd, *weights
            )
            for gemm_idx in range(num_gemms):
                new_args[weights_start + gemm_idx] = qweights[gemm_idx]
        else:
            for gemm_idx in range(num_gemms):
                pos = weights_start + gemm_idx
                new_args[pos] = self._get_weight_quantizer(gemm_idx)(args[pos])
        output = getattr(package, func_name)(*new_args)
        # TE 2.15+ returns `(out, new_workspaces)`; TE <= 2.14 returns just `out`.
        # Only the activation tensor participates in output quantization.
        if isinstance(output, tuple):
            return (self.output_quantizer(output[0]), *output[1:])
        return self.output_quantizer(output)

    # Override the quantized linear function
    _quantized_linear_fn = te_grouped_quantized_linear_fn


class _QuantLayerNormLinearFunc(torch.autograd.Function):
    """Patched version of _LayerNormLinear to quantize the input to the GEMM operation."""

    @staticmethod
    def _get_original_gemm():
        if Version("2.0") <= _TE_VERSION:
            return te_layernorm_linear.general_gemm
        else:
            return te_layernorm_linear.tex.gemm

    @staticmethod
    def _gemm_replace_args():
        if Version("2.0") <= _TE_VERSION:
            return (te_layernorm_linear, "general_gemm")
        else:
            return (te_layernorm_linear.tex, "gemm")

    @staticmethod
    def forward(ctx, inp, ln_weight, ln_bias, weight, *args, **kwargs):
        input_quantizer, weight_quantizer = _QuantLayerNormLinearFunc.modelopt_quantizers

        qweight = weight_quantizer(weight)
        qweight.requires_grad = weight.requires_grad
        if ctx is not None:
            # We need to recompute the quantized input for the backward pass, so we save the input_quantizer
            ctx.modelopt_input_quantizer = input_quantizer

        original_gemm = _QuantLayerNormLinearFunc._get_original_gemm()

        def _patched_general_gemm(weight, input, *gemm_args, **gemm_kwargs):
            qinput = input_quantizer(input)
            return original_gemm(weight, qinput, *gemm_args, **gemm_kwargs)

        with replace_function(
            *_QuantLayerNormLinearFunc._gemm_replace_args(),
            _patched_general_gemm,  # type: ignore[call-arg]
        ):
            outputs = te_layernorm_linear._og_LayerNormLinear.forward(
                ctx, inp, ln_weight, ln_bias, qweight, *args, **kwargs
            )
        return outputs

    # TODO: Support non-pass-through backward behavior for activation quantization
    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass for _QuantLayerNormLinearFunc functional.

        The backward pass input and weight gradient estimation uses straight through estimator (STE).
        We should add support for advanced gradient estimation techniques like STE with clipping.
        However this is a low priority item.
        """
        gemm_call_counter = {"count": 0}

        original_gemm = _QuantLayerNormLinearFunc._get_original_gemm()

        def _patched_general_gemm(a, b, *gemm_args, **gemm_kwargs):
            # The first time, gemm is used for dgrad calculation
            # dgrad GEMM; dx = dy * qw; Called as gemm(qw, dy, ...)
            if gemm_call_counter["count"] == 0:
                gemm_call_counter["count"] += 1
                return original_gemm(a, b, *gemm_args, **gemm_kwargs)

            # The second time, gemm is used for wgrad calculation
            # wgrad GEMM; dqw = dy^T * x; Called as gemm(x, dy, ..);

            # x should be quantized input (qinput) for the backward pass as per chain rule,
            # but gemm is called with the unquantized input (a)
            # So lets first get the quantized input (qinput) and then call the gemm
            qinput = ctx.modelopt_input_quantizer(a)
            return original_gemm(qinput, b, *gemm_args, **gemm_kwargs)

        with replace_function(
            *_QuantLayerNormLinearFunc._gemm_replace_args(),
            _patched_general_gemm,  # type: ignore[call-arg]
        ):
            # During backward, the patch does not exist; autograd will automatically use
            # _QuantLayerNormLinearFunc.backward
            outputs = te_layernorm_linear._LayerNormLinear.backward(ctx, *grad_outputs)

        delattr(ctx, "modelopt_input_quantizer")
        return outputs


@QuantModuleRegistry.register({te.pytorch.LayerNormLinear: "te_LayerNormLinear"})
class _QuantTELayerNormLinear(_ParallelLinear):
    _functionals_to_replace = []

    def _setup(self):
        super()._setup()
        if getattr(self, "fuse_wgrad_accumulation", False):
            warnings.warn(
                "fuse_wgrad_accumulation is not supported with ModelOpt quantization. "
                "Setting fuse_wgrad_accumulation to False."
            )
            self.fuse_wgrad_accumulation = False

    def forward(self, *args, **kwargs):
        """Call ModelOpt patch for _LayerNormLinear functional."""
        _assert_te_fp8_enabled()
        # This is multi-process safe (such as in torch distributed jobs), not multi-thread safe
        _QuantLayerNormLinearFunc.modelopt_quantizers = (
            self.input_quantizer,
            self.weight_quantizer,
        )
        with replace_function(
            te_layernorm_linear,
            "_LayerNormLinear",
            _QuantLayerNormLinearFunc,
            "_og_LayerNormLinear",
        ):
            outputs = super().forward(*args, **kwargs)
        delattr(_QuantLayerNormLinearFunc, "modelopt_quantizers")
        if isinstance(outputs, tuple):
            return (self.output_quantizer(outputs[0]), *outputs[1:])
        return self.output_quantizer(outputs)
