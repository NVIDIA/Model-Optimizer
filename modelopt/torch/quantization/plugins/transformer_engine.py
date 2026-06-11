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

import inspect
import warnings

import torch
import transformer_engine as te
import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear
import transformer_engine.pytorch.module.linear as te_linear
from packaging.version import Version

from modelopt.torch.quantization.utils import replace_function

from ..nn import QuantModuleRegistry, TensorQuantizer
from .custom import _ParallelLinear

_TE_VERSION = Version(te.__version__)


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


def _reshape_loaded_amax_to_buffer_shape(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Reshape a loaded `_amax` tensor to match the live buffer shape on this rank.

    PyTorch's `load_state_dict` enforces an exact-shape check at the leaf module
    level. For per-expert TEGroupedLinear quantization, the dist-checkpoint tensor
    is saved flat as `[num_gemms]` (the legacy `_process_quantizer_amax` flatten)
    while the live buffer is registered as `[num_gemms, 1, 1]` from the metadata
    path (axis=0 calibration with keepdims). Reshape the loaded tensor in the
    state_dict the strict check is about to read.

    The hook is leaf-level and rank-local — it only touches the local
    `state_dict[prefix + "_amax"]` and the local buffer's shape, and only acts
    when shapes differ but numel matches. EP>1 narrowing happens in the outer
    `_QuantMegatronTEGroupedLinear._load_from_state_dict` before this hook runs.
    """
    key = prefix + "_amax"
    if key not in state_dict:
        return
    buf = getattr(module, "_amax", None)
    if buf is None:
        return
    loaded = state_dict[key]
    if loaded.shape == buf.shape:
        return
    if loaded.numel() == buf.numel():
        state_dict[key] = loaded.reshape(buf.shape).contiguous()


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

        # Reconcile the per-expert _amax storage shape on restore. The save path keeps
        # the dist-checkpoint tensor as flat [num_gemms_global] (legacy convention for
        # quantizer amax) while _pytorch_state_metadata records the live multi-dim
        # buffer shape [num_gemms, 1, 1] (axis=0 calibration + keepdims). On restore,
        # the metadata path registers an [num_gemms, 1, 1] buffer first; load_state_dict
        # then tries to copy the flat tensor into it and raises a size mismatch.
        #
        # Reshape inside a pre-hook at the leaf quantizer level so it fires AFTER
        # PyTorch has narrowed state_dict to this submodule's prefix but BEFORE the
        # strict size check. The hook is pure-CPU and rank-local — no per-rank state
        # queries beyond the leaf's own buffer shape, so it doesn't introduce the
        # rank-asymmetric collective ordering that the parent-level state_dict
        # mutation approach did. EP>1 still relies on the outer narrow path.
        if hasattr(self, "weight_quantizer") and isinstance(self.weight_quantizer, TensorQuantizer):
            self.weight_quantizer._register_load_state_dict_pre_hook(
                _reshape_loaded_amax_to_buffer_shape, with_module=True
            )

    def _is_per_expert_weight_quant(self) -> bool:
        # Per-expert mode: axis=0 on the weight quantizer means the stacked
        # [num_gemms, out, in] weight is reduced over (1, 2), producing one amax
        # per expert. Anything else (axis=None, axis=(0, 1), ...) keeps the
        # legacy per-tensor path.
        quantizer = getattr(self, "weight_quantizer", None)
        if not isinstance(quantizer, TensorQuantizer):
            return False
        axis = quantizer.axis
        return axis in (0, (0,))

    def modelopt_post_restore(self, prefix: str = ""):
        # GroupedMLP stores the weights as weight0, weight1, etc. To run post_restore in order to
        # initialize the quantizer states, self.weight is used to extract shape, dtype etc.
        #
        # Per-expert mode (axis=0) stores _amax as [num_gemms, 1, 1] — the same shape
        # iter_weights_for_calibration and te_grouped_quantized_linear_fn produce by stacking
        # the per-expert weights first. Mirror that here so the restore path initializes the
        # placeholder _amax to the same shape; otherwise dist-checkpoint load builds _amax from
        # the un-stacked [out, in] view (shape [out, 1]) and _gather_global_per_expert_amax
        # later fails on `amax.view(num_gemms)`.
        assert not hasattr(self, "weight"), "self.weight should not exist for TEGroupedLinear"
        if self._is_per_expert_weight_quant():
            weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
            self.weight = torch.stack(weights, dim=0)
        else:
            self.weight = self.weight0
        super().modelopt_post_restore(prefix=prefix)
        # Remove self.weight after post_restore.
        delattr(self, "weight")

    def iter_weights_for_calibration(self):
        """Yield grouped weights for calibration.

        Per-tensor (``axis=None``): yields ``(weight_i, weight_quantizer)`` for each
        expert separately; the calibrator accumulates a single amax across all experts.

        Per-expert (``axis=0``): yields ``(stacked, weight_quantizer)`` once, where
        ``stacked`` has shape ``[num_gemms, out, in]``. The quantizer's axis-0 reduction
        then produces one amax per expert (shape ``[num_gemms, 1, 1]``).
        """
        if self._is_per_expert_weight_quant():
            weights = [getattr(self, f"weight{i}", None) for i in range(self.num_gemms)]
            if any(w is None for w in weights):
                return
            yield torch.stack(weights, dim=0), self.weight_quantizer
            return

        for i in range(self.num_gemms):
            weight_i = getattr(self, f"weight{i}", None)
            if weight_i is not None:
                yield weight_i, self.weight_quantizer

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
        if self._is_per_expert_weight_quant():
            # Stack the N expert weights into one [N, out, in] tensor so the
            # quantizer's axis-0 reduction sees them together; the resulting
            # _amax of shape [N, 1, 1] broadcasts to give per-expert fake-quant.
            #
            # Use unbind(0) (not q_stacked[i]) for the unpacking — unbind's
            # backward is a single fused stack op, vs N SelectBackward nodes
            # that each allocate an intermediate grad buffer per slice. At
            # high N (hundreds of experts), this collapses backward latency
            # by orders of magnitude. Measured under OMNIML-5064.
            stacked = torch.stack(list(args[weights_start : weights_start + num_gemms]), dim=0)
            q_stacked = self.weight_quantizer(stacked)
            new_args[weights_start : weights_start + num_gemms] = q_stacked.unbind(0)
        else:
            for i in range(weights_start, weights_start + num_gemms):
                new_args[i] = self.weight_quantizer(args[i])
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
