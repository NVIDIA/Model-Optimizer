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
import os
import warnings

import torch
import transformer_engine as te
import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear
import transformer_engine.pytorch.module.linear as te_linear
from packaging.version import Version

from modelopt.torch.quantization.utils import replace_function

from ..nn import GroupedQuantizer, QuantModuleRegistry, SequentialQuantizer, TensorQuantizer
from .custom import _ParallelLinear

_TE_VERSION = Version(te.__version__)

_COMPILE_TEGROUPED_WEIGHT_LOOP_ENV = "MODELOPT_TEGROUPED_COMPILE_WEIGHT_LOOP"


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


def _is_calibrating(quantizer):
    """Return whether a tensor or sequential quantizer is collecting calibration stats."""
    if isinstance(quantizer, SequentialQuantizer):
        return any(getattr(q, "_if_calib", False) for q in quantizer)
    return getattr(quantizer, "_if_calib", False)


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

        # Each fused expert gets its own weight quantizer (independent amax), stored in a
        # GroupedQuantizer (an nn.ModuleList) surfaced as ``weight_quantizer.{i}`` so the
        # fused-experts name normalizer maps them to ``*weight_quantizer`` and the stock configs
        # apply. This replaces the single shared weight quantizer ``super()._setup()`` installed.
        self.weight_quantizer = GroupedQuantizer(
            *(TensorQuantizer() for _ in range(self.num_gemms))
        )

        # Compile only the per-expert quantizer loop. The surrounding TE grouped GEMM remains
        # eager, and the opt-in flag leaves the default execution path unchanged.
        if os.getenv(_COMPILE_TEGROUPED_WEIGHT_LOOP_ENV, "0") == "1":
            quantizers = tuple(self.weight_quantizer)

            def quantize_weights(*weights):
                return tuple(quantizer(weight) for quantizer, weight in zip(quantizers, weights))

            self._compiled_weight_quantizer_loop = torch.compile(
                quantize_weights,
                backend="inductor",
                fullgraph=False,
                mode="reduce-overhead",
            )

    def modelopt_post_restore(self, prefix: str = ""):
        # GroupedMLP stores the weights as weight0, weight1, etc. To run post_restore in order to
        # initialize the quantizer states, self.weight is used to extract shape, dtype etc. Assigning
        # self.weight0 to self.weight to run the quantizer states initialization.
        assert not hasattr(self, "weight"), "self.weight should not exist for TEGroupedLinear"
        self.weight = self.weight0
        super().modelopt_post_restore(prefix=prefix)
        # Remove self.weight after post_restore.
        delattr(self, "weight")

        # Preserve the loaded (calibrated) per-expert amax. Recomputing via max_calibrate
        # replaces MSE/static-calibrated (and QAD-frozen) amax with max|W|, which corrupts
        # static recipes on export and overwrites frozen amax on every QAD resume. Only
        # re-calibrate a quantizer whose loaded amax is shape-INCOMPATIBLE with its weight
        # (a genuine TP/EP change between save and restore); otherwise keep it as-is.
        # weight_quantizer is a GroupedQuantizer (one per expert) after _setup; guard defensively.
        if not isinstance(self.weight_quantizer, GroupedQuantizer):
            return

        from modelopt.torch.quantization.model_calib import max_calibrate

        for i in range(self.num_gemms):
            weight_i = getattr(self, f"weight{i}", None)
            if weight_i is None:
                continue
            if weight_i.device.type != "cuda":
                continue  # export loads weights on CPU; the fp4 dry-run needs CUDA — keep loaded amax
            wq_i = self.weight_quantizer[i]
            q = wq_i[0] if isinstance(wq_i, SequentialQuantizer) else wq_i
            if not hasattr(q, "_amax") or q._amax is None:
                continue
            prev_fake = getattr(q, "_fake_quant", True)
            q._fake_quant = True
            try:
                wq_i(weight_i)  # dry-run: succeeds iff the loaded amax fits this weight
                shape_ok = True
            except Exception as e:
                # Only a genuine amax/weight shape mismatch (a TP/EP change between save and
                # restore) may fall through to the max|W| recompute below. A CUDA/OOM/device or
                # any other error must NOT be silently turned into a recompute -- that would
                # discard the stored MSE/static/QAD amax this block exists to preserve. Re-raise
                # anything that is not clearly a shape mismatch.
                msg = str(e).lower()
                is_shape_mismatch = (
                    isinstance(e, RuntimeError)
                    and any(
                        k in msg for k in ("size", "shape", "must match", "broadcast", "dimension")
                    )
                    and not any(k in msg for k in ("cuda", "out of memory", "device-side", "nccl"))
                )
                if not is_shape_mismatch:
                    raise
                shape_ok = False
            finally:
                q._fake_quant = prev_fake
            if shape_ok:
                continue  # loaded amax is valid -> keep it, do NOT recompute
            # Recompute is lossy for static recipes; never do it silently.
            warnings.warn(
                f"{type(self).__name__}: restored amax {tuple(q._amax.shape)} for expert {i} "
                f"weight_quantizer is shape-incompatible with weight {tuple(weight_i.shape)} "
                f"(likely a TP/EP change); recomputing as max|W| and discarding the stored "
                f"MSE/static/QAD amax for this expert."
            )
            wq_i.reset_amax()
            max_calibrate(wq_i, lambda wq, w=weight_i: wq(w), distributed_sync=False)

    def iter_weights_for_calibration(self):
        """Yield ``(weight_i, weight_quantizer)`` for each of the ``num_gemms`` grouped weights."""
        grouped = isinstance(self.weight_quantizer, GroupedQuantizer)
        for i in range(self.num_gemms):
            weight_i = getattr(self, f"weight{i}", None)
            if weight_i is not None:
                yield weight_i, (self.weight_quantizer[i] if grouped else self.weight_quantizer)

    def fold_weight(self, keep_attrs: bool = False):
        """Fold each grouped weight with its corresponding fake-quant quantizer."""
        quantizer_weights: dict[TensorQuantizer, list[torch.Tensor]] = {}
        for weight, quantizer in self.iter_weights_for_calibration():
            if isinstance(quantizer, TensorQuantizer) and quantizer.fake_quant:
                quantizer_weights.setdefault(quantizer, []).append(weight)

        for quantizer, weights in quantizer_weights.items():
            self._fold_weight_quantizer(quantizer, weights, keep_attrs)

    @staticmethod
    def te_grouped_quantized_linear_fn(package, func_name, self, *args):
        _assert_te_fp8_enabled()
        # Locate `inp` by parameter name in the un-patched `_GroupedLinear.forward`
        # signature — robust to TE versions that move the m_splits/non_tensor_args
        # slot around (e.g. `m_splits` was packed into `non_tensor_args[0]` in TE
        # 2.10-2.15, then split back out into its own arg in TE 2.16+).
        # `num_gemms` comes from `self.num_gemms` (set by the public
        # `GroupedLinear.__init__`) rather than from introspecting `*args`, so it's
        # unaffected by that churn.
        # `*weights_and_biases` is always the trailing variadic — 2 * num_gemms tensors
        # (weights, then biases).
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
        num_gemms = self.num_gemms
        weights_start = len(args) - 2 * num_gemms

        new_args = list(args)
        new_args[inp_pos] = self.input_quantizer(args[inp_pos])
        weights = tuple(args[weights_start : weights_start + num_gemms])
        # Calibration mutates collector state and must stay outside Inductor/CUDAGraph capture.
        grouped = isinstance(self.weight_quantizer, GroupedQuantizer)
        use_compiled_loop = (
            grouped
            and hasattr(self, "_compiled_weight_quantizer_loop")
            and not any(_is_calibrating(quantizer) for quantizer in self.weight_quantizer)
        )
        if use_compiled_loop:
            quantized_weights = self._compiled_weight_quantizer_loop(*weights)
        else:
            quantized_weights = tuple(
                (self.weight_quantizer[gemm_idx] if grouped else self.weight_quantizer)(weight)
                for gemm_idx, weight in enumerate(weights)
            )
        for gemm_idx, quantized_weight in enumerate(quantized_weights):
            new_args[weights_start + gemm_idx] = quantized_weight
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
