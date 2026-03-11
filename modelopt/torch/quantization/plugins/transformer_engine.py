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

import warnings

import torch
import transformer_engine as te
import transformer_engine.pytorch.module.grouped_linear as te_grouped_linear
import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear
import transformer_engine.pytorch.module.linear as te_linear
from packaging.version import Version

from modelopt.torch.quantization.utils import replace_function

from ..nn import QuantModuleRegistry
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
        if Version("2.0") <= _TE_VERSION:
            idx = 1 if func_name == "_forward" else 0
            weight, inputs = args[idx], args[idx + 1]
            remaining_args = args[idx + 2 :]
            weight = self.weight_quantizer(weight)
            inputs = self.input_quantizer(inputs)
            new_args = (weight, inputs, *remaining_args)
            new_args = (args[0], *new_args) if func_name == "_forward" else new_args
            output = getattr(package, func_name)(
                *new_args,
                **kwargs,
            )
        else:
            idx = 1 if func_name == "_forward" else 0
            weight, weight_fp8, inputs = args[idx], args[idx + 1], args[idx + 2]
            remaining_args = args[idx + 3 :]
            weight = self.weight_quantizer(weight)
            inputs = self.input_quantizer(inputs)
            new_args = (weight, weight_fp8, inputs, *remaining_args)
            new_args = (args[0], *new_args) if func_name == "_forward" else new_args
            output = getattr(package, func_name)(
                *new_args,
                **kwargs,
            )
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

        # TODO: GroupedLinear supports weights split by `num_gemms`, to support quantization
        # with static parameters beyond per-tensor, we need to support a unique quantizer for each gemm.

    def modelopt_post_restore(self, prefix: str = ""):
        # GroupedMLP stores the weights as weight0, weight1, etc. To run post_restore in order to
        # initialize the quantizer states, self.weight is used to extract shape, dtype etc. Assigning
        # self.weight0 to self.weight to run the quantizer states initialization.
        assert not hasattr(self, "weight"), "self.weight should not exist for TEGroupedLinear"
        self.weight = self.weight0
        super().modelopt_post_restore(prefix=prefix)
        # Remove self.weight after post_restore.
        delattr(self, "weight")

    @staticmethod
    def te_grouped_quantized_linear_fn(package, func_name, self, *args):
        _assert_te_fp8_enabled()
        idx = 1 if func_name == "_forward" else 0
        inp = args[idx]

        # Handle both old and new TE signatures (changed in PR #2377 in TE 2.10)
        # New signature (TE >= 2.10): forward(ctx, inp, non_tensor_args: Tuple, *weights_and_biases)
        # Old signature (TE < 2.10): forward(ctx, inp, m_splits: List[int], use_bias, ...)
        if Version("2.10") <= _TE_VERSION:
            # New signature: non_tensor_args is a tuple, m_splits is the first element
            num_gemms = len(args[idx + 1][0])
        else:
            # Old signature: m_splits is directly args[idx + 1]
            num_gemms = len(args[idx + 1])

        weights_and_biases = args[-2 * num_gemms :]
        weights, biases = weights_and_biases[:num_gemms], weights_and_biases[num_gemms:]
        quantized_inputs = self.input_quantizer(inp)
        quantized_weights = [self.weight_quantizer(weight) for weight in weights]

        output = getattr(package, func_name)(
            *(
                args[0],
                quantized_inputs,
            )
            if func_name == "_forward"
            else (quantized_inputs,),
            *args[idx + 1 : -2 * num_gemms],
            *quantized_weights,
            *biases,
        )
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
