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

"""Support quantization for VLLM layers."""

import importlib

import torch
import vllm.attention as vllm_attention
import vllm.model_executor.layers.fused_moe.layer as vllm_fused_moe_layer
import vllm.model_executor.layers.linear as vllm_linear
from vllm.attention.layers.cross_attention import CrossAttention
from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

from ...utils.distributed import ParallelState
from ..nn import QuantLinearConvBase, QuantModule, QuantModuleRegistry, TensorQuantizer

# Try multiple import paths for vLLM compatibility across versions
vllm_shared_fused_moe_layer = None
for module_path in [
    "vllm.model_executor.layers.fused_moe.shared_fused_moe",  # 0.11.0+
    "vllm.model_executor.layers.shared_fused_moe.shared_fused_moe",  # 0.10.2
]:
    try:
        vllm_shared_fused_moe_layer = importlib.import_module(module_path)
        break
    except ImportError:
        continue

try:
    from vllm.attention.layer import MLAAttention as VllmMLAAttention
except ImportError:
    VllmMLAAttention = None

vllm_fused_moe_package = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")


class FakeQuantMethod:
    """A class that implements fake quantization methods for vLLM models.

    This class provides functionality to apply quantization methods to model layers
    in a way that's compatible with vLLM's architecture.
    """

    def __init__(self, quant_method):
        """Initialize the FakeQuantMethod.

        Args:
            quant_method: The quantization method to be applied to the model layers.
        """
        self.quant_method = quant_method

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the quantization method to a given layer.

        Args:
            layer (torch.nn.Module): The neural network layer to be quantized.
            x (torch.Tensor): The input tensor to the layer.
            bias (torch.Tensor | None, optional): The bias tensor to the layer. Defaults to None.

        Returns:
            torch.Tensor: The quantized output tensor.
        """
        x = layer.input_quantizer(x)
        if layer.weight_quantizer.is_enabled:
            original_weight = layer.weight
            quantized_tensor = layer.weight_quantizer(layer.weight)
            # parameterize the quantized weight
            if isinstance(original_weight, torch.nn.Parameter) and not isinstance(
                quantized_tensor, torch.nn.Parameter
            ):
                quantized_tensor = torch.nn.Parameter(
                    quantized_tensor, requires_grad=original_weight.requires_grad
                )
            layer.weight = quantized_tensor
            output = self.quant_method.apply(layer, x, bias)
            layer.weight = original_weight
        else:
            output = self.quant_method.apply(layer, x, bias)
        output = layer.output_quantizer(output)
        return output


def create_parallel_state():
    """Create a parallel state for vLLM."""
    dp_group = get_dp_group().device_group
    tp_group = get_tp_group().device_group
    ep_group = get_ep_group().device_group
    return ParallelState(dp_group, tp_group, ep_group)


class _VLLMParallelLinear(QuantModule):
    def _setup(self):
        self.input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.output_quantizer.disable()
        assert type(self.quant_method) is vllm_linear.UnquantizedLinearMethod, (
            f"quant_method is {type(self.quant_method)}"
        )
        self.fake_quant_method = FakeQuantMethod(self.quant_method)
        self.parallel_state = create_parallel_state()

    def forward(self, input_):
        # This context manager will conflict with torch.compile
        # with replace_function(self, "quant_method", self.fake_quant_method):
        # Manually replace quant_method instead
        self._quant_method = self.quant_method
        self.quant_method = self.fake_quant_method
        output = super().forward(input_)
        self.quant_method = self._quant_method
        return output


@QuantModuleRegistry.register({vllm_linear.RowParallelLinear: "vllm_RowParallelLinear"})
class _QuantVLLMRowParallelLinear(_VLLMParallelLinear):
    pass


@QuantModuleRegistry.register({vllm_linear.ColumnParallelLinear: "vllm_ColumnParallelLinear"})
class _QuantVLLMColumnParallelLinear(_VLLMParallelLinear):
    pass


@QuantModuleRegistry.register(
    {vllm_linear.MergedColumnParallelLinear: "vllm_MergedColumnParallelLinear"}
)
class _QuantVLLMMergedColumnParallelLinear(_VLLMParallelLinear):
    pass


@QuantModuleRegistry.register({vllm_linear.QKVParallelLinear: "vllm_QKVParallelLinear"})
class _QuantVLLMQKVParallelLinear(_VLLMParallelLinear):
    pass


# ReplicatedLinear is for MoE router and should not be quantized


class _QuantFusedMoEBase(QuantModule):
    def _setup(self):
        self.w13_input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.w2_input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.w13_weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.w2_weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.w13_output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.w2_output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.w13_output_quantizer.disable()
        self.w2_output_quantizer.disable()
        assert type(self.quant_method) is vllm_fused_moe_layer.UnquantizedFusedMoEMethod, (
            f"quant_method is {type(self.quant_method)}"
        )
        self.parallel_state = create_parallel_state()

    def invoke_fused_moe_quantized(
        self,
        A: torch.Tensor,  # noqa: N803
        B: torch.Tensor,  # noqa: N803
        C: torch.Tensor,  # noqa: N803
        *args,
        **kwargs,
    ):
        if B is self.w13_weight:
            # First layer of expert
            A = self.w13_input_quantizer(A)  # noqa: N806
            if self.w13_weight_quantizer.is_enabled:
                original_weight = self.w13_weight
                self.w13_weight = self.w13_weight_quantizer(self.w13_weight)
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
                self.w13_weight = original_weight
            else:
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
            if self.w13_output_quantizer.is_enabled:
                C[:] = self.w13_output_quantizer(C)
        elif B is self.w2_weight:
            A = self.w2_input_quantizer(A)  # noqa: N806
            if self.w2_weight_quantizer.is_enabled:
                original_weight = self.w2_weight
                self.w2_weight = self.w2_weight_quantizer(self.w2_weight)
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
                self.w2_weight = original_weight
            else:
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
            if self.w2_output_quantizer.is_enabled:
                C[:] = self.w2_output_quantizer(C)
        else:
            raise ValueError("Cannot determine first or second layer of expert")

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # This is again due to the bad coding of vLLM
        # fused_moe submodule is overwritten by the fused_moe function
        # so we need to import the fused_moe module explicitly
        assert vllm_fused_moe_package.invoke_fused_moe_kernel is not None
        # This context manager will conflict with torch.compile
        # with replace_function(
        #     vllm_fused_moe_package,
        #     "invoke_fused_moe_kernel",
        #     self.invoke_fused_moe_quantized,
        # ):
        try:
            vllm_fused_moe_package._invoke_fused_moe_kernel = (  # type: ignore[attr-defined]
                vllm_fused_moe_package.invoke_fused_moe_kernel
            )
            vllm_fused_moe_package.invoke_fused_moe_kernel = self.invoke_fused_moe_quantized  # type: ignore[attr-defined]
            output = super().forward(hidden_states, router_logits)
            return output
        finally:
            vllm_fused_moe_package.invoke_fused_moe_kernel = (  # type: ignore[attr-defined]
                vllm_fused_moe_package._invoke_fused_moe_kernel
            )

    @torch.no_grad()
    def fold_weight(self):
        # the MoE weights can be super large, it consumes too much memory, so we need to fold the weight one by one
        for i in range(self.w13_weight.shape[0]):
            self.w13_weight[i].copy_(
                self.w13_weight_quantizer(self.w13_weight[i].float().contiguous()).to(
                    self.w13_weight.dtype
                )
            )
        self.w13_weight_quantizer.disable()
        for i in range(self.w2_weight.shape[0]):
            self.w2_weight[i].copy_(
                self.w2_weight_quantizer(self.w2_weight[i].float().contiguous()).to(
                    self.w2_weight.dtype
                )
            )
        self.w2_weight_quantizer.disable()

        torch.cuda.empty_cache()


@QuantModuleRegistry.register({vllm_fused_moe_layer.FusedMoE: "vllm_FusedMoE"})
class _QuantVLLMFusedMoE(_QuantFusedMoEBase):
    pass


if vllm_shared_fused_moe_layer is not None:

    @QuantModuleRegistry.register(
        {vllm_shared_fused_moe_layer.SharedFusedMoE: "vllm_SharedFusedMoE"}
    )
    class _QuantVLLMSharedFusedMoE(_QuantFusedMoEBase):
        pass


@QuantModuleRegistry.register({vllm_attention.Attention: "vllm_Attention"})
class _QuantVLLMAttention(QuantModule):
    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()
        self.parallel_state = create_parallel_state()

    def forward(self, query, key, value, *args, **kwargs):
        query = self.q_bmm_quantizer(query)
        key = self.k_bmm_quantizer(key)
        value = self.v_bmm_quantizer(value)

        return super().forward(query, key, value, *args, **kwargs)


@QuantModuleRegistry.register({CrossAttention: "vllm_CrossAttention"})
class _QuantVLLMCrossAttention(_QuantVLLMAttention):
    pass


@QuantModuleRegistry.register({EncoderOnlyAttention: "vllm_EncoderOnlyAttention"})
class _QuantVLLMEncoderOnlyAttention(_QuantVLLMAttention):
    pass


if VllmMLAAttention is not None:

    @QuantModuleRegistry.register({VllmMLAAttention: "vllm_MLAAttention"})
    class _QuantVLLMMLAAttention(QuantModule):
        def _setup(self):
            self.q_bmm_quantizer = TensorQuantizer()
            self.kv_c_bmm_quantizer = TensorQuantizer()
            self.k_pe_bmm_quantizer = TensorQuantizer()
            self.parallel_state = create_parallel_state()

        def forward(self, query, kv_c, k_pe, *args, **kwargs):
            query = self.q_bmm_quantizer(query)
            kv_c = self.kv_c_bmm_quantizer(kv_c)
            k_pe = self.k_pe_bmm_quantizer(k_pe)
            return super().forward(query, kv_c, k_pe, *args, **kwargs)
