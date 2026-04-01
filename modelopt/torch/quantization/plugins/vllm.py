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
from contextlib import contextmanager
from itertools import chain

import torch

# Try multiple import paths for vLLM compatibility across versions
if importlib.util.find_spec("vllm.attention"):
    import vllm.attention as vllm_attention  # vllm < 0.16.0
else:
    import vllm.model_executor.layers.attention as vllm_attention  # vllm >= 0.16.0

import vllm.model_executor.layers.fused_moe.layer as vllm_fused_moe_layer
import vllm.model_executor.layers.linear as vllm_linear
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

from ...utils.distributed import ParallelState
from ..nn import QuantLinearConvBase, QuantModule, QuantModuleRegistry, TensorQuantizer
from .custom import CUSTOM_MODEL_PLUGINS

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
    _has_attention_layers = importlib.util.find_spec("vllm.attention.layers") is not None
except (ModuleNotFoundError, ValueError):
    _has_attention_layers = False

if _has_attention_layers:  # vllm < 0.15.0
    from vllm.attention.layers.cross_attention import CrossAttention
    from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
else:
    try:
        from vllm.model_executor.layers.attention.cross_attention import CrossAttention
    except ImportError:
        CrossAttention = None
    try:
        from vllm.model_executor.layers.attention.encoder_only_attention import EncoderOnlyAttention
    except ImportError:
        EncoderOnlyAttention = None

try:
    _has_attention_layer = importlib.util.find_spec("vllm.attention.layer") is not None
except (ModuleNotFoundError, ValueError):
    _has_attention_layer = False

if _has_attention_layer:
    import vllm.attention.layer as vllm_attention

try:
    VllmMLAAttention = vllm_attention.MLAAttention
except (AttributeError, ImportError):
    VllmMLAAttention = None

_ATTENTION_TYPES = tuple(
    t
    for t in [vllm_attention.Attention, CrossAttention, EncoderOnlyAttention, VllmMLAAttention]
    if t is not None
)

vllm_fused_moe_package = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")


@contextmanager
def disable_compilation(model):
    """Disable compilation for a model.

    Args:
        model: The model to disable compilation for.
    """
    do_not_compile = True
    if hasattr(model, "model"):
        do_not_compile = model.model.do_not_compile
        model.model.do_not_compile = True
    elif hasattr(model, "language_model"):
        do_not_compile = model.language_model.model.do_not_compile
        model.language_model.model.do_not_compile = True
    else:
        raise ValueError("Model does not have a model or language_model attribute")

    try:
        yield
    finally:
        if hasattr(model, "model"):
            model.model.do_not_compile = do_not_compile
        elif hasattr(model, "language_model"):
            model.language_model.model.do_not_compile = do_not_compile


# vLLM Attention stores ``device``/``dtype`` as plain attrs; ``dtype`` may be a string
# (e.g. ``"float16"``, ``"auto"``). We resolve and stamp concrete torch types before
# QuantModule replacement. Priority: explicit attrs → KV-cache → shallow tensor scan.
# No model-wide fallback: a tensor from a different shard gives the wrong device under TP.


def _vllm_attr_dtype_to_torch(dtype) -> torch.dtype | None:
    """Resolve vLLM dtype attr to ``torch.dtype``; ``None`` for ``"auto"`` (caller falls through)."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and dtype != "auto":
        resolved = getattr(torch, dtype, None)
        if resolved is None:
            raise ValueError(f"Unrecognized vLLM dtype string: {dtype!r}")
        return resolved
    return None


def _get_device_dtype(module: torch.nn.Module) -> tuple:
    """Return ``(device, dtype)`` for a vLLM Attention module, or ``(None, None)`` if unresolvable."""
    # Explicit attrs set by vLLM at construction — primary path.
    dev, dt = getattr(module, "device", None), getattr(module, "dtype", None)
    if dev is not None and dt is not None:
        dt_resolved = _vllm_attr_dtype_to_torch(dt)
        if dt_resolved is not None:
            return dev, dt_resolved

    # KV-cache tensors are available after allocation; respect kv_cache_dtype when set.
    # kv_cache is a list of tensors (v0) or a single tensor (v1).
    kv = getattr(module, "kv_cache", None)
    if kv is not None:
        t0 = kv[0] if isinstance(kv, (list, tuple)) and len(kv) > 0 else kv
        if isinstance(t0, torch.Tensor) and t0.numel() > 0:
            spec = getattr(module, "kv_cache_dtype", t0.dtype)
            out_dtype = (
                t0.dtype if spec == "auto" else (_vllm_attr_dtype_to_torch(spec) or t0.dtype)
            )
            return t0.device, out_dtype

    # Shallow scan: weights often live on child modules rather than the attention module itself.
    for mod in (module, *module.children()):
        for t in chain(mod.parameters(recurse=False), mod.buffers(recurse=False)):
            return t.device, t.dtype

    return None, None


def vllm_replace_quant_module_hook(model: torch.nn.Module) -> None:
    """Stamp resolved (device, dtype) onto Attention modules before QuantModule replacement."""
    for _n, m in model.named_modules():
        if isinstance(m, _ATTENTION_TYPES):
            m.device, m.dtype = _get_device_dtype(m)


CUSTOM_MODEL_PLUGINS.add(vllm_replace_quant_module_hook)


def _vllm_attention_modelopt_post_restore(self) -> None:
    """Move Attention module to its correct device after ModelOpt state restore."""
    device, dtype = _get_device_dtype(self)
    if device is None or dtype is None:
        raise RuntimeError(
            "Could not determine device/dtype for vLLM Attention. "
            "Ensure vllm_replace_quant_module_hook runs before replace_quant_module."
        )
    self.to(device=device)


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
        if layer.input_quantizer.is_enabled:
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
    try:
        # EP group is only created for MoE models; dense models don't have one.
        ep_group = get_ep_group().device_group
    except (AssertionError, RuntimeError):
        ep_group = -1
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

    def _sync_input_pre_quant_scale_to_weight(self) -> None:
        """Align pre_quant_scale to weight (vLLM CUTLASS expects matching device/dtype)."""
        pqs = getattr(self.input_quantizer, "_pre_quant_scale", None)
        if pqs is None:
            return
        w = getattr(self, "weight", None)
        if w is None or not isinstance(w, torch.Tensor) or w.is_meta:
            return
        if pqs.device != w.device or pqs.dtype != w.dtype:
            self.input_quantizer._pre_quant_scale.data = pqs.data.to(device=w.device, dtype=w.dtype)

    def modelopt_post_restore(self, prefix: str = "") -> None:
        super().modelopt_post_restore(prefix=prefix)
        self._sync_input_pre_quant_scale_to_weight()

    def forward(self, input_):
        # This context manager will conflict with torch.compile
        # with replace_function(self, "quant_method", self.fake_quant_method):
        # Manually replace quant_method instead
        self._quant_method = self.quant_method
        self.quant_method = self.fake_quant_method
        output = super().forward(input_)
        self.quant_method = self._quant_method
        return output


def post_restore_vllm_parallel_linears(model: torch.nn.Module) -> None:
    """Re-run modelopt_post_restore on vLLM parallel linears after set_quantizer_state_dict.

    restore_quantizer_state already calls modelopt_post_restore on all QuantModules, but vLLM
    reload paths that load modelopt_state_weights via set_quantizer_state_dict do not.
    """
    for module in model.modules():
        if isinstance(module, _VLLMParallelLinear):
            module.modelopt_post_restore("")


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
    def fold_weight(self, keep_attrs: bool = False):
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

    def modelopt_post_restore(self, prefix: str = "") -> None:
        _vllm_attention_modelopt_post_restore(self)


if CrossAttention is not None:

    @QuantModuleRegistry.register({CrossAttention: "vllm_CrossAttention"})
    class _QuantVLLMCrossAttention(_QuantVLLMAttention):
        pass


if EncoderOnlyAttention is not None:

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

        def modelopt_post_restore(self, prefix: str = "") -> None:
            _vllm_attention_modelopt_post_restore(self)
