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

import contextvars
import importlib
import os
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
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
# vLLM may call one entry (e.g. ``dispatch_fused_moe_kernel``) which then calls another on the same
# module (e.g. ``invoke_fused_moe_triton_kernel``). Patching every name would otherwise apply fakequant
# twice; see ``_moe_fakequant_active`` in ``invoke_fused_moe_quantized``.
_FUSED_MOE_KERNEL_CANDIDATES = (
    "invoke_fused_moe_kernel",
    "invoke_fused_moe_triton_kernel",
    "dispatch_fused_moe_kernel",
)
_FUSED_MOE_KERNEL_FUNCS = tuple(
    n for n in _FUSED_MOE_KERNEL_CANDIDATES if hasattr(vllm_fused_moe_package, n)
)


_FORCE_TRITON_MOE = os.environ.get("MODELOPT_FORCE_TRITON_MOE", "0").lower() in (
    "1",
    "true",
    "yes",
)


def _force_triton_moe_dispatch() -> None:
    """Force vLLM's compressed-tensors WNA16 MoE dispatcher onto the Triton (non-Marlin) path.

    Marlin's fused MoE kernel runs gate→silu→up→down as a single CUDA kernel; the
    intermediate activation entering ``down_proj`` (= w2 input) never materialises as
    a Python tensor, so a wrapper at the FusedMoE module level can only observe the
    block input (w13_input). Switching to ``CompressedTensorsWNA16MoEMethod`` (Triton)
    splits the expert forward into two ``dispatch_fused_moe_kernel`` calls with the
    intermediate tensor in between — exactly where the existing modelopt kernel
    monkey-patch can intercept w2_input.

    Patched at the import site in ``compressed_tensors_moe.compressed_tensors_moe``
    so other Marlin support checks (dense Linear layers) are unaffected.
    """
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
            compressed_tensors_moe as _ct_moe_dispatcher,
        )
    except ImportError:
        return

    if not hasattr(_ct_moe_dispatcher, "check_moe_marlin_supports_layer"):
        return

    _ct_moe_dispatcher.check_moe_marlin_supports_layer = lambda *_a, **_kw: False
    print(
        "[modelopt] MODELOPT_FORCE_TRITON_MOE=1: forced compressed-tensors MoE dispatch "
        "to CompressedTensorsWNA16MoEMethod (Triton). w2 input observable for calibration.",
        flush=True,
    )


if _FORCE_TRITON_MOE:
    _force_triton_moe_dispatch()

_moe_fakequant_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "moe_fakequant_active", default=False
)


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
        # The original assertion required UnquantizedFusedMoEMethod. We now also
        # accept compressed-tensors INT4 sources (e.g.
        # CompressedTensorsWNA16MarlinMoEMethod) so a model that ships pack-
        # quantized routed experts can be re-calibrated via vLLM runtime.
        # When the source is compressed, the layer carries ``w13_weight_packed``
        # (int32) instead of ``w13_weight``, and dispatches to the Marlin / TRT-LLM
        # MoE kernel directly from ``quant_method.apply``. Modelopt cannot
        # monkey-patch ``invoke_fused_moe_kernel`` to intercept those paths, so we
        # take a different approach in ``forward`` and ``fold_weight``: collect
        # input amax via the input-quantizer call BEFORE handing off to the
        # original Marlin kernel, and skip weight folding (export-side handles
        # that path from the source packed weights).
        self._compressed_source = type(self.quant_method) is not (
            vllm_fused_moe_layer.UnquantizedFusedMoEMethod
        )
        # When the dispatcher has been forced onto the Triton (non-Marlin) WNA16 path
        # (MODELOPT_FORCE_TRITON_MOE=1), the per-expert forward goes through two
        # ``dispatch_fused_moe_kernel`` calls with the intermediate activation
        # exposed in between, so the kernel monkey-patch CAN intercept both w13
        # and w2 inputs. For Marlin (or any single-kernel fused path) the
        # intermediate is hidden and only w13_input is observable.
        qm_name = type(self.quant_method).__name__
        self._compressed_source_triton = (
            self._compressed_source and qm_name == "CompressedTensorsWNA16MoEMethod"
        )
        if self._compressed_source:
            print(
                f"[modelopt] vLLM FusedMoE wrapped with compressed source "
                f"quant_method={qm_name}; "
                f"w2_input observable={self._compressed_source_triton}",
                flush=True,
            )
            # Source weights are already INT4-packed; never fake-quant them
            # during the calibration forward. Disable up-front (mirror of
            # what fold_weight does post-calibration) so the kernel-patch
            # path doesn't try to wrap ``weight_packed`` as a BF16 tensor.
            self.w13_weight_quantizer.disable()
            self.w2_weight_quantizer.disable()
        self.parallel_state = create_parallel_state()

    def invoke_fused_moe_quantized(
        self,
        A: torch.Tensor,  # noqa: N803
        B: torch.Tensor,  # noqa: N803
        C: torch.Tensor,  # noqa: N803
        *args,
        original_kernel: Callable,
        **kwargs,
    ):
        # Nested module-level entry (e.g. dispatch -> triton): call the real kernel once, no second quant.
        if _moe_fakequant_active.get():
            return original_kernel(A, B, C, *args, **kwargs)
        token = _moe_fakequant_active.set(True)
        try:
            return self._invoke_fused_moe_quantized_function(
                A, B, C, *args, original_kernel=original_kernel, **kwargs
            )
        finally:
            _moe_fakequant_active.reset(token)

    def _invoke_fused_moe_quantized_function(
        self,
        A: torch.Tensor,  # noqa: N803
        B: torch.Tensor,  # noqa: N803
        C: torch.Tensor,  # noqa: N803
        *args,
        original_kernel: Callable,
        **kwargs,
    ):
        # Compressed-tensors source layers carry ``w13_weight_packed`` / ``w2_weight_packed``
        # (INT4 packed as int32 / uint8) instead of the unquantized ``w13_weight`` / ``w2_weight``.
        # Match either, so the same dispatch identifies the first vs. second expert GEMM.
        w13_w = getattr(self, "w13_weight", None)
        w13_w_packed = getattr(self, "w13_weight_packed", None)
        w2_w = getattr(self, "w2_weight", None)
        w2_w_packed = getattr(self, "w2_weight_packed", None)
        is_w13 = (w13_w is not None and B is w13_w) or (
            w13_w_packed is not None and B is w13_w_packed
        )
        is_w2 = (w2_w is not None and B is w2_w) or (
            w2_w_packed is not None and B is w2_w_packed
        )
        if is_w13:
            # First layer of expert
            A = self.w13_input_quantizer(A)  # noqa: N806
            # Compressed source: weights are INT4-packed (no ``self.w13_weight`` attr); skip
            # the fake-quant-wrap branch unconditionally. ``mtq.quantize`` re-enables weight
            # quantizers from the wildcard config regardless of ``_setup``-time disables.
            if self.w13_weight_quantizer.is_enabled and not self._compressed_source:  # pragma: no cover
                # Same pattern as FakeQuantMethod.apply: wrap as nn.Parameter if needed, swap
                # w13_weight, call kernel, restore (tensor cannot stay assigned to nn.Parameter slot).
                original_weight = self.w13_weight
                quantized_tensor = self.w13_weight_quantizer(original_weight)
                try:
                    if isinstance(original_weight, torch.nn.Parameter) and not isinstance(
                        quantized_tensor, torch.nn.Parameter
                    ):
                        quantized_tensor = torch.nn.Parameter(
                            quantized_tensor, requires_grad=original_weight.requires_grad
                        )
                    self.w13_weight = quantized_tensor
                    B = quantized_tensor  # noqa: N806
                    original_kernel(A, B, C, *args, **kwargs)
                finally:
                    self.w13_weight = original_weight
            else:
                original_kernel(A, B, C, *args, **kwargs)
            if self.w13_output_quantizer.is_enabled:
                C[:] = self.w13_output_quantizer(C)
        elif is_w2:
            A = self.w2_input_quantizer(A)  # noqa: N806
            if self.w2_weight_quantizer.is_enabled and not self._compressed_source:  # pragma: no cover
                original_weight = self.w2_weight
                quantized_tensor = self.w2_weight_quantizer(original_weight)
                try:
                    if isinstance(original_weight, torch.nn.Parameter) and not isinstance(
                        quantized_tensor, torch.nn.Parameter
                    ):
                        quantized_tensor = torch.nn.Parameter(
                            quantized_tensor, requires_grad=original_weight.requires_grad
                        )
                    self.w2_weight = quantized_tensor
                    B = quantized_tensor  # noqa: N806
                    original_kernel(A, B, C, *args, **kwargs)
                finally:
                    self.w2_weight = original_weight
            else:
                original_kernel(A, B, C, *args, **kwargs)
            if self.w2_output_quantizer.is_enabled:
                C[:] = self.w2_output_quantizer(C)
        else:
            raise ValueError("Cannot determine first or second layer of expert")

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self._compressed_source and not self._compressed_source_triton:
            # Marlin (or any single-kernel fused) compressed source: the kernel
            # hides the post-activation intermediate, so we can only collect
            # amax for the activation entering the MoE block. ``w2_input_quantizer``
            # is left empty along this path.
            if self.w13_input_quantizer.is_enabled:
                hidden_states = self.w13_input_quantizer(hidden_states)
            return super().forward(hidden_states, router_logits)
        # This is again due to the bad coding of vLLM
        # fused_moe submodule is overwritten by the fused_moe function
        # so we need to import the fused_moe module explicitly
        assert _FUSED_MOE_KERNEL_FUNCS and all(
            getattr(vllm_fused_moe_package, n, None) is not None for n in _FUSED_MOE_KERNEL_FUNCS
        )
        # This context manager will conflict with torch.compile
        # with replace_function(
        #     vllm_fused_moe_package,
        #     "invoke_fused_moe_kernel",
        #     self.invoke_fused_moe_quantized,
        # ):
        originals = {n: getattr(vllm_fused_moe_package, n) for n in _FUSED_MOE_KERNEL_FUNCS}
        try:
            for n in _FUSED_MOE_KERNEL_FUNCS:
                setattr(
                    vllm_fused_moe_package,
                    n,
                    partial(
                        self.invoke_fused_moe_quantized,
                        original_kernel=originals[n],
                    ),
                )
            output = super().forward(hidden_states, router_logits)
            return output
        finally:
            for n in _FUSED_MOE_KERNEL_FUNCS:
                setattr(vllm_fused_moe_package, n, originals[n])

    @torch.no_grad()
    def fold_weight(self, keep_attrs: bool = False):
        if self._compressed_source:
            # Compressed-tensors source: the layer has ``w13_weight_packed``
            # (int32) instead of ``w13_weight``, so the per-expert in-place fold
            # below would AttributeError. Skip the actual fold; the unified-HF
            # export path materializes the dequantized weight from the source
            # packed format and applies the calibrated quantization there.
            # We still disable the weight quantizers so the post-fold check in
            # fakequant_worker passes (a still-enabled weight quantizer would
            # double-quantize activations on subsequent forwards).
            self.w13_weight_quantizer.disable()
            self.w2_weight_quantizer.disable()
            return
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

        if torch.cuda.is_available():
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
