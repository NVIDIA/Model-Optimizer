# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Support quantization for huggingface layers."""

import importlib
import inspect
import logging
import warnings
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import transformers
from torch import Tensor
from torch.nn.functional import linear
from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear
from transformers.models.t5.modeling_t5 import T5Attention

from modelopt.torch.kernels.common.attention import IS_AVAILABLE as TRITON_FA_AVAILABLE
from modelopt.torch.kernels.common.attention import (
    triton_attention_forward,
    validate_triton_attention_envelope,
)
from modelopt.torch.kernels.quantization.gemm import IS_AVAILABLE as IS_TRITON_AVAILABLE
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils.distributed import ParallelState

from ..algorithms import AutoQuantizeGradientSearcher
from ..conversion import register
from ..nn import QuantInputBase, QuantModule, QuantModuleRegistry, TensorQuantizer
from ..nn.modules.quant_linear import _QuantLinear
from ..utils import replace_function, sync_moe_expert_amax
from ..utils.layerwise_calib import LayerActivationCollector
from .attention import register_attention_for_kv_quant
from .custom import CUSTOM_MODEL_PLUGINS, _ParallelLinear, _QuantFunctionalMixin

logger = logging.getLogger(__name__)

try:
    from torch.distributed.tensor import Shard
except ImportError:
    Shard = None

try:
    import kitchen
    from kitchen.fa import KitchenFlashAttentionModule
    from kitchen.triton_module import triton_fa_params
except ImportError:
    kitchen = None

if IS_TRITON_AVAILABLE:
    from modelopt.torch.kernels.quantization.gemm import weight_dequant
else:
    weight_dequant = None

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["register_hf_attentions_on_the_fly"]


class _QuantAttention(QuantModule):
    """Attention class for KV Cache quantization of modules using transformers' attention interface."""

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()
        self.p_bmm_quantizer = TensorQuantizer()
        self.kitchen_attn_fn = None
        self.use_kitchen = False

    def _init_kitchen_attn_fn(self):
        if not self.p_bmm_quantizer.is_enabled:
            self.kitchen_attn_fn = "disabled"
            return
        self.use_kitchen = True
        if self.p_bmm_quantizer.is_mxfp(8):
            qfa_params = triton_fa_params.QTritonFAParams(
                backend="triton",
                qk_dot_precisions="bf16@bf16",
                pv_dot_precisions="mxfp8_e4m3_emulation@bf16",
                dp_v_x_do_dot_precisions="bf16@bf16",
                dp_do_x_v_dot_precisions="bf16@bf16",
                dq_ds_x_k_dot_precisions="bf16@bf16",
                dk_ds_x_q_dot_precisions="bf16@bf16",
                dv_p_x_do_dot_precisions="bf16@bf16",
                use_natural_transcendental_func=False,  # Different from default
            )
        else:
            raise NotImplementedError(f"p_bmm_quantizer not supported: {self.p_bmm_quantizer}")

        self.kitchen_attn_fn = KitchenFlashAttentionModule(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=self.config.head_dim,
            num_gqa_groups=None,  # self.config.num_key_value_heads, kitchen does not support gqa.
            attention_dropout=self.config.attention_dropout,
            qkv_format="sbhd",  # this is not used at all, but in forward, this is the only supported format.
            attn_mask_type="causal",
            window_size=getattr(self.config, "sliding_window", None),
            sequence_parallel=False,
            get_rng_state_tracker=None,
            layer_number=None,
            attention_type="self",
            softmax_scale=None,  # This will be convert to the same default as sdpa: 1/sqrt(dim_q)
            qfa_params=qfa_params,
        )

    def _p_qdq_mode(self) -> str | None:
        """Map the p_bmm_quantizer config to a Triton P quant-dequant mode.

        Returns "fp8" for per-tensor E4M3, "nvfp4" for dynamic E2M1 with
        block-16 E4M3 scales, or None when the p_bmm_quantizer is disabled
        or its format is not supported by the built-in Triton kernel
        (e.g. MXFP8, which goes through kitchen).
        """
        pq = self.p_bmm_quantizer
        if not pq.is_enabled:
            return None
        if pq.is_fp8:
            return "fp8"
        # Only dynamic NVFP4 maps to the kernel, and only at block size 16 (the kernel
        # hardcodes it). Static (calibrated) NVFP4 is excluded because is_nvfp4_dynamic
        # requires dynamically-computed block scales.
        if pq.is_nvfp4_dynamic and (pq.block_sizes or {}).get(-1, None) == 16:
            return "nvfp4"
        return None

    def _triton_qdq_attention(self, p_qdq, query_states, key_states, value_states, **kwargs):
        """Quantized attention via the built-in Triton kernel (no kitchen required).

        Fake quant-dequant of the softmax probabilities (P) is fused into the
        flash-attention kernel; see ``p_qdq`` in
        :func:`modelopt.torch.kernels.common.attention.triton_fa.attention`.

        Inputs outside the kernel/wrapper envelope (sliding window, sinks,
        softcapping, non-causal masks, ...) raise ``NotImplementedError``
        instead of silently computing wrong attention; see
        :func:`validate_triton_attention_envelope
        <modelopt.torch.kernels.common.attention.hf_triton_attention.validate_triton_attention_envelope>`.
        """
        if not TRITON_FA_AVAILABLE:
            raise RuntimeError(
                f"p_bmm_quantizer ({p_qdq}) requires the Triton attention kernel. "
                "Install triton with `pip install triton` and run on a CUDA device."
            )
        assert (
            triton_attention_forward is not None and validate_triton_attention_envelope is not None
        )
        attention_mask = kwargs.pop("attention_mask", None)
        validate_triton_attention_envelope(self, query_states, key_states, attention_mask, **kwargs)

        # Forward a user-set or calibrated per-tensor amax to the kernel, which
        # converts it to the FP8 / NVFP4 scale. Without one, the kernel default
        # amax of 1.0 applies -- the theoretical upper bound of the unnormalized
        # P's amax (P lies in [0, 1]).
        p_qdq_amax = None
        pq_amax = getattr(self.p_bmm_quantizer, "_amax", None)
        if pq_amax is not None:
            if pq_amax.numel() != 1:
                raise NotImplementedError(
                    "p_bmm_quantizer via the Triton attention kernel only supports a "
                    f"per-tensor (scalar) amax, got shape {tuple(pq_amax.shape)}."
                )
            p_qdq_amax = float(pq_amax)

        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = query_states.shape[-1] ** -0.5
        return triton_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
            p_qdq=p_qdq,
            p_qdq_amax=p_qdq_amax,
        )

    def _eager_p_qdq_attention(
        self, original_attention_interface, query_states, key_states, value_states, **kwargs
    ):
        """Apply ``p_bmm_quantizer`` to the softmax output via an eager wrapper.

        For attention outside the causal-only Triton kernel's envelope (e.g. ViT's
        non-causal attention). Swapping ``F.softmax`` for a quantized version keeps
        the quantizer in the traced graph so ONNX / Torch-TRT export emits Q/DQ
        around the softmax probabilities. Requires an eager attention
        implementation; SDPA-fused softmax (computed inside the C++ kernel) is
        unaffected.
        """
        _pq = self.p_bmm_quantizer
        _orig_softmax = torch.nn.functional.softmax

        def _quantized_softmax(*s_args, **s_kwargs):
            return _pq(_orig_softmax(*s_args, **s_kwargs))

        with replace_function(torch.nn.functional, "softmax", _quantized_softmax):
            return original_attention_interface(
                self, query_states, key_states, value_states, **kwargs
            )

    @staticmethod
    def _quantized_attention(
        original_attention_interface,
        self,
        query_states,
        key_states,
        value_states,
        *args,
        **kwargs,
    ):
        query_states = self.q_bmm_quantizer(query_states)
        key_states = self.k_bmm_quantizer(key_states)
        value_states = self.v_bmm_quantizer(value_states)

        # FP8 / NVFP4 P quant-dequant runs on the built-in Triton kernel
        # and takes priority over kitchen (which handles MXFP8).
        p_qdq = self._p_qdq_mode()
        if p_qdq is not None:
            # The attention interface passes attention_mask as the only
            # positional argument after q/k/v; everything else is a kwarg.
            if args:
                kwargs["attention_mask"] = args[0]
            # The built-in Triton P kernel is causal-only. Non-causal attention
            # (e.g. ViT) applies p_bmm_quantizer through an eager softmax wrapper
            # that stays export-traceable for ONNX / Torch-TRT instead.
            if kwargs.get("is_causal") is False or getattr(self, "is_causal", True) is False:
                return self._eager_p_qdq_attention(
                    original_attention_interface, query_states, key_states, value_states, **kwargs
                )
            if self.p_bmm_quantizer._if_quant:
                return self._triton_qdq_attention(
                    p_qdq, query_states, key_states, value_states, **kwargs
                )

        # Fused P-QDQ paths bypass TensorQuantizer.forward().
        if not self.p_bmm_quantizer.is_enabled or not self.p_bmm_quantizer._if_quant:
            if args:
                kwargs.pop("attention_mask", None)
            return original_attention_interface(
                self, query_states, key_states, value_states, *args, **kwargs
            )

        if kitchen is not None and self.kitchen_attn_fn is None:
            self._init_kitchen_attn_fn()
        if not self.use_kitchen:
            return original_attention_interface(
                self, query_states, key_states, value_states, *args, **kwargs
            )

        query_sequence_length = query_states.shape[2]
        if query_states.shape[2] < key_states.shape[2]:  # For decoding stage.
            shape = list(query_states.shape)
            shape[2] = key_states.shape[2] - query_states.shape[2]
            query_states = torch.cat(
                [
                    torch.empty(shape, dtype=query_states.dtype, device=query_states.device),
                    query_states,
                ],
                dim=2,
            )

        n_repeat = self.config.num_attention_heads // self.config.num_key_value_heads
        if n_repeat > 1:
            key_states = key_states.repeat_interleave(n_repeat, dim=1)
            value_states = value_states.repeat_interleave(n_repeat, dim=1)
        # kitchen only supports sbhd. we have bhsd.
        query_states = query_states.permute(2, 0, 1, 3)
        key_states = key_states.permute(2, 0, 1, 3)
        value_states = value_states.permute(2, 0, 1, 3)
        attn_out = self.kitchen_attn_fn(query_states, key_states, value_states)
        attn_out = attn_out[-query_sequence_length:, :, :]
        # output is sb(h*d), we need bshd
        attn_out = attn_out.reshape(
            (attn_out.shape[0], attn_out.shape[1], query_states.shape[2], -1)
        ).permute(1, 0, 2, 3)
        return attn_out.contiguous(), None

    def forward(self, *args, **kwargs):
        """Forward method for KV cache quantization of modules using transformers' attention interface.

        The forward method is used to patch the attention interface with _quantized_attention.
        Once output tensors are generated, it restores the original attention interface.
        """
        # Some attention classes (e.g. BertAttention) do not store
        # `self.config` directly; fall back to searching child modules for a config attribute.
        _config = getattr(self, "config", None)
        if _config is None:
            _config = next(
                (getattr(m, "config", None) for m in self.children() if hasattr(m, "config")),
                None,
            )
        _attn_impl = getattr(_config, "_attn_implementation", None) if _config is not None else None

        def _is_eager_attention():
            if _attn_impl is None or _attn_impl == "eager":
                return True
            return bool(_attn_impl == "sdpa" and kwargs.get("output_attentions", False))

        # Get the original transformers module before wrapped in any ModelOpt DynamicModule
        module: ModuleType = inspect.getmodule(self.get_attn_type(self))

        # Preprocessing logic to patch attention interface
        original_attention_interface = (
            module.eager_attention_forward
            if _is_eager_attention()
            else module.ALL_ATTENTION_FUNCTIONS[_attn_impl]
        )
        patch_fn = partial(self._quantized_attention, original_attention_interface)

        if _is_eager_attention():
            if not hasattr(module, "eager_attention_forward"):
                raise AssertionError(
                    f"Module {module} does not have `eager_attention_forward` to enable KV Cache quantization. "
                    "Please use a different attention implementation such as `sdpa` by setting "
                    "`model.config._attn_implementation = 'sdpa'` before quantization."
                )
            module.eager_attention_forward = patch_fn  # type: ignore[attr-defined]
        else:
            module.ALL_ATTENTION_FUNCTIONS[_attn_impl] = patch_fn

        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            # Cleanup logic to restore the original attention interface
            if _is_eager_attention():
                module.eager_attention_forward = original_attention_interface  # type: ignore[attr-defined]
            else:
                module.ALL_ATTENTION_FUNCTIONS[_attn_impl] = original_attention_interface

        return outputs

    @staticmethod
    def is_compatible_attention(attn):
        # Some models (e.g. T5 before transformers 5.15) do not use the attention interface,
        # hence lets do a crude check here to see if the attention module is using it
        # This is not foolproof but should work for most cases
        module = inspect.getmodule(attn)
        return getattr(module, "ALL_ATTENTION_FUNCTIONS", None) is not None

    @staticmethod
    def get_attn_type(attn_module) -> type:
        # If this is a DynamicModule, it means that the module class has been wrapped by ModelOpt
        # Hence, we need to get the original class by level=0
        return (
            attn_module.get_original_cls_by_level(level=0)
            if isinstance(attn_module, DynamicModule)
            else type(attn_module)
        )


class _T5QuantAttention(QuantModule):
    """Attention class for KV Cache quantization compatible with T5 Model."""

    def _quantized_matmul(self, batch1, batch2):
        # T5Attention has two matmul operations, one for the query and key and one for the attention and value.
        # The first matmul is quantized with the q_bmm_quantizer and k_bmm_quantizer. The second matmul is
        # quantized with the v_bmm_quantizer.
        if self.qk_quant_matmul:
            self.qk_quant_matmul = False
            q, k = batch1, batch2
            return torch._matmul(
                self.q_bmm_quantizer(q), self.k_bmm_quantizer(k.transpose(3, 2)).transpose(3, 2)
            )
        else:
            self.qk_quant_matmul = True
            attn, v = batch1, batch2
            return torch._matmul(attn, self.v_bmm_quantizer(v))

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)

    @staticmethod
    def is_compatible_attention(attn):
        return issubclass(attn, T5Attention)

    def forward(self, *args, **kwargs):
        # self.qk_quant_matmul is used to alternate between the two matmul operations for T5Attention
        self.qk_quant_matmul = True
        with replace_function(torch, "matmul", self._quantized_matmul):
            return super().forward(*args, **kwargs)


def _wraps_nested_attention(module):
    """Return True when ``module`` contains another Attention child on this specific instance.

    Checked per-instance (not by class) so an attention class reused as both wrapper and
    leaf is not dropped everywhere. In a 3-level hierarchy (Outer → Middle → Inner), both
    Outer and Middle are treated as wrappers and only Inner is registered for KV-cache
    quantization. Used to avoid double-patching ``eager_attention_forward`` when a wrapper
    attention module (e.g. ``ViTAttention``) delegates to a nested self-attention child
    (e.g. ``ViTSelfAttention``).
    """
    return any(
        child is not module and type(child).__name__.endswith("Attention")
        for _, child in module.named_modules()
    )


def register_hf_attentions_on_the_fly(model):
    """Find HF Attention modules in the model and register them for KV Cache quantization.

    This function attempts to find child modules ending with "Attention" in the name.
    If such child modules are not found, or the corresponding class does not contain
    identifiable attention patterns, the function will not register any new modules.
    """
    if not _is_supported_hf_model(model):
        return

    attention_cls = set()
    registered_attn_module = False

    for name, module in model.named_modules():
        # Only register attention classes that are from Huggingface transformers
        if type(module).__name__.endswith("Attention"):
            if _wraps_nested_attention(module):
                continue
            attention_type = _QuantAttention.get_attn_type(module)
            # Add modules to be registered only if they arent already registered
            if (
                QuantModuleRegistry.get(attention_type) is None
                and attention_type not in attention_cls
            ):
                if _QuantAttention.is_compatible_attention(attention_type):
                    # Lets register the attention class for KV Cache quantization
                    register(attention_type, _QuantAttention)
                    registered_attn_module = True
                    print(
                        f"Registered {attention_type} to {_QuantAttention.__name__} for KV Cache quantization"
                    )
                elif _T5QuantAttention.is_compatible_attention(attention_type):
                    register(attention_type, _T5QuantAttention)
                    registered_attn_module = True
                    print(
                        f"Registered {attention_type} to {_T5QuantAttention.__name__} for KV Cache quantization"
                    )
                else:
                    attention_cls.add(attention_type)
                    print(
                        f"Registered {attention_type} to AST based quantized class for KV Cache quantization"
                    )

    # Check if the attention class has been registered
    # For T5Attention, we want to avoid registering T5LayerCrossAttention and T5LayerSelfAttention.
    # Hence we check if the attention class has been registered.
    if registered_attn_module or not attention_cls:
        return

    # this is the case for models that do not use the attention interface
    # Register the attention class for KV Cache quantization
    success = any(register_attention_for_kv_quant(cls) for cls in attention_cls)
    if not success:
        warnings.warn(
            f"Could not create a quantized attention class for  {attention_cls} from this model. "
            "To enable KV Cache quantization, please create a custom quantized attention class for this model and "
            "register it to ModelOpt using `mtq.register` "
            "(see https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#custom-quantized-module-and-quantizer-placement)"
        )


class HFParallelLinear(torch.nn.Linear, DynamicModule):
    supported_hf_tp_plans = []
    shard = None

    def _setup(self):
        # transformers>=5.16 keeps the sharded weight as a DTensor
        if isinstance(self.weight, torch.distributed.tensor.DTensor):
            assert self.weight.placements == self.shard, (
                f"Received unexpected shard {self.weight.placements} for {self}"
            )
            device_mesh = self.weight.device_mesh
            # Keep HF's TP wrapper on its local-tensor path (also in training),
            # so the quantized linear sees this rank's plain weight shard.
            self._hf_quantized_needs_local_tp = True
        else:  # transformers<5.16: weights are plain Parameters, mesh is on the module
            device_mesh = self._hf_device_mesh
        tp_group = device_mesh.get_group()
        self._parallel_state = ParallelState(data_parallel_group=-1, tensor_parallel_group=tp_group)

    @classmethod
    def is_compatible(cls, linear, tp_mesh=None) -> bool:
        if not isinstance(linear, torch.nn.Linear):
            return False
        if hasattr(linear, "_hf_tp_plan"):  # transformers<5.16
            return linear._hf_tp_plan in cls.supported_hf_tp_plans
        # transformers>=5.16 marks nothing on the module: the weight is a DTensor on the model's TP
        # mesh, sharded along the TP style's dim. Matching the mesh keeps FSDP2 shards out.
        weight = linear.weight
        return (
            tp_mesh is not None
            and isinstance(weight, torch.distributed.tensor.DTensor)
            and weight.device_mesh == tp_mesh
            and tuple(weight.placements) == cls.shard
        )

    # Defined so DMRegistry does not treat this class the same as nn.Linear.
    def forward(self, x):
        # transformers>=5.16 installs its TP input/output transforms as an instance forward, which
        # DynamicModule.convert keeps as _forward_pre_dm; it wraps the unconverted nn.Linear.forward.
        pre_fwd = getattr(self, "_forward_pre_dm", None)
        if pre_fwd is not None:
            return pre_fwd(x)
        return super().forward(x)


class HFColumnParallelLinear(HFParallelLinear):
    supported_hf_tp_plans = ["colwise", "colwise_rep"]
    shard = (Shard(0),) if Shard is not None else None


class HFRowParallelLinear(HFParallelLinear):
    supported_hf_tp_plans = ["rowwise", "rowwise_rep"]
    shard = (Shard(1),) if Shard is not None else None


class _QuantHFParallelLinear(_ParallelLinear):
    _functionals_to_replace = [(torch.nn.functional, "linear")]

    def fold_weight(self, keep_attrs: bool = False):
        with self.enable_weight_access_and_writeback():
            super().fold_weight(keep_attrs)

    @contextmanager
    def enable_weight_access_and_writeback(self):
        if isinstance(self.weight, torch.distributed.tensor.DTensor):  # transformers>=5.16
            assert self.weight.placements == self.shard, (
                f"Received unexpected shard {self.weight.placements} for {self}"
            )
            weight = self.weight
            # TODO: To support TP + FSDP, we need to redistribute the tensor with replicate instead of shard
            self.weight = nn.Parameter(weight.to_local())
            try:
                yield
            finally:
                self.weight = weight
        else:  # transformers<5.16: weights are already plain Parameters
            yield


@QuantModuleRegistry.register({HFColumnParallelLinear: "HFColumnParallelLinear"})
class QuantHFColumnParallelLinear(_QuantHFParallelLinear):
    _is_column_parallel = True


@QuantModuleRegistry.register({HFRowParallelLinear: "HFRowParallelLinear"})
class QuantHFRowParallelLinear(_QuantHFParallelLinear):
    _is_row_parallel = True


def convert_hf_parallel_linears_on_the_fly(model):
    """Convert nn.Linear layers that have been TP sharded by HF.

    Huggingface shards regular nn.Linear layers to rowwise or columnwise tensor-parallel layers dynamically.
    This method converts them to `HFColumnParallelLinear` and `HFRowParallelLinear` so that they
    can be treated as TP sharded layers and not like regular nn.Linear layers.
    """
    # transformers>=5.16 keeps the TP mesh on the PreTrainedModel, which may sit inside a wrapper.
    tp_mesh = next(
        (m._device_mesh for m in model.modules() if getattr(m, "_device_mesh", None) is not None),
        None,
    )
    converted = False
    for module in model.modules():
        if HFColumnParallelLinear.is_compatible(module, tp_mesh):
            HFColumnParallelLinear.convert(module)
            converted = True
        elif HFRowParallelLinear.is_compatible(module, tp_mesh):
            HFRowParallelLinear.convert(module)
            converted = True
    if tp_mesh is not None and not converted:
        warnings.warn(
            "Found a transformers tensor-parallel mesh but no TP-sharded linear layer; they will be"
            " quantized as unsharded layers."
        )


if transformers.pytorch_utils.Conv1D not in QuantModuleRegistry:
    # transformers.pytorch_utils.Conv1D used in HF-GPT2 is not a real Conv1D
    # It is actually a Linear layer where weight is transposed and torch.addmm is used
    @QuantModuleRegistry.register({transformers.pytorch_utils.Conv1D: "Conv1D"})
    class _QuantConv1D(_QuantLinear):
        @classmethod
        @torch.no_grad()
        def convert(cls, module: nn.Module) -> "_QuantConv1D":
            module.weight = nn.Parameter(module.weight.T.contiguous())
            module.out_features, module.in_features = module.weight.shape
            # We want the forward method of nn.Linear to be called instead of the forward method of Conv1D
            dyn_cls: QuantModule = QuantModuleRegistry.get(nn.Linear)
            return dyn_cls.convert(module)


class _TransposedQuantization(torch.autograd.Function):
    """Applies transposed quantization.

    This is useful for weight quantization of some MoEs such as gpt-oss or Llama4 which has expert weights
    of shape (num_experts, in_dim, out_dim). Per-channel/Per-block quantization from ModelOpt
    assumes that `in_dim` is -1 dim. Hence for quantizing such MoE weights, lets use transposed quantization.
    """

    # Note: TransposedQuantization uses STE with no clipping
    @staticmethod
    def forward(ctx, inputs, quantizer):
        return quantizer(inputs.transpose(-1, -2).contiguous()).transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


_transposed_quantize = _TransposedQuantization.apply


class _TransposedExpertsCalibMixin:
    """Weight-only calibration for BMM-style experts that quantize their weights transposed.

    ``_QuantGptOssExperts`` / ``_QuantLlama4TextExperts`` hold 3-D expert weights of shape
    ``(num_experts, in_dim, out_dim)`` and quantize them *transposed* in the forward (see
    :class:`_TransposedQuantization`: per-channel / per-block quantization expects the
    contraction ``in_dim`` as the last axis). Weight-only calibration
    (``max_calibrate`` -> ``weight_only_quantize``) must feed the weight quantizer the same
    transposed view; otherwise static-block NVFP4 locks ``_original_shape`` from the
    non-transposed weight and the forward then raises "Input shape has changed".
    Calibrating transposed also matches the orientation the unified HF export reads ``_amax``
    in (it transposes BMM expert weights before deriving scales).

    The transposed view is not made contiguous (unlike the forward's ``_transposed_quantize``,
    which needs it for the matmul): calibration only reads the shape and reduces for ``_amax``,
    both of which the quantizer handles on a non-contiguous view via ``reshape``.

    For ``_QuantGptOssExperts`` ``gate_up_proj``/``down_proj`` are dynamic attributes; weight-only
    calibration runs with weight quantization disabled, so they return the raw weight here.
    """

    def iter_weights_for_calibration(self):
        """Yield ``(transposed_weight, weight_quantizer)`` for each expert projection."""
        for weight_name in ("gate_up_proj", "down_proj"):
            weight = getattr(self, weight_name)
            yield weight.transpose(-1, -2), getattr(self, f"{weight_name}_weight_quantizer")


class _QuantSparseSequentialMoe(QuantModule):
    """Quantization wrapper for HuggingFace sparse MoE blocks.

    This base class is for Sequential MoEs (i.e each experts are implemented as standalone modules),
    e.g. remote-code models; native transformers MoEs use fused experts instead.

    Supports ``layer_sync_moe_local_experts_amax`` to sync input quantizer amax across experts.

    Optionally supports config-driven features (disabled by default):
    - ``_moe_calib_experts_ratio``: force-forward tokens to more experts during calibration.
      When set to a value > 0, also enables token counting per expert.

    When disabled, forward is a direct pass-through with zero overhead.
    """

    def _setup(self):
        self._moe_calib_experts_ratio = None
        self._token_counting_initialized = False

    def _num_experts(self) -> int:
        """Expert count from the gate, the block or the experts container (0 if unknown)."""
        for obj in [getattr(self, "gate", None), self, getattr(self, "experts", None)]:
            for attr in ("num_experts", "n_routed_experts"):
                if getattr(obj, attr, None):
                    return getattr(obj, attr)
        return len(self.experts) if hasattr(self.experts, "__len__") else 0

    def _init_token_counting(self):
        """Lazy-init token counting infra (buffer + gate hook). Called once from forward."""
        self._token_counting_initialized = True
        num_experts = self._num_experts()
        if num_experts == 0:
            warnings.warn(
                f"{self.__class__.__name__}: could not resolve num_experts; "
                "expert routing will not be tracked for this layer."
            )
            return

        self.register_buffer(
            "expert_token_count",
            torch.zeros(num_experts, dtype=torch.long, device=next(self.parameters()).device),
            persistent=False,
        )
        self._count_expert_tokens = False
        if hasattr(self, "gate"):
            self.gate.register_forward_hook(self._gate_forward_hook)

    def _gate_forward_hook(self, module, input, output):
        if not self._count_expert_tokens:
            return
        with torch.no_grad():
            if isinstance(output, tuple) and len(output) >= 3:
                # TopKRouter: returns (logits, scores, indices)
                indices = output[2]
            else:
                # nn.Linear-style gate (e.g. remote code): returns logits tensor
                logits = output if not isinstance(output, tuple) else output[0]
                top_k = self.gate.top_k if hasattr(self.gate, "top_k") else self.top_k
                _, indices = torch.topk(logits.float(), top_k, dim=-1)
            counts = torch.bincount(indices.reshape(-1), minlength=self.expert_token_count.shape[0])
            self.expert_token_count += counts.to(self.expert_token_count.device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._moe_calib_experts_ratio is None:
            return super().forward(hidden_states)

        is_calib = any(getattr(m, "_if_calib", False) for m in self.experts.modules())

        # During calibration, forward all tokens to a larger fraction of experts to improve
        # calibration coverage, then re-run with the original top_k for actual outputs.
        if is_calib:
            # Skip counting when all experts are calibrated (ratio == 1.0). Init first: it resets
            # the counting flag, which would otherwise drop the first batch's counts.
            count_tokens = self._moe_calib_experts_ratio < 1.0
            if count_tokens and not self._token_counting_initialized:
                self._init_token_counting()
            self._count_expert_tokens = count_tokens
            # top_k lives on the router gate, or on the block itself (e.g. remote-code MoEs)
            top_k_owner = self.gate if hasattr(getattr(self, "gate", None), "top_k") else self
            original_top_k = top_k_owner.top_k
            top_k_owner.top_k = max(
                original_top_k, round(self._num_experts() * self._moe_calib_experts_ratio)
            )
            try:
                super().forward(hidden_states)
            finally:
                top_k_owner.top_k = original_top_k
            self._count_expert_tokens = False

        output = super().forward(hidden_states)
        self._count_expert_tokens = False
        return output

    def layer_sync_moe_local_experts_amax(self, sync_weight_amax=False):
        """Sync input_quantizer amax across experts so all share the same amax per quantizer.

        Skipped when _moe_calib_experts_ratio is set, as each expert is calibrated independently.
        Also skipped when experts is a fused module (e.g. Llama4TextExperts) with shared quantizers.

        Args:
            sync_weight_amax: If True, also sync weight quantizer amax across experts.

        """
        if self._moe_calib_experts_ratio is not None:
            return
        sync_moe_expert_amax(self.experts, sync_weight_amax=sync_weight_amax)


def _get_fused_expert_intermediate_dim(module):
    """Resolve the intermediate (expert) dimension from a fused expert module.

    Different transformers versions use ``intermediate_size`` or ``intermediate_dim``.
    """
    for attr in ("intermediate_dim", "intermediate_size", "moe_intermediate_size"):
        if hasattr(module, attr):
            return getattr(module, attr)
    return module.gate_up_proj.shape[1] // 2


class _QuantFusedExperts(_QuantFunctionalMixin):
    """Generic quantized wrapper for HuggingFace fused MoE expert modules.

    Handles any module following the standard HF transformers 5.0+ fused expert
    pattern (``@use_experts_implementation``):

    * ``gate_up_proj``: 3-D ``nn.Parameter`` of shape ``(num_experts, 2*intermediate_dim, hidden_dim)``
    * ``down_proj``:    3-D ``nn.Parameter`` of shape ``(num_experts, hidden_dim, intermediate_dim)``
    * ``act_fn``:       activation function applied between gate/up and down projections
    * ``forward``:      calls ``F.linear`` exactly twice per expert (gate_up then down)

    Per-expert quantization is achieved by intercepting ``F.linear`` and recovering
    the expert index from the weight tensor's storage offset into the 3-D parameter.
    Each expert gets its own weight quantizers (``nn.ModuleList``), while input
    quantizers are shared across all experts (single ``TensorQuantizer``) to match
    the shared input quantization scale used by downstream inference frameworks.

    Verified compatible models: Mixtral, Qwen2-MoE, Qwen3-MoE, Qwen3.5-MoE,
    DeepSeek-V3, Jamba, OLMoE.

    Backends: both the ``eager`` (per-expert ``F.linear``) and ``grouped_mm``
    (``transformers.integrations.moe._grouped_linear`` →
    ``torch._grouped_mm`` / ``torch.nn.functional.grouped_mm`` / fallback)
    forwards are intercepted. The same per-expert weight quantizers and shared
    input quantizers are used for both, so calibration in either backend yields
    an identical exported checkpoint. Only one backend's hook fires per forward
    (selected by ``config._experts_implementation``), so installing both is safe.

    Limitation: the ``grouped_mm`` path assumes the non-transposed weight layout
    ``(num_experts, out_dim, in_dim)`` used by all standard fused experts. The
    transposed layout (e.g. GptOss, which has its own quantized wrapper) is not
    handled here. ``batched_mm`` (``torch.bmm``) and ``sonicmoe`` (opaque CUDA
    kernel) backends are not intercepted; ``normalize_experts_impl_on_the_fly``
    redirects them to ``grouped_mm`` before calibration.
    """

    # Name of the first (pre-activation) fused projection. ``gate_up_proj`` for gated experts
    # (Mixtral/Qwen/DeepSeek/...); subclasses override it (e.g. ``up_proj`` for the ungated
    # NemotronH experts). Drives the weight + per-expert quantizer attribute names so the same
    # eager/grouped_mm interception logic serves both layouts.
    _first_proj_attr: str = "gate_up_proj"
    # Whether the first projection packs a gate half that must be split on export.
    _is_gated = True
    # Under expert parallelism each rank's experts see only the tokens routed to them, so calibration
    # shares the layer's input amax over the expert-parallel group (see max_calibrate).
    _share_input_amax_across_ep = True

    @property
    def _first_proj_input_quantizer_attr(self) -> str:
        return f"{self._first_proj_attr}_input_quantizer"

    @property
    def _first_proj_weight_quantizers_attr(self) -> str:
        return f"{self._first_proj_attr}_weight_quantizers"

    def _get_expert_idx_from_first_proj(self, weight: torch.Tensor) -> int:
        """Recover expert index from the first-projection weight slice's storage offset.

        When HF indexes ``<first_proj>[idx]``, the result is a view sharing the
        same underlying storage.  The offset delta divided by the stride along
        dim-0 gives the expert index.

        The invariant breaks if the tensor is ``.contiguous()``-copied or
        redistributed by certain distributed wrappers (FSDP2, tensor parallel).
        """
        first_proj = getattr(self, self._first_proj_attr)
        base_offset = first_proj.storage_offset()
        stride = first_proj.stride(0)
        if stride == 0:
            return 0
        idx = (weight.storage_offset() - base_offset) // stride
        assert 0 <= idx < self.num_experts, (
            f"Computed expert index {idx} out of range [0, {self.num_experts}). "
            "This can happen if the weight was .contiguous()-copied or redistributed."
        )
        return idx

    def _setup(self):
        n = self.num_experts
        first = self._first_proj_attr
        setattr(self, f"{first}_input_quantizer", TensorQuantizer())
        setattr(
            self, f"{first}_weight_quantizers", nn.ModuleList([TensorQuantizer() for _ in range(n)])
        )
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizers = nn.ModuleList([TensorQuantizer() for _ in range(n)])

        self._register_temp_attribute("_down_proj_linear", False)
        self._register_temp_attribute("_current_expert_idx", 0)
        # Cumulative tokens-per-expert offsets of the current grouped_mm forward,
        # stashed so local-Hessian capture can split the expert-sorted batch.
        self._register_temp_attribute("_grouped_offs", None)
        # How many times an interception hook fired for this module. Still zero after a
        # calibration forward means the active experts backend reached neither `F.linear` nor
        # `_grouped_linear`, so no quantizer ever saw data -- `assert_fused_experts_intercepted`
        # turns that into an error instead of an export full of fallback scales.
        self._register_temp_attribute("_intercept_count", 0)
        # How many times this module's forward ran: an experts module that no calibration token
        # reached (e.g. vision experts under text-only calibration) has nothing to intercept.
        self._register_temp_attribute("_forward_count", 0)

    @property
    def _running_grouped_mm(self) -> bool:
        """Whether this module's forward runs its expert matmuls through ``_grouped_linear``.

        The ``grouped_mm`` backend and the ``torch_a2a`` expert-parallel forward both do, handing each
        projection one expert-sorted batch rather than one ``F.linear`` call per expert.
        """
        cfg = getattr(self, "config", None)
        return getattr(cfg, "_experts_implementation", None) in ("grouped_mm", "torch_a2a")

    @property
    def _is_transposed_experts(self) -> bool:
        """True if the fused weights are stored ``(num_experts, in_dim, out_dim)`` (e.g. llama4/gpt-oss).

        Set by transformers' ``use_experts_implementation(is_transposed=...)`` on the wrapped module;
        per-channel/block (NVFP4) quant must run on the transposed view so the contraction dim is last.
        """
        return bool(getattr(self, "is_transposed", False))

    def _local_experts(self, weight: torch.Tensor) -> torch.Tensor:
        """``weight`` indexed the way the per-expert quantizers are: by this module's own experts.

        Under expert parallelism transformers keeps a fused expert weight as a DTensor sharded over
        the experts and sets ``num_experts`` to the local count, so indexing the DTensor would read
        the GLOBAL expert: from rank 1 on, another rank's. Its local shard holds this rank's experts in
        quantizer order. A DTensor holding every expert (FSDP2) is indexed as it is.
        """
        if (
            isinstance(weight, torch.distributed.tensor.DTensor)
            and weight.shape[0] != self.num_experts
        ):
            return weight.to_local()
        return weight

    def _quantize_grouped_weight(self, weight: torch.Tensor, quantizers: nn.ModuleList):
        """Fake-quantize a fused 3-D weight per expert.

        Each expert slice passes through its own weight quantizer (collecting
        ``amax`` during calibration, fake-quantizing afterwards), matching the
        eager per-expert ``F.linear`` semantics. Returns ``weight`` untouched when
        every quantizer is disabled so passthrough stays exact (and cheap).

        For transposed experts (weight ``(num_experts, in_dim, out_dim)``) the slice is quantized
        *transposed* (``in_dim`` last) via :func:`_transposed_quantize`, so per-channel/block (NVFP4)
        blocks along the contraction dim -- matching the transposed calibration + export. For per-tensor
        (FP8) the orientation is irrelevant, so this is a no-op difference there.
        """
        if all(not q.is_enabled for q in quantizers):
            return weight
        weight = self._local_experts(weight)
        if self._is_transposed_experts:
            return torch.stack(
                [
                    _transposed_quantize(weight[idx], quantizers[idx])
                    for idx in range(self.num_experts)
                ],
                dim=0,
            )
        return torch.stack([quantizers[idx](weight[idx]) for idx in range(self.num_experts)], dim=0)

    @property
    def functionals_to_replace(self):
        _orig_linear = torch.nn.functional.linear
        # First (pre-activation) projection quantizers, by layout: gate_up_proj_* (gated) or
        # up_proj_* (ungated NemotronH). Bound once; the objects are stable across the forward.
        first_input_q = getattr(self, self._first_proj_input_quantizer_attr)
        first_weight_qs = getattr(self, self._first_proj_weight_quantizers_attr)

        # The HF fused expert forward calls F.linear exactly twice per expert
        # in strict alternation: first the up/gate_up projection, then down_proj.
        # forward() resets the toggle before each call to super().forward().
        def _quantized_linear(input, weight, bias=None):
            self._intercept_count += 1
            if self._down_proj_linear:
                idx = self._current_expert_idx
                input = self.down_proj_input_quantizer(input)
                weight = self.down_proj_weight_quantizers[idx](
                    self.get_expert_weight("down_proj", idx, input.dtype)
                )
            else:
                idx = self._get_expert_idx_from_first_proj(weight)
                self._current_expert_idx = idx
                input = first_input_q(input)
                weight = first_weight_qs[idx](
                    self.get_expert_weight(self._first_proj_attr, idx, input.dtype)
                )
            self._down_proj_linear = not self._down_proj_linear
            return _orig_linear(input, weight, bias)

        replacements: list[tuple] = [(torch.nn.functional, "linear", _quantized_linear)]

        # grouped_mm backend: HF's grouped_mm_experts_forward calls the module-level
        # _grouped_linear exactly twice (gate_up then down) with the FULL 3-D weight
        # in the same (num_experts, out, in) layout as eager F.linear, so the same
        # per-expert weight quantizers / shared input quantizers apply. Only one
        # backend's hook fires per forward, so patching both F.linear and
        # _grouped_linear is safe.
        try:
            import transformers.integrations.moe as _moe
        except ImportError:
            _moe = None
        if _moe is not None and hasattr(_moe, "_grouped_linear"):
            _orig_grouped_linear = _moe._grouped_linear

            def _quantized_grouped_linear(input, weight, offs, bias=None, is_transposed=False):
                self._intercept_count += 1
                # Recorded before quantizing input so the local-Hessian forward
                # pre-hook on the input quantizer can split the expert-sorted batch.
                self._grouped_offs = offs
                quantizers = (
                    self.down_proj_weight_quantizers if self._down_proj_linear else first_weight_qs
                )
                if self._down_proj_linear:
                    input = self.down_proj_input_quantizer(input)
                else:
                    input = first_input_q(input)
                # _quantize_grouped_weight quantizes each expert slice transposed when the experts are stored
                # (num_experts, in_dim, out_dim) (self._is_transposed_experts), so per-channel/block (NVFP4)
                # blocks along the contraction dim. ``is_transposed`` here is the same layout flag the
                # grouped matmul uses, so the quantized weight goes straight into _orig_grouped_linear.
                weight = self._quantize_grouped_weight(weight, quantizers)
                self._down_proj_linear = not self._down_proj_linear
                return _orig_grouped_linear(
                    input, weight, offs, bias=bias, is_transposed=is_transposed
                )

            replacements.append((_moe, "_grouped_linear", _quantized_grouped_linear))

        return replacements

    def forward(self, *args, **kwargs):
        self._down_proj_linear = False
        self._forward_count += 1
        return super().forward(*args, **kwargs)

    def get_expert_weight(self, name, idx, dtype=None):
        """Return one expert's projection in the calibration/export weight domain."""
        return getattr(self, name)[idx]

    def iter_weights_for_calibration(self):
        """Yield ``(weight_slice, quantizer)`` per-expert pairs.

        The base impl uses singular ``*_weight_quantizer`` and skips fused-
        experts modules, so weight-only calibration never reaches per-expert
        quantizers without this override.
        """
        transposed = self._is_transposed_experts
        for weight_name, quantizers_name in (
            (self._first_proj_attr, self._first_proj_weight_quantizers_attr),
            ("down_proj", "down_proj_weight_quantizers"),
        ):
            weight = getattr(self, weight_name, None)
            quantizers = getattr(self, quantizers_name, None)
            if weight is None or quantizers is None:
                continue
            weight = self._local_experts(weight)
            for idx, q in enumerate(quantizers):
                # Calibrate the same (transposed) view the forward quantizes, so block NVFP4 locks
                # _original_shape / _amax with the contraction dim last (and the export reads it that way).
                expert_weight = self.get_expert_weight(weight_name, idx)
                yield (expert_weight.transpose(-1, -2) if transposed else expert_weight), q

    def iter_weight_quantizers_for_calibration(self):
        """Yield the per-expert weight quantizers without slicing the fused 3-D weight.

        Overrides the base, which would go through :meth:`iter_weights_for_calibration` and
        evaluate ``weight[idx]``. Under FSDP2 that weight is a DTensor, so every expert slice
        dispatches a redistribute collective to produce a value a quantizer-only caller discards.
        """
        for weight_name, quantizers_name in (
            (self._first_proj_attr, self._first_proj_weight_quantizers_attr),
            ("down_proj", "down_proj_weight_quantizers"),
        ):
            # Same skip condition as iter_weights_for_calibration, so the two stay in lockstep.
            # Fetching the attribute is free; only indexing into it would collective.
            quantizers = getattr(self, quantizers_name, None)
            if getattr(self, weight_name, None) is None or quantizers is None:
                continue
            yield from quantizers

    def fold_weight(self, keep_attrs: bool = False):
        """Bake each per-expert weight quantizer into its slice of the fused 3-D weight.

        The base ``fold_weight`` only handles singular ``*_weight_quantizer`` attributes and
        would skip the ``nn.ModuleList`` of per-expert quantizers used here. The per-expert
        ``(weight_slice, quantizer)`` pairs are the same ones :meth:`iter_weights_for_calibration`
        yields, so we reuse it; each fake-quant quantizer's quantization and rotation are folded
        in and disabled, and calibration buffers are dropped unless ``keep_attrs``.
        """
        transposed = self._is_transposed_experts
        for weight_name, quantizers_name in (
            (self._first_proj_attr, self._first_proj_weight_quantizers_attr),
            ("down_proj", "down_proj_weight_quantizers"),
        ):
            weight = getattr(self, weight_name, None)
            quantizers = getattr(self, quantizers_name, None)
            if weight is None or quantizers is None:
                continue
            for idx, q in enumerate(quantizers):
                if not (isinstance(q, TensorQuantizer) and q.fake_quant):
                    continue
                slice_ = self._local_experts(weight.data)[idx]
                # Fold in the same (transposed) orientation the weight was calibrated/quantized in.
                folded = (
                    _transposed_quantize(slice_.float(), q) if transposed else q(slice_.float())
                )
                slice_.copy_(folded.to(weight.dtype))
                q.disable()
                q.disable_rotate()
                if keep_attrs and hasattr(q, "_pre_quant_scale"):
                    # The scale is already baked into the folded weight; disable pre-quant
                    # scaling so it is not applied a second time at inference.
                    q._enable_pre_quant_scale = False
                elif not keep_attrs:
                    for attr_name in ("_pre_quant_scale", "_amax"):
                        if hasattr(q, attr_name):
                            delattr(q, attr_name)


class _QuantNemotronHExperts(_QuantFusedExperts):
    """Quantized wrapper for NemotronH's ungated fused MoE experts (``NemotronHExperts``).

    NemotronH uses a standard (ungated) MLP per expert -- ``down_proj(act_fn(up_proj(x)))`` --
    so the fused container exposes ``up_proj``/``down_proj`` 3-D parameters instead of the gated
    ``gate_up_proj``. The forward still calls the first projection then ``down_proj`` exactly once
    per expert (eager ``F.linear``) or once per batch (``grouped_mm`` ``_grouped_linear``), so the
    only change from :class:`_QuantFusedExperts` is the first-projection name: the per-expert weight
    quantizers and shared input quantizer live under ``up_proj_*`` instead of ``gate_up_proj_*``.
    """

    _first_proj_attr = "up_proj"
    # Read by the export: an ungated up_proj is exported whole, not split into gate|up halves.
    _is_gated = False


class _QuantFP8Experts(_QuantFusedExperts):
    """Scale-aware PTQ for native Transformers FP8 experts, dequantized per expert."""

    @property
    def _first_proj_attr(self):
        return "gate_up_proj" if self.has_gate else "up_proj"

    @property
    def _is_gated(self):
        return self.has_gate

    def _setup(self):
        if getattr(self.config, "expert_dtype", "fp8") != "fp8":
            raise NotImplementedError("Only FP8 E4M3 native expert weights are supported")
        super()._setup()

    def get_expert_weight(self, name, idx, dtype=None):
        weight = getattr(self, name)[idx]
        if weight.element_size() > 1:
            return weight if dtype is None else weight.to(dtype)
        dtype = dtype or torch.get_default_dtype()
        scales = getattr(self, f"{name}_scale_inv")[idx].float()
        rows, cols = self.block_size or (weight.shape[0] // scales.shape[0], weight.shape[1])
        scales = scales.repeat_interleave(rows, 0).repeat_interleave(cols, 1)
        return (weight.float() * scales[: weight.shape[0], : weight.shape[1]]).to(dtype)

    def linear(self, input, weight, weight_scale_inv, activation_scale=None):
        # Preserve the FP8 view's storage offset for the per-expert quantizer lookup.
        return torch.nn.functional.linear(input, weight)

    def _validate_fold_weight(self):
        if any(
            getattr(self, name).element_size() == 1 for name in (self._first_proj_attr, "down_proj")
        ):
            raise ValueError("Call unpack_weight() before folding native FP8 expert weights.")
        super()._validate_fold_weight()

    def fold_weight(self, keep_attrs: bool = False):
        self._validate_fold_weight()
        super().fold_weight(keep_attrs)

    def unpack_weight(self):
        """Materialize floating weights before folding; retain scales for native forward."""
        for name in (self._first_proj_attr, "down_proj"):
            weight = getattr(self, name)
            if weight.element_size() == 1:
                unpacked = torch.empty_like(weight, dtype=torch.get_default_dtype())
                for idx in range(self.num_experts):
                    unpacked[idx].copy_(self.get_expert_weight(name, idx))
                setattr(self, name, nn.Parameter(unpacked, requires_grad=False))


def _is_quant_fused_experts_module(module):
    """Return True for a converted HF fused-MoE-experts quantization wrapper."""
    return isinstance(module, _QuantFusedExperts)


def _get_fused_experts_quantizer_attr_names(module):
    """Return quantizer attribute names for a converted fused-experts module.

    Two shared input quantizers plus two ``ModuleList`` of per-expert weight quantizers.
    The first projection is ``gate_up_proj`` for gated experts and ``up_proj`` for the
    ungated NemotronH layout, so read it from the wrapper rather than assuming.
    """
    first = getattr(module, "_first_proj_attr", "gate_up_proj")
    return (
        f"{first}_input_quantizer",
        f"{first}_weight_quantizers",
        "down_proj_input_quantizer",
        "down_proj_weight_quantizers",
    )


@contextmanager
def patch_compressed_linear_loading():
    """Context manager that patches CompressedLinear to survive custom ``_init_weights`` calls.

    When loading pack-quantized models with ``trust_remote_code=True``,
    ``compressed_tensors`` replaces ``.weight`` with ``.weight_packed`` on
    CompressedLinear modules.  Custom model code (e.g. ``modeling_deepseek.py``)
    often does ``module.weight.data.normal_(...)`` inside ``_init_weights``,
    which crashes because ``.weight`` no longer exists.

    This context manager monkey-patches ``CompressedLinear.__getattr__`` to
    return a harmless dummy for ``.weight`` accesses, and restores the original
    behaviour on exit (even if an exception is raised).

    Usage::

        from modelopt.torch.quantization.plugins.huggingface import patch_compressed_linear_loading

        with patch_compressed_linear_loading():
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, device_map="auto", trust_remote_code=True, dtype="auto"
            )
    """
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear
    except ImportError:
        yield
        return

    if getattr(CompressedLinear, "_modelopt_init_patched", False):
        yield
        return

    original_getattr = getattr(CompressedLinear, "__getattr__", None)

    class _DummyWeightData:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class _DummyWeight:
        def __init__(self):
            self.data = _DummyWeightData()

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    def patched_getattr(self, name):
        if name == "weight":
            if "_parameters" in self.__dict__ and "weight" in self._parameters:
                return self._parameters["weight"]
            if "weight" in self.__dict__:
                return self.__dict__["weight"]
            return _DummyWeight()
        if original_getattr is not None:
            return original_getattr(self, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    CompressedLinear.__getattr__ = patched_getattr
    CompressedLinear._modelopt_init_patched = True
    logger.info("Patched CompressedLinear for transformers compatibility")

    try:
        yield
    finally:
        if original_getattr is not None:
            CompressedLinear.__getattr__ = original_getattr
        elif hasattr(CompressedLinear, "__getattr__"):
            del CompressedLinear.__getattr__
        CompressedLinear._modelopt_init_patched = False
        logger.info("Restored CompressedLinear original state")


class _QuantCompressedLinear(QuantModule):
    """Quantization wrapper for ``compressed_tensors`` CompressedLinear modules.

    Handles on-the-fly decompression of pack-quantized INT4 weights during
    calibration.  This avoids fully decompressing all experts into GPU memory
    at once (which would OOM for large MoE models), and also correctly handles
    the ``weight_shape`` metadata that ``compressed_tensors`` stores as a
    tensor rather than a plain list.
    """

    def _setup(self):
        self.input_quantizer = TensorQuantizer()
        self.weight_quantizer = TensorQuantizer()

    def _build_compressed_data(self):
        """Build compressed_data dict and quantization_args from module attributes.

        Returns a (compressed_data, quant_args) tuple suitable for
        ``self.compressor.decompress_weight()``.  ``weight_shape`` is
        normalised to a plain ``list[int]`` so that ``compressed_tensors``
        does not choke on a ``torch.Tensor`` value.
        """
        compressed_data = {"weight_packed": self.weight_packed}
        if hasattr(self, "weight_scale"):
            compressed_data["weight_scale"] = self.weight_scale
        if hasattr(self, "weight_shape"):
            ws = self.weight_shape
            if isinstance(ws, torch.Tensor):
                compressed_data["weight_shape"] = [int(x) for x in ws.tolist()]
            elif isinstance(ws, (list, tuple)):
                compressed_data["weight_shape"] = [int(x) for x in ws]
            else:
                compressed_data["weight_shape"] = ws
        if hasattr(self, "weight_zero_point"):
            compressed_data["weight_zero_point"] = self.weight_zero_point

        quant_args = None
        if hasattr(self, "quantization_scheme") and self.quantization_scheme:
            if hasattr(self.quantization_scheme, "weights"):
                quant_args = self.quantization_scheme.weights

        return compressed_data, quant_args

    def forward(self, input: Tensor) -> Tensor:
        from compressed_tensors.quantization import QuantizationStatus

        if self.quantization_status == QuantizationStatus.COMPRESSED:
            # Real packed weights are int32. If it's float, it's not actually compressed.
            if self.weight_packed.dtype == torch.int32:
                compressed_data, quant_args = self._build_compressed_data()
                if not hasattr(self, "_logged_on_the_fly"):
                    logger.debug("On-the-fly decompression for %s", self.__class__.__name__)
                    self._logged_on_the_fly = True
                weight_data = self.compressor.decompress_weight(
                    compressed_data=compressed_data,
                    quantization_args=quant_args,
                )
            else:
                weight_data = self.weight_packed
        else:
            weight_data = self.weight

        return linear(self.input_quantizer(input), self.weight_quantizer(weight_data), self.bias)

    def unpack_weight(self):
        from compressed_tensors.quantization import QuantizationStatus

        if self.quantization_status == QuantizationStatus.COMPRESSED:
            compressed_data, quant_args = self._build_compressed_data()

            # Skip non-pack-quantized weights (e.g., vision modules stored as BF16)
            if isinstance(compressed_data["weight_packed"], torch.Tensor):
                if compressed_data["weight_packed"].dtype != torch.int32:
                    return

            decompressed = self.compressor.decompress_weight(
                compressed_data=compressed_data,
                quantization_args=quant_args,
            )
            # Clear any placeholder before registering the real parameter
            self._parameters.pop("weight", None)
            self._buffers.pop("weight", None)
            if "weight" in self.__dict__:
                del self.__dict__["weight"]
            param = nn.Parameter(decompressed, requires_grad=False)
            self._parameters["weight"] = param
            self.__dict__["weight"] = param

        if hasattr(self, "weight_packed"):
            del self.weight_packed
        if hasattr(self, "weight_scale"):
            del self.weight_scale
        if hasattr(self, "weight_shape"):
            if "weight_shape" in self._parameters:
                del self._parameters["weight_shape"]
            else:
                delattr(self, "weight_shape")
        if self.quantization_status == QuantizationStatus.COMPRESSED:
            self.quantization_status = QuantizationStatus.FROZEN


class _QuantFP8Linear(QuantModule):
    def _setup(self):
        self.input_quantizer = TensorQuantizer()
        self.weight_quantizer = TensorQuantizer()
        assert self.weight.ndim == 2, "Weight must be 2D"
        if self.weight_scale_inv.ndim == 0:
            self.block_size = None
        else:
            assert self.weight_scale_inv.ndim == 2, "Weight scale inverse must be 0D or 2D"
            self.block_size = max(
                self.weight.shape[0] // self.weight_scale_inv.shape[0],
                self.weight.shape[1] // self.weight_scale_inv.shape[1],
            )
            assert self.block_size == 128, "Block size must be 128"

    def _get_weight_and_scale_inv(self):
        if isinstance(self.weight, torch.distributed.tensor.DTensor):
            weight = self.weight._local_tensor.contiguous()
            scale_inv = self.weight_scale_inv._local_tensor.contiguous()
        else:
            weight = self.weight.contiguous()
            scale_inv = self.weight_scale_inv.contiguous()
        return weight, scale_inv

    def _dequantize_weight(self, dtype: torch.dtype) -> Tensor:
        weight, scale_inv = self._get_weight_and_scale_inv()
        if self.block_size is None:
            return weight.to(dtype) * scale_inv.to(dtype)
        assert weight_dequant is not None, "Triton is not available"
        return weight_dequant(weight, scale_inv, self.block_size, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight.element_size() == 1:
            with torch.cuda.device(self.weight.device):
                weight = self._dequantize_weight(input.dtype)
        else:
            weight = self.weight
        return linear(
            self.input_quantizer(input),
            self.weight_quantizer(weight),
            self.bias,
        )

    def unpack_weight(self):
        with torch.cuda.device(self.weight.device):
            self.weight = nn.Parameter(
                self._dequantize_weight(torch.get_default_dtype()),
                requires_grad=False,
            )
        if hasattr(self, "weight_scale_inv"):
            del self.weight_scale_inv


try:
    from compressed_tensors.linear.compressed_linear import CompressedLinear

    if CompressedLinear not in QuantModuleRegistry:
        QuantModuleRegistry.register({CompressedLinear: "hf.CompressedLinear"})(
            _QuantCompressedLinear
        )
except ImportError:
    pass

if FP8Experts not in QuantModuleRegistry:
    QuantModuleRegistry.register({FP8Experts: "hf.FP8Experts"})(_QuantFP8Experts)

if FP8Linear not in QuantModuleRegistry:
    QuantModuleRegistry.register({FP8Linear: "hf.FP8Linear"})(_QuantFP8Linear)


def _has_num_experts(obj):
    # n_routed_experts: NemotronH-style MoE
    return hasattr(obj, "num_experts") or hasattr(obj, "n_routed_experts")


def _is_sparse_sequaential_moe_block(module):
    """Check if a module is structurally a sparse sequential MoE block compatible with _QuantSparseSequentialMoe.

    All HuggingFace MoE blocks (Mixtral, Qwen3Moe, Qwen2Moe, Qwen3Next, Llama4, MiniMax,
    NemotronH, etc.) share a common structural pattern: a ``gate`` (TopKRouter) sub-module with
    routing attributes (``top_k`` and ``num_experts`` or ``n_routed_experts``), and an ``experts``
    sub-module.

    This function detects that pattern instead of relying on class names, making it forward-compatible
    with new MoE architectures.
    """
    if not hasattr(module, "experts"):
        return False

    if not hasattr(module.experts, "__iter__"):
        # Fused (batched) experts, no per-expert quantizers
        return False

    # Primary: gate sub-module has topk/top_k + num_experts (standard TopKRouter pattern)
    if hasattr(module, "gate"):
        gate = module.gate
        if hasattr(gate, "top_k") and _has_num_experts(gate):
            return True

    # Fallback: top_k + num_experts on the block itself (e.g. remote-code MoE blocks)
    if hasattr(module, "top_k"):
        if not _has_num_experts(module) and hasattr(module.experts, "__len__"):
            module.num_experts = len(module.experts)
        return _has_num_experts(module)

    return False


def register_sparse_moe_on_the_fly(model):
    """Auto-detect and register MOE modules as _QuantSparseSequentialMoe.

    Walks the model tree, identifies MoE blocks by their structural attributes
    (``gate`` + ``experts``), and registers unregistered ones with ``_QuantSparseSequentialMoe``.
    """
    visited_types = set()
    for name, module in model.named_modules():
        mod_type = type(module)

        # Avoid duplicate registration: skip if we already processed this type
        # in this walk, or if it was previously registered in the QuantModuleRegistry.
        if mod_type in visited_types or QuantModuleRegistry.get(mod_type) is not None:
            continue

        visited_types.add(mod_type)

        if _is_sparse_sequaential_moe_block(module):
            print(
                f"\033[1mDetected MOE module '{name}' of type {mod_type.__name__}, "
                f"registering with _QuantSparseSequentialMoe.\033[0m"
            )
            QuantModuleRegistry.register({mod_type: f"hf.{mod_type.__name__}"})(
                _QuantSparseSequentialMoe
            )


def _fused_experts_wrapper(module):
    """Return the ``_QuantFusedExperts`` subclass for a fused-MoE-experts module, else None.

    Detects the standardized HuggingFace transformers 5.0+ fused expert pattern: ``num_experts``,
    a 3-D ``down_proj`` and a 3-D first projection, either

    * gated (:class:`_QuantFusedExperts`): a 3-D ``gate_up_proj`` fusing gate+up. Matches
      ``MixtralExperts``, ``Qwen2MoeExperts``, ``Qwen3MoeExperts``,
      ``Qwen3_5MoeExperts``, ``DeepseekV3NaiveMoe``, ``JambaExperts``,
      ``OlmoeExperts``, ``MiniMaxM2Experts``, ``MiniMaxM3VLExperts``, etc.
    * ungated (:class:`_QuantNemotronHExperts`): a 3-D ``up_proj`` and neither ``gate_proj`` nor
      ``gate_up_proj``. Matches NemotronH ``NemotronHExperts``.

    Matching is by shape alone, so GptOss and Llama4TextExperts -- transposed layouts with their own
    explicit registrations -- match too; :func:`register_fused_experts_on_the_fly` never reaches
    them while those registrations stand.

    ``act_fn`` is not required: these wrappers only intercept the two ``F.linear``
    calls, so modules with a custom gated activation (e.g. ``MiniMaxM3VLExperts``)
    are still supported.
    """
    # ``act_fn`` is a common marker of the standard HF fused-experts pattern, but it is NOT required by
    # _QuantFusedExperts: the wrapper only quantizes the two projections; the activation is applied by the
    # module's OWN forward. Models with a custom activation applied inline -- e.g. MiniMax-M3's SwiGLU via
    # ``_apply_gate`` (``swiglu_alpha``/``swiglu_limit``, no ``act_fn``) -- are otherwise standard fused
    # experts, so gate on the STRUCTURAL signature (num_experts + two 3-D fused projections) and treat
    # ``act_fn`` as optional.
    if not hasattr(module, "num_experts"):
        return None
    down = getattr(module, "down_proj", None)
    if not isinstance(down, (nn.Parameter, Tensor)) or down.dim() != 3:
        return None
    gate_up = getattr(module, "gate_up_proj", None)
    if isinstance(gate_up, (nn.Parameter, Tensor)) and gate_up.dim() == 3:
        return _QuantFusedExperts
    up = getattr(module, "up_proj", None)
    if isinstance(up, (nn.Parameter, Tensor)) and up.dim() == 3:
        # Only an ungated layout: a split gated one (3-D gate_proj + up_proj) would otherwise be
        # claimed here and its gate_proj left unquantized.
        if getattr(module, "gate_proj", None) is None and gate_up is None:
            return _QuantNemotronHExperts
    return None


def _is_fused_experts_module(module):
    """Check if a module is a fused MoE expert container compatible with _QuantFusedExperts."""
    return _fused_experts_wrapper(module) is not None


def register_fused_experts_on_the_fly(model):
    """Auto-detect and register fused MoE expert modules as _QuantFusedExperts.

    Walks the model tree, identifies fused expert containers by their structural
    attributes (``gate_up_proj`` + ``down_proj`` 3-D parameters), and registers
    unregistered ones with ``_QuantFusedExperts``.

    Skips modules that are already registered (e.g. Llama4TextExperts, GptOssExperts
    have their own explicit registrations with different quantization strategies).
    """
    visited_types = set()
    for name, module in model.named_modules():
        mod_type = type(module)

        if mod_type in visited_types or QuantModuleRegistry.get(mod_type) is not None:
            continue

        visited_types.add(mod_type)

        wrapper = _fused_experts_wrapper(module)
        if wrapper is not None and getattr(module, "is_transposed", False):
            # `_fused_experts_wrapper` matches on parameter SHAPE only, so it also claims a
            # transposed `(num_experts, in_dim, out_dim)` container. The fused-experts export
            # (export/moe_utils.py) then splits gate/up by slicing dim 1 at `intermediate_dim`, the
            # wrong axis for that layout, and writes silently wrong weights. Leave it unquantized.
            warnings.warn(
                f"Fused MoE experts '{name}' ({mod_type.__name__}) store their weights transposed "
                "(num_experts, in_dim, out_dim), which the generic _QuantFusedExperts export path "
                "does not support: it splits gate/up along dim 1, which is the input dim for this "
                "layout. Leaving them unquantized; such a class needs its own quantized wrapper, as "
                "GptOssExperts and Llama4TextExperts have."
            )
            continue
        if wrapper is not None:
            print(
                f"\033[1mDetected fused MoE experts '{name}' of type {mod_type.__name__}, "
                f"registering with {wrapper.__name__}.\033[0m"
            )
            QuantModuleRegistry.register({mod_type: f"hf.{mod_type.__name__}"})(wrapper)


def normalize_experts_impl_on_the_fly(model):
    """Redirect HF fused-experts modules onto a quantization-supported backend.

    HF transformers 5.0+ decorates fused-experts forwards with
    ``@use_experts_implementation``, which dispatches on
    ``config._experts_implementation``. ``_QuantFusedExperts`` intercepts the
    ``eager`` (per-expert ``F.linear``) and ``grouped_mm``
    (``transformers.integrations.moe._grouped_linear``) forwards. The remaining
    backends — ``batched_mm`` (``torch.bmm``), ``sonicmoe`` (opaque CUDA kernel),
    ``kernels``/hub, etc. — bypass those hooks, so calibration silently collects
    nothing and the exported checkpoint produces garbage at inference.

    To avoid that while keeping the fast grouped path, this redirects any
    unsupported backend to ``grouped_mm`` (which has a CPU / old-HW fallback, so
    it is universally runnable). ``eager``, ``grouped_mm`` and ``torch_a2a`` are
    left untouched, and so is ``None``, which dispatches to the eager ``F.linear``
    forward -- but transformers resolves an unrequested backend to ``grouped_mm``
    wherever that can dispatch, when it builds the model, so a model calibrates on
    the grouped path unless ``eager`` was asked for. Applied to the model config
    and recursively to ``text_config`` / ``vision_config`` / ``audio_config`` /
    ``speech_config`` whenever a fused-experts module is present.
    """
    fused_experts = [module for module in model.modules() if _is_fused_experts_module(module)]
    if not fused_experts:
        return

    nested_cfg_attrs = ("text_config", "vision_config", "audio_config", "speech_config")
    # Backends already intercepted by _QuantFusedExperts (None => eager F.linear forward). `torch_a2a`
    # (all-to-all DP=EP) runs its local experts through the same `_grouped_linear`, so _QuantFusedExperts
    # intercepts it too -- don't clobber it back to grouped_mm.
    supported_impls = (None, "eager", "grouped_mm", "torch_a2a")
    # A class with its own quantized wrapper -- GPT-OSS's quantizes down_proj's input by intercepting
    # the eager forward's bmm/matmul -- only works on the eager forward, so keep main's forced eager
    # for such a model.
    force_eager = any(
        (wrapper := QuantModuleRegistry.get(type(module))) is not None
        and not issubclass(wrapper, _QuantFusedExperts)
        for module in fused_experts
    )

    def _normalize(cfg):
        if cfg is None:
            return
        if hasattr(cfg, "_experts_implementation"):
            if force_eager and cfg._experts_implementation != "torch_a2a":
                cfg._experts_implementation = "eager"
            elif cfg._experts_implementation not in supported_impls:
                cfg._experts_implementation = "grouped_mm"
        for sub in nested_cfg_attrs:
            if hasattr(cfg, sub):
                _normalize(getattr(cfg, sub))

    if hasattr(model, "config"):
        _normalize(model.config)

    # Some models construct auxiliary blocks from a deep copy of the root config.
    # Nemotron-H MTP does this, so updating only ``model.config`` leaves its fused
    # experts on an unsupported backend that bypasses the quantizer hooks.
    for module in fused_experts:
        _normalize(getattr(module, "config", None))


def assert_fused_experts_intercepted(model):
    """Raise if a quantized fused-experts module was never reached by an interception hook.

    :class:`_QuantFusedExperts` collects statistics by intercepting the calls the experts
    forward makes -- ``F.linear`` per expert on the eager path, ``_grouped_linear`` once per
    projection on the grouped_mm path. A backend that uses neither bypasses both hooks
    *silently*: no quantizer is invoked, no amax is collected, and the export then falls back
    to weight-derived scales while warning about calibration size -- which points at the
    dataset when the fault is in the dispatch.

    The mismatch is structural. :func:`register_fused_experts_on_the_fly` claims a module on
    the *shape* of its parameters, but the hooks depend on the *behavior* of a forward that
    nothing enforces, and "eager" is not a shared implementation -- it is whatever the model
    author wrote. :func:`normalize_experts_impl_on_the_fly` can redirect a class that carries
    ``@use_experts_implementation``, but a class without it ignores
    ``config._experts_implementation`` entirely, so no config change can rescue it. Checking
    the hit count is a direct test of the thing actually depended on.

    Only modules whose forward ran and that have at least one enabled quantizer are reported:
    an experts module no calibration token reached (vision experts under text-only calibration)
    falls back to weight-derived scales as before, and one the recipe leaves unquantized loses
    nothing by not being intercepted.
    """
    missed = []
    for name, module in model.named_modules():
        if not isinstance(module, _QuantFusedExperts):
            continue
        # A missing attribute (rather than 0) means it was already cleaned up -- say nothing.
        if getattr(module, "_intercept_count", None) != 0 or not getattr(
            module, "_forward_count", 0
        ):
            continue

        first = module._first_proj_attr
        quantizers = []
        for attr in (f"{first}_weight_quantizers", "down_proj_weight_quantizers"):
            quantizers.extend(getattr(module, attr, None) or [])
        for attr in (f"{first}_input_quantizer", "down_proj_input_quantizer"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None:
                quantizers.append(quantizer)
        if not any(getattr(q, "is_enabled", False) for q in quantizers):
            continue

        impl = getattr(getattr(module, "config", None), "_experts_implementation", None)
        missed.append(f"  {name} ({type(module).__name__}, experts_implementation={impl!r})")

    if not missed:
        return

    raise RuntimeError(
        "Fused-experts calibration collected nothing: the forward of the module(s) below reached "
        "neither `F.linear` nor `_grouped_linear`, so not one of their quantizers was invoked and "
        "every expert would be exported with a weight-derived fallback scale.\n"
        + "\n".join(missed)
        + "\n\nTheir forward ran, but the active experts backend bypasses both hooks -- "
        "`batched_mm`, `sonicmoe`, or a hand-written `bmm`/`matmul` forward on a class without "
        "`@use_experts_implementation`, which `normalize_experts_impl_on_the_fly` cannot "
        "redirect. Check the module's `forward` and `config._experts_implementation`."
    )


def _is_supported_hf_model(model):
    """Check if the model a valid model for transformers quantization specific support."""
    supported_models = [transformers.PreTrainedModel]
    try:
        from peft import PeftModel

        supported_models.append(PeftModel)
    except ImportError:
        pass
    return isinstance(model, tuple(supported_models))


def is_homogeneous_hf_model(model: nn.Module) -> bool:
    from modelopt.torch.models.nemotron_h.modeling_ptq import is_nemotron_h_model

    if is_nemotron_h_model(model):
        return False
    decoder_layers = get_homogeneous_hf_decoder_layers(model)
    if decoder_layers is None or len(decoder_layers) == 0:
        return False
    layer_classes = {type(layer) for layer in decoder_layers}
    return len(layer_classes) == 1


def get_homogeneous_hf_decoder_layers(model: nn.Module) -> nn.ModuleList | None:
    if not _is_supported_hf_model(model):
        return None

    decoder = model
    if hasattr(decoder, "model"):
        decoder = decoder.model
    if hasattr(decoder, "language_model"):
        decoder = decoder.language_model
    if hasattr(decoder, "layers"):
        return decoder.layers

    return None


@contextmanager
def setup_model_for_gradient_checkpointing(model: nn.Module):
    use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        # Disable use_cache explicitly before forward is called
        use_cache = model.config.use_cache
        model.config.use_cache = False

    if not hasattr(model, "gradient_checkpointing_enable") or not (
        hasattr(model, "supports_gradient_checkpointing") and model.supports_gradient_checkpointing
    ):
        warnings.warn(
            "AutoQuantize: Huggingface model without gradient checkpointing support detected. "
            "AutoQuantize will consume more memory."
        )
    else:
        try:
            warnings.warn(
                "AutoQuantize: Huggingface model detected - Enabling gradient checkpointing. "
                "Disable gradient checkpointing after AutoQuantize if this is not desired!"
            )
            model.gradient_checkpointing_enable({"use_reentrant": True})
            for m in model.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.train()  # Make sure the module is in training mode to enable gradient checkpointing
                else:
                    # Eval mode for non-checkpointed modules to avoid fused kernels
                    # that bypass linear layers. E.g. in nemotron-h, the Mamba layer's
                    # training path uses a fused kernel that takes out_proj weights
                    # directly, skipping the linear module's forward (and thus quantization).
                    m.eval()
        except Exception as e:
            warnings.warn(
                f"AutoQuantize: Error enabling gradient checkpointing for huggingface model due to: {e}, "
                "AutoQuantize will consume more memory."
            )
    yield
    if use_cache is not None:
        model.config.use_cache = use_cache


def _is_param_grad_enabled_for_auto_quantize(pname, model):
    # Enable grad for embedding layers to propagate gradients through the model,
    # allowing each layer to compute its input gradients during the backward pass.
    return "embed" in pname


AutoQuantizeGradientSearcher.register_custom_support(
    _is_supported_hf_model,
    setup_model_for_gradient_checkpointing,
    _is_param_grad_enabled_for_auto_quantize,
)

# Model-specific PTQ support lives with its model in
# ``modelopt/torch/models/<model_type>/modeling_ptq.py`` and registers itself on import. It is
# imported here: after the generic wrappers it builds on are defined, and before the
# homogeneous decoder discoverer below, since the first matching discoverer wins and
# Nemotron-H's is the more specific one.
for _model_type in (
    "falcon",
    "gpt_oss",
    "llama4",
    "nemotron_h",
    "step3p5",
    "step3p7",
):
    importlib.import_module(f"modelopt.torch.models.{_model_type}.modeling_ptq")

LayerActivationCollector.register_decoder_layer_support(
    is_homogeneous_hf_model, get_homogeneous_hf_decoder_layers
)


CUSTOM_MODEL_PLUGINS.update(
    [
        register_fused_experts_on_the_fly,
        normalize_experts_impl_on_the_fly,
        register_sparse_moe_on_the_fly,
        register_hf_attentions_on_the_fly,
        convert_hf_parallel_linears_on_the_fly,
    ]
)
