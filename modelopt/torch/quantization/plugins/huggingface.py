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
import torch.distributed as dist
import torch.nn as nn
import transformers
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard
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
from modelopt.torch.utils import warn_rank_0
from modelopt.torch.utils.distributed import ParallelState, recorded_distribution_plan

from ..algorithms import AutoQuantizeGradientSearcher
from ..conversion import register
from ..nn import QuantInputBase, QuantModule, QuantModuleRegistry, TensorQuantizer
from ..nn.modules.quant_linear import _QuantLinear
from ..nn.modules.tensor_quantizer import _try_stacked_fake_quant
from ..utils import replace_function, sync_moe_expert_amax
from ..utils.layerwise_calib import LayerActivationCollector
from .attention import register_attention_for_kv_quant
from .custom import CUSTOM_MODEL_PLUGINS, _ParallelLinear, _QuantFunctionalMixin

logger = logging.getLogger(__name__)

try:
    import transformers.integrations.moe as _transformers_moe
except ImportError:
    _transformers_moe = None

# Whether transformers has the grouped experts forward's `_grouped_linear`, the call _QuantFusedExperts
# intercepts on it: without it nothing intercepts grouped_mm, so calibration cannot use that backend.
_HAS_GROUPED_LINEAR = _transformers_moe is not None and hasattr(
    _transformers_moe, "_grouped_linear"
)

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
    shard = (Shard(0),)


class HFRowParallelLinear(HFParallelLinear):
    supported_hf_tp_plans = ["rowwise", "rowwise_rep"]
    shard = (Shard(1),)


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
    # An expert-parallel load keeps its experts' mesh there too, and shards no Linear by design.
    plan = recorded_distribution_plan(model)
    if tp_mesh is not None and not converted and not (plan is not None and plan.is_expert_parallel):
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
    forwards are intercepted. Both calibrate the same per-expert weight quantizers
    and shared input quantizers, and export the same keys; the activation scales
    they calibrate can differ, so ``eager`` is the way to calibrate as before
    ``grouped_mm`` was intercepted. Only one backend's hook fires per forward
    (selected by ``config._experts_implementation``), so installing both is safe:
    the ``grouped_mm`` fallback does not call ``F.linear``, which the CPU tests that
    run it pin.

    Limitation: the ``grouped_mm`` path assumes the non-transposed weight layout
    ``(num_experts, out_dim, in_dim)`` used by all standard fused experts. The
    transposed layout (e.g. GptOss, which has its own quantized wrapper) is not
    handled here. ``batched_mm`` (``torch.bmm``) and ``sonicmoe`` (opaque CUDA
    kernel) backends are not intercepted; ``normalize_experts_impl_on_the_fly``
    redirects them to ``grouped_mm`` before calibration.

    A non-gated layout (a single ``up_proj``) is handled by a subclass that overrides the
    ``_first_proj_attr`` / ``_is_gated`` hooks below; each layout names the
    first-projection quantizers after its backing parameter.
    """

    # Name of the 3-D weight parameter feeding the first ``F.linear`` per expert.
    # Gated experts fuse gate+up into ``gate_up_proj``; non-gated experts use a
    # single ``up_proj``, which their subclass names here.
    _first_proj_attr = "gate_up_proj"
    # Whether the first projection packs a gate half that must be split on export.
    _is_gated = True
    # Under expert parallelism each rank's experts see only the tokens routed to them, so calibration
    # shares the layer's input amax over the expert-parallel group (see max_calibrate).
    _share_input_amax_across_ep = True
    # Whether the wrapper quantizes only on the eager per-expert forward, so a model holding it must
    # calibrate there (see normalize_experts_impl_on_the_fly).
    _eager_only = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # The grouped_mm forward quantizes the stored weight and never reaches get_expert_weight, so
        # a wrapper that defines its weight domain there would calibrate the wrong weights on it.
        if (
            cls.get_expert_weight is not _QuantFusedExperts.get_expert_weight
            and not cls._eager_only
        ):
            raise TypeError(
                f"{cls.__name__} overrides get_expert_weight, which the grouped_mm forward bypasses; "
                "set _eager_only = True so that its models calibrate on the eager forward."
            )

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
        setattr(self, self._first_proj_input_quantizer_attr, TensorQuantizer())
        setattr(
            self,
            self._first_proj_weight_quantizers_attr,
            nn.ModuleList([TensorQuantizer() for _ in range(n)]),
        )
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizers = nn.ModuleList([TensorQuantizer() for _ in range(n)])

        self._register_temp_attribute("_down_proj_linear", False)
        self._register_temp_attribute("_current_expert_idx", 0)
        # Per forward: each projection's experts, fake-quantized together (see _quantized_weight).
        self._register_temp_attribute("_stacked_weights", {})
        # Cumulative tokens-per-expert offsets of the grouped forward in flight (grouped_mm or
        # torch_a2a), None on the eager path: tells a weight-input capture (local Hessian, imatrix,
        # GPTQ) which forward ran and how to split the expert-sorted batch. Cleared when forward
        # returns, so a capture reached outside one skips rather than splitting by stale offsets.
        self._register_temp_attribute("_grouped_offs", None)
        # How many times an interception hook fired for this module. Still zero after a
        # calibration forward means the active experts backend reached neither `F.linear` nor
        # `_grouped_linear`, so no quantizer ever saw data -- `assert_fused_experts_intercepted`
        # turns that into an error instead of an export full of fallback scales.
        self._register_temp_attribute("_intercept_count", 0)
        # How many times this module's forward ran: an experts module that no calibration token
        # reached (e.g. vision experts under text-only calibration) has nothing to intercept.
        self._register_temp_attribute("_forward_count", 0)
        # Whether this calibration captures the experts' inputs to calibrate their weights (GPTQ,
        # local Hessian, imatrix), so that the weight quantizers need the interception too.
        self._register_temp_attribute("_weight_input_capture", False)

    @property
    def _running_grouped_mm(self) -> bool:
        """Whether this module's forward runs its expert matmuls through ``_grouped_linear``.

        The ``grouped_mm`` backend and the ``torch_a2a`` expert-parallel forward both do, handing each
        projection one expert-sorted batch rather than one ``F.linear`` call per expert.
        """
        cfg = getattr(self, "config", None)
        return getattr(cfg, "_experts_implementation", None) in ("grouped_mm", "torch_a2a")

    def _local_experts(self, weight: torch.Tensor) -> torch.Tensor:
        """``weight`` indexed the way the per-expert quantizers are: by this module's own experts.

        Under expert parallelism transformers 5.16 and later keep a fused expert weight as a DTensor
        sharded over the experts (earlier ones keep a plain tensor of this rank's experts) and set
        ``num_experts`` to the local count, so indexing the DTensor would read the GLOBAL expert: from
        rank 1 on, another rank's. Its local shard holds this rank's experts in quantizer order. A
        DTensor holding every expert (FSDP2) is indexed as it is, and one whose local shard holds
        another number of experts than the quantizers is refused.
        """
        if isinstance(weight, DTensor) and weight.shape[0] != self.num_experts:
            local = weight.to_local()
            if local.shape[0] != self.num_experts:
                raise RuntimeError(
                    f"{type(self).__name__} has {self.num_experts} experts, but its expert weight, a "
                    f"DTensor placed {weight.placements}, holds {local.shape[0]} on this rank: only a "
                    "weight sharded over the experts can be indexed by local expert."
                )
            return local
        return weight

    def _quantize_grouped_weight(self, weight: torch.Tensor, quantizers: nn.ModuleList):
        """Fake-quantize a fused 3-D weight per expert.

        Each expert slice passes through its own weight quantizer (collecting
        ``amax`` during calibration, fake-quantizing afterwards), matching the
        eager per-expert ``F.linear`` semantics. Returns the weight untouched when every
        quantizer would pass its slice through unchanged, so passthrough stays exact (and cheap):
        a disabled quantizer, or one that only collects statistics, as in calibration.

        Reached only inside the experts forward, around which transformers swaps an expert-parallel
        DTensor weight for this rank's local shard, so the weight is already indexed by local expert.
        """
        # A quantizer that is disabled, or that collects statistics without quantizing, hands its
        # slice back unchanged, unless it applies a pre-quant scale or a rotation (a disabled one
        # still does, as on the eager path). Then the result is the weight itself: building it
        # slice by slice would copy every expert's weight on each calibration forward.
        if all(
            isinstance(q, TensorQuantizer)
            and not (q.is_enabled and q._if_quant)
            and q.pre_quant_scale is None
            and not q.rotate_is_enabled
            for q in quantizers
        ):
            # Only a calibrating quantizer does anything with its slice: one whose amax is constant
            # (enable_stats_collection leaves it neither quantizing nor calibrating) would not.
            collecting = [q.is_enabled and q._if_calib for q in quantizers]
            if any(collecting):
                for q, w, collects in zip(quantizers, weight, collecting, strict=True):
                    if collects:
                        q(w)  # collects this expert's statistics
            return weight
        # One kernel launch for every expert where the format has one. Unlike the eager path, which
        # without autograd quantizes only the routed experts, this one quantizes every expert either
        # way, and the kernel's one output is the buffer the slice-by-slice path below fills.
        stacked = _try_stacked_fake_quant(weight, quantizers)
        if stacked is not None:
            return stacked
        if torch.is_grad_enabled():
            # Slices written into one buffer would chain a full-size gradient copy per expert into
            # the backward (QAT), so stack them there.
            return torch.stack([q(w) for q, w in zip(quantizers, weight, strict=True)])
        # Each slice goes into the output as it is made: a stack would hold all of them beside it.
        # Strict, so every row of the output is written or the call fails. The output takes the
        # dtype the stack promotes the slices to: a pre-quant scale can widen one, and narrowing
        # it here would compute from other weights than with autograd.
        out: torch.Tensor | None = None
        for idx, (q, w) in enumerate(zip(quantizers, weight, strict=True)):
            quantized = q(w)
            if out is None:
                out = torch.empty_like(weight, dtype=quantized.dtype)
            elif quantized.dtype != out.dtype:
                out = out.to(torch.promote_types(out.dtype, quantized.dtype))
            out[idx] = quantized
            del quantized  # not alive beside the next slice
        return out

    @property
    def functionals_to_replace(self):
        _orig_linear = torch.nn.functional.linear
        # First-projection quantizers, by layout: gate_up_proj_* (gated) or up_proj_* (non-gated).
        # Bound once; the objects are stable across the forward.
        first_input_q = getattr(self, self._first_proj_input_quantizer_attr)
        first_weight_qs = getattr(self, self._first_proj_weight_quantizers_attr)

        # The HF fused expert forward calls F.linear exactly twice per expert
        # in strict alternation: first the up/gate_up projection, then down_proj.
        # forward() resets the toggle before each call to super().forward().
        def _quantized_linear(input, weight, bias=None):
            self._intercept_count += 1
            # The eager path: no expert-sorted batch for a capture hook to split.
            self._grouped_offs = None
            if self._down_proj_linear:
                idx = self._current_expert_idx
                input = self.down_proj_input_quantizer(input)
                weight = self._quantized_weight(
                    "down_proj", self.down_proj_weight_quantizers, idx, input.dtype
                )
            else:
                idx = self._get_expert_idx_from_first_proj(weight)
                self._current_expert_idx = idx
                input = first_input_q(input)
                weight = self._quantized_weight(
                    self._first_proj_attr,
                    first_weight_qs,
                    idx,
                    input.dtype,
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
        if _HAS_GROUPED_LINEAR:
            _orig_grouped_linear = _transformers_moe._grouped_linear

            def _quantized_grouped_linear(input, weight, offs, bias=None, is_transposed=False):
                if self._eager_only:
                    # Quantizing the stored weight here would bypass get_expert_weight, which defines
                    # this wrapper's weights; normalize_experts_impl_on_the_fly pins its model to eager.
                    raise RuntimeError(
                        f"{type(self).__name__} quantizes on the eager experts forward only, but its "
                        'grouped_mm forward ran. Load the model with experts_implementation="eager", '
                        "and without expert parallelism, whose forward runs it too."
                    )
                self._intercept_count += 1
                # The batch holds every expert's rows, so no expert for the eager loop's index to name.
                # The offsets, recorded before quantizing input, let a capture hook on the input
                # quantizer split the expert-sorted batch back per expert.
                self._current_expert_idx = None
                self._grouped_offs = offs
                quantizers = (
                    self.down_proj_weight_quantizers if self._down_proj_linear else first_weight_qs
                )
                input_quantizer = (
                    self.down_proj_input_quantizer if self._down_proj_linear else first_input_q
                )
                input = self._quantize_grouped_input(input, offs, input_quantizer)
                weight = self._quantize_grouped_weight(weight, quantizers)
                self._down_proj_linear = not self._down_proj_linear
                return _orig_grouped_linear(
                    input, weight, offs, bias=bias, is_transposed=is_transposed
                )

            replacements.append((_transformers_moe, "_grouped_linear", _quantized_grouped_linear))

        return replacements

    @staticmethod
    def _quantize_grouped_input(input, offs, quantizer):
        """Quantize a grouped projection's input, collecting statistics on the rows it computes only.

        Rows past ``offs[-1]`` belong to no expert of this module: under transformers' own expert
        parallelism they are the rows routed to other ranks' experts, sorted to the tail, which the
        grouped GEMM neither reads nor writes. Before transformers 5.17 the down projection's input
        holds them uninitialized, so a calibrating quantizer sees only the rows before them; the tail
        passes through unread. Finding where it starts costs a host sync, so it is skipped while the
        quantizer collects nothing.
        """
        if getattr(quantizer, "is_enabled", False) and getattr(quantizer, "_if_calib", False):
            routed = int(offs[-1]) if offs.numel() else 0
            if routed < input.shape[0]:
                computed = quantizer(input[:routed])
                return torch.cat([computed, input[routed:].to(computed.dtype)])
        return quantizer(input)

    def _quantized_weight(self, name, quantizers, idx, dtype):
        """Expert ``idx``'s ``name`` projection, fake-quantized by its own weight quantizer.

        When :func:`_try_stacked_fake_quant` supports the experts' quantizers, all experts are
        fake-quantized at once, by one kernel launch on the projection's first call of the forward,
        instead of one launch per expert.
        """
        if name not in self._stacked_weights:
            # Without autograd the per-expert path is cheaper: it quantizes only the routed experts,
            # one at a time. A subclass that rebuilds each expert's weight (FP8 dequant) keeps it too.
            stacked = (
                _try_stacked_fake_quant(getattr(self, name), quantizers)
                if torch.is_grad_enabled()
                and type(self).get_expert_weight is _QuantFusedExperts.get_expert_weight
                else None
            )
            # unbind's backward writes one gradient for all experts; weight[idx] per expert would
            # write a full-size one each.
            self._stacked_weights[name] = None if stacked is None else stacked.unbind(0)
        stacked = self._stacked_weights[name]
        if stacked is None:
            return quantizers[idx](self.get_expert_weight(name, idx, dtype))
        return stacked[idx]

    def forward(self, *args, **kwargs):
        self._down_proj_linear = False
        # Only a forward that makes a projection call is work the interception check expects. The
        # eager path makes one per expert some row was routed to: none for a forward with no rows, nor
        # for one whose rows transformers' expert parallelism routed to other ranks' experts (masked
        # to num_experts). The grouped paths call _grouped_linear whatever rows they are handed.
        # Once a hook has fired the check passes, so stop counting: the routing test syncs the host.
        if self._intercept_count == 0:
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            top_k_index = args[1] if len(args) > 1 else kwargs.get("top_k_index")
            if hidden_states is None or (
                hidden_states.numel()
                and (
                    self._running_grouped_mm
                    or not isinstance(top_k_index, torch.Tensor)
                    or bool((top_k_index < self.num_experts).any())
                )
            ):
                self._forward_count += 1
        self._stacked_weights = {}
        try:
            return super().forward(*args, **kwargs)
        finally:
            # Autograd keeps what backward needs; holding more would pin every expert's copy.
            self._stacked_weights = {}
            # A capture reached outside a forward must not split its input by this forward's offsets.
            self._grouped_offs = None

    def get_expert_weight(self, name, idx, dtype=None):
        """Return one expert's projection in the calibration/export weight domain.

        Indexed by this module's own experts, as the per-expert quantizers are (see
        :meth:`_local_experts`).
        """
        return self._local_experts(getattr(self, name))[idx]

    def iter_weights_for_calibration(self):
        """Yield ``(weight_slice, quantizer)`` per-expert pairs.

        The base impl uses singular ``*_weight_quantizer`` and skips fused-
        experts modules, so weight-only calibration never reaches per-expert
        quantizers without this override.
        """
        for weight_name, quantizers_name in (
            (self._first_proj_attr, self._first_proj_weight_quantizers_attr),
            ("down_proj", "down_proj_weight_quantizers"),
        ):
            weight = getattr(self, weight_name, None)
            quantizers = getattr(self, quantizers_name, None)
            if weight is None or quantizers is None:
                continue
            for idx, q in enumerate(quantizers):
                yield self.get_expert_weight(weight_name, idx), q

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
        for weight_slice, q in self.iter_weights_for_calibration():
            if isinstance(q, TensorQuantizer) and q.fake_quant:
                self._fold_weight_quantizer(q, (weight_slice,), keep_attrs)


class _QuantFP8Experts(_QuantFusedExperts):
    """Scale-aware PTQ for native Transformers FP8 experts, dequantized per expert."""

    # The block scales apply in get_expert_weight, which only the eager forward reaches: transformers'
    # FP8 grouped_mm forward runs its own kernels, which nothing here intercepts. So the grouped
    # interceptor's refusal never fires for these: what keeps them on eager is
    # normalize_experts_impl_on_the_fly, which runs on convert and on restore.
    _eager_only = True

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
        weight = self._local_experts(getattr(self, name))[idx]
        if weight.element_size() > 1:
            return weight if dtype is None else weight.to(dtype)
        dtype = dtype or torch.get_default_dtype()
        scales = self._local_experts(getattr(self, f"{name}_scale_inv"))[idx].float()
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
        names = [
            n for n in (self._first_proj_attr, "down_proj") if getattr(self, n).element_size() == 1
        ]
        # Sharded over the experts, one unpacked parameter would have to stay sharded too. Checked for
        # both projections before either is unpacked, so a refusal leaves the module as it was.
        if any(self._local_experts(getattr(self, n)) is not getattr(self, n) for n in names):
            raise NotImplementedError(
                f"{type(self).__name__} cannot unpack native FP8 expert weights sharded over the "
                "experts. Quantize the model without expert parallelism."
            )
        for name in names:
            weight = getattr(self, name)
            unpacked = torch.empty_like(weight, dtype=torch.get_default_dtype())
            for idx in range(self.num_experts):
                unpacked[idx].copy_(self.get_expert_weight(name, idx))
            setattr(self, name, nn.Parameter(unpacked, requires_grad=False))


def _get_fused_experts_quantizer_attr_names(module):
    """Return quantizer attribute names for a converted fused-experts module."""
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    return (
        f"{first_proj_attr}_input_quantizer",
        f"{first_proj_attr}_weight_quantizers",
        "down_proj_input_quantizer",
        "down_proj_weight_quantizers",
    )


def _is_quant_fused_experts_module(module):
    """Return True for a converted HF fused-MoE-experts quantization wrapper."""
    return isinstance(module, _QuantFusedExperts)


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
    """Return :class:`_QuantFusedExperts` for a fused-MoE-experts module, else None.

    Detects the standardized HuggingFace transformers 5.0+ fused expert pattern: ``num_experts``,
    a 3-D ``down_proj`` and a 3-D ``gate_up_proj`` fusing gate+up. Matches ``MixtralExperts``,
    ``Qwen2MoeExperts``, ``Qwen3MoeExperts``, ``Qwen3_5MoeExperts``, ``DeepseekV3NaiveMoe``,
    ``JambaExperts``, ``OlmoeExperts``, ``MiniMaxM2Experts``, ``MiniMaxM3VLExperts``, etc. Experts
    of another layout register a wrapper of their own, as Nemotron-H's non-gated ones do.

    Matching is by shape alone, so GptOss and Llama4TextExperts -- transposed layouts with their own
    explicit registrations -- match too; :func:`register_fused_experts_on_the_fly` never reaches
    them while those registrations stand.

    ``act_fn`` is not required: these wrappers only intercept the two ``F.linear``
    calls, so modules with a custom gated activation (e.g. ``MiniMaxM3VLExperts``)
    are still supported.
    """
    if not hasattr(module, "num_experts"):
        return None
    down = getattr(module, "down_proj", None)
    if not isinstance(down, (nn.Parameter, Tensor)) or down.dim() != 3:
        return None
    gate_up = getattr(module, "gate_up_proj", None)
    if isinstance(gate_up, (nn.Parameter, Tensor)) and gate_up.dim() == 3:
        return _QuantFusedExperts
    return None


def _is_fused_experts_module(module):
    """Check if a module is a fused MoE expert container compatible with _QuantFusedExperts.

    That is one :func:`_fused_experts_wrapper` detects, or one whose class registered a subclass of
    it, as Nemotron-H's non-gated experts and transformers' FP8 experts do.
    Transposed containers match too, though registration skips them:
    :func:`normalize_experts_impl_on_the_fly` must still find them, to keep GPT-OSS's own wrapper on
    eager, the only forward on which it quantizes ``down_proj``'s input (it intercepts the eager
    forward's ``bmm``/``matmul``).
    """
    if _fused_experts_wrapper(module) is not None:
        return True
    wrapper = QuantModuleRegistry.get(type(module)) if hasattr(module, "num_experts") else None
    return wrapper is not None and issubclass(wrapper, _QuantFusedExperts)


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
        if wrapper is not None and _transposed_experts([module]):
            # `_fused_experts_wrapper` matches on parameter SHAPE only, so it also claims a
            # transposed `(num_experts, in_dim, out_dim)` container. The fused-experts export
            # (export/moe_utils.py) then splits gate/up by slicing dim 1 at `intermediate_dim`, the
            # wrong axis for that layout, and writes silently wrong weights. Leave it unquantized.
            warnings.warn(
                f"Fused MoE experts '{name}' ({mod_type.__name__}) store their weights transposed "
                "(num_experts, in_dim, out_dim), which the generic _QuantFusedExperts export path "
                "does not support: it splits gate/up along dim 1, which is the input dim for this "
                "layout. Leaving them unquantized, so export_hf_checkpoint will refuse their MoE "
                "block; such a class needs its own quantized wrapper, as GptOssExperts and "
                "Llama4TextExperts have."
            )
            continue
        if wrapper is not None:
            print(
                f"\033[1mDetected fused MoE experts '{name}' of type {mod_type.__name__}, "
                f"registering with {wrapper.__name__}.\033[0m"
            )
            QuantModuleRegistry.register({mod_type: f"hf.{mod_type.__name__}"})(wrapper)


def _transposed_experts(modules) -> list[str]:
    """Names of the expert classes in ``modules`` stored ``(num_experts, in_dim, out_dim)``.

    A container may say so (``is_transposed``), and a gated fused-experts one shows it either way:
    stored ``(num_experts, out_dim, in_dim)``, its first projection takes the dim ``down_proj`` puts
    out, which stored transposed it never does.
    """
    return sorted(
        {
            type(module).__name__
            for module in modules
            if getattr(module, "is_transposed", False)
            or (
                (wrapper := _fused_experts_wrapper(module)) is not None
                and getattr(module, wrapper._first_proj_attr).shape[-1] != module.down_proj.shape[1]
            )
        }
    )


def _eager_only_experts(modules) -> list[str]:
    """Names of the expert classes in ``modules`` ModelOpt quantizes on the eager forward only.

    A class with its own quantized wrapper -- GPT-OSS's quantizes down_proj's input by intercepting
    the eager forward's bmm/matmul -- only works on the eager forward, and so does a fused-experts
    wrapper that says so (FP8's). The registry lookup caches the dynamic class it generates, as
    quantizing the module would. Those wrappers register when their module is imported;
    register_fused_experts_on_the_fly, which may run after this, registers only the generic
    wrapper, which is not eager-only.
    """
    return sorted(
        {
            type(module).__name__
            for module in modules
            if (wrapper := QuantModuleRegistry.get(type(module))) is not None
            and (not issubclass(wrapper, _QuantFusedExperts) or wrapper._eager_only)
        }
    )


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
    it is universally runnable), or to ``eager`` on a transformers without the
    ``_grouped_linear`` grouped_mm is intercepted at. ``eager``, ``grouped_mm`` and ``torch_a2a`` are
    left untouched, and so is ``None``, which dispatches to the eager ``F.linear``
    forward -- but transformers resolves an unrequested backend to ``grouped_mm``
    wherever that can dispatch, when it builds the model, so a model calibrates on
    the grouped path unless ``eager`` was asked for. Applied to the model config
    and recursively to ``text_config`` / ``vision_config`` / ``audio_config`` /
    ``speech_config`` whenever a fused-experts module is present.

    Experts that quantize only on the eager forward are refused on ``torch_a2a``, the
    expert-parallel forward, which cannot fall back to eager; no config is changed then.
    """
    fused_experts = [module for module in model.modules() if _is_fused_experts_module(module)]
    if not fused_experts:
        return

    nested_cfg_attrs = ("text_config", "vision_config", "audio_config", "speech_config")
    # Backends already intercepted by _QuantFusedExperts (None => eager F.linear forward). `torch_a2a`
    # (all-to-all DP=EP) runs its local experts through the same `_grouped_linear`, so _QuantFusedExperts
    # intercepts it too -- don't clobber it back to grouped_mm. Without `_grouped_linear` grouped_mm is
    # intercepted nowhere, so it is neither kept nor redirected to: main's eager is.
    supported_impls = (None, "eager", "torch_a2a") + (
        ("grouped_mm",) if _HAS_GROUPED_LINEAR else ()
    )
    fallback_impl = "grouped_mm" if _HAS_GROUPED_LINEAR else "eager"
    # Experts that quantize on the eager forward only keep main's forced eager.
    eager_only = _eager_only_experts(fused_experts)

    def _configs(cfg):
        if cfg is not None:
            yield cfg
            for sub in nested_cfg_attrs:
                yield from _configs(getattr(cfg, sub, None))

    # Some models construct auxiliary blocks from a deep copy of the root config.
    # Nemotron-H MTP does this, so updating only ``model.config`` leaves its fused
    # experts on an unsupported backend that bypasses the quantizer hooks.
    roots = [getattr(model, "config", None)] + [getattr(m, "config", None) for m in fused_experts]
    configs = [c for root in roots for c in _configs(root) if hasattr(c, "_experts_implementation")]
    if eager_only and any(c._experts_implementation == "torch_a2a" for c in configs):
        raise NotImplementedError(
            f"{', '.join(eager_only)} quantize only on the eager experts forward, but this model runs "
            "its experts on `torch_a2a`, the expert-parallel forward, which cannot fall back to "
            "eager. Quantize the model without expert parallelism."
        )
    moved = {}  # (was, now), once each: the root config and its sub-configs often agree
    for cfg in configs:
        was = cfg._experts_implementation
        if eager_only:
            cfg._experts_implementation = "eager"
        elif was not in supported_impls:
            cfg._experts_implementation = fallback_impl
        # None already dispatches to the eager F.linear forward: pinning it to eager moves nothing.
        if cfg._experts_implementation != was and was is not None:
            moved[(was, cfg._experts_implementation)] = None
    reason = (
        f"{', '.join(eager_only)} quantize on the eager forward only"
        if eager_only
        else "nothing intercepts that backend"
    )
    for was, now in moved:
        # The caller may have asked for it (a load's experts_implementation, hf_ptq's flag). Every
        # rank redirects alike, so one says so.
        warn_rank_0(
            f"Quantizing the fused experts on {now!r} instead of experts_implementation={was!r}: "
            f"{reason}."
        )


def _bypassed_both_hooks(module) -> bool:
    """Whether ``module``'s forward ran past both hooks, with a quantizer that needed them.

    See :func:`assert_fused_experts_intercepted`.
    """
    # A missing attribute (rather than 0) means it was already cleaned up -- say nothing.
    if getattr(module, "_intercept_count", None) != 0 or not getattr(module, "_forward_count", 0):
        return False
    first = module._first_proj_attr
    quantizers = [getattr(module, f"{first}_input_quantizer", None)]
    quantizers.append(getattr(module, "down_proj_input_quantizer", None))
    if getattr(module, "_weight_input_capture", False):
        for attr in (f"{first}_weight_quantizers", "down_proj_weight_quantizers"):
            quantizers.extend(getattr(module, attr, None) or [])
    return any(getattr(q, "is_enabled", False) for q in quantizers)


def _expert_parallel_group(module):
    """The process group ``module``'s experts are sharded over (expert parallelism), else ``None``.

    Read from the experts' weight, a DTensor sharded over the experts under the expert parallelism of
    transformers 5.16 and later. Earlier versions keep each rank's experts as a plain tensor, which
    names no group.
    """
    weight = getattr(module, module._first_proj_attr, None)
    if not isinstance(weight, DTensor) or weight.shape[0] == module.num_experts:
        return None
    dims = [dim for dim, placement in enumerate(weight.placements) if placement == Shard(0)]
    return weight.device_mesh.get_group(dims[0]) if dims else None


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

    Only modules whose forward ran and that have a quantizer needing it are reported: an enabled
    input quantizer, or an enabled weight quantizer when the calibration captured the experts'
    inputs to calibrate their weights (GPTQ, local Hessian, imatrix). Otherwise a weight quantizer
    calibrates on its weight alone (:func:`weight_only_quantize`), so weight-only experts lose
    nothing by not being intercepted, and neither do experts the recipe leaves unquantized. An
    experts module no calibration token reached (vision experts under text-only calibration) falls
    back to weight-derived scales as before.

    Under the expert parallelism of transformers 5.16 and later, a module's verdict is reduced over
    the group its experts are sharded over, which every member already joins in the experts'
    forward, so the whole group raises together. Otherwise the check is rank-local, also under
    earlier versions' expert parallelism, whose plain-tensor shards name no group: it raises on the
    ranks whose experts saw rows, and a rank whose experts saw none goes on to its next collective
    and waits there until the launcher stops it or the collective times out, so the error is in
    another rank's log.
    """
    experts = [
        (name, module, _bypassed_both_hooks(module), _expert_parallel_group(module))
        for name, module in model.named_modules()
        if isinstance(module, _QuantFusedExperts)
    ]
    bypassed = [here for _, _, here, _ in experts]
    # Each member of an expert-parallel group holds the same experts modules in the same order, so
    # one reduction per group, whatever this rank found, tells it what any member found.
    for group in dict.fromkeys(group for *_, group in experts if group is not None):
        members = [i for i, (*_, g) in enumerate(experts) if g is group]
        device = experts[members[0]][1].down_proj.device  # where the group's backend reads it
        flags = torch.tensor([bypassed[i] for i in members], dtype=torch.int32, device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
        for i, flag in zip(members, flags.tolist()):
            bypassed[i] = bool(flag)

    missed = []
    for (name, module, here, _), anywhere in zip(experts, bypassed):
        if anywhere:
            impl = getattr(getattr(module, "config", None), "_experts_implementation", None)
            where = "" if here else ", on another rank of its expert-parallel group"
            missed.append(
                f"  {name} ({type(module).__name__}, experts_implementation={impl!r}{where})"
            )

    if not missed:
        return

    raise RuntimeError(
        "Fused-experts calibration collected nothing: the forward of the module(s) below reached "
        "neither `F.linear` nor `_grouped_linear`, so no calibration batch reached their quantizers: "
        "their input scales would fall back to weight-derived values, and a weight calibration that "
        "reads the experts' inputs got none.\n"
        + "\n".join(missed)
        + "\n\nTheir forward ran, but the active experts backend bypasses both hooks -- "
        "`batched_mm`, `sonicmoe`, or a hand-written `bmm`/`matmul` forward on a class without "
        "`@use_experts_implementation`, which `normalize_experts_impl_on_the_fly` cannot "
        "redirect. Check the module's `forward` and `config._experts_implementation`. A class no "
        "backend change redirects needs its own quantized wrapper, as GptOssExperts has; to "
        "quantize the rest of the model meanwhile, disable these modules' quantizers in the quant "
        "config."
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
