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

"""Experimental AutoModel expert QAD using fixed, published NVFP4 scales."""

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from nemo_automodel.components.moe.experts import GroupedExperts
from torch import nn
from torch.distributed.tensor import DTensor

from modelopt.torch.quantization.qtensor import NVFP4QTensor

__all__ = ["NVFP4GroupedExperts", "export_hf_experts"]

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SCALE_NAMES = ("weight_scale", "weight_scale_2", "input_scale")
_COMPUTE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _qdq(inputs, block_scale, scale2):
    packed, scale, double_scale = NVFP4QTensor.quantize(
        inputs.contiguous(), 16, block_scale, scale2
    )
    return packed.dequantize(
        scale=scale, double_scale=double_scale, block_sizes={-1: 16}, dtype=inputs.dtype
    )


class _PublishedQDQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, block_scale, scale2):
        # Chunk large tensors to bound FP32/int64 temporaries without taxing small experts.
        if inputs.numel() <= 4194304:
            return _qdq(inputs, block_scale, scale2)
        blocks = inputs.reshape(-1, 16)
        output = torch.empty_like(blocks)
        for start in range(0, blocks.shape[0], 262144):
            end = start + 262144
            scale = None if block_scale is None else block_scale.reshape(-1, 1)[start:end]
            output[start:end].copy_(_qdq(blocks[start:end], scale, scale2))
        return output.reshape(inputs.shape)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def _check_tensor(tensor: torch.Tensor, shape: tuple, dtype: torch.dtype, name: str):
    if isinstance(tensor, DTensor) or tensor.is_meta:
        raise ValueError(f"{name} must be a materialized, unsharded tensor")
    if tensor.shape != shape or tensor.dtype != dtype:
        raise ValueError(f"{name}: expected {shape} / {dtype}, got {tensor.shape} / {tensor.dtype}")
    if not torch.isfinite(tensor.float()).all() or not (tensor.float() > 0).all():
        raise ValueError(f"{name} must contain finite positive scales")


class _Projection(nn.Module):
    def __init__(self, scales: Mapping[str, torch.Tensor], shape: tuple[int, int]):
        super().__init__()
        for name in _SCALE_NAMES:
            value = scales[name]
            expected = (shape[0], shape[1] // 16) if name == "weight_scale" else ()
            dtype = torch.float8_e4m3fn if name == "weight_scale" else torch.float32
            _check_tensor(value, expected, dtype, name)
            self.register_buffer(name, value.detach().clone())

    def _apply(self, fn, recurse=True):
        scales = {name: self._buffers[name] for name in _SCALE_NAMES}
        super()._apply(fn, recurse)
        # Module dtype casts must not round the checkpoint's FP32 multipliers.
        for name, value in scales.items():
            self._buffers[name] = value.to(device=self._buffers[name].device)
        return self

    def forward(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply W4A4 to inputs [tokens, in] and weight [out, in]."""
        inputs = _PublishedQDQ.apply(inputs, None, self.input_scale)
        weight = _PublishedQDQ.apply(weight, self.weight_scale, self.weight_scale_2)
        return F.linear(inputs, weight)


class NVFP4GroupedExperts(GroupedExperts):
    """Reference W4A4 adapter for AutoModel's bias-free SwiGLU expert loop.

    Construct from an unsharded ``GroupedExperts`` before FSDP2 wrapping. Parameters
    retain their identity; published scales are cloned and stay frozen. This is an
    explicit example adapter, not an ``mtq.quantize``/``mto.restore`` plugin.
    """

    def __init__(
        self, module: GroupedExperts, hf_state: Mapping[str, torch.Tensor], prefix: str = ""
    ):
        """Import HF scales ``<prefix>.<expert>.<projection>.<scale_name>``.

        ``hf_state`` holds FP8 [out, in/16] block scales and FP32 scalar weight/input
        multipliers. The BF16/FP32 master weights remain those of ``module``.
        """
        if type(module) is not GroupedExperts or module.use_torch_mm:
            raise ValueError("Requires an unwrapped GroupedExperts with experts='torch'")
        cfg = module.config
        if cfg.expert_bias or cfg.expert_activation != "swiglu" or cfg.swiglu_limit:
            raise ValueError("Only bias-free, unclamped SwiGLU experts are supported")
        if not cfg.apply_router_weight_after_down:
            raise ValueError(
                "Set apply_router_weight_after_down=True before importing NVFP4 scales"
            )
        if (
            min(cfg.n_routed_experts, cfg.expert_dim, cfg.moe_inter_dim) <= 0
            or cfg.expert_dim % 16
            or cfg.moe_inter_dim % 16
        ):
            raise ValueError("Expert dimensions must be divisible by the NVFP4 block size (16)")
        if (
            module._forward_hooks
            or module._forward_pre_hooks
            or module._backward_hooks
            or module._backward_pre_hooks
        ):
            raise ValueError("Attach the adapter before installing module hooks or sharding")
        for weight in (module.gate_and_up_projs, module.down_projs):
            if isinstance(weight, DTensor) or weight.is_meta or weight.dtype not in _COMPUTE_DTYPES:
                raise ValueError("Requires materialized, unsharded floating-point master weights")
        if (
            module.gate_and_up_projs.shape
            != (cfg.n_routed_experts, cfg.expert_dim, 2 * cfg.moe_inter_dim)
            or module.down_projs.shape != (cfg.n_routed_experts, cfg.moe_inter_dim, cfg.expert_dim)
            or module.gate_and_up_projs.device != module.down_projs.device
        ):
            raise ValueError("Master weights must match the native expert layout on one device")
        shapes = {
            "gate_proj": (cfg.moe_inter_dim, cfg.expert_dim),
            "up_proj": (cfg.moe_inter_dim, cfg.expert_dim),
            "down_proj": (cfg.expert_dim, cfg.moe_inter_dim),
        }
        stem = f"{prefix}." if prefix else ""
        quantizers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        proj: _Projection(
                            {
                                name: hf_state[f"{stem}{expert}.{proj}.{name}"]
                                for name in _SCALE_NAMES
                            },
                            shapes[proj],
                        )
                        for proj in _PROJECTIONS
                    }
                )
                for expert in range(cfg.n_routed_experts)
            ]
        )
        # Avoid allocating a second copy of the fused expert weights.
        with torch.device("meta"):
            super().__init__(cfg)
        self.gate_and_up_projs = module.gate_and_up_projs
        self.down_projs = module.down_projs
        self.quantizers = quantizers.to(device=module.gate_and_up_projs.device)
        self.train(module.training)

    def forward(self, x, token_mask, weights, indices, post_expert_output=None):
        """Run after FSDP2 unsharding; reject EP before AutoModel issues collectives."""
        if any(isinstance(p, DTensor) for p in (self.gate_and_up_projs, self.down_projs)):
            raise ValueError("Expert parallelism is not supported by the reference NVFP4 adapter")
        if x.dtype not in _COMPUTE_DTYPES:
            raise ValueError("Inputs must use float32, float16, or bfloat16")
        return super().forward(x, token_mask, weights, indices, post_expert_output)

    def _forward_loop(
        self,
        x,
        weights,
        indices,
        token_mask,
        gate_and_up_projs,
        down_projs,
        gate_up_proj_bias,
        down_proj_bias,
        n_local_experts,
        experts_start_idx,
        experts_end_idx,
        post_expert_output,
    ):
        """Compute [tokens, top_k, hidden] outputs with routing after the down projection.

        x: [tokens, hidden]; weights/indices: [tokens, top_k]; token_mask: [tokens].
        FSDP2 has gathered the weights as regular tensors [experts, hidden, 2*inter]
        and [experts, inter, hidden]. Expert-parallel DTensor layouts are unsupported.
        """
        if n_local_experts != self.n_routed_experts or experts_start_idx != 0:
            raise ValueError("Expert parallelism is not supported by the reference NVFP4 adapter")
        y = torch.zeros(
            (x.shape[0], weights.shape[1], x.shape[1]), device=x.device, dtype=torch.float32
        )
        # Unbind accumulates the fused gradient once, instead of once per expert slice.
        up_weights, down_weights = gate_and_up_projs.unbind(), down_projs.unbind()
        active = False
        for expert, (up, down, q) in enumerate(zip(up_weights, down_weights, self.quantizers)):
            rows, slots = torch.where((indices == expert) & token_mask[:, None])
            if rows.numel() == 0:
                continue
            active = True
            gate, up = up.T.chunk(2, dim=0)
            inputs = x[rows]
            activated = F.silu(q["gate_proj"](inputs, gate)) * q["up_proj"](inputs, up)
            out = q["down_proj"](activated, down.T)
            if post_expert_output is not None:
                out = post_expert_output(out)
            y[rows, slots] = out.float() * weights[rows, slots, None]
        if not active:
            # One element per input keeps zero gradients connected without overflowing FP16 sums.
            y = y + x[:1, :1].sum() * 0 + weights[:1, :1].sum() * 0
            y = y + gate_and_up_projs[0, 0, 0] * 0 + down_projs[0, 0, 0] * 0
        return y


def export_hf_experts(
    state: Mapping[str, torch.Tensor], prefix: str = "", dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Export one adapter's full state to per-expert HF NVFP4 tensors.

    ``state`` must contain regular full tensors, e.g. from FSDP2
    ``get_model_state_dict(full_state_dict=True, cpu_offload=True)`` on rank 0.
    ``prefix`` is the module path in both the native state and exported HF keys.
    ``dtype`` must match the training compute dtype, before quantizing master weights.
    No collectives or device transfers are performed; this exports experts only.
    """
    stem = f"{prefix}." if prefix else ""
    up, down = state[stem + "gate_and_up_projs"], state[stem + "down_projs"]
    if dtype not in _COMPUTE_DTYPES:
        raise ValueError("Export dtype must be float32, float16, or bfloat16")
    for tensor in state.values():
        if isinstance(tensor, DTensor) or tensor.is_meta:
            raise ValueError(
                "Export requires a materialized full state; gather outside the exporter"
            )
    if (
        up.ndim != 3
        or down.ndim != 3
        or up.shape != (down.shape[0], down.shape[2], down.shape[1] * 2)
    ):
        raise ValueError("Invalid fused expert projection shapes")
    if up.dtype not in _COMPUTE_DTYPES or down.dtype not in _COMPUTE_DTYPES:
        raise ValueError("Export requires floating-point master weights")
    if min(down.shape) <= 0 or down.shape[1] % 16 or down.shape[2] % 16:
        raise ValueError("Invalid expert dimensions for NVFP4 export")
    output = {}
    for expert, (up_weight, down_weight) in enumerate(zip(up.unbind(), down.unbind())):
        matrices = (*up_weight.T.chunk(2, dim=0), down_weight.T)
        for proj, matrix in zip(_PROJECTIONS, matrices):
            key = f"{stem}quantizers.{expert}.{proj}."
            scales = {name: state[key + name] for name in _SCALE_NAMES}
            # Validate serialized state, including dtype casts performed outside this adapter.
            for name, value in scales.items():
                shape = (matrix.shape[0], matrix.shape[1] // 16) if name == "weight_scale" else ()
                scale_dtype = torch.float8_e4m3fn if name == "weight_scale" else torch.float32
                _check_tensor(value, shape, scale_dtype, name)
            packed, _, _ = NVFP4QTensor.quantize(
                matrix.to(dtype).contiguous(), 16, scales["weight_scale"], scales["weight_scale_2"]
            )
            hf_key = f"{stem}{expert}.{proj}."
            output[hf_key + "weight"] = packed._quantized_data
            output.update({hf_key + name: value.clone() for name, value in scales.items()})
    return output
