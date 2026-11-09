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


"""Differentiable V4.1 MoE components with published checkpoint parameter names.

Architecture reference: deepseek-ai/DeepSeek-V4.1-Flash, revision
314113f8a7dadb388215396ba5fbee2c610a04d6, inference/model.py (Gate, Expert, MoE)
and inference/kernel.py (act_quant). No reference implementation is imported.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["DeepseekV41MoE", "DeepseekV41MoEConfig"]


@dataclass(frozen=True)
class DeepseekV41MoEConfig:
    """V4.1 component facts, independent of optional Transformers/NeMo packages."""

    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    vision_routing_bias: bool = False
    checkpoint_quantized_projections: tuple[str, ...] = ()
    model_type: ClassVar[str] = "deepseek_v41"

    def __post_init__(self):
        if min(self.hidden_size, self.moe_intermediate_size, self.n_routed_experts) <= 0:
            raise ValueError("V4.1 MoE dimensions must be positive.")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("V4.1 top-k must lie within the routed expert count.")
        if self.n_shared_experts != 1:
            raise ValueError("The published V4.1 MoE requires exactly one shared expert.")
        if self.scoring_func not in ("sqrtsoftplus", "softmax", "sigmoid") or self.gate_temp <= 0:
            raise ValueError("Invalid V4.1 routing activation or gate temperature.")
        object.__setattr__(
            self, "checkpoint_quantized_projections", tuple(self.checkpoint_quantized_projections)
        )

    def to_dict(self):
        """Return serializable component metadata, including checkpoint activation precision."""
        return {"model_type": self.model_type, **asdict(self)}

    @classmethod
    def from_hf_config(cls, config):
        """Extract component facts from a published HF root/text config or dictionary."""
        if isinstance(config, cls):
            return config
        raw = config if isinstance(config, Mapping) else vars(config)
        text = raw.get("text_config", raw)
        text = text if isinstance(text, Mapping) else vars(text)
        values = {item.name: text[item.name] for item in fields(cls) if item.name in text}
        values.setdefault("vision_routing_bias", bool(raw.get("vision_config")))
        return cls(**values)


def _fp8_inputs(value):
    """Round checkpoint-quantized GEMM inputs with the published MXFP8 scale rule."""
    if value.shape[-1] % 32:
        raise ValueError("Checkpoint FP8 activation widths must divide into groups of 32.")
    grouped = value.float().unflatten(-1, (-1, 32))
    amax = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    rounded = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    # Quantized values match inference; STE leaves repair gradients through their inputs.
    return (grouped + (rounded - grouped).detach()).flatten(-2).to(value.dtype)


class _V41Router(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, device=device, dtype=torch.float32
            )
        )
        self.bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, device=device, dtype=torch.float32)
        )
        self.bias_vl = (
            nn.Parameter(torch.zeros(config.n_routed_experts, device=device, dtype=torch.float32))
            if config.vision_routing_bias
            else None
        )
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states, image_mask=None):
        with torch.autocast(hidden_states.device.type, enabled=False):
            logits = F.linear(hidden_states.float(), self.weight.float()) / self.config.gate_temp
        if self.config.scoring_func == "softmax":
            scores = logits.softmax(-1)
        elif self.config.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = F.softplus(logits).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask[..., None], self.bias_vl, bias)
        indices = (scores + bias).topk(self.config.num_experts_per_tok, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.config.norm_topk_prob and self.config.num_experts_per_tok > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * self.config.routed_scaling_factor, indices


class _V41Expert(nn.Module):
    def __init__(self, config, device, dtype):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.w2 = nn.Linear(
            config.moe_intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype
        )
        self.w3 = nn.Linear(
            config.hidden_size, config.moe_intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.swiglu_limit = config.swiglu_limit
        self.checkpoint_quantized_projections = frozenset()

    def _project(self, name, value):
        if name in self.checkpoint_quantized_projections:
            value = _fp8_inputs(value)
        return getattr(self, name)(value)

    def forward(self, hidden_states, weights=None):
        gate = self._project("w1", hidden_states).float()
        up = self._project("w3", hidden_states).float()
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        value = F.silu(gate) * up
        if weights is not None:
            value = value * weights
        return self._project("w2", value.to(hidden_states.dtype))


def _decode_weight(weight, scale, shape):
    if weight.dtype in (torch.uint8, torch.int8, getattr(torch, "float4_e2m1fn_x2", None)):
        if weight.shape != (shape[0], shape[1] // 2) or shape[1] % 32:
            raise ValueError("Invalid packed V4.1 FP4 weight shape.")
        if scale is None or scale.shape != (shape[0], shape[1] // 32):
            raise ValueError("V4.1 FP4 weight scale must be [out_features, in_features / 32].")
        table = weight.new_tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32
        )
        packed = weight.view(torch.uint8).long()
        values = torch.stack((table[packed & 15], table[packed >> 4]), -1).flatten(-2)
        return values * scale.float().repeat_interleave(32, -1), True
    if weight.shape != shape:
        raise ValueError(
            f"Invalid V4.1 weight shape {tuple(weight.shape)}; expected {tuple(shape)}."
        )
    if weight.dtype == torch.float8_e4m3fn:
        expected = ((shape[0] + 31) // 32, (shape[1] + 31) // 32)
        if scale is None or scale.shape != expected:
            raise ValueError("V4.1 FP8 weight scale must use 32x32 blocks.")
        expanded = scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        return weight.float() * expanded[: shape[0], : shape[1]], True
    if scale is not None or weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("Unexpected V4.1 weight dtype or scale on an unquantized weight.")
    return weight, False


class DeepseekV41MoE(nn.Module):
    """Run one V4.1 MoE component on hidden states, without a full HF model.

    ``image_mask`` selects the published VL routing correction; it does not run a
    vision encoder. Ordinary ``nn.Linear`` experts support ModelOpt PTQ and explicit
    repair units, e.g. ``units=["moe"]`` on a caller-owned container.
    """

    def __init__(self, config, *, device=None, dtype=None):
        """Build one component with ordinary floating linear weights, optionally on meta."""
        super().__init__()
        self.config = DeepseekV41MoEConfig.from_hf_config(config)
        self.num_experts = self.config.n_routed_experts
        self.gate = _V41Router(self.config, device)
        self.experts = nn.ModuleList(
            _V41Expert(self.config, device, dtype) for _ in range(self.num_experts)
        )
        self.shared_experts = _V41Expert(self.config, device, dtype)
        for path in self.config.checkpoint_quantized_projections:
            owner, _, projection = path.rpartition(".")
            if projection not in ("w1", "w2", "w3"):
                raise ValueError(f"Invalid V4.1 checkpoint projection {path!r}.")
            expert = self.get_submodule(owner)
            expert.checkpoint_quantized_projections |= {projection}

    def forward(self, hidden_states, image_mask=None):
        """Return routed plus shared outputs with the original input shape and dtype."""
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("V4.1 MoE hidden width does not match its configuration.")
        if image_mask is not None and (
            image_mask.shape != hidden_states.shape[:-1] or image_mask.dtype != torch.bool
        ):
            raise ValueError("V4.1 image_mask must be boolean and match token dimensions.")
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        weights, indices = self.gate(flat, None if image_mask is None else image_mask.reshape(-1))
        result = torch.zeros_like(flat, dtype=torch.float32)
        for index, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == index)
            if not tokens.numel():
                continue
            values = expert(flat[tokens], weights[tokens, slots, None])
            result = result.index_add(0, tokens, values.float())
        result = result + self.shared_experts(flat).float()
        return result.to(hidden_states.dtype).reshape_as(hidden_states)

    @classmethod
    def from_checkpoint(cls, config, state_dict, *, prefix="", device=None, dtype=torch.bfloat16):
        """Extract one supplied MoE mapping, reconstructing raw BF16/FP8/FP4 weights.

        Pass the published prefix, e.g. ``layers.7.ffn.``. FP8/FP4 projections
        retain dynamic MXFP8 input rounding; reconstructed parameters remain normal
        floating tensors so subsequent ModelOpt PTQ owns its own packed formats.
        No files, weights, process groups, or full model are loaded implicitly.
        """
        model = cls(config, device="meta", dtype=dtype)
        parameters = dict(model.named_parameters())
        expected = set()
        decoded = {}
        quantized = {}
        for name, parameter in parameters.items():
            key = prefix + name
            if key not in state_dict:
                raise ValueError(f"Missing V4.1 MoE checkpoint key {key!r}.")
            expected.add(key)
            raw = state_dict[key]
            target_dtype = torch.float32 if name.startswith("gate.") else dtype
            if name.endswith(".weight") and not name.startswith("gate."):
                scale_keys = [
                    key.removesuffix("weight") + suffix for suffix in ("weight_scale_inv", "scale")
                ]
                scales = [k for k in scale_keys if k in state_dict]
                if len(scales) > 1:
                    raise ValueError(f"Ambiguous V4.1 weight scale for {key!r}.")
                expected.update(scales)
                value, is_quantized = _decode_weight(
                    raw, state_dict[scales[0]] if scales else None, parameter.shape
                )
                if is_quantized:
                    owner, projection, _ = name.rsplit(".", 2)
                    quantized.setdefault(owner, set()).add(projection)
            else:
                value, _ = _decode_weight(raw, None, parameter.shape)
            decoded[name] = value.to(device=device, dtype=target_dtype).detach().clone()
        extra = {key for key in state_dict if key.startswith(prefix)} - expected
        if extra:
            raise ValueError(f"Unexpected V4.1 MoE checkpoint keys: {sorted(extra)}")
        model.load_state_dict(decoded, assign=True)
        for owner, projections in quantized.items():
            model.get_submodule(owner).checkpoint_quantized_projections = frozenset(projections)
        model.config = replace(
            model.config,
            checkpoint_quantized_projections=tuple(
                path + "." + projection
                for path, expert in model.named_modules()
                if isinstance(expert, _V41Expert)
                for projection in sorted(expert.checkpoint_quantized_projections)
            ),
        )
        return model
