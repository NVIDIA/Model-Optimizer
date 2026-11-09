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

# Architecture reference: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/kernel.py
# Copyright (c) 2023 DeepSeek. MIT License.
# SPDX-License-Identifier: Apache-2.0 AND MIT

"""Differentiable tensor expressions for published V4.1 kernel mathematics."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .moe import _fp8_inputs

__all__ = []


def fp4_inputs(value, group_size=32, e4m3_scale=False):
    shape = value.shape
    grouped = value.float().unflatten(-1, (-1, group_size))
    maximum = grouped.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        scale = torch.exp2(torch.ceil(torch.log2(maximum.clamp_min(6 * 2**-126) / 6)))
    scaled = (grouped / scale).clamp(-6, 6)
    magnitude = scaled.abs().contiguous()
    levels = value.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    boundaries = (levels[1:] + levels[:-1]) / 2
    lower = torch.bucketize(magnitude, boundaries)
    at_tie = magnitude == boundaries[lower.clamp_max(6)]
    index = lower + (at_tie & (lower % 2 == 1)).long()
    rounded = torch.copysign(levels[index], scaled) * scale
    return (grouped + (rounded - grouped).detach()).reshape(shape).to(value.dtype)


class V41Linear(nn.Linear):
    def __init__(self, in_features, out_features, *, fp32=False):
        super().__init__(
            in_features, out_features, bias=False, dtype=torch.float32 if fp32 else None
        )
        self.checkpoint_quantized = False
        self.compute_in_fp32 = fp32

    def forward(self, value):
        if self.checkpoint_quantized:
            value = _fp8_inputs(value)
        if self.compute_in_fp32:
            with torch.autocast(value.device.type, enabled=False):
                return F.linear(value.float(), self.weight.float())
        return F.linear(value.to(self.weight.dtype), self.weight)


class V41LMHead(nn.Linear):
    def forward(self, hidden_states):
        with torch.autocast(hidden_states.device.type, enabled=False):
            return F.linear(hidden_states.float(), self.weight.float())


class V41GroupedProjection(nn.Module):
    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, hidden_states):
        weight = self.weight.reshape(self.groups, -1, self.weight.shape[-1])
        return torch.einsum("bsgd,grd->bsgr", hidden_states, weight.to(hidden_states.dtype))


class V41RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight).to(value.dtype)


def rotary(value, positions, config, compressed, inverse=False):
    width = config.qk_rope_head_dim
    base = config.compress_rope_theta if compressed else config.rope_theta
    frequency = base ** (
        -torch.arange(0, width, 2, device=value.device, dtype=torch.float32) / width
    )
    scaling = config.rope_scaling if compressed else None
    original = scaling.get("original_max_position_embeddings", 0) if scaling else 0
    if original and scaling is not None:

        def correction(rotations):
            return width * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(scaling.get("beta_fast", 32))), 0)
        high = min(math.ceil(correction(scaling.get("beta_slow", 1))), width - 1)
        ramp = (
            (torch.arange(width // 2, device=value.device) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        frequency = frequency * (1 - ramp) + frequency / scaling["factor"] * ramp
    angles = positions.float().unsqueeze(-1) * frequency
    if value.ndim == 4:
        angles = angles.unsqueeze(-2)
    if inverse:
        angles = -angles
    pairs = value[..., -width:].float().unflatten(-1, (-1, 2))
    real, imaginary = pairs.unbind(-1)
    rotated = torch.stack(
        (
            real * angles.cos() - imaginary * angles.sin(),
            imaginary * angles.cos() + real * angles.sin(),
        ),
        -1,
    ).flatten(-2)
    return torch.cat((value[..., :-width], rotated.to(value.dtype)), -1)


def hyper_mixes(value, projection, scale, bias, config):
    flat = value.flatten(-2).float()
    with torch.autocast(value.device.type, enabled=False):
        mixes = F.linear(flat, projection.float()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + config.rms_norm_eps
        )
    count = config.hc_mult
    pre = torch.sigmoid(mixes[..., :count] * scale[0] + bias[:count]) + config.hc_eps
    post = 2 * torch.sigmoid(mixes[..., count : 2 * count] * scale[1] + bias[count : 2 * count])
    comb = (mixes[..., 2 * count :] * scale[2] + bias[2 * count :]).unflatten(-1, (count, count))
    comb = comb.softmax(-1) + config.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    for _ in range(config.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + config.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + config.hc_eps)
    return pre, post, comb


def hyper_pre(value, pre):
    return (value.float() * pre.unsqueeze(-1)).sum(-2).to(value.dtype)


def hyper_post(value, residual, post, comb):
    with torch.autocast(value.device.type, enabled=False):
        mixed = torch.einsum("bsij,bsid->bsjd", comb, residual.float())
    return (post.unsqueeze(-1) * value.float().unsqueeze(-2) + mixed).to(value.dtype)


def select_candidates(scores, lengths, topk, block_size):
    width = scores.shape[-1]
    blocked = (
        F.pad(scores, (0, -width % block_size), value=-torch.inf)
        .unflatten(-1, (-1, block_size))
        .amax(-1)
    )
    newest = (lengths - 1) // block_size
    blocked = blocked.masked_fill(
        torch.arange(blocked.shape[-1], device=scores.device) == newest, torch.inf
    )
    values, indices = blocked.topk(min(topk, blocked.shape[-1]), -1)
    keep = torch.zeros_like(blocked, dtype=torch.bool).scatter(-1, indices, values > -torch.inf)
    return keep.repeat_interleave(block_size, -1)[..., :width]


def sparse_attention(query, kv, sink, indices):
    selected = kv[torch.arange(kv.shape[0], device=kv.device)[:, None, None], indices.clamp_min(0)]
    with torch.autocast(query.device.type, enabled=False):
        scores = torch.einsum("bshd,bskd->bshk", query.float(), selected.float()) / math.sqrt(
            query.shape[-1]
        )
        scores = scores.masked_fill(indices[:, :, None, :] < 0, -torch.inf)
        maximum = torch.maximum(scores.amax(-1, keepdim=True), sink[None, None, :, None].float())
        numerator = (scores - maximum).exp()
        denominator = numerator.sum(-1, keepdim=True) + (sink[None, None, :, None] - maximum).exp()
        probabilities = numerator / denominator
        return torch.einsum("bshk,bskd->bshd", probabilities, selected.float()).to(query.dtype)
