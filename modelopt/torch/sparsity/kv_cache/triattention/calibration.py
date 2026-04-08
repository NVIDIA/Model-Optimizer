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

"""Calibration for TriAttention: compute per-head Q/K frequency statistics.

Hooks into attention layers during a forward pass, captures pre-RoPE Q vectors,
inverts RoPE, converts to frequency domain, and computes per-head mean statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .rope_utils import invert_rope, rotate_half, to_complex_pairs
from .scoring import HeadFrequencyStats

__all__ = [
    "CalibrationData",
    "compute_head_stats_from_q",
    "run_calibration",
]


@dataclass
class CalibrationData:
    """Container for TriAttention calibration output.

    Stores per-head frequency statistics computed during calibration, along with
    model metadata needed for scoring at inference time.
    """

    head_stats: dict[tuple[int, int], HeadFrequencyStats]  # (layer, head) -> stats
    head_dim: int
    rope_style: str
    num_layers: int
    num_kv_heads: int

    def state_dict(self) -> dict[str, Any]:
        """Serialize to state dict for checkpoint embedding."""
        stats_serialized = {}
        for (layer, head), hs in self.head_stats.items():
            key = f"layer{layer:02d}_head{head:02d}"
            stats_serialized[key] = {
                "q_mean_real": hs.q_mean_complex.real.cpu(),
                "q_mean_imag": hs.q_mean_complex.imag.cpu(),
                "q_abs_mean": hs.q_abs_mean.cpu(),
            }
        return {
            "metadata": {
                "head_dim": self.head_dim,
                "rope_style": self.rope_style,
                "num_layers": self.num_layers,
                "num_kv_heads": self.num_kv_heads,
                "sampled_heads": [[layer, head] for layer, head in self.head_stats],
            },
            "stats": stats_serialized,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> CalibrationData:
        """Deserialize from state dict."""
        metadata = state["metadata"]
        stats_raw = state["stats"]
        sampled_heads = [tuple(pair) for pair in metadata["sampled_heads"]]
        head_stats: dict[tuple[int, int], HeadFrequencyStats] = {}
        for layer, head in sampled_heads:
            key = f"layer{layer:02d}_head{head:02d}"
            entry = stats_raw[key]
            q_mean_complex = torch.complex(
                entry["q_mean_real"].to(torch.float32),
                entry["q_mean_imag"].to(torch.float32),
            )
            q_abs_mean = entry["q_abs_mean"].to(torch.float32)
            head_stats[(int(layer), int(head))] = HeadFrequencyStats(
                q_mean_complex=q_mean_complex,
                q_abs_mean=q_abs_mean,
            )
        return cls(
            head_stats=head_stats,
            head_dim=metadata["head_dim"],
            rope_style=metadata["rope_style"],
            num_layers=metadata["num_layers"],
            num_kv_heads=metadata["num_kv_heads"],
        )


def compute_head_stats_from_q(
    q_pre_rope: torch.Tensor,
    style: str = "half",
) -> HeadFrequencyStats:
    """Compute frequency statistics for a single head from pre-RoPE Q vectors.

    Args:
        q_pre_rope: Pre-RoPE query vectors for one head, shape (seq_len, head_dim).
        style: RoPE pairing style.

    Returns:
        HeadFrequencyStats with q_mean_complex and q_abs_mean.
    """
    q_complex = to_complex_pairs(q_pre_rope, style=style)  # (seq_len, freq_count)
    q_mean_complex = q_complex.mean(dim=0)  # (freq_count,)
    q_abs_mean = q_complex.abs().mean(dim=0)  # (freq_count,)
    return HeadFrequencyStats(
        q_mean_complex=q_mean_complex,
        q_abs_mean=q_abs_mean,
    )


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------


def _find_attention_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Find attention sub-modules in HF model (model.model.layers[i].self_attn)."""
    backbone = getattr(model, "model", model)
    layer_list = getattr(backbone, "layers", None)
    if layer_list is None:
        raise RuntimeError(
            "Cannot locate transformer layers. Expected model.model.layers attribute."
        )
    layers = []
    for layer_module in layer_list:
        attn = getattr(layer_module, "self_attn", None)
        if attn is None:
            raise RuntimeError("Layer missing self_attn attribute.")
        layers.append(attn)
    return layers


def _get_rotary_embedding(model: torch.nn.Module) -> torch.nn.Module:
    """Find the rotary embedding module in the model."""
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "rotary_emb"):
        return backbone.rotary_emb
    # Some models put rotary_emb on individual attention layers
    attn_layers = _find_attention_layers(model)
    if attn_layers and hasattr(attn_layers[0], "rotary_emb"):
        return attn_layers[0].rotary_emb
    raise RuntimeError("Cannot locate rotary_emb on model.model or self_attn.")


def _get_model_config(model: torch.nn.Module) -> dict[str, Any]:
    """Extract model config parameters needed for calibration."""
    config = getattr(model, "config", None)
    if config is not None:
        num_layers = getattr(config, "num_hidden_layers", None)
        num_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        head_dim = getattr(config, "head_dim", None)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        if head_dim is None and hidden_size and num_heads:
            head_dim = hidden_size // num_heads
        if all(v is not None for v in [num_layers, num_heads, head_dim, num_kv_heads]):
            return {
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "num_kv_heads": num_kv_heads,
            }

    # Fallback: infer from model structure
    attn_layers = _find_attention_layers(model)
    num_layers = len(attn_layers)
    attn0 = attn_layers[0]
    num_heads = getattr(attn0, "num_heads", None)
    head_dim = getattr(attn0, "head_dim", None)
    num_kv_heads = getattr(attn0, "num_key_value_heads", num_heads)

    if num_heads is None or head_dim is None:
        # Infer from q_proj weight shape
        q_proj = getattr(attn0, "q_proj", None)
        if q_proj is not None:
            out_features = q_proj.out_features
            # Guess: if num_heads not available, try common head_dims
            for hd in [128, 96, 64, 32]:
                if out_features % hd == 0:
                    head_dim = head_dim or hd
                    num_heads = num_heads or (out_features // hd)
                    break

    if num_heads is None or head_dim is None:
        raise RuntimeError("Cannot determine num_heads and head_dim from model.")

    num_kv_heads = num_kv_heads or num_heads
    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
    }


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------


def run_calibration(
    model: torch.nn.Module,
    forward_loop: Any = None,
    rope_style: str = "half",
) -> CalibrationData:
    """Run TriAttention calibration on a model.

    Hooks into attention layers, captures post-RoPE Q states during a forward
    pass, inverts RoPE, and computes per-head frequency statistics.

    Args:
        model: The model to calibrate. Must follow HF structure
            (model.model.layers[i].self_attn with q_proj).
        forward_loop: Callable that takes the model and runs forward passes
            on calibration data. Signature: ``forward_loop(model) -> None``.
        rope_style: RoPE pairing style ('half' or 'interleaved').

    Returns:
        CalibrationData with per-head frequency statistics.
    """
    model_cfg = _get_model_config(model)
    num_layers = model_cfg["num_layers"]
    num_heads = model_cfg["num_heads"]
    head_dim = model_cfg["head_dim"]
    num_kv_heads = model_cfg["num_kv_heads"]

    attn_layers = _find_attention_layers(model)
    rotary = _get_rotary_embedding(model)
    attn_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # Storage for captured Q states
    captured_q: dict[int, torch.Tensor] = {}

    def _make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            bsz, q_len, _ = hidden_states.shape
            q = module.q_proj(hidden_states)
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

            # Apply RoPE — ensure cos/sin are on the same device as hidden_states
            device = hidden_states.device
            pos_ids = torch.arange(q_len, device=device).unsqueeze(0)
            probe = torch.zeros(1, q_len, head_dim, device=device, dtype=hidden_states.dtype)
            cos, sin = rotary(probe, pos_ids)
            cos = cos.to(device=device)
            sin = sin.to(device=device)
            q_rot = (q * cos.unsqueeze(1)) + (rotate_half(q, style=rope_style) * sin.unsqueeze(1))
            q_rot = q_rot * attn_scale
            captured_q[layer_idx] = q_rot.detach()

        return hook_fn

    # Register hooks
    handles = []
    for layer_idx, attn in enumerate(attn_layers):
        handle = attn.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True)
        handles.append(handle)

    # Run forward pass
    try:
        if forward_loop is not None:
            forward_loop(model)
    finally:
        # Always remove hooks
        for handle in handles:
            handle.remove()

    # Compute per-head frequency statistics
    head_stats: dict[tuple[int, int], HeadFrequencyStats] = {}

    for layer_idx in range(num_layers):
        q_rot = captured_q.get(layer_idx)
        if q_rot is None:
            continue

        # q_rot: (batch, num_heads, seq_len, head_dim)
        # Build cos/sin for RoPE inversion — ensure same device as q_rot
        device = q_rot.device
        seq_len = q_rot.shape[2]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        probe = torch.zeros(1, seq_len, head_dim, device=device, dtype=q_rot.dtype)
        cos, sin = rotary(probe, pos_ids)
        cos = cos.to(device=device).unsqueeze(1)  # (1, 1, seq_len, head_dim)
        sin = sin.to(device=device).unsqueeze(1)

        # Invert RoPE
        q_base = invert_rope(q_rot, cos, sin, attn_scale, style=rope_style)

        for head_idx in range(num_heads):
            q_head = q_base[0, head_idx]  # (seq_len, head_dim)
            head_stats[(layer_idx, head_idx)] = compute_head_stats_from_q(q_head, style=rope_style)

        # Free memory
        del captured_q[layer_idx]

    return CalibrationData(
        head_stats=head_stats,
        head_dim=head_dim,
        rope_style=rope_style,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
    )
