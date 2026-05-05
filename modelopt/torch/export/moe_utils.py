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

"""Utilities for Mixture-of-Experts (MoE) model export."""

import copy
import warnings
from pathlib import Path

import torch
import torch.nn as nn


def _slice_fused_expert_amax(
    amax: torch.Tensor,
    weight_slice: torch.Tensor,
    fused_start: int,
    fused_total: int,
    block_size: int | None,
) -> torch.Tensor:
    """Return ``amax`` reshaped/sliced to match one exported expert projection.

    MSE-calibrated NVFP4 fused experts may store per-block amax for the fused
    gate+up tensor in flattened form, for example ``[1024 * 128, 1]`` for a
    ``[1024, 2048]`` fused weight. Serving/export splits that into two
    ``[512, 2048]`` projections, so the amax must be reshaped to the fused
    block grid first and then row-sliced.
    """
    if amax.numel() <= 1:
        return amax

    if block_size is not None and weight_slice.shape[-1] % block_size == 0:
        num_blocks_per_row = weight_slice.shape[-1] // block_size
        expected_shape = (*weight_slice.shape[:-1], num_blocks_per_row)
        expected_numel = torch.tensor(expected_shape).prod().item()
        fused_numel = fused_total * num_blocks_per_row

        if amax.numel() == fused_numel:
            return (
                amax.reshape(fused_total, num_blocks_per_row)[
                    fused_start : fused_start + weight_slice.shape[0]
                ]
                .contiguous()
                .reshape(expected_shape)
            )
        if amax.numel() == expected_numel:
            return amax.reshape(expected_shape).contiguous()

    amax_dim0 = amax.shape[0]
    if amax_dim0 == fused_total:
        return amax[fused_start : fused_start + weight_slice.shape[0]].contiguous()
    if amax_dim0 > fused_total and amax_dim0 % fused_total == 0:
        row_factor = amax_dim0 // fused_total
        slice_start = fused_start * row_factor
        slice_end = (fused_start + weight_slice.shape[0]) * row_factor
        return amax[slice_start:slice_end].contiguous()
    if fused_total % amax_dim0 == 0:
        slice_start = fused_start * amax_dim0 // fused_total
        slice_end = (fused_start + weight_slice.shape[0]) * amax_dim0 // fused_total
        return amax[slice_start:slice_end].contiguous()
    return amax


def _export_fused_experts(module: nn.Module, dtype: torch.dtype) -> None:
    """Split fused MoE expert weights and export per-expert quantization scales.

    Works with any module wrapped by ``_QuantFusedExperts`` — i.e. any HF
    transformers 5.0+ fused expert container that stores ``gate_up_proj`` and
    ``down_proj`` as 3-D ``nn.Parameter`` tensors with per-expert quantizer
    ``nn.ModuleList`` s.

    Steps:

    1. Handle amax fallback for uncalibrated expert input quantizers.
    2. Split fused 3-D weights into per-expert 2-D projections
       (``gate_proj``, ``up_proj``, ``down_proj``).
    3. Call ``_export_quantized_weight`` on each projection.
    4. Register results under the standard naming convention::

           {E}.gate_proj.weight, {E}.gate_proj.weight_scale, ...
           {E}.up_proj.weight, {E}.up_proj.weight_scale, ...
           {E}.down_proj.weight, {E}.down_proj.weight_scale, ...
    """
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight
    from modelopt.torch.quantization.plugins.huggingface import _get_fused_expert_intermediate_dim

    n = module.num_experts
    expert_dim = _get_fused_expert_intermediate_dim(module)

    # 1. Shared input quantizers — one per projection type, shared across all experts.
    gate_up_input_q = module.gate_up_proj_input_quantizer
    down_input_q = module.down_proj_input_quantizer

    gate_up = module.gate_up_proj.data
    down = module.down_proj.data

    # 2-3. Split + export each per-expert projection.
    fused_dim0 = gate_up.shape[1]  # 2 * expert_dim

    for idx in range(n):
        expert = nn.Module()

        projections = [
            ("gate_proj", gate_up[idx, :expert_dim, :], 0, fused_dim0, True),
            ("up_proj", gate_up[idx, expert_dim:, :], expert_dim, fused_dim0, True),
            ("down_proj", down[idx], 0, down.shape[1], False),
        ]

        for proj_name, weight_slice, fused_start, fused_total, is_gate_up in projections:
            w_quantizer_src = (
                module.gate_up_proj_weight_quantizers[idx]
                if is_gate_up
                else module.down_proj_weight_quantizers[idx]
            )
            i_quantizer = gate_up_input_q if is_gate_up else down_input_q

            # gate/up share a quantizer — deepcopy so gate_proj and up_proj get
            # independent quantizers that can hold different amax slices.
            if is_gate_up:
                w_quantizer = copy.deepcopy(w_quantizer_src)
            else:
                w_quantizer = w_quantizer_src

            # For per-channel amax (dim >= 1), proportionally slice dim-0
            # to match the split weight.
            if (
                hasattr(w_quantizer, "_amax")
                and w_quantizer._amax is not None
                and w_quantizer._amax.dim() >= 1
            ):
                amax = w_quantizer._amax
                block_size = (getattr(w_quantizer, "block_sizes", None) or {}).get(-1)
                w_quantizer._amax = _slice_fused_expert_amax(
                    amax, weight_slice, fused_start, fused_total, block_size
                )
                if w_quantizer._amax is amax:
                    warnings.warn(
                        f"Expert {idx} {proj_name}: fused amax shape ({tuple(amax.shape)}) "
                        f"cannot be mapped to projection shape ({tuple(weight_slice.shape)}). "
                        f"Skipping amax slicing, "
                        f"which may produce incorrect quantization scales.",
                        stacklevel=2,
                    )

            # Patch invalid per-block amax entries (NaN/inf/negative/zero/too-small/too-large)
            # with weight-derived fallback values.
            min_valid_amax = 2e-3  # floor matches FP8 E4M3FN minimum subnormal (2^-9 ≈ 0.00195)
            max_valid_amax = 1e6
            if (
                hasattr(w_quantizer, "_amax")
                and w_quantizer._amax is not None
                and w_quantizer._amax.numel() > 1
                and (getattr(w_quantizer, "block_sizes", None) or {}).get(-1) is not None
            ):
                amax_cpu = w_quantizer._amax
                invalid_mask = ~(
                    torch.isfinite(amax_cpu)
                    & (amax_cpu >= min_valid_amax)
                    & (amax_cpu <= max_valid_amax)
                )
                if invalid_mask.any():
                    _block_size = (getattr(w_quantizer, "block_sizes", None) or {}).get(-1, 16)
                    per_block_fallback = (
                        weight_slice.detach()
                        .reshape(-1, _block_size)
                        .abs()
                        .amax(dim=1, keepdim=True)
                        .cpu()
                        .float()
                        .clamp(min=2e-3)
                        .reshape(amax_cpu.shape)
                    )
                    amax_cpu[invalid_mask] = per_block_fallback[invalid_mask]
                    w_quantizer._amax = amax_cpu

            # For uncalibrated experts (amax missing or invalid scalar), fall back to
            # per-block amax from weights so the static export path can reshape it correctly.
            # Only applies to per-block (NVFP4) quantizers — non-block quantizers have
            # no block_sizes and should not be routed to the static NVFP4 export path.
            if (
                hasattr(w_quantizer, "is_enabled")
                and w_quantizer.is_enabled
                and (getattr(w_quantizer, "block_sizes", None) or {}).get(-1) is not None
                and (
                    not hasattr(w_quantizer, "_amax")
                    or w_quantizer._amax is None
                    or (
                        w_quantizer._amax.numel() == 1
                        and not (
                            torch.isfinite(w_quantizer._amax)
                            and w_quantizer._amax >= min_valid_amax
                            and w_quantizer._amax <= max_valid_amax
                        )
                    )
                )
            ):
                _block_size = (getattr(w_quantizer, "block_sizes", None) or {}).get(-1, 16)
                fallback_per_block = (
                    weight_slice.detach()
                    .reshape(-1, _block_size)
                    .abs()
                    .amax(dim=1, keepdim=True)
                    .cpu()
                    .float()
                    .clamp(min=2e-3)
                    .reshape(*weight_slice.shape[:-1], weight_slice.shape[-1] // _block_size)
                )
                w_quantizer._amax = fallback_per_block
                warnings.warn(
                    f"Expert {idx} {proj_name} weight quantizer was not calibrated "
                    f"(amax missing or zero). Using weight-derived per-block amax as fallback. "
                    f"Consider using more calibration data to activate all experts.",
                    stacklevel=2,
                )

            wrapper = nn.Module()
            wrapper.weight = nn.Parameter(weight_slice.contiguous(), requires_grad=False)
            wrapper.weight_quantizer = w_quantizer
            wrapper.input_quantizer = i_quantizer

            # Set global_amax to route to the static NVFP4 export path (reads per-block _amax).
            # Always recompute from the current (possibly patched) _amax — a stale zero
            # global_amax causes division-by-zero in the per-block scale formula.
            # Guard: only per-block (NVFP4) quantizers have block_sizes; skip for others.
            wq = wrapper.weight_quantizer
            if (
                hasattr(wq, "_amax")
                and wq._amax is not None
                and wq._amax.numel() > 1
                and (getattr(wq, "block_sizes", None) or {}).get(-1) is not None
            ):
                wq._amax = wq._amax.to(weight_slice.device)
                wq.global_amax = wq._amax.float().amax().clamp(min=2e-3)

            _export_quantized_weight(wrapper, dtype)

            proj = nn.Module()
            proj.weight = wrapper.weight
            for attr in ("weight_scale", "weight_scale_2", "input_scale"):
                if hasattr(wrapper, attr):
                    proj.register_buffer(attr, getattr(wrapper, attr))

            expert.add_module(proj_name, proj)

        module.add_module(str(idx), expert)

    # 4. Remove fused params and quantizer lists — replaced by per-expert submodules
    for attr in (
        "gate_up_proj",
        "down_proj",
        "gate_up_proj_weight_quantizers",
        "gate_up_proj_input_quantizer",
        "down_proj_weight_quantizers",
        "down_proj_input_quantizer",
    ):
        if hasattr(module, attr):
            delattr(module, attr)


def save_expert_token_count_table(model: nn.Module, output_dir: str | Path | None = None):
    """Collect expert_token_count from all quantized MoE layers and save as an HTML table.

    The table has rows for each MoE layer and columns for each expert, with cell values
    showing the number of tokens routed to that expert during calibration.

    Args:
        model: The model containing quantized MoE layers with ``expert_token_count`` attributes.
        output_dir: Directory to save the HTML file. Defaults to current directory.
    """
    rows = []
    for name, module in model.named_modules():
        if hasattr(module, "expert_token_count") and module.expert_token_count.numel() > 0:
            rows.append((name, module.expert_token_count))

    if not rows:
        return

    num_experts = rows[0][1].shape[0]
    assert all(r[1].shape[0] == num_experts for r in rows), (
        "All MoE layers must have the same number of experts"
    )
    html_parts = [
        "<html><head><style>",
        "table { border-collapse: collapse; font-family: monospace; }",
        "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }",
        "th { background: #f0f0f0; }",
        "</style></head><body>",
        "<h2>Expert Calib Token Counts (per MoE layer)</h2>",
        "<table><tr><th>Layer/Expert</th>",
    ]
    html_parts.extend(f"<th>{i}</th>" for i in range(num_experts))
    html_parts.append("</tr>")

    for name, counts in rows:
        avg = counts.float().mean().item()
        html_parts.append(f"<tr><td>{name}</td>")
        for c in counts.tolist():
            if avg > 0 and c < avg * 0.05:
                style = ' style="background: #ff6666;"'
            elif avg > 0 and c < avg * 0.1:
                style = ' style="background: #ffcccc;"'
            else:
                style = ""
            html_parts.append(f"<td{style}>{c}</td>")
        html_parts.append("</tr>")

    html_parts.append("</table></body></html>")
    html_content = "\n".join(html_parts)

    if output_dir is None:
        output_dir = Path(".")
    output_path = Path(output_dir) / ".moe.html"
    output_path.write_text(html_content, encoding="utf-8")
    print(f"\033[1mExpert token count table saved to {output_path}\033[0m")
