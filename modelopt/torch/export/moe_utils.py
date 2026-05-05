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

    def _safe_cpu_amax(quantizer_src: nn.Module) -> torch.Tensor | None:
        """Extract _amax to CPU float32, surfacing and clearing any pending CUDA error first."""
        amax = getattr(quantizer_src, "_amax", None)
        if amax is None or not isinstance(amax, torch.Tensor):
            return None
        try:
            if amax.is_cuda:
                torch.cuda.synchronize(amax.device)
            return amax.detach().cpu().float()
        except Exception:
            return None

    for idx in range(n):
        expert = nn.Module()

        # Pre-extract both quantizer amaxes to CPU *before* deepcopy.
        # deepcopy calls .clone() on CUDA tensors — if the stored _amax has corrupt
        # bfloat16 storage (common for under-calibrated experts), that clone triggers an
        # async CUDA illegal-memory-access error.  By extracting first (with an explicit
        # synchronize to surface+clear the error) and then nulling _amax on the source
        # before deepcopy, we guarantee no CUDA kernel ever touches the corrupt tensor.
        gu_amax_cpu = _safe_cpu_amax(module.gate_up_proj_weight_quantizers[idx])
        down_amax_cpu = _safe_cpu_amax(module.down_proj_weight_quantizers[idx])

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

            # gate/up share a weight quantizer — clone so each gets independent amax.
            # Null _amax on source before deepcopy so the corrupt CUDA tensor is never
            # cloned; restore afterwards for the sibling projection (up_proj reuses src).
            if is_gate_up:
                _saved_amax = getattr(w_quantizer_src, "_amax", None)
                w_quantizer_src._amax = None
                w_quantizer = copy.deepcopy(w_quantizer_src)
                w_quantizer_src._amax = _saved_amax
                # Inject the pre-extracted CPU amax (already float32, no CUDA ops).
                w_quantizer._amax = gu_amax_cpu
            else:
                w_quantizer = w_quantizer_src
                w_quantizer._amax = down_amax_cpu

            # For per-channel amax (dim >= 1), proportionally slice dim-0
            # to match the split weight.
            if (
                hasattr(w_quantizer, "_amax")
                and w_quantizer._amax is not None
                and w_quantizer._amax.dim() >= 1
            ):
                amax = w_quantizer._amax  # CPU float32
                amax_dim0 = amax.shape[0]
                if amax_dim0 % fused_total == 0:
                    slice_start = fused_start * amax_dim0 // fused_total
                    slice_end = (fused_start + weight_slice.shape[0]) * amax_dim0 // fused_total
                    # Bypass amax.setter (which forbids shape changes); w_quantizer is a
                    # deepcopy for gate/up so mutating it is safe.
                    w_quantizer._amax = amax[slice_start:slice_end].contiguous()
                else:
                    warnings.warn(
                        f"Expert {idx} {proj_name}: fused amax dim0 ({amax_dim0}) does not "
                        f"evenly divide fused_total ({fused_total}). Skipping amax slicing, "
                        f"which may produce incorrect quantization scales.",
                        stacklevel=2,
                    )

            # Patch invalid blocks in the (possibly per-block) amax.
            # A block is invalid if it is: NaN, inf, negative, zero, subnormal (too small),
            # or a garbage-huge value from uninitialized memory.  All ops here are on CPU.
            _min_valid_amax = 2e-3  # ≈ FP8 E4M3FN minimum subnormal (2^-9); matches fallback clamp
            _max_valid_amax = 1e6  # reject clearly-corrupt huge values (uninitialized memory, etc.)
            if (
                hasattr(w_quantizer, "_amax")
                and w_quantizer._amax is not None
                and w_quantizer._amax.numel() > 1
            ):
                amax_cpu = w_quantizer._amax  # CPU float32, per-block shape e.g. (H, W)
                invalid_mask = ~(
                    torch.isfinite(amax_cpu)
                    & (amax_cpu >= _min_valid_amax)
                    & (amax_cpu <= _max_valid_amax)
                )
                if invalid_mask.any():
                    block_size = 16
                    per_block_fallback = (
                        weight_slice.detach()
                        .cpu()
                        .float()
                        .reshape(-1, block_size)
                        .abs()
                        .amax(dim=1, keepdim=True)
                        .clamp(min=2e-3)
                        .reshape(amax_cpu.shape)
                    )
                    amax_cpu[invalid_mask] = per_block_fallback[invalid_mask]
                    w_quantizer._amax = amax_cpu

            # If the weight quantizer was never calibrated (scalar amax missing or invalid),
            # compute per-block amax from weights entirely on CPU.
            # Use per-block (not scalar) so the static NVFP4 export path can reshape _amax
            # via .view(H, num_blocks_per_row) without a shape mismatch.
            if (
                hasattr(w_quantizer, "is_enabled")
                and w_quantizer.is_enabled
                and (
                    not hasattr(w_quantizer, "_amax")
                    or w_quantizer._amax is None
                    or (
                        w_quantizer._amax.numel() == 1
                        and not (
                            torch.isfinite(w_quantizer._amax)
                            and w_quantizer._amax >= _min_valid_amax
                            and w_quantizer._amax <= _max_valid_amax
                        )
                    )
                )
            ):
                _block_size = 16
                fallback_per_block = (
                    weight_slice.detach()
                    .cpu()
                    .float()
                    .reshape(-1, _block_size)
                    .abs()
                    .amax(dim=1, keepdim=True)
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

            # For NVFP4 MSE-calibrated experts, _amax is per-block (shape [H, W]).
            # Setting global_amax routes to the static export path which correctly reshapes
            # _amax via .view(expected_shape) rather than treating it as a per-tensor scale.
            # Move _amax to the weight's device here (it was kept on CPU until now).
            # IMPORTANT: always recompute global_amax from the current (possibly patched)
            # per-block _amax — do not skip when global_amax is already set, because a
            # stale zero value from an uncalibrated expert causes division-by-zero in the
            # FP8 per-block scale formula (per_block_scale * 448 / (global_amax/6)).
            wq = wrapper.weight_quantizer
            if hasattr(wq, "_amax") and wq._amax is not None and wq._amax.numel() > 1:
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
