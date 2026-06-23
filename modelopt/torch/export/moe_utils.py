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


def _alias_per_expert_subtree_from_prior(module: nn.Module, prior: nn.Module, n: int) -> None:
    """Build per-expert subtree on ``module`` by aliasing ``prior``'s packed buffers.

    For each expert ``idx`` in ``0..n-1``, creates ``module.{idx}.{gate,up,down}_proj``
    sub-modules whose ``weight`` / ``weight_scale`` / ``weight_scale_2`` /
    ``input_scale`` are aliased to the prior side's already-packed tensors.
    data_ptr equality is preserved so the downstream
    ``postprocess_state_dict`` dedup collapses the duplicates at write time.
    Called by ``_export_fused_experts`` on the tied-experts cache-hit fast path.
    """
    for _idx in range(n):
        _prior_expert = getattr(prior, str(_idx), None)
        if _prior_expert is None:
            continue
        _cur_expert = nn.Module()
        for _proj_name in ("gate_proj", "up_proj", "down_proj"):
            _prior_proj = getattr(_prior_expert, _proj_name, None)
            if _prior_proj is None:
                continue
            _cur_proj = nn.Module()
            if hasattr(_prior_proj, "weight"):
                _cur_proj.weight = _prior_proj.weight
            for _attr in ("weight_scale", "weight_scale_2", "input_scale"):
                if hasattr(_prior_proj, _attr):
                    _cur_proj.register_buffer(_attr, getattr(_prior_proj, _attr))
            _cur_expert.add_module(_proj_name, _cur_proj)
        module.add_module(str(_idx), _cur_expert)


def _delete_fused_moe_source_attrs(module: nn.Module) -> None:
    """Remove the 3-D fused source params and per-expert quantizer ModuleLists.

    Called once the per-expert subtree exists (either via the fast-path
    aliases or via the full unpack/pack path) so the redundant fused form
    doesn't appear in the exported state_dict alongside the per-expert form.
    """
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    for attr in (
        "gate_up_proj",
        first_proj_attr,
        "down_proj",
        "gate_up_proj_weight_quantizers",
        "gate_up_proj_input_quantizer",
        "down_proj_weight_quantizers",
        "down_proj_input_quantizer",
    ):
        if hasattr(module, attr):
            delattr(module, attr)


def _export_fused_experts(
    module: nn.Module,
    dtype: torch.dtype,
    _moe_tied_cache: dict[tuple[int, int], nn.Module] | None = None,
    _tied_cache: dict[int, nn.Module] | None = None,
) -> None:
    """Split fused MoE expert weights and export per-expert quantization scales.

    Works with any module wrapped by ``_QuantFusedExperts`` (gated, with a fused
    ``gate_up_proj``) or ``_QuantNonGatedFusedExperts`` (non-gated, with a single
    ``up_proj`` — e.g. NemotronH). Both store their projections as 3-D
    ``nn.Parameter`` tensors with per-expert quantizer ``nn.ModuleList`` s.

    Steps:

    1. Handle amax fallback for uncalibrated expert weight quantizers.
    2. Split fused 3-D weights into per-expert 2-D projections — gated:
       (``gate_proj``, ``up_proj``, ``down_proj``); non-gated: (``up_proj``,
       ``down_proj``).
    3. Call ``_export_quantized_weight`` on each projection.
    4. Register results under the standard naming convention::

           {E}.gate_proj.weight, {E}.gate_proj.weight_scale, ...  # gated only
           {E}.up_proj.weight, {E}.up_proj.weight_scale, ...
           {E}.down_proj.weight, {E}.down_proj.weight_scale, ...

    Tied-experts dedup is opt-in via ``_moe_tied_cache``: when multiple
    fused-expert modules share their 3-D source params via HF
    ``_tied_weights_keys``, the unpacking creates fresh per-expert tensors
    that break the tie. With ``_moe_tied_cache`` provided (tuple-keyed by
    ``(<first_proj>.data_ptr(), down_proj.data_ptr())``), the alias step
    at the end re-points the per-expert ``weight`` / ``weight_scale`` /
    ``weight_scale_2`` / ``input_scale`` buffers at a previously-processed
    module sharing the same source memory. ``_tied_cache`` (int-keyed) is
    threaded through to the per-projection ``_export_quantized_weight``
    calls so wrapper-level dedup uses the same scope as standalone Linears.
    Both caches are owned by the caller (typically
    ``_export_transformers_checkpoint``) and scoped to one export
    invocation; when ``None`` the corresponding alias step is skipped.
    """
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight
    from modelopt.torch.quantization.plugins.huggingface import _get_fused_expert_intermediate_dim

    n = module.num_experts
    # Gated experts fuse gate+up into ``gate_up_proj`` and must be split on export;
    is_gated = getattr(module, "_is_gated", True)
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    # Only the gated split needs the per-expert intermediate dim (gate|up boundary).
    expert_dim = _get_fused_expert_intermediate_dim(module) if is_gated else None

    # Capture source tensor identities BEFORE unpacking (the source
    # attrs are deleted at the end of this function).
    _source_key = (
        getattr(module, first_proj_attr).data_ptr(),
        module.down_proj.data_ptr(),
    )

    # Tied-experts fast path: if this exact (first_proj, down) source-tensor pair
    # has been processed before, alias all per-expert buffers directly from the
    # prior module — no unpacking, no per-expert packing, no transient buffers
    # thrown away. Cache miss falls through to the full unpack/pack below and
    # registers this module as the prior for any later tied module.
    if _moe_tied_cache is not None:
        _prior = _moe_tied_cache.get(_source_key)
        if _prior is not None and _prior is not module:
            _alias_per_expert_subtree_from_prior(module, _prior, n)
            _delete_fused_moe_source_attrs(module)
            return

    # 1. Shared input quantizers — one per projection type, shared across all experts.
    gate_up_input_q = module.gate_up_proj_input_quantizer
    down_input_q = module.down_proj_input_quantizer

    gate_up = getattr(module, first_proj_attr).data  # first projection (gate_up or up)
    down = module.down_proj.data

    # 2-3. Split + export each per-expert projection.
    fused_dim0 = gate_up.shape[1]  # gated: 2 * expert_dim; non-gated: expert_dim

    for idx in range(n):
        expert = nn.Module()

        # If the gate_up source quantizer was never calibrated (rare expert
        # that received no calibration tokens), derive its amax once from the
        # FUSED tensor so gate and up share the same weight_scale_2 below.
        # Why: vLLM fuses W1 (gate) and W3 (up) at load time and asserts a
        # single per-tensor scale across the fusion. The per-projection
        # fallback further down would otherwise compute amax independently from
        # each half — gate's max and up's max generally differ — producing
        # mismatched weight_scale_2 and garbled MoE output at inference.
        # Non-gated experts have no gate/up fusion, so this shared-amax step is
        # skipped — their single up_proj uses the generic per-projection fallback.
        gate_up_q = module.gate_up_proj_weight_quantizers[idx]
        if (
            is_gated
            and getattr(gate_up_q, "is_enabled", False)
            and (
                not hasattr(gate_up_q, "_amax")
                or gate_up_q._amax is None
                or torch.all(gate_up_q._amax == 0)
            )
        ):
            gate_up_q.amax = gate_up[idx].abs().amax().to(torch.float32)
            warnings.warn(
                f"Expert {idx} gate_up_proj weight quantizer was not calibrated "
                f"(amax missing or zero). Using fused-tensor amax as fallback "
                f"(shared by gate and up so weight_scale_2 stays consistent). "
                f"Consider increasing calibration size to activate all experts.",
                stacklevel=2,
            )

        if is_gated:
            projections = [
                ("gate_proj", gate_up[idx, :expert_dim, :], 0, fused_dim0, True),
                ("up_proj", gate_up[idx, expert_dim:, :], expert_dim, fused_dim0, True),
                ("down_proj", down[idx], 0, down.shape[1], False),
            ]
        else:
            # Non-gated: the single up_proj maps 1:1 to its weight quantizer, so it
            # is exported whole (no dim-0 split, no shared gate/up weight_scale_2).
            projections = [
                ("up_proj", gate_up[idx], 0, fused_dim0, True),
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
            w_quantizer = copy.deepcopy(w_quantizer_src) if is_gate_up else w_quantizer_src

            # For per-channel amax (dim >= 1), proportionally slice dim-0
            # to match the split weight.
            if (
                hasattr(w_quantizer, "_amax")
                and w_quantizer._amax is not None
                and w_quantizer._amax.dim() >= 1
            ):
                amax = w_quantizer._amax
                # Per-block _amax (NVFP4 static) collapses the row axis we want
                # to slice on; restore it so dim-0 slicing splits gate/up.
                if amax.numel() != fused_total and amax.numel() % fused_total == 0:
                    amax = amax.contiguous().view(fused_total, amax.numel() // fused_total)
                amax_dim0 = amax.shape[0]
                if fused_total % amax_dim0 == 0:
                    slice_start = fused_start * amax_dim0 // fused_total
                    slice_end = (fused_start + weight_slice.shape[0]) * amax_dim0 // fused_total
                    sliced = amax[slice_start:slice_end].contiguous()
                    # The amax setter refuses shape changes; drop _amax first.
                    if hasattr(w_quantizer, "_amax"):
                        delattr(w_quantizer, "_amax")
                    w_quantizer.amax = sliced
                else:
                    warnings.warn(
                        f"Expert {idx} {proj_name}: fused amax dim0 ({amax_dim0}) does not "
                        f"evenly divide fused_total ({fused_total}). Skipping amax slicing, "
                        f"which may produce incorrect quantization scales.",
                        stacklevel=2,
                    )

            # If the weight quantizer was never calibrated, compute amax from weights.
            if (
                hasattr(w_quantizer, "is_enabled")
                and w_quantizer.is_enabled
                and (
                    not hasattr(w_quantizer, "_amax")
                    or w_quantizer._amax is None
                    or torch.all(w_quantizer._amax == 0)
                )
            ):
                w_quantizer.amax = weight_slice.abs().amax().to(torch.float32)
                warnings.warn(
                    f"Expert {idx} {proj_name} weight quantizer was not calibrated "
                    f"(amax missing or zero). Using weight-derived amax as fallback. "
                    f"Consider using more calibration data to activate all experts.",
                    stacklevel=2,
                )

            wrapper = nn.Module()
            wrapper.weight = nn.Parameter(weight_slice.contiguous(), requires_grad=False)
            wrapper.weight_quantizer = w_quantizer
            wrapper.input_quantizer = i_quantizer

            _export_quantized_weight(wrapper, dtype, _tied_cache=_tied_cache)

            proj = nn.Module()
            proj.weight = wrapper.weight
            for attr in ("weight_scale", "weight_scale_2", "input_scale"):
                if hasattr(wrapper, attr):
                    proj.register_buffer(attr, getattr(wrapper, attr))

            expert.add_module(proj_name, proj)

        module.add_module(str(idx), expert)

    # 4. Remove fused params and quantizer lists — replaced by per-expert submodules
    _delete_fused_moe_source_attrs(module)

    # 5. Register this module in the dedup cache so any later tied module
    # (same source data_ptr pair) takes the fast path at the top of this
    # function. Reached only on cache miss; cache-hit modules early-exited
    # above before any unpack work.
    if _moe_tied_cache is not None:
        _moe_tied_cache[_source_key] = module


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
