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
    first_proj_weight_quantizers_attr = f"{first_proj_attr}_weight_quantizers"
    first_proj_input_quantizer_attr = f"{first_proj_attr}_input_quantizer"
    for attr in (
        first_proj_attr,
        first_proj_weight_quantizers_attr,
        first_proj_input_quantizer_attr,
        "down_proj",
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
    first_proj_input_q = getattr(module, f"{first_proj_attr}_input_quantizer")
    first_proj_weight_quantizers = getattr(module, f"{first_proj_attr}_weight_quantizers")
    down_input_q = module.down_proj_input_quantizer

    first_proj = getattr(module, first_proj_attr).data  # gate_up_proj or up_proj
    down = module.down_proj.data

    # 2-3. Split + export each per-expert projection.
    fused_dim0 = first_proj.shape[1]  # gated: 2 * expert_dim; non-gated: expert_dim

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
        first_proj_q = first_proj_weight_quantizers[idx]
        if (
            is_gated
            and getattr(first_proj_q, "is_enabled", False)
            and (
                not hasattr(first_proj_q, "_amax")
                or first_proj_q._amax is None
                or torch.all(first_proj_q._amax == 0)
            )
        ):
            first_proj_q.amax = first_proj[idx].abs().amax().to(torch.float32)
            warnings.warn(
                f"Expert {idx} gate_up_proj weight quantizer was not calibrated "
                f"(amax missing or zero). Using fused-tensor amax as fallback "
                f"(shared by gate and up so weight_scale_2 stays consistent). "
                f"Consider increasing calibration size to activate all experts.",
                stacklevel=2,
            )

        if is_gated:
            projections = [
                ("gate_proj", first_proj[idx, :expert_dim, :], 0, fused_dim0, True),
                ("up_proj", first_proj[idx, expert_dim:, :], expert_dim, fused_dim0, True),
                ("down_proj", down[idx], 0, down.shape[1], False),
            ]
        else:
            # Non-gated: the single up_proj maps 1:1 to its weight quantizer, so it
            # is exported whole (no dim-0 split, no shared gate/up weight_scale_2).
            projections = [
                ("up_proj", first_proj[idx], 0, fused_dim0, True),
                ("down_proj", down[idx], 0, down.shape[1], False),
            ]

        for (
            proj_name,
            weight_slice,
            fused_start,
            fused_total,
            uses_first_proj_quantizers,
        ) in projections:
            w_quantizer_src = (
                first_proj_weight_quantizers[idx]
                if uses_first_proj_quantizers
                else module.down_proj_weight_quantizers[idx]
            )
            i_quantizer = first_proj_input_q if uses_first_proj_quantizers else down_input_q

            # gate/up share a weight quantizer — clone so each gets independent amax.
            w_quantizer = (
                copy.deepcopy(w_quantizer_src) if uses_first_proj_quantizers else w_quantizer_src
            )

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


def _export_fused_experts_keep_fused(module, dtype):
    """Shard-local keep-fused NVFP4/FP8 pack for one rank's experts.

    Pack THIS rank's experts IN PLACE and keep the weight FUSED (``Shard(0)`` on E),
    instead of splitting into per-expert child modules.

    Self-adapts to FSDP2 vs single-process via ``module._shard_local_start`` (set by
    :func:`fsdp2_shard_local_pack`):

    * FSDP2 shard-local: the context has already ``to_local``'d ``gate_up_proj``/``down_proj`` to this
      rank's ``[E/world, ...]`` block and recorded ``start`` (this rank's global expert offset), so the
      per-expert weight quantizers are indexed ``[start + i]`` (the quantizer ModuleList is global
      length-E and replicated -- verified in ``spike_moe_indexing.py``).
    * single-process: ``start == 0`` and the tensors are the full ``[E, ...]`` block.

    Each local expert's fused first-projection ``[2I, H]`` is packed WHOLE by reusing the standard
    ``_export_quantized_weight`` (so the per-tensor ``weight_scale_2`` is naturally shared by the gate
    and up halves -- what vLLM's load-time fusion requires -- and per-block scales are row-local). The
    packed per-expert results are stacked back into fused ``[local_n, ...]`` tensors + per-expert
    scale buffers; the gate/up split is deferred to write time (``_split_fused_experts_state_dict``).
    Non-destructive: the fused params survive, so :func:`fsdp2_shard_local_pack` re-registers them.
    """
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    start = getattr(module, "_shard_local_start", {}).get(first_proj_attr, 0)
    first_proj_wq = getattr(module, f"{first_proj_attr}_weight_quantizers")
    down_wq = module.down_proj_weight_quantizers
    first_proj_iq = getattr(module, f"{first_proj_attr}_input_quantizer")
    down_iq = module.down_proj_input_quantizer

    first_proj = getattr(
        module, first_proj_attr
    ).data  # local [local_n, 2I, H] (or full [E, 2I, H])
    down = module.down_proj.data
    local_n = first_proj.shape[0]

    def _pack_one(weight_2d, w_quant_src, i_quant):
        wrapper = nn.Module()
        wrapper.weight = nn.Parameter(weight_2d.contiguous(), requires_grad=False)
        # deepcopy so packing does not mutate the shared calibrated quantizer state.
        wrapper.weight_quantizer = copy.deepcopy(w_quant_src)
        wrapper.input_quantizer = i_quant
        wq = wrapper.weight_quantizer
        if getattr(wq, "is_enabled", False) and (
            not hasattr(wq, "_amax") or wq._amax is None or torch.all(wq._amax == 0)
        ):
            # Uncalibrated expert (received no tokens): fall back to the weight's own amax.
            wq.amax = weight_2d.abs().amax().to(torch.float32)
        _export_quantized_weight(wrapper, dtype)
        return (
            wrapper.weight.data,
            getattr(wrapper, "weight_scale", None),
            getattr(wrapper, "weight_scale_2", None),
            getattr(wrapper, "input_scale", None),
        )

    fp_w, fp_s, fp_s2 = [], [], []
    dp_w, dp_s, dp_s2 = [], [], []
    fp_input_scale = dp_input_scale = None
    for i in range(local_n):
        g = start + i
        w, s, s2, isc = _pack_one(first_proj[i], first_proj_wq[g], first_proj_iq)
        fp_w.append(w)
        fp_s.append(s)
        fp_s2.append(s2)
        fp_input_scale = isc if isc is not None else fp_input_scale
        w, s, s2, isc = _pack_one(down[i], down_wq[g], down_iq)
        dp_w.append(w)
        dp_s.append(s)
        dp_s2.append(s2)
        dp_input_scale = isc if isc is not None else dp_input_scale

    def _register(attr, weights, scales, scale2s):
        setattr(module, attr, nn.Parameter(torch.stack(weights), requires_grad=False))
        if scales[0] is not None:
            module.register_buffer(f"{attr}_weight_scale", torch.stack(scales))
        if scale2s[0] is not None:
            module.register_buffer(
                f"{attr}_weight_scale_2", torch.stack([x.reshape(()) for x in scale2s])
            )

    _register(first_proj_attr, fp_w, fp_s, fp_s2)
    _register("down_proj", dp_w, dp_s, dp_s2)
    if fp_input_scale is not None:
        module.register_buffer(f"{first_proj_attr}_input_scale", fp_input_scale)
    if dp_input_scale is not None:
        module.register_buffer("down_proj_input_scale", dp_input_scale)


def _split_fused_experts_state_dict(state_dict, model):
    """Split keep-fused expert tensors in a gathered ``state_dict`` into per-expert keys.

    Converts the ``_export_fused_experts_keep_fused`` output (fused ``{name}.gate_up_proj [E,2I,H/2]``
    + per-expert scales) into the same per-expert deployment keys ``_export_fused_experts`` emits
    (``{name}.{e}.gate_proj.weight`` / ``up_proj`` / ``down_proj`` + scales). Byte-identical: packing
    is row-independent, so slicing the packed fused tensor equals packing each half; gate/up share the
    per-tensor ``weight_scale_2`` (packed together with one amax). Runs on the gathered (full) dict.
    """
    from modelopt.torch.quantization.plugins.huggingface import _get_fused_expert_intermediate_dim

    for name, module in model.named_modules():
        first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
        if not hasattr(module, f"{first_proj_attr}_weight_quantizers"):
            continue
        prefix = f"{name}." if name else ""
        fp_w = state_dict.pop(prefix + first_proj_attr, None)
        if fp_w is None:
            continue  # not keep-fused here (e.g. already split) -> nothing to do
        is_gated = getattr(module, "_is_gated", True)
        n = module.num_experts
        fp_s = state_dict.pop(prefix + f"{first_proj_attr}_weight_scale", None)
        fp_s2 = state_dict.pop(prefix + f"{first_proj_attr}_weight_scale_2", None)
        fp_in = state_dict.pop(prefix + f"{first_proj_attr}_input_scale", None)
        dp_w = state_dict.pop(prefix + "down_proj", None)
        dp_s = state_dict.pop(prefix + "down_proj_weight_scale", None)
        dp_s2 = state_dict.pop(prefix + "down_proj_weight_scale_2", None)
        dp_in = state_dict.pop(prefix + "down_proj_input_scale", None)

        # Drop leftover fused quantizer buffers (keep-fused does not delete them, unlike the
        # per-expert path which calls _delete_fused_moe_source_attrs).
        for k in [
            k for k in state_dict if k.startswith(prefix) and "_quantizer" in k[len(prefix) :]
        ]:
            state_dict.pop(k)

        edim = _get_fused_expert_intermediate_dim(module) if is_gated else None

        def _emit(e, proj, w, s, s2, insc):
            # clone/contiguous so each per-expert/projection key is a DISTINCT tensor object. The
            # shared scales (one input_scale across all experts; one weight_scale_2 across gate|up)
            # would otherwise share a data_ptr and get collapsed by postprocess_state_dict's tied-
            # weight dedup -- producing fewer keys than the per-expert path, which builds separate
            # equal-valued objects. Cloning matches that format exactly (values are identical).
            p = f"{prefix}{e}.{proj}."
            state_dict[p + "weight"] = w.contiguous()
            if s is not None:
                state_dict[p + "weight_scale"] = s.contiguous()
            if s2 is not None:
                state_dict[p + "weight_scale_2"] = s2.clone()
            if insc is not None:
                state_dict[p + "input_scale"] = insc.clone()

        for e in range(n):
            if is_gated:
                _emit(
                    e,
                    "gate_proj",
                    fp_w[e, :edim],
                    fp_s[e, :edim] if fp_s is not None else None,
                    fp_s2[e] if fp_s2 is not None else None,
                    fp_in,
                )
                _emit(
                    e,
                    "up_proj",
                    fp_w[e, edim:],
                    fp_s[e, edim:] if fp_s is not None else None,
                    fp_s2[e] if fp_s2 is not None else None,
                    fp_in,
                )
            else:
                _emit(
                    e,
                    "up_proj",
                    fp_w[e],
                    fp_s[e] if fp_s is not None else None,
                    fp_s2[e] if fp_s2 is not None else None,
                    fp_in,
                )
            _emit(
                e,
                "down_proj",
                dp_w[e],
                dp_s[e] if dp_s is not None else None,
                dp_s2[e] if dp_s2 is not None else None,
                dp_in,
            )
    return state_dict


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
