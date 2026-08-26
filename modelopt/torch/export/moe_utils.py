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

from modelopt.torch.utils.logging import warn_rank_0


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

    Tied experts are not deduped here: when multiple fused-expert modules share their
    3-D source params via HF ``_tied_weights_keys``, each is split and packed
    independently to byte-identical per-expert tensors, and the duplicate keys are
    dropped by name in ``postprocess_state_dict`` (the single dedup authority).
    """
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight
    from modelopt.torch.quantization.plugins.huggingface import _get_fused_expert_intermediate_dim

    n = module.num_experts
    # Gated experts fuse gate+up into ``gate_up_proj`` and must be split on export;
    is_gated = getattr(module, "_is_gated", True)
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    # Only the gated split needs the per-expert intermediate dim (gate|up boundary).
    expert_dim = _get_fused_expert_intermediate_dim(module) if is_gated else None

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

            _export_quantized_weight(wrapper, dtype)

            proj = nn.Module()
            proj.weight = wrapper.weight
            for attr in ("weight_scale", "weight_scale_2", "input_scale"):
                if hasattr(wrapper, attr):
                    proj.register_buffer(attr, getattr(wrapper, attr))

            expert.add_module(proj_name, proj)

        module.add_module(str(idx), expert)

    # 4. Remove fused params and quantizer lists — replaced by per-expert submodules
    _delete_fused_moe_source_attrs(module)


def _weight_amax_missing(quantizer) -> bool:
    """Whether an enabled weight quantizer has no usable amax (expert saw no calibration tokens)."""
    return getattr(quantizer, "is_enabled", False) and (
        not hasattr(quantizer, "_amax")
        or quantizer._amax is None
        or torch.all(quantizer._amax == 0)
    )


def _pack_one_expert(weight, weight_quantizer, input_quantizer, dtype, needs_amax_fallback):
    """Quantize one expert's matrix with the same packer a normal linear layer uses.

    Returns the packed weight and its scales, each None if this format has no such scale. Set
    ``needs_amax_fallback`` for an expert that saw no calibration data, to scale it off its weights.
    """
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    wrapper = nn.Module()
    wrapper.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    # deepcopy so packing does not mutate the shared calibrated quantizer state.
    wrapper.weight_quantizer = copy.deepcopy(weight_quantizer)
    wrapper.input_quantizer = input_quantizer

    if needs_amax_fallback:
        # Uncalibrated expert (received no tokens): fall back to the weight's own amax.
        wrapper.weight_quantizer.amax = weight.abs().amax().to(torch.float32)

    _export_quantized_weight(wrapper, dtype)
    return (
        wrapper.weight.data,
        getattr(wrapper, "weight_scale", None),
        getattr(wrapper, "weight_scale_2", None),
        getattr(wrapper, "input_scale", None),
    )


def _pack_projection(module, proj_name, first_expert, dtype):
    """Quantize every expert this rank owns for one projection, and attach the result.

    ``first_expert`` says where this rank's experts start in the full list, which is how the
    quantizers are numbered.
    """
    experts = getattr(module, proj_name).data
    weight_quantizers = getattr(module, f"{proj_name}_weight_quantizers")
    input_quantizer = getattr(module, f"{proj_name}_input_quantizer")

    weights, weight_scales, weight_scale_2s, input_scale = [], [], [], None
    uncalibrated = []
    for local_index in range(experts.shape[0]):
        quantizer = weight_quantizers[first_expert + local_index]
        needs_amax_fallback = _weight_amax_missing(quantizer)
        if needs_amax_fallback:
            uncalibrated.append(first_expert + local_index)
        weight, scale, scale_2, expert_input_scale = _pack_one_expert(
            experts[local_index],
            quantizer,
            input_quantizer,
            dtype,
            needs_amax_fallback,
        )
        weights.append(weight)
        weight_scales.append(scale)
        weight_scale_2s.append(scale_2)
        if expert_input_scale is not None:
            input_scale = expert_input_scale

    if uncalibrated:
        # Mirrors the warning _export_fused_experts emits on the non-sharded path, so an
        # under-calibrated MoE does not export silently. One line per projection rather than per
        # expert, and rank 0 only, since every rank hits the same experts.
        warn_rank_0(
            f"{proj_name} weight quantizers for experts {uncalibrated} were not calibrated "
            f"(amax missing or zero). Using weight-derived amax as fallback. Consider increasing "
            f"calibration size to activate all experts."
        )

    setattr(module, proj_name, nn.Parameter(torch.stack(weights), requires_grad=False))
    if weight_scales[0] is not None:
        module.register_buffer(f"{proj_name}_weight_scale", torch.stack(weight_scales))
    if weight_scale_2s[0] is not None:
        module.register_buffer(
            f"{proj_name}_weight_scale_2",
            torch.stack([s.reshape(()) for s in weight_scale_2s]),
        )
    if input_scale is not None:
        module.register_buffer(f"{proj_name}_input_scale", input_scale)


def _pack_fused_experts_shard_local(module, dtype):
    """Quantize this rank's experts in place, keeping the weight fused on the expert axis.

    ``module._shard_local_start`` gives this rank's global expert offset, so the per-expert
    quantizers (a replicated global-length list) are indexed by it; it is 0 single-process, where the
    tensors are the full block. Each expert's first projection is packed whole so gate and up share
    one ``weight_scale_2``, which vLLM's load-time fusion expects; the gate/up split happens later in
    :func:`_split_packed_fused_experts`.
    """
    first_proj_name = getattr(module, "_first_proj_attr", "gate_up_proj")
    # Both projections are sharded on the expert axis, so one offset covers both.
    first_expert = getattr(module, "_shard_local_start", {}).get(first_proj_name, 0)

    for proj_name in (first_proj_name, "down_proj"):
        _pack_projection(module, proj_name, first_expert, dtype)


def _emit_expert_projection(state_dict, key_prefix, weight, scale, scale_2, input_scale):
    """Write one expert's projection and its scales into the state dict.

    Every key gets its own tensor object: experts share some scales, and reusing one object would
    let the tied-weight dedup drop keys.
    """
    state_dict[key_prefix + "weight"] = weight.contiguous()
    if scale is not None:
        state_dict[key_prefix + "weight_scale"] = scale.contiguous()
    if scale_2 is not None:
        state_dict[key_prefix + "weight_scale_2"] = scale_2.clone()
    if input_scale is not None:
        state_dict[key_prefix + "input_scale"] = input_scale.clone()


def _split_packed_fused_experts(state_dict, model):
    """Split keep-fused expert tensors in a gathered ``state_dict`` into per-expert keys.

    Converts the ``_pack_fused_experts_shard_local`` output (fused ``{name}.gate_up_proj [E,2I,H/2]``
    + per-expert scales) into the same per-expert deployment keys ``_export_fused_experts`` emits
    (``{name}.{e}.gate_proj.weight`` / ``up_proj`` / ``down_proj`` + scales). Byte-identical: packing
    is row-independent, so slicing the packed fused tensor equals packing each half; gate/up share the
    per-tensor ``weight_scale_2`` (packed together with one amax). Runs on the gathered (full) dict.
    """
    from modelopt.torch.quantization.plugins.huggingface import _get_fused_expert_intermediate_dim

    for name, module in model.named_modules():
        first_proj_name = getattr(module, "_first_proj_attr", "gate_up_proj")
        if not hasattr(module, f"{first_proj_name}_weight_quantizers"):
            continue
        prefix = f"{name}." if name else ""

        fused_weight = state_dict.pop(prefix + first_proj_name, None)
        if fused_weight is None:
            continue  # not keep-fused here (e.g. already split) -> nothing to do
        fused_scale = state_dict.pop(prefix + f"{first_proj_name}_weight_scale", None)
        fused_scale_2 = state_dict.pop(prefix + f"{first_proj_name}_weight_scale_2", None)
        fused_input_scale = state_dict.pop(prefix + f"{first_proj_name}_input_scale", None)
        down_weight = state_dict.pop(prefix + "down_proj", None)
        down_scale = state_dict.pop(prefix + "down_proj_weight_scale", None)
        down_scale_2 = state_dict.pop(prefix + "down_proj_weight_scale_2", None)
        down_input_scale = state_dict.pop(prefix + "down_proj_input_scale", None)

        # Drop leftover fused quantizer buffers (keep-fused does not delete them, unlike the
        # per-expert path which calls _delete_fused_moe_source_attrs).
        for key in [
            k for k in state_dict if k.startswith(prefix) and "_quantizer" in k[len(prefix) :]
        ]:
            state_dict.pop(key)

        # A gated module packs gate and up together, so its rows split in half; a non-gated one
        # holds up_proj alone and takes every row.
        if getattr(module, "_is_gated", True):
            half = _get_fused_expert_intermediate_dim(module)
            first_proj_rows = [("gate_proj", slice(None, half)), ("up_proj", slice(half, None))]
        else:
            first_proj_rows = [("up_proj", slice(None))]

        for expert in range(module.num_experts):
            for proj_name, rows in first_proj_rows:
                _emit_expert_projection(
                    state_dict,
                    f"{prefix}{expert}.{proj_name}.",
                    fused_weight[expert, rows],
                    fused_scale[expert, rows] if fused_scale is not None else None,
                    fused_scale_2[expert] if fused_scale_2 is not None else None,
                    fused_input_scale,
                )
            _emit_expert_projection(
                state_dict,
                f"{prefix}{expert}.down_proj.",
                down_weight[expert],
                down_scale[expert] if down_scale is not None else None,
                down_scale_2[expert] if down_scale_2 is not None else None,
                down_input_scale,
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
