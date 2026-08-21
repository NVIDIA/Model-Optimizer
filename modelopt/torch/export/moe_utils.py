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


def _export_fused_experts_keep_fused(module: nn.Module, dtype: torch.dtype) -> None:
    """Fold per-expert quantizers into the FUSED 3-D expert weights *in place*, staying sharded.

    Distributed (DCP) counterpart to :func:`_export_fused_experts`. That function slices the fused
    weight into full per-expert tensors -- under FSDP2 it must first all-gather the ``Shard(0)``
    fused weight (via the export handler's ``unshard``) and then materializes every expert on every
    rank, putting the whole model on each GPU (OOMs at 235B/480B). This keeps each projection a
    single fused param: on a ``Shard(0)`` DTensor it quantizes ONLY this rank's local experts
    (``to_local()`` + the DTensor's dim-0 global offset for the matching per-expert weight quantizer)
    and rewraps the result ``Shard(0)``, so the model stays sharded and DCP writes per-rank shards.
    On a plain tensor (single process) it quantizes all ``num_experts`` -- identical to the in-model
    path. Quantizing the fused ``<proj>[i]`` slice with its per-expert quantizer is byte-identical to
    the split-then-quantize, so ``_export_fused_experts_keep_fused`` + :func:`split_fused_experts_state_dict`
    reproduce :func:`_export_fused_experts`'s output exactly. Registers, per projection ``P``:
    ``module.<P>`` (quantized fused weight), ``module.<P>_weight_scale`` (stacked per-expert scales),
    ``module.<P>_weight_scale_2`` (NVFP4 per-expert scalar), ``module.<P>_input_scale`` (shared).
    """
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    try:
        from torch.distributed.tensor import DTensor, Replicate, Shard
    except Exception:  # pragma: no cover - non-distributed install
        DTensor = None
    try:
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    except Exception:  # pragma: no cover - older torch: fall back to even-split offset
        compute_local_shape_and_global_offset = None

    first = getattr(module, "_first_proj_attr", "gate_up_proj")
    n = module.num_experts
    for proj in (first, "down_proj"):
        weight = getattr(module, proj)
        weight_quantizers = getattr(module, f"{proj}_weight_quantizers")
        input_quantizer = getattr(module, f"{proj}_input_quantizer", None)

        # Under FSDP2 the fused 3-D weight is a DTensor sharded on dim 0 (the expert axis). Quantize
        # ONLY this rank's local experts and rewrap the result sharded identically -- indexing the
        # DTensor (weight[i]) would all-gather the whole fused weight onto every rank, de-sharding
        # the model and OOMing the GPU at scale. Plain tensor (single process): all experts.
        wdata = weight.data
        is_dt = DTensor is not None and isinstance(wdata, DTensor)
        if is_dt:
            mesh, placements = wdata.device_mesh, wdata.placements
            local_w = wdata.to_local()
            offset = None
            if compute_local_shape_and_global_offset is not None:
                try:
                    _, global_offset = compute_local_shape_and_global_offset(
                        tuple(wdata.shape), mesh, placements
                    )
                    offset = int(global_offset[0])
                except Exception:
                    offset = None
            if offset is None:  # even-split fallback (dim-0 Shard on a 1-D mesh)
                offset, nshards = 0, 1
                for mdim, p in enumerate(placements):
                    if isinstance(p, Shard) and p.dim == 0:
                        nshards = mesh.size(mdim)
                        offset = mesh.get_local_rank(mdim) * ((n + nshards - 1) // nshards)
                        break
            scale_placements = [
                p if (isinstance(p, Shard) and p.dim == 0) else Replicate() for p in placements
            ]
            local_indices = list(range(offset, offset + local_w.shape[0]))
        else:
            local_w = wdata
            local_indices = list(range(n))

        fp8_weights: list[torch.Tensor] = []
        weight_scales: list[torch.Tensor] = []
        weight_scale_2s: list[torch.Tensor] = []
        input_scale: torch.Tensor | None = None
        for local_idx, global_idx in enumerate(local_indices):
            w_quantizer = weight_quantizers[global_idx]
            w_slice = local_w[local_idx]
            # Uncalibrated-expert fallback: derive amax from this expert's local weight slice
            # (matches _export_fused_experts), so a never-routed expert still exports sane scales.
            if (
                hasattr(w_quantizer, "is_enabled")
                and w_quantizer.is_enabled
                and (
                    not hasattr(w_quantizer, "_amax")
                    or w_quantizer._amax is None
                    or torch.all(w_quantizer._amax == 0)
                )
            ):
                w_quantizer.amax = w_slice.abs().amax().to(torch.float32)
                warnings.warn(
                    f"Expert {global_idx} {proj} weight quantizer was not calibrated (amax missing "
                    f"or zero); using weight-derived amax. Increase calibration size to activate all "
                    f"experts.",
                    stacklevel=2,
                )
            wrapper = nn.Module()
            wrapper.weight = nn.Parameter(w_slice.contiguous(), requires_grad=False)
            wrapper.weight_quantizer = w_quantizer
            if input_quantizer is not None:
                wrapper.input_quantizer = input_quantizer
            _export_quantized_weight(wrapper, dtype)
            fp8_weights.append(wrapper.weight.data)
            weight_scales.append(wrapper.weight_scale)
            # NVFP4 carries a SECOND per-tensor weight scale (weight_scale_2) that dequantizes the
            # per-block weight_scale; FP8 has none. Keep it so the fused buffer (and the per-expert
            # split) stay dequantizable.
            if hasattr(wrapper, "weight_scale_2"):
                weight_scale_2s.append(wrapper.weight_scale_2)
            if hasattr(wrapper, "input_scale"):
                input_scale = wrapper.input_scale
        local_fp8 = torch.stack(fp8_weights, dim=0)
        local_ws = torch.stack(weight_scales, dim=0)
        if is_dt:
            local_fp8 = DTensor.from_local(local_fp8, mesh, placements, run_check=False)
            local_ws = DTensor.from_local(local_ws, mesh, scale_placements, run_check=False)
        setattr(module, proj, nn.Parameter(local_fp8, requires_grad=False))
        module.register_buffer(f"{proj}_weight_scale", local_ws)
        if weight_scale_2s:
            # Per-expert scalar (E,) -- shards on the expert axis exactly like the weight scale.
            local_ws2 = torch.stack(weight_scale_2s, dim=0)
            if is_dt:
                local_ws2 = DTensor.from_local(local_ws2, mesh, scale_placements, run_check=False)
            module.register_buffer(f"{proj}_weight_scale_2", local_ws2)
        if input_scale is not None:
            module.register_buffer(f"{proj}_input_scale", input_scale)

    # Drop the quantizer modules -- their info now lives in the fused weight + scale buffers.
    for attr in (
        f"{first}_weight_quantizers",
        f"{first}_input_quantizer",
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

# ----- Distributed-export helpers (fused-experts split for the no-gather FSDP2 write) -----
_FUSED_FIRST_PROJ = ("gate_up_proj", "up_proj")  # gated first / ungated first
_FUSED_PROJ = ("gate_up_proj", "up_proj", "down_proj")


def _fused_experts_prefixes(state_dict: dict) -> list[str]:
    """Return the sorted set of ``...experts`` prefixes that carry a fused 3-D expert weight
    (``<prefix>.gate_up_proj`` / ``up_proj`` / ``down_proj``) in ``state_dict``.
    """
    prefixes: set[str] = set()
    for key, tensor in state_dict.items():
        for proj in _FUSED_PROJ:
            if key.endswith(f".experts.{proj}") and hasattr(tensor, "dim") and tensor.dim() == 3:
                prefixes.add(key[: -(len(proj) + 1)])
                break
    return sorted(prefixes)


def split_fused_experts_state_dict(state_dict: dict) -> dict:
    """Rewrite fused-experts entries in a (consolidated) HF state dict into per-expert entries.

    The distributed export writes experts *fused* (3-D ``experts.<proj>`` weights + fused scale
    buffers, kept sharded for DCP). This post-pass splits them into the same per-expert layout the
    in-model :func:`_export_fused_experts` produces, so the two export paths yield identical
    checkpoints. Operates on the (already folded/quantized) tensors only -- pure slicing, no
    quantization. Non-fused-experts keys pass through unchanged.

    Per fused-experts prefix ``P`` (``...experts``), inputs are (gated shown; ungated drops the gate split):
      ``P.gate_up_proj`` ``(E, 2*I, H)`` + ``P.gate_up_proj_weight_scale`` + ``P.gate_up_proj_input_scale``
      ``P.down_proj``    ``(E, O, I)``   + ``P.down_proj_weight_scale``    + ``P.down_proj_input_scale``
    Outputs, for each expert ``i``:
      ``P.{i}.gate_proj.weight`` ``(I, H)`` (+ ``.weight_scale``/``.input_scale``)   [gated only]
      ``P.{i}.up_proj.weight``   ``(I, H)`` (+ ``.weight_scale``/``.input_scale``)
      ``P.{i}.down_proj.weight`` ``(O, I)`` (+ ``.weight_scale``/``.input_scale``)
    """
    prefixes = _fused_experts_prefixes(state_dict)
    if not prefixes:
        return state_dict

    # Keys consumed (and thus removed) while emitting per-expert keys.
    consumed: set[str] = set()
    out: dict = {}

    for prefix in prefixes:
        gated = f"{prefix}.gate_up_proj" in state_dict
        first = "gate_up_proj" if gated else "up_proj"
        first_w = state_dict[f"{prefix}.{first}"]
        down_w = state_dict[f"{prefix}.down_proj"]
        n_experts = first_w.shape[0]

        def _scales(proj_name):
            ws = state_dict.get(f"{prefix}.{proj_name}_weight_scale")
            ins = state_dict.get(f"{prefix}.{proj_name}_input_scale")
            # NVFP4 second-level (per-tensor) weight scale; None for FP8. (E,) one scalar per expert.
            ws2 = state_dict.get(f"{prefix}.{proj_name}_weight_scale_2")
            return ws, ins, ws2

        first_ws, first_in, first_ws2 = _scales(first)
        down_ws, down_in, down_ws2 = _scales("down_proj")

        def _slice_weight_scale(ws, row_slice, fused_rows):
            # per-tensor-per-expert scale -> scalar; per-output-channel -> slice the rows.
            if ws is None:
                return None
            if ws.dim() <= 1:  # (E,) one scalar per expert
                return ws[expert_idx]
            sub = ws[expert_idx]  # (fused_rows,) or (fused_rows, ...)
            return sub if row_slice is None else sub[row_slice]

        for expert_idx in range(n_experts):
            if gated:
                inter = first_w.shape[1] // 2
                # gate and up are halves of the fused gate_up: they share its single per-tensor
                # weight_scale_2 (first_ws2), matching the in-model export + vLLM's W1/W3 fusion.
                projections = [
                    (
                        "gate_proj",
                        first_w[expert_idx, :inter, :],
                        first_ws,
                        first_in,
                        first_ws2,
                        slice(0, inter),
                    ),
                    (
                        "up_proj",
                        first_w[expert_idx, inter:, :],
                        first_ws,
                        first_in,
                        first_ws2,
                        slice(inter, None),
                    ),
                    ("down_proj", down_w[expert_idx], down_ws, down_in, down_ws2, None),
                ]
            else:
                projections = [
                    ("up_proj", first_w[expert_idx], first_ws, first_in, first_ws2, None),
                    ("down_proj", down_w[expert_idx], down_ws, down_in, down_ws2, None),
                ]
            for proj_name, weight, ws, ins, ws2, row_slice in projections:
                base = f"{prefix}.{expert_idx}.{proj_name}"
                # .clone() every emitted tensor so no two keys share storage. gate/up share the
                # per-tensor scale object (and slices alias the fused parent); without cloning,
                # safetensors/save_pretrained shared-tensor dedup silently drops the duplicate
                # (e.g. up_proj.weight_scale/input_scale would go missing).
                out[f"{base}.weight"] = weight.detach().clone().contiguous()
                sliced_ws = _slice_weight_scale(ws, row_slice, weight.shape[0])
                if sliced_ws is not None:
                    out[f"{base}.weight_scale"] = sliced_ws.detach().clone().contiguous()
                # weight_scale_2 is a per-expert scalar (shared by gate/up): _slice_weight_scale's
                # dim<=1 branch returns ws2[expert_idx] regardless of row_slice. NVFP4 only; None for FP8.
                sliced_ws2 = _slice_weight_scale(ws2, row_slice, weight.shape[0])
                if sliced_ws2 is not None:
                    out[f"{base}.weight_scale_2"] = sliced_ws2.detach().clone().contiguous()
                if ins is not None:
                    out[f"{base}.input_scale"] = ins.detach().clone()

        # Mark the fused tensors + their scale buffers consumed.
        for proj in (first, "down_proj"):
            for suffix in ("", "_weight_scale", "_weight_scale_2", "_input_scale"):
                consumed.add(f"{prefix}.{proj}{suffix}")

    for key, tensor in state_dict.items():
        if key not in consumed:
            out[key] = tensor
    return out



def _dtensor_dim0_offset(dt) -> int:
    """Global dim-0 start index of this rank's local shard of a ``Shard(0)`` DTensor."""
    from torch.distributed.tensor import Shard

    mesh, placements = dt.device_mesh, dt.placements
    try:
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        _, global_offset = compute_local_shape_and_global_offset(tuple(dt.shape), mesh, placements)
        return int(global_offset[0])
    except Exception:
        n = dt.shape[0]
        for mdim, p in enumerate(placements):
            if isinstance(p, Shard) and p.dim == 0:
                nshards = mesh.size(mdim)
                return mesh.get_local_rank(mdim) * ((n + nshards - 1) // nshards)
        return 0


def _split_local_fused_module(
    prefix: str, sharded_sd: dict, ep_rank: int = 0, total_experts: int = 0
) -> dict:
    """Split THIS rank's *local* shard of one fused-experts module into per-expert keys with GLOBAL
    expert indices (plain, local tensors -- no all-gather).

    FSDP2/EP shard the fused 3-D expert weight on dim 0 (the expert axis), so every expert this rank
    owns is whole and local. We reuse :func:`split_fused_experts_state_dict` on the ``to_local()``
    shard (which numbers experts ``0..E_local-1``) and then shift each expert index by the rank's
    global dim-0 offset, so the resulting per-expert keys match the single-process export and stay
    on this rank for a direct distributed write (no consolidation).

    The fused expert tensor can arrive in two EP layouts, distinguished by its dim-0 size vs
    ``total_experts`` (the config expert count):
      * **per-EP-group** (size < total): transformers pre-sliced the param to this rank's EP group
        (FSDP x EP -> a dp-sharded DTensor of the group; or a plain ep-shard). ``_dtensor_dim0_offset``
        only gives the WITHIN-group offset, so we add the EP block base ``ep_rank * group_size``.
      * **all-experts** (size == total): an EP DTensor sharded across the ep mesh spanning all experts
        (classic DP x EP). ``_dtensor_dim0_offset`` already returns the TRUE global offset -> no base.
    ``total_experts == 0`` keeps the legacy per-EP-group behavior.
    """
    from torch.distributed.tensor import DTensor

    gated = f"{prefix}.gate_up_proj" in sharded_sd
    first = "gate_up_proj" if gated else "up_proj"
    fw = sharded_sd[f"{prefix}.{first}"]
    # dp_offset: this rank's dim-0 start WITHIN its DTensor; n_local: experts it owns.
    dp_offset = _dtensor_dim0_offset(fw) if isinstance(fw, DTensor) else 0
    n_local = (
        fw.to_local().shape[0]
        if isinstance(fw, DTensor)
        else (fw.shape[0] if fw is not None else 0)
    )
    group_size = int(fw.shape[0]) if fw is not None else 0
    # Add the EP block base only when the tensor is a per-EP-group slice (size < total_experts) or
    # total is unknown (legacy). When it spans all experts, dp_offset is already global.
    add_ep_base = (
        bool(ep_rank) and group_size > 0 and (total_experts == 0 or group_size < total_experts)
    )
    offset = dp_offset + (ep_rank * group_size if add_ep_base else 0)

    def _loc(key):
        v = sharded_sd.get(key)
        if v is None:
            return None
        return v.to_local() if isinstance(v, DTensor) else v

    def _loc_scale(key):
        # Weight scales are dim-0 (per-expert) tensors. A DTensor scale shards with the weight ->
        # to_local() aligns. But under EP the keep-fused fold emits the scale as a PLAIN buffer
        # spanning the whole EP group (not dp-sharded like the weight param); slice it by the
        # weight's within-group dp_offset so each scale pairs with its own dp-local expert.
        v = sharded_sd.get(key)
        if v is None:
            return None
        if isinstance(v, DTensor):
            return v.to_local()
        if v.dim() >= 1 and v.shape[0] > n_local and dp_offset + n_local <= v.shape[0]:
            return v[dp_offset : dp_offset + n_local]
        return v

    # Build a fused sub-dict of this rank's local experts (plain), then split + reindex.
    local_fused: dict = {
        f"{prefix}.{first}": _loc(f"{prefix}.{first}"),
        f"{prefix}.down_proj": _loc(f"{prefix}.down_proj"),
    }
    for proj in (first, "down_proj"):
        ws = _loc_scale(f"{prefix}.{proj}_weight_scale")
        if ws is not None:
            local_fused[f"{prefix}.{proj}_weight_scale"] = ws
        # NVFP4 second-level per-tensor weight scale: also a per-expert (dim-0) tensor, so the same
        # dp-offset slicing pairs each scalar with its dp-local expert. None for FP8.
        ws2 = _loc_scale(f"{prefix}.{proj}_weight_scale_2")
        if ws2 is not None:
            local_fused[f"{prefix}.{proj}_weight_scale_2"] = ws2
        ins = sharded_sd.get(f"{prefix}.{proj}_input_scale")  # shared scalar (replicated)
        if ins is not None:
            local_fused[f"{prefix}.{proj}_input_scale"] = ins

    split_local = split_fused_experts_state_dict(
        local_fused
    )  # keys {prefix}.{i}.* (i in 0..E_local-1)
    if offset == 0:
        return split_local

    plen = len(prefix) + 1
    out: dict = {}
    for key, val in split_local.items():
        i_str, tail = key[plen:].split(".", 1)
        out[f"{prefix}.{int(i_str) + offset}.{tail}"] = val
    return out


