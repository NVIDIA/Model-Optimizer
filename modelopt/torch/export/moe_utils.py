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
from types import SimpleNamespace

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
        f"{first_proj_attr}_scale_inv",
        "down_proj_scale_inv",
        "gate_up_proj_activation_scale",
        "up_proj_activation_scale",
        "down_proj_activation_scale",
    ):
        if hasattr(module, attr):
            delattr(module, attr)


def _tensor_quantizers(quantizer) -> list:
    """The ``TensorQuantizer`` stages of ``quantizer`` (several for W4A8's SequentialQuantizer).

    A SequentialQuantizer has no ``_amax`` of its own and forwards ``amax`` writes to every
    stage, so each stage must be checked and set on its own.
    """
    # Deferred: ``export/__init__`` star-imports this module, so keep quantization out of its
    # import-time dependencies.
    from modelopt.torch.quantization.nn import SequentialQuantizer

    return list(quantizer) if isinstance(quantizer, SequentialQuantizer) else [quantizer]


def _is_uncalibrated(quantizer) -> bool:
    amax = getattr(quantizer, "_amax", None)
    return getattr(quantizer, "is_enabled", False) and (amax is None or bool(torch.all(amax == 0)))


def _fallback_amax(weight: torch.Tensor, quantizer) -> torch.Tensor:
    """Weight-derived amax for an uncalibrated expert, in the shape export expects.

    The amax setter rejects shape changes, and a block/channel-wise quantizer (e.g. the INT4
    stage of W4A8 AWQ) keeps a zero ``_amax`` of that shape, so a scalar fallback would raise.
    A missing ``_amax`` on an INT4 block quantizer still needs per-block ``(rows, blocks)``.
    """
    like = getattr(quantizer, "_amax", None)
    block_sizes = getattr(quantizer, "block_sizes", None)
    block = block_sizes.get(-1) if block_sizes and block_sizes.get("type") == "static" else None
    # Integer static-block stages (INT4 of W4A8) need a per-block amax. Float-block formats
    # (NVFP4) keep a per-tensor amax plus separate block scales, so they take the scalar path.
    if like is None and isinstance(quantizer.num_bits, int) and block:
        if weight.shape[-1] % block:
            raise ValueError(
                f"Weight of shape {tuple(weight.shape)} is not divisible into blocks of {block}."
            )
        like = weight.new_empty(weight.shape[0], weight.shape[-1] // block)
    if like is None or like.numel() == 1:
        amax = torch.maximum(weight.amax(), -weight.amin()).float()
        return amax if like is None else amax.view_as(like)
    # Block-quant calibration records the flat (rows * blocks, 1); also accept (rows, blocks) and
    # a per-row (rows,). Refuse any other layout (e.g. per input channel) rather than guess its
    # grouping.
    rows = weight.shape[0]
    if like.dim() == 1 and like.shape[0] == rows:
        blocks = 1
    elif like.dim() == 2 and like.shape[-1] == 1 and like.shape[0] % rows == 0:
        blocks = like.shape[0] // rows
    elif like.dim() == 2 and like.shape[0] == rows:
        blocks = like.shape[1]
    else:
        blocks = 0
    if not blocks or weight.shape[-1] % blocks:
        raise ValueError(
            f"Cannot derive amax of shape {tuple(like.shape)} from a weight of shape "
            f"{tuple(weight.shape)}."
        )
    groups = weight.reshape(rows, blocks, -1)
    return torch.maximum(groups.amax(dim=-1), -groups.amin(dim=-1)).float().reshape(like.shape)


def _rewrap_expert_rows(local: torch.Tensor, mesh, placements, num_experts: int):
    """Wrap this rank's experts as a DTensor spanning ``num_experts`` rows on the expert axis."""
    from torch.distributed.tensor import DTensor

    shape = torch.Size((num_experts, *local.shape[1:]))
    stride = torch.empty(shape, device="meta").stride()
    return DTensor.from_local(local, mesh, placements, run_check=False, shape=shape, stride=stride)


def _local_expert_weight(module: nn.Module, proj: str, local: torch.Tensor, idx: int, dtype):
    """``module.get_expert_weight(proj, idx, dtype)``, read from ``local`` rather than the module.

    The per-expert export packs each expert as that accessor returns it, which casts unpacked native
    FP8 experts, held at the default dtype, to the export dtype. The accessor indexes the fused
    parameter itself, a sharded DTensor here, so it is handed this rank's rows instead.
    """
    return type(module).get_expert_weight(SimpleNamespace(**{proj: local}), proj, idx, dtype)


def _refuse_unfoldable_shards(module: nn.Module, proj: str, weight, num_quantizers: int) -> None:
    """Refuse a sharded fused expert weight the keep-fused fold cannot fold, on every rank alike.

    Every rank reaches the same verdict: the global shape and the mesh are the same on all of them,
    and their quantizer counts differ only where each keeps its own experts' quantizers of an uneven
    shard, which the checks below refuse on every rank.
    """
    from torch.distributed.tensor import Shard

    name = f"{type(module).__name__}.{proj}"
    sharded = [placement for placement in weight.placements if isinstance(placement, Shard)]
    if len(sharded) > 1 or any(placement.dim != 0 for placement in sharded):
        # The fold shards the scales on the expert axis alone, and the ranks agree on their experts'
        # quantization over the one mesh dim that shards it.
        raise NotImplementedError(
            f"{name}: placements {weight.placements}; the keep-fused fold needs the experts (dim 0) "
            "sharded over one mesh dim and no other dim sharded."
        )
    shards = 1
    for mdim, placement in enumerate(weight.placements):
        if isinstance(placement, Shard) and placement.dim == 0:
            size = weight.device_mesh.size(mdim)
            # A rank holding no experts would stack none and register no input scale, so the ranks'
            # state dicts would differ for the DCP write. The experts are chunked as torch does.
            chunk = -(-weight.shape[0] // size)
            if not chunk or -(-weight.shape[0] // chunk) < size:
                raise NotImplementedError(
                    f"{name}: {weight.shape[0]} experts over {size} shards leaves a rank none; the "
                    "keep-fused fold needs every rank to hold an expert."
                )
            shards *= size
    # FSDP2 keeps every expert's quantizer on every rank; expert parallelism keeps only this rank's,
    # in the order of its shard, as _QuantFusedExperts._local_experts reads them.
    if num_quantizers != weight.shape[0] and num_quantizers * shards != weight.shape[0]:
        raise NotImplementedError(
            f"{name}: {num_quantizers} weight quantizers for {weight.shape[0]} experts in {shards} "
            "shards; the keep-fused fold needs one per expert, or one per expert of this rank's shard."
        )


def _export_fused_experts_keep_fused(module: nn.Module, dtype: torch.dtype) -> None:
    """Fold per-expert quantizers into the FUSED 3-D expert weights *in place*, staying sharded.

    Distributed (DCP) counterpart to :func:`_export_fused_experts`. That function slices the fused
    weight into full per-expert tensors -- under FSDP2 it must first all-gather the ``Shard(0)``
    fused weight (via the export handler's ``unshard``) and then materializes every expert on every
    rank, putting the whole model on each GPU (OOMs at 235B/480B). This keeps each projection a
    single fused param: on a ``Shard(0)`` DTensor it quantizes ONLY this rank's local experts
    (``to_local()``, each with its own weight quantizer: by global index where every rank keeps every
    expert's, by position in the shard where a rank keeps only its own) and rewraps the result
    ``Shard(0)``, so the model stays sharded and DCP writes per-rank shards. Where a rank keeps only
    its own experts' quantizers, the ranks first gather each projection's quantizer states
    (collective over the expert mesh dim), so every rank folds, leaves or refuses it alike.
    On a plain tensor (single process) it quantizes all ``num_experts`` -- identical to the in-model
    path. Quantizing the fused ``<proj>[i]`` slice with its per-expert quantizer is byte-identical to
    the split-then-quantize, so ``_export_fused_experts_keep_fused`` + :func:`split_fused_experts_state_dict`
    reproduce :func:`_export_fused_experts`'s output exactly. Registers, per quantized projection ``P``:
    ``module.<P>`` (quantized fused weight), ``module.<P>_weight_scale`` (stacked per-expert scales),
    ``module.<P>_weight_scale_2`` (NVFP4 per-expert scalar), ``module.<P>_input_scale`` (shared).
    """
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    try:
        from torch.distributed.tensor import DTensor, Replicate, Shard
    except Exception:  # pragma: no cover - non-distributed install
        DTensor = None  # noqa: N806 (the optional-import fallback for a class)

    first = getattr(module, "_first_proj_attr", "gate_up_proj")
    n = module.num_experts
    # One fused tensor cannot hold quantized and unquantized experts side by side, so a projection
    # is folded only when all its experts are quantized; one with none quantized stays as it is,
    # which is what the per-expert export writes for it. Every refusal comes before either is
    # folded, so it leaves the module untouched.
    folded = []
    for proj in (first, "down_proj"):
        weight_quantizers = getattr(module, f"{proj}_weight_quantizers")
        enabled = [q.is_enabled for q in weight_quantizers]
        wdata = getattr(module, proj).data
        if wdata.element_size() == 1:
            # get_expert_weight dequantizes these with their scale_inv, which the fold drops below
            # whether it folds the projection or leaves it as it is.
            raise NotImplementedError(
                f"{type(module).__name__}.{proj}: native FP8 expert weights cannot be exported as "
                "stored, since the fold drops their scale_inv; dequantize them first "
                "(unpack_weight())."
            )
        is_dt = DTensor is not None and isinstance(wdata, DTensor)
        if is_dt:
            _refuse_unfoldable_shards(module, proj, wdata, len(weight_quantizers))
            if len(weight_quantizers) != wdata.shape[0]:
                # Expert parallelism leaves each rank only its own experts' quantizers, so the ranks
                # decide on the projection together: alone, one rank could fold it while another
                # leaves it as it is or refuses it. Joined by offset rather than by group rank.
                mdim = next(i for i, p in enumerate(wdata.placements) if isinstance(p, Shard))
                gathered: list = [None] * wdata.device_mesh.size(mdim)
                torch.distributed.all_gather_object(
                    gathered,
                    (_dtensor_dim0_offset(wdata), enabled),
                    group=wdata.device_mesh.get_group(mdim),
                )
                enabled = [e for _, shard in sorted(gathered) for e in shard]
        if all(enabled):
            if not is_dt and not len(weight_quantizers) == wdata.shape[0] == n:
                raise NotImplementedError(
                    f"{type(module).__name__}.{proj}: num_experts is {n}, but the unsharded weight "
                    f"holds {wdata.shape[0]} experts and there are {len(weight_quantizers)} weight "
                    "quantizers; the keep-fused fold needs the three equal."
                )
            folded.append(proj)
        elif any(enabled):
            raise NotImplementedError(
                f"{type(module).__name__}.{proj}: weight quantizers "
                f"{[i for i, e in enumerate(enabled) if not e]} of {len(enabled)} are disabled and "
                "the rest are not; a fused expert weight cannot mix quantized and unquantized experts."
            )
    for proj in folded:
        weight = getattr(module, proj)
        weight_quantizers = getattr(module, f"{proj}_weight_quantizers")
        input_quantizer = getattr(module, f"{proj}_input_quantizer", None)

        # Under FSDP2 the fused 3-D weight is a DTensor sharded on dim 0 (the expert axis). Quantize
        # ONLY this rank's local experts and rewrap the result sharded identically -- indexing the
        # DTensor (weight[i]) would all-gather the whole fused weight onto every rank, de-sharding
        # the model and OOMing the GPU at scale. Plain tensor (single process): all experts.
        wdata = weight.data
        is_dt = DTensor is not None and isinstance(wdata, DTensor)
        quantizers_are_global = True
        if is_dt:
            mesh, placements = wdata.device_mesh, wdata.placements
            local_w = wdata.to_local()
            offset = _dtensor_dim0_offset(wdata)
            scale_placements = [
                p if (isinstance(p, Shard) and p.dim == 0) else Replicate() for p in placements
            ]
            local_indices = list(range(offset, offset + local_w.shape[0]))
            # Every expert's quantizers (FSDP2), or only this rank's, in shard order (expert
            # parallelism): _refuse_unfoldable_shards allows no other count.
            quantizers_are_global = len(weight_quantizers) == wdata.shape[0]
        else:
            local_w = wdata
            local_indices = list(range(n))

        fp8_weights: list[torch.Tensor] = []
        weight_scales: list[torch.Tensor] = []
        weight_scale_2s: list[torch.Tensor] = []
        input_scale: torch.Tensor | None = None
        for local_idx, global_idx in enumerate(local_indices):
            w_quantizer = weight_quantizers[global_idx if quantizers_are_global else local_idx]
            w_slice = _local_expert_weight(module, proj, local_w, local_idx, dtype)
            # Uncalibrated-expert fallback: derive amax from this expert's local weight slice
            # (matches _export_fused_experts), so a never-routed expert still exports sane scales.
            # Per stage, in the shape each expects (W4A8's SequentialQuantizer, block-wise INT4).
            uncalibrated = [q for q in _tensor_quantizers(w_quantizer) if _is_uncalibrated(q)]
            for q in uncalibrated:
                q.amax = _fallback_amax(w_slice, q)
            if uncalibrated:
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
        # On the weight's device: per-tensor FP8 builds each expert's scale from a Python float, on
        # the CPU, and a DTensor rewrap would label a CPU shard with the weight's device mesh.
        local_ws = torch.stack(weight_scales, dim=0).to(local_w.device)
        if is_dt:
            # Keep the global expert count: from_local would infer it as local x world, which is
            # wrong for an uneven shard (3 experts on 2 ranks) and misnumbers the experts.
            local_fp8 = _rewrap_expert_rows(local_fp8, mesh, placements, wdata.shape[0])
            local_ws = _rewrap_expert_rows(local_ws, mesh, scale_placements, wdata.shape[0])
        setattr(module, proj, nn.Parameter(local_fp8, requires_grad=False))
        module.register_buffer(f"{proj}_weight_scale", local_ws)
        if weight_scale_2s:
            # Per-expert scalar (E,) -- shards on the expert axis exactly like the weight scale.
            local_ws2 = torch.stack(weight_scale_2s, dim=0).to(local_w.device)
            if is_dt:
                local_ws2 = _rewrap_expert_rows(local_ws2, mesh, scale_placements, wdata.shape[0])
            module.register_buffer(f"{proj}_weight_scale_2", local_ws2)
        if input_scale is not None:
            module.register_buffer(f"{proj}_input_scale", input_scale)

    # A projection left as it is goes out in the weight domain the per-expert export writes it in.
    for proj in (first, "down_proj"):
        if proj in folded:
            continue
        wdata = getattr(module, proj).data
        is_dt = DTensor is not None and isinstance(wdata, DTensor)
        local = wdata.to_local() if is_dt else wdata
        domain = _local_expert_weight(module, proj, local, 0, dtype).dtype
        if domain != local.dtype:
            local = local.to(domain)
            if is_dt:
                local = _rewrap_expert_rows(
                    local, wdata.device_mesh, wdata.placements, wdata.shape[0]
                )
            setattr(module, proj, nn.Parameter(local, requires_grad=False))

    # Drop the quantizer modules -- their info now lives in the fused weight + scale buffers -- and
    # the source checkpoint's scales, which described the weights the fold replaced, as the
    # per-expert export does (_delete_fused_moe_source_attrs).
    for attr in (
        f"{first}_weight_quantizers",
        f"{first}_input_quantizer",
        "down_proj_weight_quantizers",
        "down_proj_input_quantizer",
        f"{first}_scale_inv",
        "down_proj_scale_inv",
        f"{first}_activation_scale",
        "down_proj_activation_scale",
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

    n = module.num_experts
    # Gated experts fuse gate+up into ``gate_up_proj`` and must be split on export;
    is_gated = getattr(module, "_is_gated", True)
    first_proj_attr = getattr(module, "_first_proj_attr", "gate_up_proj")
    # Only the gated split needs the per-expert intermediate dim (gate|up boundary).
    # Deferred: the huggingface plugin imports transformers at module scope, and transformers is
    # an optional extra.
    from modelopt.torch.quantization.plugins.huggingface import _get_fused_expert_intermediate_dim

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
        first_weight = module.get_expert_weight(first_proj_attr, idx, dtype)
        down_weight = module.get_expert_weight("down_proj", idx, dtype)

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
        uncalibrated_stages = (
            [
                q
                for q in _tensor_quantizers(first_proj_weight_quantizers[idx])
                if _is_uncalibrated(q)
            ]
            if is_gated
            else []
        )
        for q in uncalibrated_stages:
            q.amax = _fallback_amax(first_weight, q)
        if uncalibrated_stages:
            warnings.warn(
                f"Expert {idx} gate_up_proj weight quantizer was not calibrated "
                f"(amax missing or zero). Using fused-tensor amax as fallback "
                f"(shared by gate and up so weight_scale_2 stays consistent). "
                f"Consider increasing calibration size to activate all experts.",
                stacklevel=2,
            )

        if is_gated:
            projections = [
                ("gate_proj", first_weight[:expert_dim, :], 0, fused_dim0, True),
                ("up_proj", first_weight[expert_dim:, :], expert_dim, fused_dim0, True),
                ("down_proj", down_weight, 0, down.shape[1], False),
            ]
        else:
            # Non-gated: the single up_proj maps 1:1 to its weight quantizer, so it
            # is exported whole (no dim-0 split, no shared gate/up weight_scale_2).
            projections = [
                ("up_proj", first_weight, 0, fused_dim0, True),
                ("down_proj", down_weight, 0, down.shape[1], False),
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

            for stage in _tensor_quantizers(w_quantizer):
                # For per-channel amax (dim >= 1), proportionally slice dim-0
                # to match the split weight.
                if getattr(stage, "_amax", None) is not None and stage._amax.numel() > 1:
                    amax = stage._amax
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
                        delattr(stage, "_amax")
                        stage.amax = sliced
                        # Block-quant calibration recorded the fused amax's row count here, and
                        # export_amax() reshapes to it, undoing the slice.
                        export_shape = getattr(stage, "_amax_shape_for_export", None)
                        if export_shape is not None and export_shape[0] == amax_dim0:
                            stage._amax_shape_for_export = (sliced.shape[0], *export_shape[1:])
                    else:
                        warnings.warn(
                            f"Expert {idx} {proj_name}: fused amax dim0 ({amax_dim0}) does not "
                            f"evenly divide fused_total ({fused_total}). Skipping amax slicing, "
                            f"which may produce incorrect quantization scales.",
                            stacklevel=2,
                        )

                # If the weight quantizer was never calibrated, compute amax from weights.
                if _is_uncalibrated(stage):
                    stage.amax = _fallback_amax(weight_slice, stage)
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
_FUSED_PROJ = ("gate_up_proj", "up_proj", "down_proj")


def _fused_experts_prefixes(state_dict: dict) -> list[str]:
    """Return the prefixes of the modules holding a fused 3-D expert weight.

    Return the sorted set of ``...experts`` prefixes that carry a fused 3-D expert weight
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
    """Rewrite the fused-experts entries of a state dict into per-expert entries.

    The distributed export writes experts *fused* (3-D ``experts.<proj>`` weights + fused scale
    buffers, kept sharded for DCP). This post-pass splits them into the same per-expert layout the
    in-model :func:`_export_fused_experts` produces, so the two export paths yield identical
    checkpoints. Operates on the (already folded/quantized) tensors only -- pure slicing, no
    quantization. Non-fused-experts keys pass through unchanged. Every emitted tensor is a copy and
    the input keeps its fused tensors, so the experts are held twice until it returns: pass one
    module's shard at a time, as :func:`_split_local_fused_module` does, not a whole model's.

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

        def _slice_weight_scale(ws, expert_idx, row_slice, rows):
            # A per-expert scalar as is; else the expert's scale rows covering ``row_slice`` of its
            # ``rows`` weight rows, in proportion: an INT4 weight packs two rows a byte while its
            # scales stay per row, and an FP8 2-D block scale covers 128 rows.
            if ws is None:
                return None
            if ws.dim() <= 1:  # (E,) one scalar per expert
                return ws[expert_idx]
            sub = ws[expert_idx]
            if row_slice is None:
                return sub
            start, stop, _ = row_slice.indices(rows)
            n = sub.shape[0]
            # Exact only for scale rows spread evenly over the weight rows, with the cut between
            # two of them; otherwise a block of rows sharing one scale straddles gate and up.
            if (rows % n and n % rows) or start * n % rows or stop * n % rows:
                raise NotImplementedError(
                    f"A {n}-row weight scale over {rows} weight rows cannot be split at the gate|up "
                    f"boundary, row {start or stop}: a block of rows sharing one scale crosses it."
                )
            return sub[start * n // rows : stop * n // rows]

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
                sliced_ws = _slice_weight_scale(ws, expert_idx, row_slice, first_w.shape[1])
                if sliced_ws is not None:
                    out[f"{base}.weight_scale"] = sliced_ws.detach().clone().contiguous()
                # weight_scale_2 is a per-expert scalar (shared by gate/up): _slice_weight_scale's
                # dim<=1 branch returns ws2[expert_idx] regardless of row_slice. NVFP4 only; None for FP8.
                sliced_ws2 = _slice_weight_scale(ws2, expert_idx, row_slice, first_w.shape[1])
                if sliced_ws2 is not None:
                    out[f"{base}.weight_scale_2"] = sliced_ws2.detach().clone().contiguous()
                if ins is not None:
                    out[f"{base}.input_scale"] = ins.detach().clone()

        # Mark the fused tensors + their scale buffers consumed.
        for proj in (first, "down_proj"):
            for suffix in ("", "_weight_scale", "_weight_scale_2", "_input_scale"):
                consumed.add(f"{prefix}.{proj}{suffix}")

    out.update({key: tensor for key, tensor in state_dict.items() if key not in consumed})
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
    """Split this rank's local shard of a fused-experts module into per-expert keys.

    Split THIS rank's *local* shard of one fused-experts module into per-expert keys with GLOBAL
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
        # to_local() aligns. The keep-fused fold emits DTensor scales for a sharded weight, but a
        # scale handed in as a PLAIN tensor over every expert of the group is accepted too: slice
        # it by the weight's within-group dp_offset so each scale pairs with its own dp-local expert.
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
