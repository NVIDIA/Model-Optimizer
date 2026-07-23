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

"""Forward-only NeMo AutoModel recipe for replace-1-block (solution) scoring.

Subclasses :class:`ActivationScoringRecipe` to reuse the proven setup (sharded model
build, puzzletron dataloader, PP kwarg sanitizers, ``_forward_batch``) without building
pruning scorers. It instead captures, per batch, the model's **final hidden state** and
the **LM-head weight**, from which the metric (``solution_metrics``/``flash_dual``)
reconstructs logits chunk-by-chunk — so neither full-vocab logits nor a second model are
needed. Only the last pipeline stage holds these; non-last stages still drive the
collective forward but capture nothing.

Capture strategy (the in-container-sensitive part): the final hidden is taken from the
model-level final norm's output (always produced; robust even when NeMo computes the loss
fused via ``te_parallel_ce`` and never calls ``lm_head``). The LM-head weight is read from
the ``lm_head`` parameter (or the tied embedding) and gathered across TP once.
"""

import logging
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

from ...pruning.mamba2_surgery import Mamba2TensorLayout, mamba2_projected_prefix_mask
from .scoring_recipe import ActivationScoringRecipe

logger = logging.getLogger(__name__)

__all__ = ["ReplaceBlockScoringRecipe"]


def _rank_tag() -> str:
    import torch.distributed as torch_dist

    if torch_dist.is_available() and torch_dist.is_initialized():
        return f"rank={torch_dist.get_rank()}"
    return "rank=0"


def _full(t: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return t.full_tensor() if isinstance(t, DTensor) else t


def _zero_like_output(output):
    if torch.is_tensor(output):
        return torch.zeros_like(output)
    if isinstance(output, tuple):
        return tuple(_zero_like_output(item) for item in output)
    if isinstance(output, list):
        return [_zero_like_output(item) for item in output]
    return output


def _module_output_zero_hook(module, args, output):
    return _zero_like_output(output)


@contextmanager
def _temporary_attention_window(layer: nn.Module, target_window: int | str):
    """Apply one layer-local attention window and restore structural attributes.

    Native AutoModel attention kernels consume ``self_attn.sliding_window`` at
    forward time. Hybrid models can additionally select a prebuilt full/sliding
    mask through a layer-level attention-type field, so both representations are
    changed together.
    """

    attention = getattr(layer, "self_attn", None)
    if attention is None:
        candidate = getattr(layer, "mixer", None)
        if candidate is not None and hasattr(candidate, "sliding_window"):
            attention = candidate
    if attention is None or not hasattr(attention, "sliding_window"):
        raise RuntimeError(
            "windowed-attention candidate requires a live attention module with "
            f"a sliding_window attribute; layer={type(layer).__name__}"
        )

    value = None if target_window == "full" else int(target_window)
    attention_type = "full_attention" if value is None else "sliding_attention"
    updates = [(attention, "sliding_window", value)]
    if hasattr(attention, "is_sliding"):
        updates.append((attention, "is_sliding", value is not None))
    for owner in (layer, attention):
        for name in ("attention_type", "layer_type"):
            current = getattr(owner, name, None)
            if current in ("full_attention", "sliding_attention"):
                updates.append((owner, name, attention_type))

    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in updates]
    try:
        for owner, name, new_value in updates:
            setattr(owner, name, new_value)
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _bool_prefix_mask(total: int, keep: int, *, device=None) -> torch.Tensor:
    mask = torch.zeros(int(total), dtype=torch.bool, device=device)
    mask[: max(0, min(int(keep), int(total)))] = True
    return mask


def _first_existing_submodule(layer: nn.Module, names: tuple[str, ...]):
    for name in names:
        try:
            module = layer.get_submodule(name)
        except AttributeError:
            continue
        if module is not None:
            return name, module
    return None, None


def _load_checkpoint_tensors(checkpoint_dir: str | Path, keys: set[str]) -> dict[str, torch.Tensor]:
    """Load a small key set from a HF/safetensors checkpoint."""
    from safetensors import safe_open
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

    checkpoint_dir = Path(checkpoint_dir)
    if not keys:
        return {}

    out: dict[str, torch.Tensor] = {}
    index_path = checkpoint_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", {})
        by_file: dict[str, list[str]] = {}
        for key in keys:
            filename = weight_map.get(key)
            if filename is not None:
                by_file.setdefault(str(filename), []).append(key)
        for filename, wanted in by_file.items():
            tensor_path = checkpoint_dir / filename
            if not tensor_path.is_file():
                raise FileNotFoundError(f"checkpoint tensor shard is missing: {tensor_path}")
            with safe_open(str(tensor_path), framework="pt", device="cpu") as f:
                available = set(f.keys())
                for key in wanted:
                    if key in available:
                        out[key] = f.get_tensor(key)
        return out

    single_file = checkpoint_dir / SAFE_WEIGHTS_NAME
    if single_file.is_file():
        with safe_open(str(single_file), framework="pt", device="cpu") as f:
            available = set(f.keys())
            for key in keys:
                if key in available:
                    out[key] = f.get_tensor(key)
        return out

    raise FileNotFoundError(
        f"checkpoint has no {SAFE_WEIGHTS_INDEX_NAME} or {SAFE_WEIGHTS_NAME}: {checkpoint_dir}"
    )


def _local_tensor_geometry(target: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    """Return local shape/offset for a tensor and whether it is distributed."""

    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    except Exception:  # noqa: BLE001
        DTensor = ()  # type: ignore[assignment]

    if isinstance(target, DTensor):
        local_shape, global_offset = compute_local_shape_and_global_offset(
            tuple(int(size) for size in target.shape),
            target.device_mesh,
            target.placements,
        )
        return tuple(local_shape), tuple(global_offset), True
    return tuple(int(size) for size in target.shape), (0,) * target.ndim, False


def _load_split_expert_overlay_tensor(
    checkpoint_dir: str | Path,
    checkpoint_key: str,
    target: torch.Tensor,
) -> tuple[torch.Tensor, bool] | None:
    """Merge one HF split-expert projection into a native grouped tensor.

    AutoModel exposes grouped expert parameters while canonical HF checkpoints
    store one matrix per expert.  For a distributed target, load only the
    rectangular slice owned by this rank instead of materializing the complete
    MoE tensor on every EP/FSDP rank.
    """

    gate_and_up_suffix = ".experts.gate_and_up_projs"
    down_suffix = ".experts.down_projs"
    if checkpoint_key.endswith(gate_and_up_suffix):
        expert_prefix = checkpoint_key[: -len("gate_and_up_projs")]
        projection_names = ("gate_proj", "up_proj")
    elif checkpoint_key.endswith(down_suffix):
        expert_prefix = checkpoint_key[: -len("down_projs")]
        projection_names = ("down_proj",)
    else:
        return None

    local_shape, global_offset, is_local = _local_tensor_geometry(target)
    if len(local_shape) != 3:
        return None
    expert_ids = range(global_offset[0], global_offset[0] + local_shape[0])
    keys = {
        f"{expert_prefix}{expert_idx}.{projection}.weight"
        for expert_idx in expert_ids
        for projection in projection_names
    }
    loaded = _load_checkpoint_tensors(checkpoint_dir, keys)

    def expert_key(expert_idx: int, projection: str) -> str:
        return f"{expert_prefix}{expert_idx}.{projection}.weight"

    if projection_names == ("down_proj",):
        required = {expert_key(idx, "down_proj") for idx in expert_ids}
        if not loaded:
            return None
        missing = sorted(required - set(loaded))
        if missing:
            raise KeyError(f"split-expert checkpoint is missing tensors: {missing[:5]}")
        value = torch.stack(
            [loaded[expert_key(idx, "down_proj")].T for idx in expert_ids],
            dim=0,
        )
    else:
        required_up = {expert_key(idx, "up_proj") for idx in expert_ids}
        if not loaded:
            return None
        missing_up = sorted(required_up - set(loaded))
        if missing_up:
            raise KeyError(f"split-expert checkpoint is missing tensors: {missing_up[:5]}")
        gate_keys = {expert_key(idx, "gate_proj") for idx in expert_ids}
        present_gate = gate_keys & set(loaded)
        if present_gate and present_gate != gate_keys:
            missing_gate = sorted(gate_keys - present_gate)
            raise KeyError(f"split-expert checkpoint is missing tensors: {missing_gate[:5]}")
        values = []
        for idx in expert_ids:
            up = loaded[expert_key(idx, "up_proj")].T
            if present_gate:
                up = torch.cat((loaded[expert_key(idx, "gate_proj")].T, up), dim=-1)
            values.append(up)
        value = torch.stack(values, dim=0)

    local_slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(global_offset[1:], local_shape[1:])
    )
    value = value[(slice(None), *local_slices)].contiguous()
    if tuple(value.shape) != local_shape:
        raise ValueError(
            "split-expert overlay shape mismatch for "
            f"{checkpoint_key}: checkpoint={tuple(value.shape)} live_local={local_shape}"
        )
    return value, is_local


def _redistribute_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return ``value`` with dtype/device/DTensor layout matching ``target``."""
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except Exception:  # noqa: BLE001
        DTensor = ()  # type: ignore[assignment]
        distribute_tensor = None

    if isinstance(target, DTensor):
        assert distribute_tensor is not None
        if isinstance(value, DTensor):
            return value.to(dtype=target.dtype)
        full_value = value.to(device=target.device, dtype=target.dtype)
        return distribute_tensor(
            full_value,
            device_mesh=target.device_mesh,
            placements=target.placements,
        )
    return value.to(device=target.device, dtype=target.dtype)


def _copy_tensor_value(target: torch.Tensor, value: torch.Tensor) -> None:
    with torch.no_grad():
        target.copy_(_redistribute_like(value, target))


def _copy_overlay_tensor_value(
    target: torch.Tensor,
    value: torch.Tensor,
    *,
    value_is_local: bool,
) -> None:
    """Copy either a global checkpoint tensor or a rank-local overlay slice."""

    if not value_is_local:
        _copy_tensor_value(target, value)
        return
    with torch.no_grad():
        target.to_local().copy_(value.to(device=target.device, dtype=target.dtype))


@contextmanager
def _temporary_parameter_mask(parameter: torch.Tensor | None, keep_mask: torch.Tensor):
    """Mask a possibly distributed parameter in place and restore it exactly.

    Native Mamba kernels consume ``conv1d.weight``/``conv1d.bias`` directly and
    therefore bypass module forward hooks.  The projected B/C channels are
    zero before the fused kernel, but a non-zero convolution bias would bring
    removed state channels back.  A short-lived value mask is consequently
    required for exact elastic prefix semantics.  DTensors are changed through
    their local shard so PP stages do not introduce full-tensor collectives in
    different orders.
    """
    with _temporary_parameter_multiplier(parameter, keep_mask):
        yield


@contextmanager
def _temporary_parameter_multiplier(
    parameter: torch.Tensor | None,
    multiplier: torch.Tensor,
):
    """Apply a reversible leading-dimension multiplier to a native parameter."""

    if parameter is None or bool((multiplier == 1).all()):
        yield
        return

    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    except Exception:  # noqa: BLE001
        DTensor = ()  # type: ignore[assignment]

    if isinstance(parameter, DTensor):
        global_shape = tuple(int(size) for size in parameter.shape)
        if int(multiplier.numel()) != global_shape[0]:
            raise ValueError(
                "parameter multiplier length must match the leading dimension: "
                f"{multiplier.numel()} != {global_shape[0]}"
            )
        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_shape,
            parameter.device_mesh,
            parameter.placements,
        )
        local_parameter = parameter.to_local()
        if tuple(int(size) for size in local_parameter.shape) != tuple(local_shape):
            raise RuntimeError(
                "DTensor local parameter shape disagrees with its placement geometry: "
                f"actual={tuple(local_parameter.shape)} expected={tuple(local_shape)}"
            )
        full_view = multiplier.reshape((-1,) + (1,) * (len(global_shape) - 1)).expand(
            global_shape
        )
        local_slices = tuple(
            slice(int(offset), int(offset) + int(size))
            for offset, size in zip(global_offset, local_shape)
        )
        local_multiplier = full_view[local_slices].to(
            device=local_parameter.device,
            dtype=local_parameter.dtype,
        )
        original_local = local_parameter.detach().clone()
        with torch.no_grad():
            local_parameter.copy_(original_local * local_multiplier)
        try:
            yield
        finally:
            with torch.no_grad():
                local_parameter.copy_(original_local)
        return

    original = _full(parameter.detach()).clone()
    if int(multiplier.numel()) != int(original.shape[0]):
        raise ValueError(
            "parameter multiplier length must match the leading dimension: "
            f"{multiplier.numel()} != {original.shape[0]}"
        )
    view = multiplier.to(device=original.device, dtype=original.dtype).reshape(
        (-1,) + (1,) * (original.ndim - 1)
    )
    _copy_tensor_value(parameter, original * view)
    try:
        yield
    finally:
        _copy_tensor_value(parameter, original)


@contextmanager
def _temporary_attrs(obj, updates: dict[str, object]):
    sentinel = object()
    saved = {key: getattr(obj, key, sentinel) for key in updates}
    try:
        for key, value in updates.items():
            if saved[key] is not sentinel:
                setattr(obj, key, value)
        yield
    finally:
        for key, value in saved.items():
            if value is not sentinel:
                setattr(obj, key, value)


def _masked_native_gate_forward(
    gate,
    *,
    target_num_experts: int | None,
    target_top_k: int | None,
    kept_expert_indices: tuple[int, ...] | None = None,
):
    """Return a Gate.forward replacement that keeps the sorted expert prefix.

    The sorted-teacher contract means expert pruning keeps global ids
    ``[0, target_num_experts)``.  Native AutoModel EP still routes with global
    expert ids, so masking the replicated gate before top-k is the right runtime
    analogue of materializing a smaller router/expert set.
    """

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor = None, cp_mesh=None):
        original_dtype = x.dtype
        compute_dtype = getattr(self, "gate_precision", None)
        x_compute = x.to(dtype=compute_dtype) if compute_dtype is not None else x
        weight = self.weight.to(dtype=compute_dtype or x.dtype)
        bias = getattr(self, "bias", None)
        bias = bias.to(dtype=compute_dtype or x.dtype) if bias is not None else None
        scores = F.linear(x_compute, weight, bias=bias)

        score_func = getattr(self, "score_func", "softmax")
        top_k = int(target_top_k or getattr(self, "topk", 1))
        effective_count = (
            len(kept_expert_indices)
            if kept_expert_indices is not None
            else target_num_experts
        )
        if effective_count is not None:
            top_k = min(top_k, int(effective_count))
        top_k = max(1, top_k)

        def _compact(scores_for_choice):
            if kept_expert_indices is not None:
                keep = torch.as_tensor(
                    kept_expert_indices,
                    dtype=torch.long,
                    device=scores_for_choice.device,
                )
                if keep.numel() == 0 or keep.min() < 0 or keep.max() >= scores_for_choice.shape[-1]:
                    raise ValueError(
                        f"kept_expert_indices={kept_expert_indices} are invalid for "
                        f"num_experts={scores_for_choice.shape[-1]}"
                    )
                scores_for_choice = scores_for_choice.index_select(-1, keep)
            elif target_num_experts is not None and int(target_num_experts) < scores_for_choice.shape[-1]:
                keep = torch.arange(int(target_num_experts), device=scores_for_choice.device)
                scores_for_choice = scores_for_choice[..., : int(target_num_experts)]
            else:
                keep = torch.arange(scores_for_choice.shape[-1], device=scores_for_choice.device)
            return scores_for_choice, keep

        scores, expert_ids = _compact(scores)
        correction = getattr(self, "e_score_correction_bias", None)
        if correction is not None:
            correction = correction.index_select(0, expert_ids)

        if score_func == "softmax":
            if getattr(self, "softmax_before_topk", False):
                probs = scores.softmax(dim=-1, dtype=compute_dtype or torch.float32)
                compact_indices = torch.topk(probs, k=top_k, dim=-1).indices
                weights = probs.gather(1, compact_indices)
            else:
                values, compact_indices = torch.topk(scores, k=top_k, dim=-1)
                weights = values.softmax(dim=1, dtype=compute_dtype or torch.float32)
        elif score_func == "softmax_with_bias":
            probs = scores.softmax(dim=-1, dtype=compute_dtype or torch.float32)
            scores_for_choice = probs
            if correction is not None:
                scores_for_choice = scores_for_choice + correction
            n_groups = int(getattr(self, "n_groups", 1) or 1)
            topk_groups = int(getattr(self, "topk_groups", 1) or 1)
            if n_groups > 1:
                grouped = scores_for_choice.view(x.size(0), n_groups, -1)
                group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(dim=-1)
                group_idx = group_scores.topk(min(topk_groups, group_scores.shape[-1]), dim=-1).indices
                mask = torch.zeros_like(grouped[..., 0]).scatter_(1, group_idx, True)
                scores_for_choice = (grouped * mask.unsqueeze(-1)).flatten(1)
            compact_indices = torch.topk(scores_for_choice, k=top_k, dim=-1).indices
            weights = probs.gather(1, compact_indices)
        elif score_func == "sqrtsoftplus":
            probs = torch.sqrt(F.softplus(scores.float()))
            scores_for_choice = probs if correction is None else probs + correction
            compact_indices = torch.topk(scores_for_choice, k=top_k, dim=-1).indices
            weights = probs.gather(1, compact_indices)
        elif score_func == "sigmoid_with_bias":
            probs = torch.sigmoid(scores.float())
            scores_for_choice = probs if correction is None else probs + correction
            n_groups = int(getattr(self, "n_groups", 1) or 1)
            topk_groups = int(getattr(self, "topk_groups", 1) or 1)
            if n_groups > 1:
                grouped = scores_for_choice.view(x.size(0), n_groups, -1)
                group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(dim=-1)
                group_idx = group_scores.topk(
                    min(topk_groups, group_scores.shape[-1]), dim=-1
                ).indices
                group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
                    1, group_idx, True
                )
                scores_for_choice = grouped.flatten(1).masked_fill(
                    ~group_mask.unsqueeze(-1).expand_as(grouped).flatten(1),
                    float("-inf"),
                )
            compact_indices = torch.topk(scores_for_choice, k=top_k, dim=-1).indices
            weights = probs.gather(1, compact_indices)
        else:
            probs = torch.sigmoid(scores.float())
            scores_for_choice = probs if correction is None else probs + correction
            n_groups = int(getattr(self, "n_groups", 1) or 1)
            topk_groups = int(getattr(self, "topk_groups", 1) or 1)
            if n_groups > 1:
                grouped = scores_for_choice.view(x.size(0), n_groups, -1)
                if correction is None:
                    group_scores = grouped.amax(dim=-1)
                else:
                    group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(dim=-1)
                group_idx = group_scores.topk(min(topk_groups, group_scores.shape[-1]), dim=-1).indices
                mask = torch.zeros_like(grouped[..., 0]).scatter_(1, group_idx, True)
                scores_for_choice = (grouped * mask.unsqueeze(-1)).flatten(1)
            compact_indices = torch.topk(scores_for_choice, k=top_k, dim=-1).indices
            weights = probs.gather(1, compact_indices)

        indices = expert_ids[compact_indices]

        if getattr(self, "norm_topk_prob", False) and top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * float(getattr(self, "route_scale", 1.0))
        return weights.to(dtype=original_dtype), indices, None

    return MethodType(forward, gate)


@contextmanager
def _patched_gate(
    gate,
    *,
    target_num_experts: int | None,
    target_top_k: int | None,
    kept_expert_indices: tuple[int, ...] | None = None,
):
    if gate is None:
        yield
        return
    effective_experts = target_num_experts
    if effective_experts is None:
        weight = getattr(gate, "weight", None)
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            effective_experts = int(weight.shape[0])
        else:
            for key in ("num_experts", "n_routed_experts"):
                value = getattr(gate, key, None)
                if value is not None:
                    effective_experts = int(value)
                    break
    if target_num_experts is not None and int(target_num_experts) <= 0:
        raise ValueError(f"target_num_experts must be positive, got {target_num_experts}")
    if target_top_k is not None and (
        int(target_top_k) <= 0
        or (effective_experts is not None and int(target_top_k) > int(effective_experts))
    ):
        raise ValueError(
            f"target_top_k={target_top_k} must be positive and no larger than "
            f"target_num_experts={effective_experts}"
        )
    attr_updates = {
        key: int(target_top_k)
        for key in ("topk", "top_k", "n_activated_experts", "num_experts_per_tok")
        if target_top_k is not None and hasattr(gate, key)
    }
    saved_attrs = {key: getattr(gate, key) for key in attr_updates}
    saved_forward = getattr(gate, "forward", None)
    should_patch_forward = target_num_experts is not None and hasattr(gate, "weight")
    try:
        if attr_updates:
            for key, value in attr_updates.items():
                setattr(gate, key, value)
        if should_patch_forward:
            gate.forward = _masked_native_gate_forward(
                gate,
                target_num_experts=target_num_experts,
                target_top_k=target_top_k,
                kept_expert_indices=kept_expert_indices,
            )
        yield
    finally:
        if saved_forward is not None:
            gate.forward = saved_forward
        for key, value in saved_attrs.items():
            setattr(gate, key, value)


@contextmanager
def _mask_expert_activation(experts, *, orig_intermediate: int, target_intermediate: int):
    orig = getattr(experts, "expert_activation_grouped", None)
    if orig is None:
        yield
        return

    keep_mask = _bool_prefix_mask(orig_intermediate, target_intermediate)

    def masked_activation(gate_and_up_out, route_weight):
        out = orig(gate_and_up_out, route_weight)
        return out * keep_mask.to(dtype=out.dtype, device=out.device).reshape((1,) * (out.ndim - 1) + (-1,))

    experts.expert_activation_grouped = masked_activation
    try:
        yield
    finally:
        experts.expert_activation_grouped = orig


def _mamba_prefix_masks(
    mamba_module,
    *,
    orig_heads: int,
    orig_head_dim: int,
    orig_groups: int,
    orig_state_dim: int,
    target_heads: int,
    target_head_dim: int,
    target_state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Mask native Nemotron3 ``in_proj`` output before the fused Mamba kernel.

    The native no-cache path passes ``outproj_weight`` directly into
    ``mamba_split_conv1d_scan_combined`` and never calls ``out_proj.forward``.
    Its projected layout is ``[gate, x, B, C, dt]``.  The mask keeps groupwise
    head and state prefixes without changing distributed parameter shapes.
    """
    intermediate = int(getattr(mamba_module, "intermediate_size", orig_heads * orig_head_dim))
    if intermediate != int(orig_heads * orig_head_dim):
        return None
    expected_conv = intermediate + 2 * orig_groups * orig_state_dim
    if int(getattr(mamba_module, "conv_dim", expected_conv)) != expected_conv:
        return None
    logical = Mamba2TensorLayout(
        in_proj_key="in_proj.weight",
        out_proj_key="out_proj.weight",
        conv_weight_key="conv1d.weight",
        conv_bias_key="conv1d.bias",
        norm_key="norm.weight",
        a_log_key="A_log",
        d_key="D",
        dt_bias_key="dt_bias",
        num_heads=orig_heads,
        head_dim=orig_head_dim,
        num_groups=orig_groups,
        state_dim=orig_state_dim,
    )
    projected = mamba2_projected_prefix_mask(
        logical,
        target_heads=target_heads,
        target_head_dim=target_head_dim,
        target_state_dim=target_state_dim,
    )
    state = orig_groups * orig_state_dim
    conv_keep = projected[intermediate : 2 * intermediate + 2 * state]
    return projected[:intermediate], projected, conv_keep


def _supports_compact_fused_mamba(mamba_module) -> bool:
    """Return whether the module uses the native no-cache fused Mamba path."""

    return getattr(mamba_module, "cp", object()) is None and all(
        hasattr(mamba_module, name)
        for name in ("in_proj", "conv1d", "norm", "out_proj")
    )


@contextmanager
def _compact_fused_mamba_forward(
    mamba_module,
    *,
    projected_keep: torch.Tensor,
    conv_keep: torch.Tensor,
    inner_keep: torch.Tensor,
    orig_heads: int,
    target_head_dim: int,
):
    """Run one native fused mixer with the same compact geometry as export."""

    from mamba_ssm.ops.triton import ssd_combined

    projected_indices = projected_keep.nonzero(as_tuple=True)[0]
    conv_indices = conv_keep.nonzero(as_tuple=True)[0]
    inner_indices = inner_keep.nonzero(as_tuple=True)[0]
    head_indices = projected_keep[-orig_heads:].nonzero(as_tuple=True)[0]
    original_forward = mamba_module.forward

    def compact_forward(module, *args, **kwargs):
        original_kernel = ssd_combined.mamba_split_conv1d_scan_combined

        def compact_kernel(
            projected,
            conv_weight,
            conv_bias,
            dt_bias,
            A,
            *kernel_args,
            **kernel_kwargs,
        ):
            def index_first(tensor, indices):
                if tensor is None:
                    return None
                return tensor.index_select(0, indices.to(tensor.device)).contiguous()

            def index_last(tensor, indices):
                if tensor is None:
                    return None
                return tensor.index_select(-1, indices.to(tensor.device)).contiguous()

            kernel_kwargs["D"] = index_first(kernel_kwargs.get("D"), head_indices)
            kernel_kwargs["rmsnorm_weight"] = index_first(
                kernel_kwargs.get("rmsnorm_weight"), inner_indices
            )
            kernel_kwargs["outproj_weight"] = index_last(
                kernel_kwargs.get("outproj_weight"), inner_indices
            )
            kernel_kwargs["headdim"] = target_head_dim
            return original_kernel(
                index_last(projected, projected_indices),
                index_first(conv_weight, conv_indices),
                index_first(conv_bias, conv_indices),
                index_first(dt_bias, head_indices),
                index_first(A, head_indices),
                *kernel_args,
                **kernel_kwargs,
            )

        ssd_combined.mamba_split_conv1d_scan_combined = compact_kernel
        try:
            return original_forward(*args, **kwargs)
        finally:
            ssd_combined.mamba_split_conv1d_scan_combined = original_kernel

    mamba_module.forward = MethodType(compact_forward, mamba_module)
    try:
        yield
    finally:
        mamba_module.forward = original_forward


def _gdn_prefix_masks(
    *,
    orig_groups: int,
    orig_heads: int,
    orig_key_dim: int,
    orig_value_dim: int,
    target_groups: int,
    target_heads: int,
    target_key_dim: int,
    target_value_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_ratio = orig_heads // orig_groups
    if target_heads % target_groups:
        raise ValueError(
            f"GDN target heads={target_heads} must be divisible by groups={target_groups}"
        )
    target_ratio = target_heads // target_groups
    if target_ratio > orig_ratio:
        raise ValueError(f"GDN target ratio={target_ratio} exceeds teacher ratio={orig_ratio}")
    qkeep = torch.zeros(orig_groups, orig_key_dim, dtype=torch.bool)
    qkeep[:target_groups, :target_key_dim] = True
    vkeep = torch.zeros(orig_heads, orig_value_dim, dtype=torch.bool)
    for group in range(target_groups):
        start = group * orig_ratio
        vkeep[start : start + target_ratio, :target_value_dim] = True
    qflat = qkeep.reshape(-1)
    vflat = vkeep.reshape(-1)
    return torch.cat((qflat, qflat, vflat)), vflat


def _mask_last_dim_forward_hook(mask: torch.Tensor):
    def hook(module, args, output):
        if not torch.is_tensor(output):
            return output
        return output * mask.to(dtype=output.dtype, device=output.device).reshape(
            (1,) * (output.ndim - 1) + (-1,)
        )

    return hook


def _gdn_norm_dim_prehook(mask: torch.Tensor, scale: float):
    """Zero tail value-head-dim channels entering the GDN norm and compensate energy.

    Matches the ``_norm_prefix_prehook`` convention in elastic_supernet: the first
    arg (x, the signal) is zeroed and rescaled; subsequent args (z, gate) are zeroed
    only, so the norm sees the same per-channel energy as a physically smaller model.
    """
    def hook(module, args):
        out = []
        for i, a in enumerate(args):
            if torch.is_tensor(a):
                m = mask.to(dtype=a.dtype, device=a.device).reshape(
                    (1,) * (a.ndim - 1) + (-1,)
                )
                out.append(a * m * scale if i == 0 else a * m)
            else:
                out.append(a)
        return tuple(out)

    return hook


@contextmanager
def _scale_gdn_kernel_query_output(mamba_module: nn.Module, scale: float):
    """Match a physically sliced GDN kernel's implicit query scale.

    GDN kernels normalize Q/K internally and derive the query scale from the
    tensor's last dimension. Runtime masking keeps the teacher tensor shape, so
    a reduced child key dimension needs an explicit output correction. The
    recurrent state is independent of the query scale and must remain unchanged.
    """
    originals = {}

    def scaled_kernel(kernel):
        def wrapped(*args, **kwargs):
            output = kernel(*args, **kwargs)
            if torch.is_tensor(output):
                return output * scale
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                return (output[0] * scale, *output[1:])
            if isinstance(output, list) and output and torch.is_tensor(output[0]):
                return [output[0] * scale, *output[1:]]
            raise TypeError(
                f"Unsupported GDN kernel output from {getattr(kernel, '__name__', type(kernel).__name__)}"
            )

        return wrapped

    for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
        kernel = getattr(mamba_module, name, None)
        if callable(kernel):
            originals[name] = kernel
            setattr(mamba_module, name, scaled_kernel(kernel))
    try:
        yield
    finally:
        for name, kernel in originals.items():
            setattr(mamba_module, name, kernel)


def _mask_last_dim_forward_hook_scaled(mask: torch.Tensor, scale: float):
    """Like ``_mask_last_dim_forward_hook`` but multiplies the output by ``scale``."""
    def hook(module, args, output):
        if not torch.is_tensor(output):
            return output
        return output * mask.to(dtype=output.dtype, device=output.device).reshape(
            (1,) * (output.ndim - 1) + (-1,)
        ) * scale

    return hook


def _scale_tensor_output_forward_hook(scale: float):
    """Scale a tensor module output without touching any model parameter."""
    def hook(module, args, output):
        if not torch.is_tensor(output):
            return output
        return output * scale

    return hook


def _prefix_rms_norm_forward_hook(keep: int):
    """Emulate a physically sliced RMSNorm in a full-width runtime envelope."""

    def hook(module, args, output):
        if not args or not torch.is_tensor(args[0]) or not torch.is_tensor(output):
            return output
        x = args[0][..., :keep]
        compute = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        eps = getattr(
            module,
            "variance_epsilon",
            getattr(module, "eps", getattr(module, "epsilon", 1e-6)),
        )
        normalized = compute * torch.rsqrt(compute.square().mean(dim=-1, keepdim=True) + eps)
        normalized = normalized.to(dtype=x.dtype)
        weight = getattr(module, "weight", None)
        if weight is not None:
            normalized = normalized * weight[:keep].to(dtype=normalized.dtype)
        return torch.cat((normalized, torch.zeros_like(output[..., keep:])), dim=-1)

    return hook


def _layer_runtime_fingerprint(layer: nn.Module) -> dict[str, tuple]:
    """Structural state that every dynamic candidate must restore exactly."""
    # FSDP2 may legitimately replace every DTensor Parameter object when a
    # fully-sharded layer completes its first direct replay forward.  Object
    # identity is therefore not a structural invariant.  The parameter
    # contract (name/global shape/dtype/trainability) and registered
    # runtime hooks are stable and catch the actual slicing/config leaks.
    from torch.distributed.fsdp import FSDPModule

    is_fsdp_module = isinstance(layer, FSDPModule)

    def _dtype_fingerprint(parameter: torch.Tensor) -> str:
        # FSDP2/DTensor parameters can expose their mixed-precision compute
        # view between forward and backward.  That transient dtype is runtime
        # state, not a candidate-owned structural mutation; shape and
        # trainability remain invariant.  Plain parameters still retain dtype
        # as part of the structural contract.
        if is_fsdp_module or (
            hasattr(parameter, "device_mesh") and hasattr(parameter, "placements")
        ):
            return "<distributed-compute-dtype>"
        return str(parameter.dtype)

    parameters = tuple(
        (
            name,
            tuple(parameter.shape),
            _dtype_fingerprint(parameter),
            bool(parameter.requires_grad),
        )
        for name, parameter in layer.named_parameters()
    )
    hooks = []
    for name, module in layer.named_modules():
        hooks.append(
            (
                name,
                tuple(module._forward_pre_hooks),
                tuple(module._forward_hooks),
            )
        )
    return {"parameters": parameters, "hooks": tuple(hooks)}


class ReplaceBlockScoringRecipe(ActivationScoringRecipe):
    """Forward-only recipe that yields per-batch (final hidden, lm_head weight, targets)."""

    def __init__(
        self,
        cfg,
        *,
        pruning_cfg=None,
        eval_iters: int | None = None,
        use_puzzletron_dataloader: bool = True,
        data_cfg: dict | None = None,
    ):
        super().__init__(
            cfg,
            pruning_mixin=None,
            hook_kwargs={},
            pruning_cfg=pruning_cfg,
            eval_iters=eval_iters,
            use_puzzletron_dataloader=use_puzzletron_dataloader,
            data_cfg=data_cfg,
        )
        self._final_norm = None
        self._capture_module = None
        self._lm_head_param: torch.Tensor | None = None
        self._captured_hidden: torch.Tensor | None = None
        self._capture_enabled = True
        self._handle = None

    # ---- setup: reuse parent (no scorers), then locate + hook the capture points ----
    def setup(self):
        super().setup()
        self._final_norm = self._find_final_norm()
        self._lm_head_param = self._find_lm_head_weight()
        # Native PP model forwards may bypass the descriptor final-norm module
        # after their vocabulary head is replaced for forward-only scoring.
        # The last model part still returns the exact hidden tensor that would
        # enter the removed head, so capture that part's output directly.  The
        # cached LM-head marker identifies the last part generically across
        # Qwen, Nemotron, and other causal-LM families.
        if self.pp is not None:
            self._capture_module = next(
                (
                    part
                    for part in self.model_parts
                    if getattr(part, "_puzzletron_removed_lm_head_weight", None)
                    is not None
                ),
                None,
            )
        else:
            self._capture_module = self._final_norm
        if self._capture_module is not None:
            self._handle = self._capture_module.register_forward_hook(self._capture_hook)
        from ...tools.logger import aprint

        aprint(
            "[solution/automodel] this rank "
            f"{'OWNS' if self.has_outputs else 'does not own'} the final hidden/lm_head"
        )

    @property
    def has_outputs(self) -> bool:
        """True on the rank(s) that hold the final norm + LM head (the last pipeline stage)."""
        return self._capture_module is not None and self._lm_head_param is not None

    def _descriptor_cls(self):
        descriptor_name = None
        if self._pruning_cfg is not None:
            descriptor_name = self._pruning_cfg.get("descriptor", None)
        if descriptor_name is None:
            model_cfg = getattr(self.cfg, "model", None)
            descriptor_name = getattr(model_cfg, "anymodel_descriptor", None)
        if descriptor_name is None:
            return None
        from ...anymodel.model_descriptor import ModelDescriptorFactory

        return ModelDescriptorFactory.get(descriptor_name)

    def _find_final_norm(self):
        """The descriptor-declared model-level final RMSNorm, not per-layer norms."""
        descriptor = self._descriptor_cls()
        if descriptor is not None:
            canonical_name = descriptor.final_norm_name()
            if canonical_name:
                for part in self.model_parts:
                    names = [str(canonical_name)]
                    adapt_name = getattr(descriptor, "adapt_module_name_for_model", None)
                    if callable(adapt_name):
                        names.append(str(adapt_name(str(canonical_name), part)))
                    for name in dict.fromkeys(names):
                        try:
                            return part.get_submodule(name)
                        except AttributeError:
                            continue
        final_norm_leafs = {"norm"}
        if descriptor is not None:
            final_norm_name = descriptor.final_norm_name()
            if final_norm_name:
                final_norm_leafs.add(str(final_norm_name).rsplit(".", 1)[-1])
        fallback = []
        for part in self.model_parts:
            for name, module in part.named_modules():
                leaf = name.rsplit(".", 1)[-1]
                components = set(name.split("."))
                if (
                    leaf in final_norm_leafs
                    and ".layers." not in name
                    and "mtp" not in components
                ):
                    fallback.append((len(name.split(".")), name, module))
        return min(fallback, default=(None, None, None))[2]

    def _find_lm_head_weight(self):
        """LM-head weight ``[vocab, d]`` gathered to full; falls back to the tied embedding."""
        for part in self.model_parts:
            cached = getattr(part, "_puzzletron_removed_lm_head_weight", None)
            if cached is not None:
                return _full(cached.detach())
            for name, module in part.named_modules():
                if name.rsplit(".", 1)[-1] == "lm_head" and hasattr(module, "weight"):
                    return _full(module.weight.detach())
        # Tied embeddings: lm_head shares embed_tokens.weight (present iff this stage has it).
        descriptor = self._descriptor_cls()
        embedding_leafs = {"embed_tokens", "embeddings"}
        if descriptor is not None:
            input_embedding_name = descriptor.input_embedding_name()
            if input_embedding_name:
                embedding_leafs.add(str(input_embedding_name).rsplit(".", 1)[-1])
        for part in self.model_parts:
            for name, module in part.named_modules():
                if name.rsplit(".", 1)[-1] in embedding_leafs and hasattr(module, "weight"):
                    return _full(module.weight.detach())
        return None

    def _capture_hook(self, module, args, output):
        if not self._capture_enabled:
            return
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden):
            for field in ("last_hidden_state", "logits", "hidden_states"):
                value = getattr(hidden, field, None)
                if torch.is_tensor(value):
                    hidden = value
                    break
        if not torch.is_tensor(hidden):
            raise RuntimeError(
                "native solution scoring could not extract a hidden tensor from "
                f"{type(output).__name__} returned by {type(module).__name__}"
            )
        hidden = _full(hidden).detach()
        if self._captured_hidden is None:
            self._captured_hidden = hidden
        else:
            if self._captured_hidden.shape[1:] != hidden.shape[1:]:
                raise RuntimeError(
                    "pipeline microbatch hidden-state shapes disagree outside the batch "
                    f"dimension: first={tuple(self._captured_hidden.shape)} "
                    f"next={tuple(hidden.shape)}"
                )
            self._captured_hidden = torch.cat((self._captured_hidden, hidden), dim=0)

    def lm_head_weight(self) -> torch.Tensor | None:
        return self._lm_head_param

    def _local_targets_for_batch(self, batch, *, device=None):
        """Return the labels matching the DP/CP-local hidden states produced by ``_forward_batch``."""
        if self._data_spec is not None:
            targets = self._last_canonical_labels
            return targets.to(device) if targets is not None and device is not None else targets
        batch = self._dp_slice_batch(batch)
        targets = batch.get("targets", batch.get("labels"))
        if targets is None:
            return None
        cp_rank, cp_size = self._cp_info()
        targets = self._shard_seq_for_cp(targets, cp_rank, cp_size)
        return targets.to(device) if device is not None else targets

    def _metric_masks_for_batch(self, batch, *, device=None):
        if self._data_spec is not None:
            masks = {
                "ce_mask": self._last_canonical_ce_mask,
                "kd_mask": self._last_canonical_kd_mask,
                "hidden_mask": self._last_canonical_hidden_mask,
            }
            return {
                name: value.to(device) if value is not None and device is not None else value
                for name, value in masks.items()
            }
        batch = self._dp_slice_batch(batch)
        targets = batch.get("targets", batch.get("labels"))
        if targets is None:
            return {"ce_mask": None, "kd_mask": None, "hidden_mask": None}
        cp_rank, cp_size = self._cp_info()
        targets = self._shard_seq_for_cp(targets, cp_rank, cp_size)
        ce_mask = targets.ne(-100)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            hidden_mask = self._shard_seq_for_cp(attention_mask, cp_rank, cp_size).bool()
        else:
            hidden_mask = torch.ones_like(targets, dtype=torch.bool)
        masks = {"ce_mask": ce_mask, "kd_mask": ce_mask, "hidden_mask": hidden_mask}
        return {
            name: value.to(device) if device is not None else value
            for name, value in masks.items()
        }

    def current_metric_masks(self) -> dict[str, torch.Tensor | None]:
        return dict(getattr(self, "_current_metric_masks", {}))

    def _trim_pp_padding(self, hidden, targets, masks):
        valid_rows = int(getattr(self, "_last_unpadded_batch_size", hidden.shape[0]))
        hidden = hidden[:valid_rows]
        targets = None if targets is None else targets[:valid_rows]
        masks = {
            name: None if value is None else value[:valid_rows]
            for name, value in masks.items()
        }
        return hidden, targets, masks

    # ---- forward driver: yield captures per batch (last stage), None elsewhere ----
    def iterate_captures(self):
        """Generator over the calibration set, yielding ``(hidden, targets)`` per batch.

        ``hidden`` is the full ``[b, t, d]`` final hidden state and ``targets`` the next-token
        labels; both are ``None`` on non-output stages (which still run the collective forward).
        Mirrors ``ActivationScoringRecipe.run_scoring``: a hook-disabled warmup primes the PP
        shapes, then one batch per iteration up to ``eval_iters``.
        """
        total = self._eval_iters
        if self.pp is not None:
            self._capture_enabled = False
            with torch.no_grad():
                self._forward_batch(next(iter(self.dataloader)))
            self._capture_enabled = True

        with torch.no_grad():
            for step, batch in enumerate(self.dataloader):
                if total is not None and step >= total:
                    break
                self._captured_hidden = None
                self._forward_batch(batch)
                if self.has_outputs:
                    device = self._captured_hidden.device if self._captured_hidden is not None else None
                    targets = self._local_targets_for_batch(batch, device=device)
                    self._current_metric_masks = self._metric_masks_for_batch(
                        batch,
                        device=device,
                    )
                    hidden, targets, self._current_metric_masks = self._trim_pp_padding(
                        self._captured_hidden,
                        targets,
                        self._current_metric_masks,
                    )
                    yield hidden, targets
                else:
                    self._current_metric_masks = {}
                    yield None, None

    # ---- dynamic single-block prune on the loaded (sorted) model ----
    def _find_decoder_layer(self, layer_idx: int):
        """Return this stage's decoder layer for global ``layer_idx`` (or None if not owned)."""
        descriptor = self._descriptor_cls()
        canonical_name = descriptor.layer_block_name(int(layer_idx))
        for part in self.model_parts:
            adapted_name = descriptor.adapt_module_name_for_model(canonical_name, part)
            for module_name in dict.fromkeys((adapted_name, canonical_name)):
                try:
                    return part.get_submodule(module_name)
                except AttributeError:
                    pass
        return None

    @staticmethod
    def _maybe_submodule(layer, name: str):
        try:
            return layer.get_submodule(name)
        except AttributeError:
            return None

    @contextmanager
    def block_checkpoint_overlay_context(
        self,
        checkpoint_dir: str | Path | None,
        layer_idx: int,
        *,
        offload_restore: bool = False,
    ):
        """Temporarily overlay one owned decoder block with tensors from a checkpoint.

        Nested bypass checkpoints contain trained weights for every block.  Replace-one-block
        diagnostics must still evaluate exactly one changed block, so the scorer keeps the sorted
        teacher resident and overlays only the target layer before applying runtime slicing.
        DTensor parameters are re-distributed using the live parameter's mesh/placements, which
        keeps the context compatible with TP/FSDP layouts built by AutoModel.
        """
        if checkpoint_dir is None:
            yield
            return

        layer = self._find_decoder_layer(int(layer_idx))
        if layer is None:
            yield
            return

        descriptor = self._descriptor_cls()
        if descriptor is None:
            raise ValueError("block checkpoint overlay requires an AnyModel descriptor")
        checkpoint_prefix = descriptor.layer_block_name(int(layer_idx))

        local_state_keys = set(layer.state_dict().keys())
        tensors_by_name: dict[str, torch.Tensor] = {}
        tensors_by_name.update(dict(layer.named_parameters(recurse=True)))
        tensors_by_name.update(dict(layer.named_buffers(recurse=True)))
        tensors_by_name = {
            name: tensor for name, tensor in tensors_by_name.items() if name in local_state_keys
        }
        # Some native AnyModel implementations keep numerically sensitive
        # parameters in an internal fp32 holder while their HF state-dict
        # adapter deliberately publishes the canonical bare key.  Overlay from
        # the published HF checkpoint, not from private native module names.
        checkpoint_key_by_local_name = {
            name: f"{checkpoint_prefix}.{name.replace('._fp32_params.', '.')}"
            for name in tensors_by_name
        }
        checkpoint_keys = set(checkpoint_key_by_local_name.values())
        loaded = _load_checkpoint_tensors(checkpoint_dir, checkpoint_keys)
        loaded_by_local_name: dict[str, tuple[torch.Tensor, bool]] = {}
        missing = []
        for local_name, target in tensors_by_name.items():
            checkpoint_key = checkpoint_key_by_local_name[local_name]
            if checkpoint_key in loaded:
                loaded_by_local_name[local_name] = (loaded[checkpoint_key], False)
                continue
            split_expert = _load_split_expert_overlay_tensor(
                checkpoint_dir,
                checkpoint_key,
                target,
            )
            if split_expert is None:
                missing.append(checkpoint_key)
            else:
                loaded_by_local_name[local_name] = split_expert
        if missing:
            raise KeyError(
                f"bypass overlay checkpoint {checkpoint_dir} is missing {len(missing)} tensor(s) "
                f"for layer {layer_idx}; first missing: {sorted(missing)[:5]}"
            )

        saved: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
        try:
            for local_name, target in tensors_by_name.items():
                checkpoint_key = checkpoint_key_by_local_name[local_name]
                value, value_is_local = loaded_by_local_name[local_name]
                expected_shape = (
                    tuple(target.to_local().shape) if value_is_local else tuple(target.shape)
                )
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        "bypass overlay shape mismatch for "
                        f"{checkpoint_key}: checkpoint={tuple(value.shape)} live={expected_shape}"
                    )
                if offload_restore:
                    _, _, is_distributed = _local_tensor_geometry(target)
                    original = (
                        target.to_local().detach().cpu().clone()
                        if is_distributed
                        else target.detach().cpu().clone()
                    )
                    saved.append((target, original, is_distributed))
                else:
                    saved.append((target, target.detach().clone(), False))
                _copy_overlay_tensor_value(target, value, value_is_local=value_is_local)
            yield
        finally:
            for target, original, original_is_local in reversed(saved):
                _copy_overlay_tensor_value(
                    target,
                    original,
                    value_is_local=original_is_local,
                )

    @contextmanager
    def prune_block_context(
        self,
        layer_idx: int,
        *,
        teacher_block_config=None,
        child_block_config=None,
        orig_intermediate=None,
        target_intermediate=None,
        orig_num_q=None,
        orig_num_kv=None,
        target_num_q=None,
        target_num_kv=None,
        head_dim=None,
        expert_keep_ids=None,
    ):
        """Make one block behave as pruned for the enclosed forward(s), then restore.

        On the owning PP stage: masks down_proj (FFN removal) / o_proj (attention removal) inputs.
        No-op on other stages (the pruned
        activations still propagate through the pipeline to the scoring stage). Sorted-teacher
        targets are prefix slices, so removal masks the prefix.

        Spec selection is shared with the realize side via ``build_block_prune_specs`` so the two
        paths cannot drift; this method only resolves the specs' module names on the owning layer.
        """
        from ...pruning.dynamic_block_prune import (
            AttnRemovalSpec,
            FFNRemovalSpec,
            build_block_prune_specs,
            register_mask_hook,
        )

        layer = self._find_decoder_layer(layer_idx)
        if layer is None:
            yield
            return

        baseline_fingerprint = _layer_runtime_fingerprint(layer)

        down_proj_name, _ = _first_existing_submodule(layer, ("mlp.down_proj", "mixer.down_proj"))
        o_proj_name, _ = _first_existing_submodule(layer, ("self_attn.o_proj", "mixer.o_proj"))
        specs = build_block_prune_specs(
            down_proj_name=down_proj_name,
            o_proj_name=o_proj_name,
            orig_intermediate=orig_intermediate,
            target_intermediate=target_intermediate if down_proj_name is not None else None,
            orig_num_q=orig_num_q,
            orig_num_kv=orig_num_kv,
            target_num_q=target_num_q,
            target_num_kv=target_num_kv if o_proj_name is not None else None,
            head_dim=head_dim,
        )

        handles = []
        try:
            for spec in specs:
                if isinstance(spec, (FFNRemovalSpec, AttnRemovalSpec)):
                    module = self._maybe_submodule(layer, spec.module_name)
                    if module is not None:
                        handles.append(register_mask_hook(module, spec.keep_mask))
            from ...pruning.runtime_candidate import apply_runtime_candidate

            runtime_candidate_handle = apply_runtime_candidate(
                layer,
                teacher_block_config,
                child_block_config,
                expert_keep_ids=expert_keep_ids,
            )
            handles.append(runtime_candidate_handle)
            yield
        finally:
            for h in reversed(handles):
                h.remove()
            restored_fingerprint = _layer_runtime_fingerprint(layer)
            if restored_fingerprint != baseline_fingerprint:
                baseline_parameters = {
                    entry[0]: entry[1:] for entry in baseline_fingerprint["parameters"]
                }
                restored_parameters = {
                    entry[0]: entry[1:] for entry in restored_fingerprint["parameters"]
                }
                changed_parameters = [
                    name
                    for name in sorted(set(baseline_parameters) | set(restored_parameters))
                    if baseline_parameters.get(name) != restored_parameters.get(name)
                ]
                baseline_hooks = {entry[0]: entry[1:] for entry in baseline_fingerprint["hooks"]}
                restored_hooks = {entry[0]: entry[1:] for entry in restored_fingerprint["hooks"]}
                changed_hooks = [
                    name
                    for name in sorted(set(baseline_hooks) | set(restored_hooks))
                    if baseline_hooks.get(name) != restored_hooks.get(name)
                ]
                changed_parameter_details = {
                    name: {
                        "before": baseline_parameters.get(name),
                        "after": restored_parameters.get(name),
                    }
                    for name in changed_parameters
                }
                raise RuntimeError(
                    "dynamic block context leaked structural state after candidate "
                    f"layer={layer_idx} rank={_rank_tag()} "
                    f"changed_parameters={changed_parameter_details} changed_hooks={changed_hooks}"
                )
            print(
                "[solution/automodel] "
                f"{_rank_tag()} restore verified layer={layer_idx}",
                flush=True,
            )

    @contextmanager
    def architecture_context(self, prune_targets):
        """Apply a deterministic set of per-layer dynamic candidates, then restore all."""
        targets = tuple(prune_targets or ())
        layer_ids = [int(target["layer_idx"]) for target in targets]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError(f"architecture context has duplicate layers: {layer_ids}")
        with ExitStack() as stack:
            for target in sorted(targets, key=lambda item: int(item["layer_idx"])):
                stack.enter_context(self.prune_block_context(**target))
            yield

    @contextmanager
    def hidden_width_context(self, width: int | None):
        """Apply one descriptor-defined residual-width slice across locally owned layers."""

        if width is None:
            yield
            return
        descriptor = self._descriptor_cls()
        if descriptor is None:
            raise ValueError("hidden-width scoring requires an AnyModel descriptor")
        config = next(
            (getattr(part, "config", None) for part in self.model_parts if getattr(part, "config", None)),
            None,
        )
        if config is None:
            raise ValueError("hidden-width scoring could not resolve the loaded model config")
        lm = descriptor.get_language_model_config(config)
        source_width = int(lm.hidden_size)
        width = int(width)
        spec = descriptor.embedding_pruning_spec(
            config,
            widths=(source_width, width),
            alignment=1,
        )
        block_configs = getattr(lm, "block_configs", getattr(config, "block_configs", ()))
        from ...pruning.runtime_hidden_width import (
            hidden_width_layer_context,
            hidden_width_module_context,
        )

        with ExitStack() as stack:
            for layer_idx in range(len(block_configs)):
                layer = self._find_decoder_layer(layer_idx)
                if layer is None:
                    continue
                stack.enter_context(
                    hidden_width_layer_context(
                        layer,
                        canonical_layer_name=descriptor.layer_block_name(layer_idx),
                        spec=spec,
                        width=width,
                    )
                )
            if self._final_norm is not None:
                stack.enter_context(
                    hidden_width_module_context(
                        self._final_norm,
                        canonical_module_name=descriptor.final_norm_name(),
                        spec=spec,
                        width=width,
                    )
                )
            yield

    @staticmethod
    def _typed_subblock_runtime_hooks(
        layer,
        *,
        teacher_block_config=None,
        child_block_config=None,
        expert_keep_ids=None,
    ):
        """Runtime slicing hooks for typed MoE/Mamba candidates on the sorted teacher.

        These hooks deliberately operate on generic subblock semantics rather
        than model-family names: MoE expert pruning is gate-prefix masking,
        MoE channel/latent pruning masks intermediate activations, and Mamba
        head/head-dim pruning masks the sorted inner state before ``out_proj``.
        """

        if teacher_block_config is None or child_block_config is None:
            return [], []
        handles = []
        contexts = []

        teacher_ffn = teacher_block_config.get_subblock("ffn")
        child_ffn = child_block_config.get_subblock("ffn")
        if (
            teacher_ffn is not None
            and child_ffn is not None
            and not getattr(teacher_ffn, "no_op", False)
            and getattr(child_ffn, "no_op", False)
        ):
            ffn_module = getattr(layer, "mlp", None)
            if ffn_module is None:
                candidate = getattr(layer, "mixer", None)
                if candidate is not None and hasattr(candidate, "down_proj"):
                    ffn_module = candidate
            if ffn_module is None:
                raise RuntimeError(
                    f"FFN no-op requested but no FFN module was found on {type(layer).__name__}"
                )
            handles.append(ffn_module.register_forward_hook(_module_output_zero_hook))

        teacher_attention = teacher_block_config.get_subblock("attention")
        child_attention = child_block_config.get_subblock("attention")
        if teacher_attention is not None and child_attention is not None:
            teacher_window = getattr(teacher_attention, "sliding_window_size", None)
            child_window = getattr(child_attention, "sliding_window_size", None)
            if child_window is not None and child_window != teacher_window:
                contexts.append(_temporary_attention_window(layer, child_window))
        if (
            teacher_attention is not None
            and child_attention is not None
            and not getattr(teacher_attention, "no_op", False)
            and getattr(child_attention, "no_op", False)
        ):
            attention_module = getattr(layer, "self_attn", None)
            if attention_module is None:
                raise RuntimeError(
                    "attention no-op requested but no self_attn module was found on "
                    f"{type(layer).__name__}"
                )
            handles.append(attention_module.register_forward_hook(_module_output_zero_hook))

        teacher_mla = teacher_block_config.get_subblock("mla")
        child_mla = child_block_config.get_subblock("mla")
        if teacher_mla is not None and child_mla is not None:
            mla_module = getattr(layer, "self_attn", None)
            if mla_module is None:
                raise RuntimeError(
                    f"MLA candidate requested but no self_attn module was found on {type(layer).__name__}"
                )
            if (
                not getattr(teacher_mla, "no_op", False)
                and getattr(child_mla, "no_op", False)
            ):
                handles.append(mla_module.register_forward_hook(_module_output_zero_hook))
            else:
                original_heads = getattr(teacher_mla, "num_heads", None)
                target_heads = getattr(child_mla, "num_heads", None)
                if (
                    original_heads is not None
                    and target_heads is not None
                    and target_heads < original_heads
                ):
                    o_proj = getattr(mla_module, "o_proj", None)
                    if o_proj is None or not hasattr(o_proj, "weight"):
                        raise RuntimeError(
                            f"MLA head slicing requires o_proj on {type(mla_module).__name__}"
                        )
                    input_features = int(o_proj.weight.shape[1])
                    if input_features % int(original_heads):
                        raise RuntimeError(
                            f"MLA o_proj input features={input_features} are not divisible by "
                            f"teacher heads={original_heads}"
                        )
                    features_per_head = input_features // int(original_heads)
                    keep = torch.zeros(input_features, dtype=torch.bool)
                    keep[: int(target_heads) * features_per_head] = True
                    from ...pruning.dynamic_block_prune import register_mask_hook

                    handles.append(register_mask_hook(o_proj, keep))
                for rank_field, norm_name in (
                    ("q_lora_rank", "q_a_layernorm"),
                    ("kv_lora_rank", "kv_a_layernorm"),
                ):
                    original_rank = getattr(teacher_mla, rank_field, None)
                    target_rank = getattr(child_mla, rank_field, None)
                    if (
                        original_rank is None
                        or target_rank is None
                        or target_rank >= original_rank
                    ):
                        continue
                    norm = getattr(mla_module, norm_name, None)
                    if norm is None:
                        raise RuntimeError(
                            f"MLA {rank_field} slicing requires {norm_name} on "
                            f"{type(mla_module).__name__}"
                        )
                    handles.append(
                        norm.register_forward_hook(
                            _prefix_rms_norm_forward_hook(int(target_rank))
                        )
                    )

        teacher_moe = teacher_block_config.get_subblock("moe")
        child_moe = child_block_config.get_subblock("moe")
        if teacher_moe is not None and child_moe is not None:
            moe_module = None
            for candidate_name in ("mixer", "mlp"):
                candidate = getattr(layer, candidate_name, None)
                if candidate is not None and (
                    hasattr(candidate, "gate")
                    or hasattr(candidate, "experts")
                    or hasattr(candidate, "fc1_latent_proj")
                ):
                    moe_module = candidate
                    break
            if moe_module is not None:
                if (
                    not getattr(teacher_moe, "no_op", False)
                    and getattr(child_moe, "no_op", False)
                ):
                    handles.append(moe_module.register_forward_hook(_module_output_zero_hook))
                else:
                    orig_experts = getattr(teacher_moe, "num_experts", None)
                    target_experts = getattr(child_moe, "num_experts", None)
                    target_top_k = getattr(child_moe, "top_k", None)
                    if target_experts is not None and orig_experts is not None and target_experts >= orig_experts:
                        target_experts = None
                    if target_top_k is not None and getattr(teacher_moe, "top_k", None) == target_top_k:
                        target_top_k = None
                    if target_experts is not None or target_top_k is not None:
                        contexts.append(
                            _patched_gate(
                                getattr(moe_module, "gate", None),
                                target_num_experts=target_experts,
                                target_top_k=target_top_k,
                                kept_expert_indices=(
                                    tuple(int(item) for item in expert_keep_ids)
                                    if expert_keep_ids is not None
                                    else None
                                ),
                            )
                        )

                    orig_inter = getattr(teacher_moe, "expert_intermediate_size", None)
                    target_inter = getattr(child_moe, "expert_intermediate_size", None)
                    experts = getattr(moe_module, "experts", None)
                    if (
                        experts is not None
                        and orig_inter is not None
                        and target_inter is not None
                        and target_inter < orig_inter
                    ):
                        if hasattr(experts, "expert_activation_grouped"):
                            contexts.append(
                                _mask_expert_activation(
                                    experts,
                                    orig_intermediate=orig_inter,
                                    target_intermediate=target_inter,
                                )
                            )
                        elif isinstance(experts, nn.ModuleList):
                            keep_mask = _bool_prefix_mask(orig_inter, target_inter)
                            for expert in experts:
                                down = getattr(expert, "down_proj", None)
                                if down is not None:
                                    handles.append(down.register_forward_pre_hook(lambda module, args, km=keep_mask: (
                                        args[0] * km.to(dtype=args[0].dtype, device=args[0].device).reshape((1,) * (args[0].ndim - 1) + (-1,)),
                                        *args[1:],
                                    )))

                    orig_shared = getattr(teacher_moe, "shared_expert_intermediate_size", None)
                    target_shared = getattr(child_moe, "shared_expert_intermediate_size", None)
                    shared = getattr(moe_module, "shared_experts", None)
                    shared_down = getattr(shared, "down_proj", None) if shared is not None else None
                    if (
                        shared_down is not None
                        and orig_shared is not None
                        and target_shared is not None
                        and target_shared < orig_shared
                    ):
                        handles.append(
                            shared_down.register_forward_pre_hook(
                                lambda module, args, km=_bool_prefix_mask(orig_shared, target_shared): (
                                    args[0]
                                    * km.to(dtype=args[0].dtype, device=args[0].device).reshape(
                                        (1,) * (args[0].ndim - 1) + (-1,)
                                    ),
                                    *args[1:],
                                )
                            )
                        )

                    orig_latent = getattr(teacher_moe, "latent_dim", None)
                    target_latent = getattr(child_moe, "latent_dim", None)
                    fc1_latent = getattr(moe_module, "fc1_latent_proj", None)
                    fc2_latent = getattr(moe_module, "fc2_latent_proj", None)
                    if (
                        fc2_latent is not None
                        and orig_latent is not None
                        and target_latent is not None
                        and target_latent < orig_latent
                    ):
                        latent_keep = _bool_prefix_mask(orig_latent, target_latent)
                        if fc1_latent is not None:
                            handles.append(
                                fc1_latent.register_forward_hook(
                                    _mask_last_dim_forward_hook(latent_keep)
                                )
                            )
                        handles.append(
                            fc2_latent.register_forward_pre_hook(
                                lambda module, args, km=latent_keep: (
                                    args[0]
                                    * km.to(dtype=args[0].dtype, device=args[0].device).reshape(
                                        (1,) * (args[0].ndim - 1) + (-1,)
                                    ),
                                    *args[1:],
                                )
                            )
                        )

        teacher_mamba = teacher_block_config.get_subblock("mamba")
        child_mamba = child_block_config.get_subblock("mamba")
        if teacher_mamba is not None and child_mamba is not None:
            mamba_module = (
                getattr(layer, "linear_attn", None)
                or getattr(layer, "mixer", None)
                or getattr(layer, "self_attn", None)
            )
            if mamba_module is not None:
                if (
                    not getattr(teacher_mamba, "no_op", False)
                    and getattr(child_mamba, "no_op", False)
                ):
                    handles.append(mamba_module.register_forward_hook(_module_output_zero_hook))
                else:
                    orig_heads = getattr(teacher_mamba, "num_heads", None)
                    orig_head_dim = getattr(teacher_mamba, "head_dim", None)
                    orig_groups = getattr(teacher_mamba, "num_groups", None)
                    orig_state_dim = getattr(teacher_mamba, "state_dim", None)
                    target_heads = getattr(child_mamba, "num_heads", None) or orig_heads
                    target_head_dim = getattr(child_mamba, "head_dim", None) or orig_head_dim
                    target_state_dim = getattr(child_mamba, "state_dim", None) or orig_state_dim
                    is_gdn = hasattr(mamba_module, "num_k_heads") and hasattr(
                        mamba_module, "in_proj_qkv"
                    )
                    if is_gdn:
                        orig_groups = int(getattr(teacher_mamba, "num_groups"))
                        orig_key_dim = int(getattr(teacher_mamba, "state_dim"))
                        target_groups = int(getattr(child_mamba, "num_groups") or orig_groups)
                        target_key_dim = int(getattr(child_mamba, "state_dim") or orig_key_dim)
                        qkv_keep, value_keep = _gdn_prefix_masks(
                            orig_groups=orig_groups,
                            orig_heads=int(orig_heads),
                            orig_key_dim=orig_key_dim,
                            orig_value_dim=int(orig_head_dim),
                            target_groups=target_groups,
                            target_heads=int(target_heads),
                            target_key_dim=target_key_dim,
                            target_value_dim=int(target_head_dim),
                        )
                        if not bool(qkv_keep.all()):
                            handles.append(
                                mamba_module.in_proj_qkv.register_forward_hook(
                                    _mask_last_dim_forward_hook(qkv_keep)
                                )
                            )
                        if target_key_dim < orig_key_dim:
                            contexts.append(
                                _scale_gdn_kernel_query_output(
                                    mamba_module,
                                    (float(orig_key_dim) / float(target_key_dim)) ** 0.5,
                                )
                            )
                        if not bool(value_keep.all()):
                            handles.append(
                                mamba_module.in_proj_z.register_forward_hook(
                                    _mask_last_dim_forward_hook(value_keep)
                                )
                            )
                            handles.append(
                                mamba_module.out_proj.register_forward_pre_hook(
                                    lambda module, args, km=value_keep: (
                                        args[0]
                                        * km.to(dtype=args[0].dtype, device=args[0].device).reshape(
                                            (1,) * (args[0].ndim - 1) + (-1,)
                                        ),
                                        *args[1:],
                                    )
                                )
                            )
                        # in_proj_a / in_proj_b output one scalar per value-head;
                        # derive the head-level keep mask from the flat value mask.
                        _head_keep = value_keep.reshape(
                            int(orig_heads), int(orig_head_dim)
                        ).any(dim=-1)
                        if not bool(_head_keep.all()):
                            for _pn in ("in_proj_a", "in_proj_b"):
                                _pr = getattr(mamba_module, _pn, None)
                                if _pr is not None:
                                    handles.append(
                                        _pr.register_forward_hook(
                                            _mask_last_dim_forward_hook(_head_keep)
                                        )
                                    )
                        # norm: compensate energy when the value head-dim is reduced so
                        # the normalisation sees the same per-channel variance as a
                        # physically smaller model (matches _apply_gdn_prefix_hooks).
                        if int(target_head_dim) < int(orig_head_dim):
                            _norm = getattr(mamba_module, "norm", None)
                            if _norm is not None:
                                _vd_mask = (
                                    torch.arange(int(orig_head_dim)) < int(target_head_dim)
                                )
                                _scale_in = (
                                    float(orig_head_dim) / float(target_head_dim)
                                ) ** 0.5
                                _scale_out = (
                                    float(target_head_dim) / float(orig_head_dim)
                                ) ** 0.5
                                handles.append(
                                    _norm.register_forward_pre_hook(
                                        _gdn_norm_dim_prehook(_vd_mask, _scale_in)
                                    )
                                )
                                handles.append(
                                    _norm.register_forward_hook(
                                        _mask_last_dim_forward_hook_scaled(
                                            _vd_mask, _scale_out
                                        )
                                    )
                                )
                    if not is_gdn and all(
                        value is not None
                        for value in (
                            orig_heads,
                            orig_head_dim,
                            orig_groups,
                            orig_state_dim,
                            target_heads,
                            target_head_dim,
                            target_state_dim,
                        )
                    ) and (
                        target_heads < orig_heads
                        or target_head_dim < orig_head_dim
                        or target_state_dim < orig_state_dim
                    ):
                        masks = _mamba_prefix_masks(
                            mamba_module,
                            orig_heads=int(orig_heads),
                            orig_head_dim=int(orig_head_dim),
                            orig_groups=int(orig_groups),
                            orig_state_dim=int(orig_state_dim),
                            target_heads=int(target_heads),
                            target_head_dim=int(target_head_dim),
                            target_state_dim=int(target_state_dim),
                        )
                        if masks is None:
                            raise RuntimeError(
                                "Mamba runtime slicing geometry does not match the packed "
                                f"projection on {type(mamba_module).__name__}"
                            )
                        keep_mask, projected_keep, conv_keep = masks
                        if _supports_compact_fused_mamba(mamba_module):
                            contexts.append(
                                _compact_fused_mamba_forward(
                                    mamba_module,
                                    projected_keep=projected_keep,
                                    conv_keep=conv_keep,
                                    inner_keep=keep_mask,
                                    orig_heads=int(orig_heads),
                                    target_head_dim=int(target_head_dim),
                                )
                            )
                        else:
                            in_proj = getattr(mamba_module, "in_proj", None)
                            if in_proj is not None and not bool(projected_keep.all()):
                                handles.append(
                                    in_proj.register_forward_hook(
                                        _mask_last_dim_forward_hook(projected_keep)
                                    )
                                )

                            # Static-shape fallbacks need an RMS compensation because
                            # their normalization still includes masked teacher channels.
                            original_inner = int(orig_heads) * int(orig_head_dim)
                            target_inner = int(target_heads) * int(target_head_dim)
                            norm = getattr(mamba_module, "norm", None)
                            norm_weight = getattr(norm, "weight", None)
                            norm_scale = 1.0
                            if target_inner < original_inner and norm_weight is not None:
                                norm_scale = (float(target_inner) / float(original_inner)) ** 0.5
                                for epsilon_name in ("variance_epsilon", "eps", "epsilon"):
                                    if hasattr(norm, epsilon_name):
                                        epsilon = getattr(norm, epsilon_name)
                                        contexts.append(
                                            _temporary_attrs(
                                                norm,
                                                {
                                                    epsilon_name: float(epsilon)
                                                    * float(target_inner)
                                                    / float(original_inner)
                                                },
                                            )
                                        )
                                        break

                            out_proj = getattr(mamba_module, "out_proj", None)
                            if out_proj is not None and not bool(keep_mask.all()):
                                handles.append(
                                    out_proj.register_forward_pre_hook(
                                        lambda module, args, km=keep_mask: (
                                            args[0]
                                            * km.to(
                                                dtype=args[0].dtype, device=args[0].device
                                            ).reshape((1,) * (args[0].ndim - 1) + (-1,)),
                                            *args[1:],
                                        )
                                    )
                                )
                            can_scale_output = (
                                out_proj is not None
                                and getattr(out_proj, "bias", None) is None
                            )
                            if norm_scale != 1.0 and can_scale_output:
                                handles.append(
                                    mamba_module.register_forward_hook(
                                        _scale_tensor_output_forward_hook(norm_scale)
                                    )
                                )
                            if in_proj is None and out_proj is None:
                                raise RuntimeError(
                                    "Mamba runtime slicing requested but neither in_proj nor "
                                    f"out_proj exists on {type(mamba_module).__name__}"
                                )

                            if norm_scale != 1.0 and not can_scale_output:
                                multiplier = torch.ones(original_inner, dtype=torch.float64)
                                multiplier[keep_mask] = norm_scale
                                contexts.append(
                                    _temporary_parameter_multiplier(norm_weight, multiplier)
                                )

                            conv1d = getattr(mamba_module, "conv1d", None)
                            conv_bias = (
                                getattr(conv1d, "bias", None) if conv1d is not None else None
                            )
                            if conv_bias is not None and not bool(conv_keep.all()):
                                contexts.append(
                                    _temporary_parameter_mask(conv_bias, conv_keep)
                                )

        return handles, contexts

    def teardown_capture(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        # Drop the gathered LM-head weight (~[vocab, d]) and the last captured hidden so they
        # are freed before the next model is built (the teacher cache already holds CPU copies).
        self._lm_head_param = None
        self._captured_hidden = None
        self._final_norm = None
        self._capture_module = None
