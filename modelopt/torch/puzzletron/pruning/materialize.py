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

"""Materialize a pruned variant from the sorted teacher — slice/merge, no realized checkpoints.

This is the realize-side counterpart of :func:`.sorted_teacher.sort_state_dict` and the static
twin of the dynamic prune (:mod:`.dynamic_block_prune`): given the sorted teacher's state dict and a
per-layer target ``(ffn K / attn (q, kv))``, it produces the **physically smaller** weights for the
chosen variant by prefix-slicing (removal) or merging (q-preserving KV reduction). It backs the
block library (a variant = a slice/merge spec into the sorted teacher, not realized ``weight_paths``)
and the bypass / final-realize steps.

Because the sorted teacher orders channels/heads by importance, slicing ``[:K]`` (and the blocked
attention keep-set) yields exactly the importance-pruned weights. Reuses the model-agnostic
primitives in :mod:`.attention_ffn_surgery`; the descriptor supplies the per-layer keys/geometry via
:func:`.sorted_teacher.build_layer_layouts`.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from ..block_config import AttentionConfig, BlockConfig, FFNConfig, MLAConfig, MambaConfig, MoEConfig
from .attention_ffn_surgery import (
    slice_query_rows_by_head,
    sorted_attention_keep_indices,
)
from .gated_delta_net import GDNShape, slice_gated_delta_net_state_dict
from .mamba2_surgery import Mamba2TensorLayout, slice_mamba2_state_dict

__all__ = [
    "BlockTarget",
    "materialize_solution_state_dict",
    "block_targets_from_replacements",
    "materialize_model_from_sorted",
    "materialize_checkpoint_from_sorted",
    "materialize_hidden_width_checkpoint",
]

_WEIGHT = ".weight"


def _shard_requires_rewrite(
    tensor_keys,
    *,
    target_layer_prefixes: tuple[str, ...],
    global_slice: bool,
) -> bool:
    """Return whether a checkpoint shard can change under this realization.

    Layer-local replacements only touch tensors below the replaced layer prefixes.
    Hidden-width and PLE slicing are model-global and therefore conservatively
    rewrite every shard.
    """

    if global_slice:
        return True
    prefixes = tuple(f"{prefix}." for prefix in target_layer_prefixes)
    return any(str(key).startswith(prefixes) for key in tensor_keys) if prefixes else False


@dataclass
class BlockTarget:
    """Target dims for one layer's replaced subblock (None = unchanged for that axis)."""

    target_intermediate: int | None = None
    target_num_q: int | None = None
    target_num_kv: int | None = None
    target_mla_heads: int | None = None
    target_q_lora_rank: int | None = None
    target_kv_lora_rank: int | None = None
    target_num_experts: int | None = None
    expert_keep_indices: tuple[int, ...] | None = None
    target_expert_intermediate: int | None = None
    target_shared_expert_intermediate: int | None = None
    target_latent_dim: int | None = None
    target_mamba_heads: int | None = None
    target_mamba_groups: int | None = None
    target_mamba_head_dim: int | None = None
    target_mamba_state_dim: int | None = None
    remove_ffn: bool = False
    remove_attention: bool = False
    remove_mamba: bool = False
    remove_moe: bool = False


def _bias_key(weight_key):
    return None if weight_key is None else weight_key[: -len(_WEIGHT)] + ".bias"


def _slice_gated_up(up: torch.Tensor, keep: torch.Tensor, intermediate: int) -> torch.Tensor:
    if up.shape[0] == 2 * intermediate:
        return up[torch.cat([keep, keep + intermediate])]
    return up[keep]


def _newly_removed_subblock(child, teacher) -> bool:
    """Return whether an active teacher subblock became a child no-op."""

    return bool(
        child is not None
        and child.no_op
        and teacher is not None
        and not teacher.no_op
    )


def block_targets_from_replacements(
    layer_replacements,
    teacher_block_configs,
    num_attention_heads: int,
) -> dict[int, BlockTarget]:
    """Convert replacement-library ``layer_replacements`` -> ``{layer_idx: BlockTarget}``.

    Each replacement pairs ``parent_layer_indices`` with ``child_block_configs`` (parsed
    ``BlockConfig`` objects). The child's FFN ``intermediate_size`` / attention
    ``num_query_heads`` + ``num_kv_heads`` become the per-layer target dims that
    :func:`materialize_solution_state_dict` slices. If a child only reduces
    ``num_kv_heads``, materialization also reduces ``num_query_heads`` by removing
    the corresponding sorted query-head groups.
    """
    targets: dict[int, BlockTarget] = {}
    for rep in layer_replacements:
        diagnostic = rep.get("diagnostic") or {}
        for layer_idx, child in zip(rep["parent_layer_indices"], rep["child_block_configs"]):
            layer_idx = int(layer_idx)
            if isinstance(child, dict):
                child = BlockConfig(**child)
            if not isinstance(child, BlockConfig):
                raise TypeError(f"Expected BlockConfig child replacement, got {type(child).__name__}")
            teacher_block_config = teacher_block_configs[layer_idx]
            if isinstance(teacher_block_config, dict):
                teacher_block_config = BlockConfig(**teacher_block_config)
            if child.to_dict() == teacher_block_config.to_dict():
                continue
            ffn = child.get_subblock("ffn")
            attn = child.get_subblock("attention")
            if ffn is not None and not isinstance(ffn, FFNConfig):
                raise TypeError(f"Expected FFNConfig for 'ffn', got {type(ffn).__name__}")
            if attn is not None and not isinstance(attn, AttentionConfig):
                raise TypeError(f"Expected AttentionConfig for 'attention', got {type(attn).__name__}")
            moe = child.get_subblock("moe")
            mamba = child.get_subblock("mamba")
            mla = child.get_subblock("mla")
            if moe is not None and not isinstance(moe, MoEConfig):
                raise TypeError(f"Expected MoEConfig for 'moe', got {type(moe).__name__}")
            if mamba is not None and not isinstance(mamba, MambaConfig):
                raise TypeError(f"Expected MambaConfig for 'mamba', got {type(mamba).__name__}")
            if mla is not None and not isinstance(mla, MLAConfig):
                raise TypeError(f"Expected MLAConfig for 'mla', got {type(mla).__name__}")
            t_kv = attn.num_kv_heads if attn is not None else None
            t_q = attn.num_query_heads if attn is not None else None
            teacher_attn = teacher_block_config.get_subblock("attention")
            if teacher_attn is not None and not isinstance(teacher_attn, AttentionConfig):
                raise TypeError(
                    f"Expected AttentionConfig for teacher 'attention', got {type(teacher_attn).__name__}"
                )
            orig_kv = teacher_attn.num_kv_heads if teacher_attn is not None else None
            teacher_ffn = teacher_block_config.get_subblock("ffn")
            teacher_mamba = teacher_block_config.get_subblock("mamba")
            teacher_moe = teacher_block_config.get_subblock("moe")
            if t_kv is not None:
                if t_q is None and orig_kv is not None and t_kv < orig_kv:
                    t_q = t_kv * (num_attention_heads // orig_kv)
                elif t_q is None:
                    t_q = num_attention_heads
            targets[layer_idx] = BlockTarget(
                target_intermediate=ffn.intermediate_size if ffn is not None else None,
                target_num_q=t_q,
                target_num_kv=t_kv,
                target_mla_heads=mla.num_heads if mla is not None else None,
                target_q_lora_rank=mla.q_lora_rank if mla is not None else None,
                target_kv_lora_rank=mla.kv_lora_rank if mla is not None else None,
                target_num_experts=moe.num_experts if moe is not None else None,
                # ``sorted_dir`` has already moved the ranked original expert
                # ids into its leading prefix.  The ids remain useful report
                # provenance, but indexing the sorted tensors by them again
                # would select a different expert set.
                expert_keep_indices=None,
                target_expert_intermediate=moe.expert_intermediate_size if moe is not None else None,
                target_shared_expert_intermediate=(
                    moe.shared_expert_intermediate_size if moe is not None else None
                ),
                target_latent_dim=moe.latent_dim if moe is not None else None,
                target_mamba_heads=mamba.num_heads if mamba is not None else None,
                target_mamba_groups=mamba.num_groups if mamba is not None else None,
                target_mamba_head_dim=mamba.head_dim if mamba is not None else None,
                target_mamba_state_dim=mamba.state_dim if mamba is not None else None,
                remove_ffn=_newly_removed_subblock(ffn, teacher_ffn),
                remove_attention=_newly_removed_subblock(attn, teacher_attn),
                remove_mamba=_newly_removed_subblock(mamba, teacher_mamba),
                remove_moe=_newly_removed_subblock(moe, teacher_moe),
            )
    return targets


def _load_sorted_state(sorted_dir) -> dict[str, torch.Tensor]:
    """Read the full state dict from a sorted-teacher checkpoint."""
    from pathlib import Path

    from safetensors.torch import load_file

    from .sorted_teacher import iter_safetensor_weight_files

    sorted_dir = Path(sorted_dir)
    state: dict[str, torch.Tensor] = {}
    for rel_path in iter_safetensor_weight_files(sorted_dir):
        state.update(load_file(str(sorted_dir / rel_path)))
    return state


def _drop_descriptor_no_op_tensors(
    state_dict: dict[str, torch.Tensor],
    targets: dict[int, BlockTarget],
    descriptor,
    num_layers: int,
) -> dict[str, torch.Tensor]:
    predicates = descriptor.layer_name_predicates(int(num_layers))
    drop_patterns = []
    for layer_idx, target in targets.items():
        if target.remove_attention or target.remove_mamba:
            pattern = predicates.get(f"block_{layer_idx}_attention")
            if pattern is not None:
                drop_patterns.append(pattern)
        if target.remove_ffn or target.remove_moe:
            pattern = predicates.get(f"block_{layer_idx}_ffn")
            if pattern is not None:
                drop_patterns.append(pattern)
    if not drop_patterns:
        return state_dict
    return {
        key: value
        for key, value in state_dict.items()
        if not any(pattern.fullmatch(key) for pattern in drop_patterns)
    }


def materialize_hidden_width_checkpoint(
    sorted_dir,
    descriptor,
    hidden_width: int,
    output_dir,
    *,
    alignment: int = 1,
    overwrite: bool = False,
) -> Path:
    """Physically realize only the descriptor-owned residual width.

    This creates the width-specific parent used by nested replacement scoring:
    block configurations remain identical to the sorted teacher while every
    language residual tensor, projector output, LM-head input, and MTP residual
    axis is sliced together.
    """
    from ..identity import stable_hash
    from ..tools.checkpoint_utils import load_model_config

    sorted_dir = Path(sorted_dir)
    teacher_config = load_model_config(
        sorted_dir,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    teacher_hidden_width = int(
        descriptor.get_language_model_config(teacher_config).hidden_size
    )
    spec = descriptor.embedding_pruning_spec(
        teacher_config,
        widths=(teacher_hidden_width, int(hidden_width)),
        alignment=int(alignment),
    )
    spec.validate_width(int(hidden_width))
    child_config = spec.update_config_object(teacher_config, int(hidden_width))
    return materialize_checkpoint_from_sorted(
        sorted_dir,
        [],
        descriptor,
        child_config,
        output_dir,
        overwrite=overwrite,
        solution_identity=stable_hash(
            {
                "kind": "hidden_width_parent",
                "teacher_hidden_width": teacher_hidden_width,
                "hidden_width": int(hidden_width),
                "alignment": int(alignment),
            },
            prefix="hidden_width_parent",
        ),
    )


def materialize_checkpoint_from_sorted(
    sorted_dir,
    layer_replacements,
    descriptor,
    child_model_config,
    output_dir,
    *,
    overwrite: bool = False,
    solution_identity: str | None = None,
) -> Path:
    """Stream one sorted-teacher shard at a time into a realized HF checkpoint."""

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from ..identity import stable_hash
    from ..utils.vllm_adapter import (
        configure_anymodel_metadata,
        convert_block_configs_to_per_layer_config,
    )
    from .sorted_teacher import build_layer_layouts, iter_safetensor_weight_files

    sorted_dir = Path(sorted_dir)
    output_dir = Path(output_dir)
    source_config_path = sorted_dir / "config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"sorted teacher is missing config.json: {source_config_path}")

    # Realized checkpoints are a shared interchange format: patched HF loaders
    # instantiate heterogeneous tensors from ``block_configs``, while vLLM
    # AnyModel consumes ``per_layer_config``.  Persist both equivalent views.
    configure_anymodel_metadata(child_model_config, descriptor)
    convert_block_configs_to_per_layer_config(child_model_config, keep_block_configs=True)
    child_model_config.block_configs = [
        block.to_dict() if hasattr(block, "to_dict") else block
        for block in child_model_config.block_configs
    ]
    config_payload = child_model_config.to_dict()
    source_index_path = sorted_dir / "model.safetensors.index.json"
    source_weight_files = iter_safetensor_weight_files(sorted_dir)
    source_identity_payload = {
        "config": json.loads(source_config_path.read_text(encoding="utf-8")),
        "weight_index": (
            json.loads(source_index_path.read_text(encoding="utf-8"))
            if source_index_path.is_file()
            else None
        ),
        # Index files identify key placement, not the concrete checkpoint
        # revision.  File metadata makes an in-place replacement invalidate a
        # previous completed-manifest match without reading hundreds of GB.
        "weight_files": [
            {
                "path": Path(relative).as_posix(),
                "size": (sorted_dir / relative).stat().st_size,
                "mtime_ns": (sorted_dir / relative).stat().st_mtime_ns,
            }
            for relative in source_weight_files
        ],
    }
    source_identity = stable_hash(source_identity_payload, prefix="sorted_teacher")
    solution_identity = solution_identity or stable_hash(
        layer_replacements, prefix="puzzle_solution"
    )
    config_identity = stable_hash(config_payload, prefix="child_config")
    expected_identity = {
        "sorted_teacher_identity": source_identity,
        "solution_identity": solution_identity,
        "config_identity": config_identity,
    }
    from ..checkpoint_transactions import (
        REALIZATION_MANIFEST,
        REALIZATION_TMP_SUFFIX,
        prepare_realization_retry,
        quarantine_incomplete_realization,
        realization_is_complete,
        remove_realization_temp_dir,
    )

    manifest_path = output_dir / REALIZATION_MANIFEST
    if not overwrite:
        if realization_is_complete(output_dir):
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if all(existing.get(key) == value for key, value in expected_identity.items()):
                return output_dir
            raise FileExistsError(
                f"realization destination exists with a different identity: {output_dir}"
            )
        if not prepare_realization_retry(output_dir, expected_identity=expected_identity):
            return output_dir
        overwrite = True
    elif output_dir.exists() and not realization_is_complete(output_dir):
        quarantine_incomplete_realization(output_dir)
        remove_realization_temp_dir(output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"realization destination already exists: {output_dir}")

    teacher_config = __import__(
        "modelopt.torch.puzzletron.tools.checkpoint_utils",
        fromlist=["load_model_config"],
    ).load_model_config(sorted_dir, trust_remote_code=descriptor.requires_trust_remote_code())
    teacher_block_configs = __import__(
        "modelopt.torch.puzzletron.block_config", fromlist=["maybe_cast_block_configs"]
    ).maybe_cast_block_configs(teacher_config.block_configs)
    lm = descriptor.get_language_model_config(teacher_config)
    child_lm = descriptor.get_language_model_config(child_model_config)
    target_hidden_width = int(getattr(child_lm, "hidden_size", lm.hidden_size))
    embedding_spec = None
    if target_hidden_width != int(lm.hidden_size):
        embedding_spec = descriptor.embedding_pruning_spec(
            teacher_config,
            widths=(int(lm.hidden_size), target_hidden_width),
            alignment=1,
        )
        embedding_spec.validate_width(target_hidden_width)
    ple_spec = descriptor.ple_pruning_spec(teacher_config)
    teacher_ple_width = int(getattr(lm, "hidden_size_per_layer_input", 0) or 0)
    target_ple_width = int(
        getattr(child_lm, "hidden_size_per_layer_input", teacher_ple_width) or 0
    )
    if target_ple_width > teacher_ple_width:
        raise ValueError(
            f"PLE target width {target_ple_width} exceeds teacher width {teacher_ple_width}"
        )
    num_q = lm.num_attention_heads
    head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // num_q)
    layer_prefix_tmpl = descriptor.layer_block_name(0).rsplit(".", 1)[0] + ".{i}"
    layout_kwargs = {}
    if hasattr(descriptor, "sorted_teacher_layout_kwargs"):
        layout_kwargs.update(descriptor.sorted_teacher_layout_kwargs(lm))
    layouts = build_layer_layouts(
        teacher_block_configs,
        layer_prefix_tmpl=layer_prefix_tmpl,
        num_attention_heads=num_q,
        head_dim=head_dim,
        **layout_kwargs,
    )
    targets = block_targets_from_replacements(layer_replacements, teacher_block_configs, num_q)
    weight_files = source_weight_files
    source_is_indexed = (sorted_dir / "model.safetensors.index.json").is_file()
    source_index = (
        json.loads(source_index_path.read_text(encoding="utf-8"))
        if source_is_indexed
        else {}
    )
    source_weight_map = dict(source_index.get("weight_map") or {})
    source_total_size = (source_index.get("metadata") or {}).get("total_size")
    can_link_unchanged = (
        source_is_indexed
        and isinstance(source_total_size, int)
        and bool(source_weight_map)
    )
    source_keys_by_shard: dict[str, set[str]] = {}
    for key, shard in source_weight_map.items():
        source_keys_by_shard.setdefault(str(shard), set()).add(str(key))
    target_layer_prefixes = tuple(
        layer_prefix_tmpl.format(i=int(layer_idx)) for layer_idx in sorted(targets)
    )
    global_slice = embedding_spec is not None or (
        ple_spec is not None and target_ple_width < teacher_ple_width
    )

    tmp_dir = output_dir.with_name(output_dir.name + REALIZATION_TMP_SUFFIX)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    weight_rel_paths = {Path(path) for path in weight_files}
    for source in sorted(sorted_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(sorted_dir)
        if relative in weight_rel_paths or source.name in {
            "model.safetensors.index.json",
            "config.json",
            "puzzletron_realization.json",
        }:
            continue
        destination = tmp_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    weight_map: dict[str, str] = dict(source_weight_map) if can_link_unchanged else {}
    tensor_count = len(weight_map)
    total_size = int(source_total_size) if can_link_unchanged else 0
    output_shards = 0
    hardlinked_shards = 0
    try:
        for relative in weight_files:
            source_shard = sorted_dir / relative
            relative_name = relative.as_posix()
            source_shard_keys = source_keys_by_shard.get(relative_name, set())
            if can_link_unchanged and not _shard_requires_rewrite(
                source_shard_keys,
                target_layer_prefixes=target_layer_prefixes,
                global_slice=global_slice,
            ):
                destination = tmp_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_shard, destination)
                    hardlinked_shards += 1
                except OSError:
                    shutil.copy2(source_shard, destination)
                output_shards += 1
                continue

            tensors = load_file(str(source_shard))
            if can_link_unchanged:
                for key, tensor in tensors.items():
                    if weight_map.pop(key, None) is not None:
                        tensor_count -= 1
                        total_size -= tensor.numel() * tensor.element_size()
            realized = materialize_solution_state_dict(tensors, layouts, targets)
            realized = _drop_descriptor_no_op_tensors(
                realized,
                targets,
                descriptor,
                int(lm.num_hidden_layers),
            )
            if embedding_spec is not None:
                realized = embedding_spec.slice_state_dict(realized, target_hidden_width)
            if ple_spec is not None and target_ple_width < teacher_ple_width:
                realized = ple_spec.slice_state_dict(realized, target_ple_width)
            if not realized:
                continue
            realized = {key: tensor.contiguous() for key, tensor in realized.items()}
            destination = tmp_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with safe_open(str(source_shard), framework="pt") as handle:
                metadata = handle.metadata()
            save_file(realized, str(destination), metadata=metadata)
            output_shards += 1
            for key, tensor in realized.items():
                if key in weight_map:
                    raise RuntimeError(f"tensor {key} was written by multiple output shards")
                weight_map[key] = relative_name
                tensor_count += 1
                total_size += tensor.numel() * tensor.element_size()
            del tensors, realized

        if source_is_indexed:
            index_metadata = dict(source_index.get("metadata") or {})
            index_metadata["total_size"] = total_size
            index = {"metadata": index_metadata, "weight_map": weight_map}
            (tmp_dir / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        child_model_config.save_pretrained(tmp_dir)
        manifest = {
            "format": "puzzletron_streaming_realization",
            "version": 1,
            "status": "complete",
            **expected_identity,
            "source_shards": len(weight_files),
            "output_shards": output_shards,
            "hardlinked_shards": hardlinked_shards,
            "tensor_count": tensor_count,
            "total_size": total_size,
            "hidden_width": target_hidden_width,
            "ple_width": target_ple_width,
        }
        (tmp_dir / REALIZATION_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_dir.replace(output_dir)
    except BaseException:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        if output_dir.exists() and not realization_is_complete(output_dir):
            try:
                quarantine_incomplete_realization(output_dir)
            except FileExistsError:
                pass
        raise
    return output_dir


def materialize_model_from_sorted(sorted_dir, layer_replacements, descriptor, child_model_config):
    """Build a pruned model by materializing ``layer_replacements`` from the sorted teacher.

    The realize/library counterpart of the dynamic prune: load the sorted teacher's weights, slice/
    merge the replaced blocks to their target dims, then load the result into a model built from
    ``child_model_config`` (the solution's heterogeneous block configs). No realized intermediate
    checkpoint. For large models this materializes densely — the sharded path writes the result via
    the normal save and reloads with ``load_and_shard_model`` (an in-container follow-up).
    """
    from .sorted_teacher import build_layer_layouts
    from ..block_config import maybe_cast_block_configs
    from ..anymodel.puzzformer import deci_x_patcher
    from ..tools.checkpoint_utils import load_model_config
    from ..tools.checkpoint_utils_hf import init_model_from_config

    teacher_config = load_model_config(
        sorted_dir, trust_remote_code=descriptor.requires_trust_remote_code()
    )
    teacher_block_configs = maybe_cast_block_configs(teacher_config.block_configs)
    lm = descriptor.get_language_model_config(teacher_config)
    child_lm = descriptor.get_language_model_config(child_model_config)
    target_hidden_width = int(getattr(child_lm, "hidden_size", lm.hidden_size))
    teacher_ple_width = int(getattr(lm, "hidden_size_per_layer_input", 0) or 0)
    target_ple_width = int(
        getattr(child_lm, "hidden_size_per_layer_input", teacher_ple_width) or 0
    )
    num_q = lm.num_attention_heads
    head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // num_q)
    layer_prefix_tmpl = descriptor.layer_block_name(0).rsplit(".", 1)[0] + ".{i}"

    layout_kwargs = {}
    if hasattr(descriptor, "sorted_teacher_layout_kwargs"):
        layout_kwargs.update(descriptor.sorted_teacher_layout_kwargs(lm))
    layouts = build_layer_layouts(
        teacher_block_configs,
        layer_prefix_tmpl=layer_prefix_tmpl,
        num_attention_heads=num_q,
        head_dim=head_dim,
        **layout_kwargs,
    )
    targets = block_targets_from_replacements(layer_replacements, teacher_block_configs, num_q)
    realized = materialize_solution_state_dict(_load_sorted_state(sorted_dir), layouts, targets)
    realized = _drop_descriptor_no_op_tensors(
        realized,
        targets,
        descriptor,
        int(lm.num_hidden_layers),
    )
    if target_hidden_width != int(lm.hidden_size):
        embedding_spec = descriptor.embedding_pruning_spec(
            teacher_config,
            widths=(int(lm.hidden_size), target_hidden_width),
            alignment=1,
        )
        realized = embedding_spec.slice_state_dict(realized, target_hidden_width)
    ple_spec = descriptor.ple_pruning_spec(teacher_config)
    if ple_spec is not None and target_ple_width < teacher_ple_width:
        realized = ple_spec.slice_state_dict(realized, target_ple_width)

    with deci_x_patcher(
        model_descriptor=descriptor,
        block_configs=getattr(child_model_config, "block_configs", None),
    ):
        model = init_model_from_config(
            child_model_config, trust_remote_code=descriptor.requires_trust_remote_code()
        )
    realized = descriptor.adapt_materialized_state_dict_for_model(
        realized,
        model=model,
        config=child_model_config,
    )
    # ``strict=False`` because the sorted-teacher state dict may carry tied/derived keys the child
    # module tree does not expose; surface real gaps (every model param must be filled) instead.
    incompatible = model.load_state_dict(realized, strict=False, assign=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    lm_child_config = descriptor.get_language_model_config(child_model_config)
    tied_embeddings = bool(
        getattr(lm_child_config, "tie_word_embeddings", getattr(child_model_config, "tie_word_embeddings", False))
    )
    allowed_missing = {"lm_head.weight"} if tied_embeddings else set()
    missing_keys = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if missing_keys:
        raise RuntimeError(
            f"materialize_model_from_sorted: {len(missing_keys)} model param(s) were not "
            f"filled from the sorted teacher, e.g. {missing_keys[:5]}"
        )
    return model


def materialize_solution_state_dict(state_dict, layouts, targets: dict[int, BlockTarget]):
    """Return a new state dict with each targeted layer's block sliced/merged to its variant.

    ``state_dict`` is the (sorted) teacher's; ``targets`` maps ``layer_idx -> BlockTarget``. Untargeted
    layers are copied unchanged. The result has smaller tensors for materialized blocks, so it loads
    into a model configured with the corresponding (heterogeneous) block configs.
    """
    sd = dict(state_dict)
    for layout in layouts:
        target = targets.get(layout.layer_idx)
        if target is None:
            continue

        if target.remove_ffn:
            for weight_key in (layout.gate_key, layout.up_key, layout.down_key):
                sd.pop(weight_key, None)
                sd.pop(_bias_key(weight_key), None)
        if target.remove_attention:
            for weight_key in (layout.q_key, layout.k_key, layout.v_key, layout.o_key):
                sd.pop(weight_key, None)
                sd.pop(_bias_key(weight_key), None)
        if target.remove_mamba and layout.mamba_prefix:
            prefix = layout.mamba_prefix + "."
            sd = {key: value for key, value in sd.items() if not key.startswith(prefix)}
        if target.remove_moe and layout.moe_prefix:
            prefix = layout.moe_prefix + "."
            sd = {key: value for key, value in sd.items() if not key.startswith(prefix)}

        # ---- FFN: keep the most-important prefix [:K] (sorted) ----
        k = target.target_intermediate
        if k is not None and k < (layout.ffn_intermediate or 0):
            keep = torch.arange(k)
            for key in (layout.gate_key, layout.up_key):
                if key in sd:
                    sd[key] = sd[key][keep]
            if layout.down_key in sd:
                sd[layout.down_key] = sd[layout.down_key][:, keep]
            for wkey in (layout.gate_key, layout.up_key):
                bkey = _bias_key(wkey)
                if bkey in sd:
                    sd[bkey] = sd[bkey][keep]  # down bias (output) unchanged

        # ---- Attention: prefix removal of sorted KV groups and query heads ----
        tkv = target.target_num_kv
        if tkv is not None and layout.num_kv_heads:
            orig_q, orig_kv, hd = layout.num_q_heads, layout.num_kv_heads, layout.head_dim
            tq = target.target_num_q if target.target_num_q is not None else orig_q
            assert tq % tkv == 0, f"target_num_q {tq} not divisible by target_num_kv {tkv}"
            if tq < orig_q or tkv < orig_kv:
                m = tq // tkv
                orig_m = orig_q // orig_kv
                assert m <= orig_m, (
                    "Attention materialization removes whole KV groups and/or a uniform number of "
                    f"query heads per group; got target_heads_per_group={m} > "
                    f"orig_heads_per_group={orig_m}."
                )
                keep_q, keep_kv = sorted_attention_keep_indices(tkv, m, orig_m)
                if layout.q_key in sd:
                    sd[layout.q_key] = slice_query_rows_by_head(
                        sd[layout.q_key], keep_q, hd, orig_q
                    )
                for key in (layout.k_key, layout.v_key):
                    if key in sd:
                        tensor = sd[key]
                        sd[key] = tensor.view(orig_kv, hd, -1)[keep_kv].reshape(-1, tensor.shape[1])
                if layout.o_key in sd:
                    tensor = sd[layout.o_key]
                    sd[layout.o_key] = tensor.view(tensor.shape[0], orig_q, hd)[:, keep_q].reshape(
                        tensor.shape[0], -1
                    )
                for wkey, kept in ((layout.q_key, keep_q), (layout.k_key, keep_kv), (layout.v_key, keep_kv)):
                    bkey = _bias_key(wkey)
                    if bkey in sd:
                        if wkey == layout.q_key:
                            sd[bkey] = slice_query_rows_by_head(sd[bkey], kept, hd, orig_q).reshape(-1)
                        else:
                            sd[bkey] = sd[bkey].view(-1, hd)[kept].reshape(-1)
                        # o bias (output) unchanged

        # ---- MLA: prefix slice coupled heads and latent bases ----
        target_heads = target.target_mla_heads
        original_heads = int(layout.mla_num_heads or 0)
        if target_heads is not None and 0 < target_heads < original_heads:
            for key in (layout.mla_q_b_key, layout.mla_kv_b_key):
                if key in sd:
                    tensor = sd[key]
                    if tensor.shape[0] % original_heads:
                        raise ValueError(
                            f"{key} rows={tensor.shape[0]} are not divisible by "
                            f"MLA heads={original_heads}"
                        )
                    rows_per_head = tensor.shape[0] // original_heads
                    sd[key] = tensor[: target_heads * rows_per_head]
                bias_key = _bias_key(key)
                if bias_key in sd:
                    bias = sd[bias_key]
                    rows_per_head = bias.shape[0] // original_heads
                    sd[bias_key] = bias[: target_heads * rows_per_head]
            if layout.mla_o_key in sd:
                tensor = sd[layout.mla_o_key]
                if tensor.shape[1] % original_heads:
                    raise ValueError(
                        f"{layout.mla_o_key} columns={tensor.shape[1]} are not divisible "
                        f"by MLA heads={original_heads}"
                    )
                cols_per_head = tensor.shape[1] // original_heads
                sd[layout.mla_o_key] = tensor[:, : target_heads * cols_per_head]

        # Prefix-slice the sorted latent bases, preserving KV RoPE rows.
        q_rank = target.target_q_lora_rank
        orig_q_rank = int(layout.mla_q_lora_rank or 0)
        if q_rank is not None and 0 < q_rank < orig_q_rank:
            if layout.mla_q_a_key in sd:
                sd[layout.mla_q_a_key] = sd[layout.mla_q_a_key][:q_rank]
            q_a_bias = _bias_key(layout.mla_q_a_key)
            if q_a_bias in sd:
                sd[q_a_bias] = sd[q_a_bias][:q_rank]
            if layout.mla_q_norm_key in sd:
                sd[layout.mla_q_norm_key] = sd[layout.mla_q_norm_key][:q_rank]
            if layout.mla_q_b_key in sd:
                sd[layout.mla_q_b_key] = sd[layout.mla_q_b_key][:, :q_rank]

        kv_rank = target.target_kv_lora_rank
        orig_kv_rank = int(layout.mla_kv_lora_rank or 0)
        if kv_rank is not None and 0 < kv_rank < orig_kv_rank:
            if layout.mla_kv_a_key in sd:
                tensor = sd[layout.mla_kv_a_key]
                sd[layout.mla_kv_a_key] = torch.cat((tensor[:kv_rank], tensor[orig_kv_rank:]), dim=0)
            kv_a_bias = _bias_key(layout.mla_kv_a_key)
            if kv_a_bias in sd:
                tensor = sd[kv_a_bias]
                sd[kv_a_bias] = torch.cat((tensor[:kv_rank], tensor[orig_kv_rank:]), dim=0)
            if layout.mla_kv_norm_key in sd:
                sd[layout.mla_kv_norm_key] = sd[layout.mla_kv_norm_key][:kv_rank]
            if layout.mla_kv_b_key in sd:
                sd[layout.mla_kv_b_key] = sd[layout.mla_kv_b_key][:, :kv_rank]

        # ---- MoE: sorted expert/channel/latent prefixes ----
        if layout.moe_prefix:
            n = target.target_num_experts
            if n is not None and layout.moe_num_experts and n < layout.moe_num_experts:
                keep_e = list(target.expert_keep_indices or range(n))
                if len(keep_e) != n or len(set(keep_e)) != n:
                    raise ValueError(
                        f"expert_keep_indices must contain {n} unique ids, got {keep_e}"
                    )
                if layout.moe_gate_key in sd:
                    sd[layout.moe_gate_key] = sd[layout.moe_gate_key][keep_e]
                if layout.moe_gate_bias_key in sd:
                    sd[layout.moe_gate_bias_key] = sd[layout.moe_gate_bias_key][keep_e]
                for key in layout.moe_router_aux_keys:
                    if key in sd:
                        sd[key] = sd[key][keep_e]
                for key in layout.moe_fused_expert_keys:
                    if key in sd:
                        sd[key] = sd[key][keep_e]
                if layout.moe_expert_up_keys and layout.moe_expert_down_keys:
                    for e in range(n, layout.moe_num_experts):
                        sd.pop(layout.moe_expert_up_keys[e], None)
                        sd.pop(layout.moe_expert_down_keys[e], None)
                        sd.pop(_bias_key(layout.moe_expert_up_keys[e]), None)
                        sd.pop(_bias_key(layout.moe_expert_down_keys[e]), None)

            k = target.target_expert_intermediate
            if k is not None and layout.moe_fused_gate_up_keys and layout.moe_fused_down_keys:
                original = int(layout.moe_expert_intermediate or 0)
                group_size = int(layout.moe_expert_intermediate_group_size)
                if original and k < original:
                    keep = torch.arange(k)
                    for key in layout.moe_fused_gate_up_keys:
                        if key in sd:
                            tensor = sd[key]
                            if layout.moe_fused_gate_layout == "interleaved":
                                index = torch.stack((2 * keep, 2 * keep + 1), dim=1).reshape(-1)
                            elif layout.moe_fused_gate_layout == "concatenated":
                                index = torch.cat((keep, keep + original))
                            else:
                                raise ValueError(
                                    f"unsupported fused gate layout "
                                    f"{layout.moe_fused_gate_layout!r}"
                                )
                            index = index.to(tensor.device)
                            sd[key] = tensor.index_select(1, index)
                    for key in layout.moe_fused_down_keys:
                        if key in sd:
                            sd[key] = sd[key][:, :, : k // group_size]
            if k is not None and layout.moe_expert_up_keys and layout.moe_expert_down_keys:
                keep = torch.arange(k)
                num_e = min(target.target_num_experts or layout.moe_num_experts or 0, len(layout.moe_expert_up_keys))
                for e in range(num_e):
                    up_key, down_key = layout.moe_expert_up_keys[e], layout.moe_expert_down_keys[e]
                    orig_intermediate = int(layout.moe_expert_intermediate or 0)
                    if orig_intermediate and k < orig_intermediate:
                        if up_key in sd:
                            sd[up_key] = _slice_gated_up(sd[up_key], keep, orig_intermediate)
                        if down_key in sd:
                            sd[down_key] = sd[down_key][:, keep]
                        bkey = _bias_key(up_key)
                        if bkey in sd:
                            sd[bkey] = _slice_gated_up(sd[bkey], keep, orig_intermediate)

            sk = target.target_shared_expert_intermediate
            if (
                sk is not None
                and sk < int(layout.moe_shared_intermediate or 0)
            ):
                keep = torch.arange(sk)
                orig_intermediate = int(layout.moe_shared_intermediate)
                if layout.moe_shared_gate_key and layout.moe_shared_gate_key in sd:
                    # Unfused shared expert (Qwen): gate/up are independent row tensors.
                    sd[layout.moe_shared_gate_key] = sd[layout.moe_shared_gate_key][keep]
                    gate_bias = _bias_key(layout.moe_shared_gate_key)
                    if gate_bias in sd:
                        sd[gate_bias] = sd[gate_bias][keep]
                if layout.moe_shared_up_key in sd:
                    sd[layout.moe_shared_up_key] = _slice_gated_up(
                        sd[layout.moe_shared_up_key], keep, orig_intermediate
                    )
                shared_bias = _bias_key(layout.moe_shared_up_key)
                if shared_bias in sd:
                    sd[shared_bias] = _slice_gated_up(sd[shared_bias], keep, orig_intermediate)
                if layout.moe_shared_down_key in sd:
                    sd[layout.moe_shared_down_key] = sd[layout.moe_shared_down_key][:, keep]

            lk = target.target_latent_dim
            if (
                lk is not None
                and lk < int(layout.moe_latent_dim or 0)
            ):
                if layout.moe_fc1_latent_key in sd:
                    sd[layout.moe_fc1_latent_key] = sd[layout.moe_fc1_latent_key][:lk]
                bkey = _bias_key(layout.moe_fc1_latent_key)
                if bkey in sd:
                    sd[bkey] = sd[bkey][:lk]
                if layout.moe_fc2_latent_key in sd:
                    sd[layout.moe_fc2_latent_key] = sd[layout.moe_fc2_latent_key][:, :lk]
                if layout.moe_expert_up_keys:
                    for key in layout.moe_expert_up_keys:
                        if key in sd:
                            sd[key] = sd[key][:, :lk]
                if layout.moe_expert_down_keys:
                    for key in layout.moe_expert_down_keys:
                        if key in sd:
                            sd[key] = sd[key][:lk, :]
                        bkey = _bias_key(key)
                        if bkey in sd:
                            sd[bkey] = sd[bkey][:lk]

        # ---- Qwen GatedDeltaNet: one coupled semantic prefix slice ----
        if layout.gated_delta_net and layout.mamba_prefix:
            teacher_shape = GDNShape(
                num_key_heads=layout.mamba_num_groups,
                num_value_heads=layout.mamba_num_heads,
                key_head_dim=layout.mamba_state_dim,
                value_head_dim=layout.mamba_head_dim,
            )
            target_shape = GDNShape(
                num_key_heads=target.target_mamba_groups or teacher_shape.num_key_heads,
                num_value_heads=target.target_mamba_heads or teacher_shape.num_value_heads,
                key_head_dim=target.target_mamba_state_dim or teacher_shape.key_head_dim,
                value_head_dim=target.target_mamba_head_dim or teacher_shape.value_head_dim,
            )
            if target_shape != teacher_shape:
                slice_gated_delta_net_state_dict(
                    sd,
                    prefix=layout.mamba_prefix,
                    shape=teacher_shape,
                    target=target_shape,
                )

        # ---- Generic Mamba: sorted head/head-dim/state prefixes ----
        mamba_resident_keys = {
            key
            for key in (
                layout.mamba_in_key,
                layout.mamba_out_key,
                layout.mamba_conv_key,
                layout.mamba_conv_bias_key,
                layout.mamba_a_key,
                layout.mamba_d_key,
                layout.mamba_dt_bias_key,
                layout.mamba_norm_key,
            )
            if key
        }
        if layout.mamba_prefix and mamba_resident_keys.intersection(sd) and not layout.gated_delta_net:
            orig_heads = layout.mamba_num_heads
            orig_hd = layout.mamba_head_dim
            orig_groups = layout.mamba_num_groups or 1
            orig_state_dim = layout.mamba_state_dim
            th = target.target_mamba_heads or orig_heads
            thd = target.target_mamba_head_dim or orig_hd
            tsd = target.target_mamba_state_dim or orig_state_dim
            needs_mamba_slice = (
                (target.target_mamba_heads is not None and orig_heads is not None and th < orig_heads)
                or (target.target_mamba_head_dim is not None and orig_hd is not None and thd < orig_hd)
                or (
                    target.target_mamba_state_dim is not None
                    and orig_state_dim is not None
                    and tsd < orig_state_dim
                )
            )
            if needs_mamba_slice and orig_heads and orig_hd and orig_state_dim:
                tensor_layout = Mamba2TensorLayout(
                    in_proj_key=layout.mamba_in_key,
                    out_proj_key=layout.mamba_out_key,
                    conv_weight_key=layout.mamba_conv_key,
                    conv_bias_key=layout.mamba_conv_bias_key,
                    norm_key=layout.mamba_norm_key,
                    a_log_key=layout.mamba_a_key,
                    d_key=layout.mamba_d_key,
                    dt_bias_key=layout.mamba_dt_bias_key,
                    num_heads=orig_heads,
                    head_dim=orig_hd,
                    num_groups=orig_groups,
                    state_dim=orig_state_dim,
                )
                sd = slice_mamba2_state_dict(
                    sd,
                    tensor_layout,
                    target_heads=th,
                    target_head_dim=thd,
                    target_state_dim=tsd,
                )
    return sd
