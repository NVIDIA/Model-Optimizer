# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ParallelCapabilities",
    "SubblockCapabilities",
    "AxisCapabilities",
    "MagnitudeFallbackSpec",
    "StageCapabilities",
    "ExportCapabilities",
    "PuzzletronCapabilities",
    "CapabilityValidationError",
    "default_capabilities",
    "validate_capabilities",
    "resolve_score_method",
]


@dataclass(frozen=True)
class ParallelCapabilities:
    tp: bool = True
    pp: bool = True
    cp: bool = True
    fsdp: bool = True
    ep: bool = False
    sequence_parallel: bool = False
    invalid_combinations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubblockCapabilities:
    kind: str
    config_class: str
    names: tuple[str, ...]
    no_op: bool = False
    replacement: bool = False
    bypass: bool = False
    tensor_bindings: bool = False


@dataclass(frozen=True)
class MagnitudeFallbackSpec:
    """Descriptor-owned observation contract for the generic ``|activation|`` metric."""

    observation_module: str
    tensor_selector: str
    scored_dim: int
    output_field: str
    expected_size: int

    def __post_init__(self) -> None:
        if not self.observation_module:
            raise ValueError("magnitude fallback requires an observation module")
        if self.tensor_selector.split(".", 1)[0] not in {"input", "output"}:
            raise ValueError("magnitude tensor_selector must start with input or output")
        if not self.output_field:
            raise ValueError("magnitude fallback requires an output field")
        if int(self.expected_size) < 1:
            raise ValueError("magnitude fallback expected_size must be positive")


@dataclass(frozen=True)
class AxisCapabilities:
    axis_id: str
    subblock_kind: str
    field: str
    sortable: bool = True
    variant_only: bool = False
    score_hooks: tuple[str, ...] = ()
    sort_impl: str | None = None
    materialize_impl: str | None = None
    runtime_slice_impl: str | None = None
    vllm_export: bool = False
    force_hf: bool = True
    native_automodel_required: bool = False
    values: tuple[Any, ...] = ()
    constraints: tuple[str, ...] = ()
    magnitude_fallback: MagnitudeFallbackSpec | None = None


@dataclass(frozen=True)
class StageCapabilities:
    convert: bool = True
    activation: bool = True
    sort: bool = True
    bypass: bool = False
    library: bool = True
    scoring: bool = True
    rpc: bool = False
    mip: bool = True
    materialize: bool = True
    global_kd: bool = False
    aiperf: bool = False
    evaluation: bool = False


@dataclass(frozen=True)
class ExportCapabilities:
    hf: bool = True
    vllm: bool = True
    per_layer_config: bool = True
    no_op: bool = True
    mamba_cache: bool = False
    anymodel_arch_info_required: bool = False


@dataclass(frozen=True)
class PuzzletronCapabilities:
    descriptor_name: str
    descriptor_version: str
    model_family: str
    force_hf_supported: bool
    native_automodel_supported: bool
    parallelism: ParallelCapabilities
    subblocks: dict[str, SubblockCapabilities]
    axes: dict[str, AxisCapabilities]
    stages: StageCapabilities
    export: ExportCapabilities
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class CapabilityValidationError(ValueError):
    pass


def resolve_score_method(axis: AxisCapabilities) -> str:
    """Choose the specialized metric first, then an explicit generic fallback."""

    if axis.score_hooks:
        return axis.score_hooks[0]
    if axis.magnitude_fallback is not None:
        return "magnitude_fallback"
    raise ValueError(f"Prunable axis {axis.axis_id!r} has no activation scorer")


def default_capabilities(
    *,
    descriptor_name: str,
    model_family: str | None = None,
    native_automodel_supported: bool = False,
) -> PuzzletronCapabilities:
    """Conservative capability declaration for current HF-backed descriptors."""
    axes = {
        "ffn_intermediate": AxisCapabilities(
            axis_id="ffn_intermediate",
            subblock_kind="ffn",
            field="intermediate_size",
            score_hooks=("independent", "iterative"),
            sort_impl="sorted_teacher.ffn",
            materialize_impl="materialize.ffn",
            runtime_slice_impl="dynamic_block_prune.ffn",
            vllm_export=True,
        ),
        "query_heads": AxisCapabilities(
            axis_id="query_heads",
            subblock_kind="attention",
            field="num_query_heads",
            score_hooks=("grouped_attention_contribution",),
            sort_impl="sorted_teacher.attention",
            materialize_impl="materialize.attention",
            runtime_slice_impl="dynamic_block_prune.attention",
            vllm_export=True,
            constraints=("gqa_grouping",),
        ),
        "q_heads_per_group": AxisCapabilities(
            axis_id="q_heads_per_group",
            subblock_kind="attention",
            field="num_query_heads",
            score_hooks=("grouped_attention_contribution",),
            sort_impl="sorted_teacher.attention",
            materialize_impl="materialize.attention",
            runtime_slice_impl="dynamic_block_prune.attention",
            vllm_export=True,
            constraints=("gqa_grouping",),
        ),
        "kv_heads": AxisCapabilities(
            axis_id="kv_heads",
            subblock_kind="attention",
            field="num_kv_heads",
            score_hooks=("grouped_attention_contribution",),
            sort_impl="sorted_teacher.attention",
            materialize_impl="materialize.attention",
            runtime_slice_impl="dynamic_block_prune.attention",
            vllm_export=True,
            constraints=("gqa_grouping",),
        ),
        "kv_groups": AxisCapabilities(
            axis_id="kv_groups",
            subblock_kind="attention",
            field="num_kv_heads",
            score_hooks=("grouped_attention_contribution",),
            sort_impl="sorted_teacher.attention",
            materialize_impl="materialize.attention",
            runtime_slice_impl="dynamic_block_prune.attention",
            vllm_export=True,
            constraints=("gqa_grouping",),
        ),
        "qk_head_dim": AxisCapabilities(
            axis_id="qk_head_dim",
            subblock_kind="attention",
            field="qk_head_dim",
            score_hooks=("qk_head_dim_contribution",),
            sort_impl="sorted_teacher.attention_qk_dim",
            materialize_impl="materialize.attention_qk_dim",
            runtime_slice_impl="dynamic_block_prune.attention_qk_dim",
            vllm_export=False,
            constraints=("rope_pairing", "qk_norm_pairing"),
        ),
        "moe_experts": AxisCapabilities(
            axis_id="moe_experts",
            subblock_kind="moe",
            field="num_experts",
            score_hooks=("removed_expert_diff",),
            vllm_export=True,
        ),
        "moe_expert_intermediate": AxisCapabilities(
            axis_id="moe_expert_intermediate",
            subblock_kind="moe",
            field="expert_intermediate_size",
            score_hooks=("expert_intermediate_contribution",),
            sort_impl="sorted_teacher.moe_expert_intermediate",
            materialize_impl="materialize.moe_expert_intermediate",
            runtime_slice_impl="dynamic_block_prune.moe_expert_intermediate",
            vllm_export=True,
        ),
        "moe_shared_expert_intermediate": AxisCapabilities(
            axis_id="moe_shared_expert_intermediate",
            subblock_kind="moe",
            field="shared_expert_intermediate_size",
            score_hooks=("shared_expert_intermediate_contribution",),
            sort_impl="sorted_teacher.moe_shared_expert",
            materialize_impl="materialize.moe_shared_expert",
            runtime_slice_impl="dynamic_block_prune.moe_shared_expert",
            vllm_export=True,
        ),
        "moe_latent_dim": AxisCapabilities(
            axis_id="moe_latent_dim",
            subblock_kind="moe",
            field="latent_dim",
            score_hooks=("latent_moe_contribution",),
            sort_impl="sorted_teacher.latent_moe_rotation",
            materialize_impl="materialize.latent_moe",
            runtime_slice_impl="dynamic_block_prune.latent_moe",
            vllm_export=True,
            constraints=("rotation_metadata",),
        ),
        "moe_top_k": AxisCapabilities(
            axis_id="moe_top_k",
            subblock_kind="moe",
            field="top_k",
            sortable=False,
            variant_only=True,
            score_hooks=(),
            runtime_slice_impl="dynamic_block_prune.moe_top_k",
            vllm_export=True,
        ),
        "mamba_heads": AxisCapabilities(
            axis_id="mamba_heads",
            subblock_kind="mamba",
            field="num_heads",
            score_hooks=("mamba_head_contribution",),
            sort_impl="sorted_teacher.mamba_heads",
            materialize_impl="materialize.mamba_heads",
            runtime_slice_impl="dynamic_block_prune.mamba_heads",
            vllm_export=True,
            constraints=("mamba_cache_shape",),
        ),
        "mamba_head_dim": AxisCapabilities(
            axis_id="mamba_head_dim",
            subblock_kind="mamba",
            field="head_dim",
            score_hooks=("mamba_head_dim_contribution",),
            sort_impl="sorted_teacher.mamba_head_dim",
            materialize_impl="materialize.mamba_head_dim",
            runtime_slice_impl="dynamic_block_prune.mamba_head_dim",
            vllm_export=True,
            constraints=("mamba_cache_shape",),
        ),
        "sliding_window_size": AxisCapabilities(
            axis_id="sliding_window_size",
            subblock_kind="attention",
            field="sliding_window_size",
            sortable=False,
            variant_only=True,
            vllm_export=True,
        ),
    }
    return PuzzletronCapabilities(
        descriptor_name=descriptor_name,
        descriptor_version="1",
        model_family=model_family or descriptor_name,
        force_hf_supported=True,
        native_automodel_supported=native_automodel_supported,
        parallelism=ParallelCapabilities(ep=native_automodel_supported),
        subblocks={
            "attention": SubblockCapabilities(
                kind="attention",
                config_class="AttentionConfig",
                names=("attention",),
                no_op=True,
                replacement=True,
                bypass=True,
                tensor_bindings=True,
            ),
            "ffn": SubblockCapabilities(
                kind="ffn",
                config_class="FFNConfig",
                names=("ffn",),
                no_op=True,
                replacement=True,
                bypass=True,
                tensor_bindings=True,
            ),
            "moe": SubblockCapabilities(
                kind="moe",
                config_class="MoEConfig",
                names=("moe",),
                no_op=True,
                replacement=True,
                bypass=False,
                tensor_bindings=False,
            ),
            "mamba": SubblockCapabilities(
                kind="mamba",
                config_class="MambaConfig",
                names=("mamba",),
                no_op=True,
                replacement=False,
                bypass=False,
                tensor_bindings=False,
            ),
        },
        axes=axes,
        stages=StageCapabilities(),
        export=ExportCapabilities(),
        notes=("default conservative capabilities",),
    )


def validate_capabilities(
    capabilities: PuzzletronCapabilities,
    *,
    enabled_axes: list[str] | tuple[str, ...] = (),
    force_hf: bool = True,
    ep: int = 1,
    require_vllm: bool = False,
    require_complete_pipeline: bool = False,
) -> None:
    errors: list[str] = []
    if force_hf and not capabilities.force_hf_supported:
        errors.append(f"{capabilities.descriptor_name} does not support force_hf=True")
    # ``force_hf=False`` in NeMo AutoModel means "prefer a custom/native model when one
    # exists"; for architectures without a native registry entry NeMo falls back to the HF
    # implementation.  Treat native support as mandatory only when EP is requested or a selected
    # axis explicitly requires native AutoModel.
    native_required = not force_hf and ep > 1
    if force_hf and ep > 1:
        errors.append("force_hf=True cannot be used with ep > 1")
    if ep > 1 and not capabilities.parallelism.ep:
        errors.append(f"{capabilities.descriptor_name} does not declare EP support")

    for axis in enabled_axes:
        axis_caps = capabilities.axes.get(axis)
        if axis_caps is None:
            errors.append(f"Unsupported Puzzletron axis '{axis}'")
            continue
        if force_hf and axis_caps.native_automodel_required:
            errors.append(f"Axis '{axis}' requires native AutoModel")
        if not force_hf and axis_caps.native_automodel_required:
            native_required = True
        if require_complete_pipeline:
            if not axis_caps.variant_only:
                if not axis_caps.score_hooks and axis_caps.magnitude_fallback is None:
                    errors.append(f"Axis '{axis}' has no activation scorer")
                if axis_caps.sortable and not axis_caps.sort_impl:
                    errors.append(f"Axis '{axis}' has no sort_impl")
            if not axis_caps.materialize_impl:
                errors.append(f"Axis '{axis}' has no materialize_impl")
            if not axis_caps.runtime_slice_impl:
                errors.append(f"Axis '{axis}' has no runtime_slice_impl")
        if require_vllm and not axis_caps.vllm_export:
            errors.append(f"Axis '{axis}' cannot be represented in vLLM export")

    if (
        native_required
        and not capabilities.native_automodel_supported
        and not any("native AutoModel" in error for error in errors)
    ):
        errors.append(f"{capabilities.descriptor_name} does not support native AutoModel")

    if require_vllm and not capabilities.export.vllm:
        errors.append(f"{capabilities.descriptor_name} does not support vLLM export")

    if errors:
        raise CapabilityValidationError("; ".join(errors))
