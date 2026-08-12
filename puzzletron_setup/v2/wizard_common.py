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

"""Shared defaults and prompt helpers for the setup-v2 wizard."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped, unused-ignore]

from puzzletron_setup import WORKER_REPOSITORY_PLACEHOLDER, WORKER_VENV_PLACEHOLDER, SetupError

from .defaults import DefaultsResolver, load_defaults
from .prompts import BACK

if TYPE_CHECKING:
    from pathlib import Path

    from puzzletron_setup.profiles import CandidateCounts

    from .presets import SetupPreset
    from .session import WizardSession
    from .state import WizardState

__all__ = [
    "BUILTINS",
    "CANONICAL_STAGE_STRATEGIES",
    "STATIC_MODEL_BATCH_PATHS",
    "STATIC_MODEL_STAGES",
]

BUILTINS = {
    "data": {"layout": "fixed", "sequence_length": 4096},
    "infrastructure": {
        "gpus_per_node": 8,
        "execution_contract": {
            "repository": WORKER_REPOSITORY_PLACEHOLDER,
            "venv": WORKER_VENV_PLACEHOLDER,
            "container": None,
            "container_mounts": None,
            "prerun_commands": [],
            "postrun_commands": [],
        },
        "runner": {
            "kind": "slurm",
            "slurm": {
                "account": "",
                "partition_interactive": "interactive",
                "partition_batch": "batch",
                "partition_cpu": None,
                "time_limit": "4:00:00",
                "qos": None,
                "max_nodes": 64,
            },
        },
    },
    "pruning": {
        "depth_granularity": "subblock",
        "depth_remove": 4,
        "replacement_granularity": "subblock",
        "width_importance_samples": 32768,
        "sort_sanity": False,
        "sort_sanity_samples": 128,
        "width_sanity": False,
        "width_sanity_samples": 128,
        "width_sanity_layer_count": 3,
        "width_sanity_targets_per_axis": 2,
        "slicing_sanity": False,
        "replacement_samples": 128,
        "bypass": {
            "enabled": True,
            "granularity": "subblock",
            "samples": 4096,
            "sequence_length": 4096,
            "batch_size": 8,
            "grad_accumulation_steps": 1,
        },
    },
    "vllm": {
        "enabled": False,
        "granularity": "subblock",
        "prefill_seq_len": 4096,
        "generation_seq_len": 1024,
        "batch_size": 1,
        "max_num_seqs": 1,
        "topology": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": False,
            "distributed_executor_backend": "mp",
        },
    },
    "mip": {
        "goal_metric": "params",
        "goal_value": "75%",
        "objective": "metrics.cosine_embedding_loss_hidden_states",
        "num_solutions": 8,
    },
}

STATIC_MODEL_STAGES = (
    "depth_importance",
    "width_importance",
    "bypass",
    "replacement_scoring",
)
STATIC_MODEL_BATCH_PATHS = {
    "depth_importance": "depth_importance.micro_batch_size",
    "width_importance": "pruning.micro_batch_size",
    "sort_sanity": "sort_sanity.micro_batch_size",
    "width_sanity": "width_sanity.micro_batch_size",
    "bypass": "bypass.training.micro_batch_size",
    "replacement_scoring": "replacement_scoring.micro_batch_size",
}
CANONICAL_STAGE_STRATEGIES = {
    "depth_importance": "persistent_pool",
    "width_importance": "single",
    "sort_sanity": "single",
    "width_sanity": "single",
    "bypass": "single",
    "replacement_scoring": "persistent_pool",
}


def _nested_records(state: WizardState) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, record in state.records().items():
        current = nested
        parts = path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = deepcopy(record.effective)
    return nested


def _resolver(
    state: WizardState,
    defaults_path: Path | None,
    preset: SetupPreset | None = None,
    family_config: str | Path | None = None,
    model_inventory: Any | None = None,
) -> DefaultsResolver:
    preset_defaults = {}
    model_profile_defaults = {}
    if preset is not None and family_config is not None:
        preset_defaults, model_profile_defaults = preset.resolved_default_layers(
            family_config,
            model_inventory,
        )
    return DefaultsResolver(
        builtins=BUILTINS,
        model_derived={},
        preset_defaults=preset_defaults,
        model_profile_defaults=model_profile_defaults,
        file_defaults=load_defaults(defaults_path),
        preserved=_nested_records(state),
    )


def _resolved(
    state: WizardState,
    resolver: DefaultsResolver,
    path: str,
    fallback: Any = None,
) -> Any:
    value = resolver.resolve(path, fallback)
    print(f"  {path}: {value.value!r} ({value.source})")
    return value


def _record_default(
    state: WizardState,
    resolver: DefaultsResolver,
    path: str,
    fallback: Any = None,
    *,
    dependencies: tuple[str, ...] = (),
) -> Any:
    resolved = resolver.resolve_default(path, fallback)
    state.set_field(
        path,
        resolved.value,
        source=resolved.source,
        dependencies=dependencies,
    )
    return resolved.value


def _section_action(
    session: WizardSession,
    section: str,
    summary: str,
    defaults: Mapping[str, Any],
    *,
    prompt_in_guided: bool = False,
) -> Any:
    session.begin(section)
    if session.guided and not prompt_in_guided:
        return "defaults"
    print(f"\n[{section}] {summary}")
    _print_default_decisions(defaults)
    return session.select(
        f"{section}.action",
        f"{section.replace('_', ' ').title()}:",
        [
            ("Use defaults shown above", "defaults"),
            ("Customize", "customize"),
        ],
        default="defaults",
    )


def _print_default_decisions(defaults: Mapping[str, Any]) -> None:
    print("  Resolved defaults:")
    rendered = yaml.safe_dump(
        _plain_review_value(defaults),
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    for line in rendered.splitlines():
        print(f"    {line}")


def _plain_review_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_review_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_review_value(item) for item in value]
    return value


def _vllm_granularity_choices(counts: CandidateCounts) -> list[tuple[str, str]]:
    width_count = counts.effective_vllm_width_count
    if width_count == 1:
        return [
            (
                f"Sublayer — {counts.vllm_subblock_total} unique configurations",
                "subblock",
            ),
            (
                f"Whole block — {counts.vllm_block_total} unique configurations",
                "block",
            ),
        ]
    return [
        (
            f"Sublayer — {counts.vllm_subblock} configurations/width, "
            f"{counts.vllm_subblock_total} total across {width_count} widths",
            "subblock",
        ),
        (
            f"Whole block — {counts.vllm_block} configurations/width, "
            f"{counts.vllm_block_total} total across {width_count} widths",
            "block",
        ),
    ]


def _replacement_granularity_choices(
    counts: CandidateCounts,
) -> list[tuple[str, str]]:
    width_count = counts.width_count

    def label(name: str, per_width: int, total: int) -> str:
        if width_count == 1:
            return f"{name} — {total} options"
        return f"{name} — {per_width} options/width, {total} total across {width_count} widths"

    return [
        (
            label(
                "One sublayer at a time",
                counts.replacement_subblock_per_width,
                counts.replacement_subblock_total,
            ),
            "subblock",
        ),
        (
            label(
                "One layer at a time",
                counts.replacement_block_per_width,
                counts.replacement_block_total,
            ),
            "block",
        ),
    ]


def _depth_granularity_choices(inventory: Any) -> list[tuple[str, str]]:
    return [
        (f"Sublayer — {inventory.num_sublayers} available", "subblock"),
        (f"Whole layer — {inventory.num_layers} available", "block"),
    ]


def _default_axis_values(axis: Any) -> list[int]:
    legal_values = tuple(int(value) for value in axis.values)
    teacher = int(axis.teacher_value)
    half = min(
        legal_values,
        key=lambda value: (abs(value - teacher // 2), -value),
    )
    return list(dict.fromkeys((teacher, half)))


def _text_field(
    session: WizardSession,
    resolver: DefaultsResolver,
    path: str,
    label: str,
    fallback: str = "",
    *,
    validate: Callable[[Any], bool | str] | None = None,
) -> Any:
    resolved = _resolved(session.state, resolver, path, fallback)
    value = session.text(
        path,
        label,
        default=str(resolved.value or ""),
        validate=validate,
    )
    if value is not BACK:
        session.state.set_field(path, value, source="user")
    return value


def _integer_field(
    session: WizardSession,
    resolver: DefaultsResolver,
    path: str,
    label: str,
    fallback: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> Any:
    resolved = _resolved(session.state, resolver, path, fallback)
    value = session.integer(
        path,
        label,
        default=int(resolved.value),
        minimum=minimum,
        maximum=maximum,
    )
    if value is not BACK:
        session.state.set_field(path, value, source="user")
    return value


def _guided_integer_default(value: Any, path: str, *, minimum: int) -> int:
    """Validate one non-interactive integer default with an actionable error."""
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise SetupError(
            f"Guided setup default {path} must be an integer; got {value!r}."
        ) from error
    if parsed < minimum:
        raise SetupError(f"Guided setup default {path} must be at least {minimum}; got {value!r}.")
    return parsed


def _mapping_copy(value: Any) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}
