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

"""Main entry point for running the puzzle optimization to find optimal layer configurations."""

# mypy: ignore-errors
import argparse
import dataclasses
import enum
import json
import sys
from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from modelopt.torch.utils import json_dump

from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..block_config import AttentionConfig, BlockConfig, FFNConfig, MambaConfig, MoEConfig
from ..replacement_library.replacement_utils import (
    extract_block_configs_and_locations,
    parse_layer_replacement,
    replacement_is_teacher,
)
from ..replacement_library.score_composition import (
    compose_full_block_metrics,
    composed_table_to_gathered_metrics,
)
from ..tools.checkpoint_utils import load_model_config
from ..tools.logger import mprint
from ..utils.misc import block_config_to_str, solution_to_str
from ..utils.parsing import get_nested_key, parse_json, parse_path
from .search_space import filter_replacements_by_axes, rank_homogeneous_solutions
from .solver_backend import run_mip_with_backend

__all__ = [
    "PuzzleMetrics",
    "MultiLayerPuzzleMetrics",
    "run_puzzle",
    "gather_multi_layer_puzzle_metrics",
    "filter_subblock_stats_by_args",
]

"""
Usage:
Must specify either --single_block_replacement_validation_dir and --subblock_stats_path (in which case the metrics will
be gathered from the validation output files) or --gathered_metrics_path (in which case the metrics will be read from
this json file).

Constraints can be specified either as 'mip_constraints' (the actual constraints that go into the MIP, e.g. 'stats.memory_mib', 'stats.runtime_ms'),
or as "human constraints" (e.g. 'target_memory', 'target_throughput', for the full list see PuzzleConstraints._ALLOWED_HUMAN_CONSTRAINTS).

"""

PuzzleMetrics: TypeAlias = dict[Hashable, dict[Hashable, dict[str, float]]]
MultiLayerPuzzleMetrics: TypeAlias = dict[str, dict[str, Hashable]]

_ATTENTION_LIKE_KINDS = frozenset(("attention", "mla", "mamba"))
_FFN_LIKE_KINDS = frozenset(("ffn", "moe"))


def _block_config_tp_compatible(block_config: BlockConfig, tp_size: int) -> bool:
    """Return whether a physically realized block can use the requested TP mesh.

    Colwise Q/K/V projection sharding requires complete heads on every rank.
    Runtime engines may support replicated KV heads, but native AutoModel's
    current generic TP plan shards these projections, so a MIP must not select
    a geometry that would split one head across ranks.
    """
    tp_size = int(tp_size)
    if tp_size <= 1:
        return True
    for subblock in block_config.subblock_configs:
        if subblock.no_op or not isinstance(subblock, AttentionConfig):
            continue
        for value in (subblock.num_query_heads, subblock.num_kv_heads):
            if value is not None and int(value) % tp_size:
                return False
    return True


def _filter_tp_incompatible_replacements(gathered_metrics: dict, tp_size: int) -> dict:
    if int(tp_size) <= 1:
        return gathered_metrics
    filtered = {
        key: value
        for key, value in gathered_metrics.items()
        if value.get("is_teacher", False)
        or _block_config_tp_compatible(value["block_config"], tp_size)
    }
    removed = len(gathered_metrics) - len(filtered)
    if removed:
        mprint(
            f"Excluded {removed} replacement candidates that cannot be physically "
            f"sharded with materialization_tp={tp_size}."
        )
    layers_before = {
        int(value["block_idx"]) for value in gathered_metrics.values() if "block_idx" in value
    }
    layers_after = {int(value["block_idx"]) for value in filtered.values() if "block_idx" in value}
    if layers_before != layers_after:
        raise RuntimeError(
            "Topology filtering removed every candidate for layers "
            f"{sorted(layers_before - layers_after)} at tp={tp_size}."
        )
    return filtered


@dataclasses.dataclass
class PuzzleConstraints:
    """A set of puzzle constraints can be expressed either directly as the mip constraints (e.g. 'stats.memory_mib') or as human constraints (e.g. 'target_throughput')"""

    class Type(enum.Enum):
        MIP = "mip"
        HUMAN = "human"

    _ALLOWED_HUMAN_CONSTRAINTS = {
        "target_memory",
        "target_memory_ratio",
        "target_throughput",
        "target_latency_seconds",
        "target_latency_ratio",
        "target_time_to_first_token",
        "target_num_kv_heads",
        "num_params",
        "num_params_ratio",
        "stats.has_attention",
    }
    type: Type
    name: str = dataclasses.field(init=False)
    constraints: dict[str, Any]

    @staticmethod
    def sizeof_fmt(num, suffix=""):
        for unit in ("", "K", "M", "G", "T"):
            if abs(num) < 1000.0:
                return f"{num:g}{unit}{suffix}"
            num /= 1000.0
        return f"{num:.1f}P{suffix}"

    def _validate_human_constraints(self):
        illegal_constraints = set(self.constraints.keys()) - self._ALLOWED_HUMAN_CONSTRAINTS
        if illegal_constraints:
            raise ValueError(
                f"The following human_constraints are illegal: {','.join(illegal_constraints)}"
            )

    @staticmethod
    def _parse_ratio(value: float | int | str) -> float:
        if isinstance(value, str):
            value = value.strip()
            if value.endswith("%"):
                value = float(value[:-1]) / 100
            else:
                value = float(value)
        value = float(value)
        if not 0 < value <= 1:
            raise ValueError(f"Ratio constraints must be in (0, 1], got {value}")
        return value

    def has_ratio_constraints(self) -> bool:
        return any(key.endswith("_ratio") for key in self.constraints)

    def resolve_ratio_constraints(self, teacher_stats: dict[str, Any]) -> "PuzzleConstraints":
        if self.type != PuzzleConstraints.Type.HUMAN or not self.has_ratio_constraints():
            return self

        constraints = deepcopy(self.constraints)
        ratio_specs = {
            "target_memory_ratio": ("target_memory", "memory_mib"),
            "target_latency_ratio": ("target_latency_seconds", "runtime_ms"),
            "num_params_ratio": ("num_params", "num_params"),
        }
        for ratio_key, (target_key, teacher_stat_key) in ratio_specs.items():
            if ratio_key not in constraints:
                continue
            if target_key in constraints:
                raise ValueError(
                    f"Specify only one of {ratio_key!r} and {target_key!r} in human_constraints."
                )
            teacher_value = teacher_stats.get(teacher_stat_key)
            if teacher_value is None:
                raise ValueError(
                    f"Cannot resolve {ratio_key!r}: teacher stats do not include "
                    f"{teacher_stat_key!r}. Make sure the matching subblock_stats entry "
                    "contains that metric."
                )
            ratio = self._parse_ratio(constraints.pop(ratio_key))
            target_value = teacher_value * ratio
            if target_key == "target_latency_seconds":
                target_value /= 1000
            constraints[target_key] = target_value
            mprint(
                f"Resolved {ratio_key}={ratio:g} to {target_key}={target_value:g} "
                f"from teacher {teacher_stat_key}={teacher_value:g}"
            )

        return PuzzleConstraints(type=self.type, constraints=constraints)

    def format_num_params_to_float(self, num_params):
        if isinstance(num_params, list):
            return [self.format_num_params_to_float(x) for x in num_params]
        if isinstance(num_params, str):
            # we only deal with Billions of params scale
            return float(num_params.replace("B", "")) * 1e9
        return num_params

    def format_num_params_to_str(self, num_params):
        if isinstance(num_params, list):
            return [self.format_num_params_to_str(x) for x in num_params]
        if isinstance(num_params, float) or isinstance(num_params, int):
            return f"{num_params / 1e9}B"
        return num_params

    def __post_init__(self):
        if self.type == PuzzleConstraints.Type.HUMAN:
            self._validate_human_constraints()

        if "stats.active_params" in self.constraints:
            self.constraints["stats.active_params"] = self.format_num_params_to_float(
                self.constraints["stats.active_params"]
            )

        # Set self.name
        constraints = deepcopy(self.constraints)  # going to override with "human readable" versions
        if "stats.active_params" in constraints:
            constraints["stats.active_params"] = self.format_num_params_to_str(
                constraints["stats.active_params"]
            )

        if self.type == PuzzleConstraints.Type.HUMAN:
            # change values to a more human string form
            if "target_memory" in constraints:
                constraints["target_memory"] = str(constraints["target_memory"]) + "MiB"
            if "num_params" in constraints:
                constraints["num_params"] = self.sizeof_fmt(constraints["num_params"])

        def build_constraint_name(constraint_name, constraint_value):
            if isinstance(constraint_value, Iterable) and not isinstance(constraint_value, str):
                return "-".join(f"{constraint_name}_{x}" for x in constraint_value)
            else:
                return f"{constraint_name}_{constraint_value}"

        self.name = "-".join(build_constraint_name(k, v) for k, v in constraints.items()).replace(
            ".", "_"
        )

    def to_mip_constraints(self, subblock_stats_args) -> dict[str, Any]:
        if self.type == PuzzleConstraints.Type.MIP:
            return self.constraints

        assert all(key in subblock_stats_args for key in ("batch_size", "generation_seq_len")), (
            "Can't realize human constraints without 'batch_size' and 'generation_seq_len' in subblock_stats_args."
        )
        batch_size = subblock_stats_args["batch_size"]
        generation_seq_len = subblock_stats_args["generation_seq_len"]

        mip_constraints = {}

        # Memory constraints
        if "target_memory" in self.constraints:
            mip_constraints["stats.memory_mib"] = self.constraints["target_memory"]

        # Total KV-heads constraint (sum across attention layers; used for KV-cache-only sweeps)
        if "target_num_kv_heads" in self.constraints:
            mip_constraints["stats.num_kv_heads"] = self.constraints["target_num_kv_heads"]

        # Throughput constraints
        throughput_constraints = []
        if "target_throughput" in self.constraints:
            if self.constraints["target_throughput"] == 0:
                raise ValueError("target_throughput must not be zero")
            throughput_constraints.append(
                batch_size * generation_seq_len / self.constraints["target_throughput"]
            )
        if "target_latency_seconds" in self.constraints:
            throughput_constraints.append(self.constraints["target_latency_seconds"])
        if throughput_constraints:
            mip_constraints["stats.runtime_ms"] = 1000 * min(throughput_constraints)

        # Prefill runtime constraint
        if "target_time_to_first_token" in self.constraints:
            mip_constraints["stats.prefill_runtime_ms"] = (
                1000 * self.constraints["target_time_to_first_token"]
            )

        # Num params
        if "num_params" in self.constraints:
            mip_constraints["stats.num_params"] = self.constraints["num_params"]
        if "stats.has_attention" in self.constraints:
            mip_constraints["stats.has_attention"] = self.constraints["stats.has_attention"]
        return mip_constraints


def parse_args() -> DictConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--puzzle_profile", type=parse_path)

    parser.add_argument("--single_block_replacement_validation_dir", type=parse_path, default=None)
    parser.add_argument(
        "--gathered_metrics_path",
        type=parse_path,
        default=None,
        help="Can be given explicitly instead of --single_block_replacement_validation_dir",
    )

    parser.add_argument("--subblock_stats_path", type=parse_path)
    parser.add_argument("--subblock_stats_args", type=parse_json)

    parser.add_argument("--objective", type=str)
    parser.add_argument("--mip_constraints", type=parse_json)
    parser.add_argument("--human_constraints", type=parse_json)
    parser.add_argument("--solver_backend", type=str, default=None)
    parser.add_argument("--report_additional_costs", type=str, action="append", default=[])

    parser.add_argument(
        "--output_path",
        type=parse_path,
        help="The main folder under which all results will be stored.",
    )

    parser.add_argument("--max_seconds_per_solution", type=float, default=60.0)
    parser.add_argument("--metric_overrides", type=parse_json, default=None)
    parser.add_argument(
        "--bigger_is_better",
        action="store_true",
        help="Set this if using accuracy objective, don't set if using loss objective",
    )

    args = parser.parse_args()
    return DictConfig(vars(args))


def run_single_puzzle_config(
    args: DictConfig,
    gathered_metrics: dict,
    subblock_stats: dict,
    subblock_stats_args: dict,
    constraints: PuzzleConstraints,
    output_folder,
) -> Path:
    # we override the constraints and subblock_stats_args for this run to keep reporting out the same way.
    args = deepcopy(args)
    report_additional_costs = args.get("report_additional_costs", []) or []

    all_subblock_stats = subblock_stats
    subblock_stats = filter_subblock_stats_by_args(all_subblock_stats, subblock_stats_args)
    _merge_namespaced_workload_stats(
        subblock_stats,
        all_subblock_stats,
        dict(args.get("workload_stats_args", {}) or {}),
    )
    teacher_reference_metrics = deepcopy(gathered_metrics)
    gathered_metrics = _filter_tp_incompatible_replacements(
        gathered_metrics,
        int(args.get("materialization_tp", 1) or 1),
    )
    _add_block_stats_to_gathered_metrics(teacher_reference_metrics, subblock_stats)
    forced_removals = list(args.get("forced_removals", []) or [])
    if forced_removals:
        gathered_metrics = _apply_forced_removals(
            gathered_metrics,
            forced_removals,
            objective=str(args.objective),
            bigger_is_better=bool(args.bigger_is_better),
        )
    gathered_metrics = filter_replacements_by_axes(
        gathered_metrics,
        axes_default=str(args.get("axes_default", "all")),
        axis_options=dict(args.get("axis_options", {}) or {}),
        teacher_replacements=teacher_reference_metrics,
    )
    _add_block_stats_to_gathered_metrics(gathered_metrics, subblock_stats)

    output_folder.mkdir(parents=True, exist_ok=True)
    _dump_gathered_metrics(gathered_metrics, output_folder)

    non_block_stats = {"stats": _get_block_stats(subblock_stats, "non_block")}
    if constraints.has_ratio_constraints():
        teacher_stats = _get_teacher_total_stats(
            teacher_reference_metrics,
            non_block_stats["stats"],
        )
        constraints = constraints.resolve_ratio_constraints(teacher_stats)
    batch_size = subblock_stats["args"]["batch_size"]
    generation_seq_len = subblock_stats["args"]["generation_seq_len"]

    mip_constraints = constraints.to_mip_constraints(subblock_stats["args"])
    orig_mip_constraints = deepcopy(mip_constraints)
    mprint(f"Solving for the following MIP constraints: {mip_constraints}")
    args.mip_constraints = orig_mip_constraints
    args.human_constraints = (
        constraints.constraints if constraints.type == PuzzleConstraints.Type.HUMAN else None
    )
    args.subblock_stats_args = subblock_stats_args

    for stat_name, max_cost in mip_constraints.items():
        try:
            non_block_cost = get_nested_key(non_block_stats, stat_name)
        except KeyError:
            non_block_cost = 0

        is_min_max = isinstance(max_cost, Iterable)
        min_cost = None
        if is_min_max:
            min_cost, max_cost = max_cost

        min_cost = min_cost - non_block_cost if (min_cost is not None) else None
        max_cost = max_cost - non_block_cost if (max_cost is not None) else None

        if is_min_max:
            mip_constraints[stat_name] = (min_cost, max_cost)
        else:
            mip_constraints[stat_name] = max_cost

    # If there's an additional cost that is not a constraint - set it to "inf" so MIP report the actual value of it.
    for cost in set(report_additional_costs) - set(orig_mip_constraints.keys()):
        mip_constraints[cost] = np.inf

    mprint(f"After non-block adjustments: {mip_constraints=}")

    solutions = run_mip_with_backend(
        replacements=gathered_metrics,
        objective=args.objective,
        constraints=mip_constraints,
        bigger_is_better=args.bigger_is_better,
        max_seconds_per_solution=args.get("max_seconds_per_solution", None),
        num_solutions=int(args.get("num_solutions", 1)),
        min_hamming_distance=int(args.get("min_hamming_distance", 1)),
        solver_backend=args.get("solver_backend", args.get("use_cuopt", None)),
    )

    homogeneous_solutions = rank_homogeneous_solutions(
        gathered_metrics,
        objective=args.objective,
        constraints=mip_constraints,
        bigger_is_better=args.bigger_is_better,
        num_solutions=int(args.get("num_homogeneous_solutions", 0) or 0),
        rank_by=str(args.get("homogeneous_rank_by", "objective")),
        constraint_weights=dict(args.get("homogeneous_constraint_weights", {}) or {}),
    )

    for solution in [*solutions, *homogeneous_solutions]:
        for stat_name in set([*orig_mip_constraints.keys(), *report_additional_costs]):
            try:
                non_block_cost = get_nested_key(non_block_stats, stat_name)
            except KeyError:
                non_block_cost = 0
            solution["total_costs"][stat_name] += non_block_cost

        # Calculate throughput from runtime_ms
        if "stats.runtime_ms" in solution["total_costs"]:
            total_runtime = solution["total_costs"]["stats.runtime_ms"]
            solution["total_costs"]["throughput"] = (
                1000 * batch_size * generation_seq_len / total_runtime
            )

        solution["total_value"] = {args.objective: solution["total_value"]}
        solution["puzzle_args"] = (
            OmegaConf.to_container(args, resolve=True)
            if isinstance(args, DictConfig)
            else vars(args)
        )
        solution["subblock_stats"] = subblock_stats["args"]
        chosen_block_configs, _ = extract_block_configs_and_locations(
            solution["chosen_replacements"]
        )
        solution["chosen_block_configs"] = chosen_block_configs
        solution["solution_repr"] = solution_to_str(chosen_block_configs)

    if len(solutions) > 0:
        solution_repr_0 = solutions[0]["solution_repr"]
        mprint(f"\n{solution_repr_0}")
        mprint(f"Total costs: {solutions[0]['total_costs']}")
        (output_folder / "solution_repr_0.txt").write_text(solution_repr_0)

    solutions_file = output_folder / "solutions.json"
    json_dump(solutions, solutions_file)
    if int(args.get("num_homogeneous_solutions", 0) or 0) != 0:
        json_dump(homogeneous_solutions, output_folder / "homogeneous_solutions.json")
    mprint(solutions_file)
    return solutions_file


def _dump_gathered_metrics(gathered_metrics: PuzzleMetrics, output_folder: Path) -> None:
    for replacement_id, replacement_info in gathered_metrics.items():
        replacement_info["block_repr"] = block_config_to_str(replacement_info["block_config"])
    gathered_metrics_for_dump = gathered_metrics

    json_dump(gathered_metrics_for_dump, output_folder / "replacement_metrics_and_stats.json")


def _apply_forced_removals(
    gathered_metrics: dict,
    forced_removals: list[dict],
    *,
    objective: str,
    bigger_is_better: bool,
) -> dict:
    """Create scenario-local no-op configs while keeping the canonical library no-op-free."""
    import dataclasses

    removals_by_layer: dict[int, set[str]] = defaultdict(set)
    for removal in forced_removals:
        removals_by_layer[int(removal["layer_idx"])].add(str(removal["kind"]))

    teacher_blocks: dict[int, BlockConfig] = {}
    for replacement in gathered_metrics.values():
        if replacement.get("is_teacher", False):
            layer_idx = int(replacement["parent_layer_indices"][0])
            teacher_blocks[layer_idx] = replacement["block_config"]
    missing = sorted(set(removals_by_layer) - set(teacher_blocks))
    if missing:
        raise ValueError(f"forced-depth scenarios are missing teacher blocks for layers {missing}")

    transformed: dict[str, dict] = {}
    dedupe: dict[tuple[int, str], tuple[str, float]] = {}
    serial = 0
    for replacement_id, replacement in gathered_metrics.items():
        layer_indices = replacement.get("parent_layer_indices", [])
        if len(layer_indices) != 1:
            raise ValueError("forced-depth MIP currently requires one-layer replacements")
        layer_idx = int(layer_indices[0])
        kinds = removals_by_layer.get(layer_idx)
        candidate = deepcopy(replacement)
        if kinds:
            block = candidate["block_config"]
            teacher = teacher_blocks[layer_idx]
            compare_kinds = kinds - {"block"}
            if "block" in kinds and block != teacher:
                continue
            if any(
                block.get_subblock(kind) != teacher.get_subblock(kind) for kind in compare_kinds
            ):
                continue
            if "block" in kinds:
                for subblock in tuple(block.subblock_configs):
                    block = block.with_subblock(dataclasses.replace(subblock, no_op=True))
            else:
                for kind in sorted(kinds):
                    subblock = block.require_subblock(kind)
                    block = block.with_subblock(dataclasses.replace(subblock, no_op=True))
            candidate["block_config"] = block
            candidate["is_teacher"] = False
            layer_replacement = candidate.get("layer_replacement")
            if layer_replacement is not None:
                layer_replacement = deepcopy(layer_replacement)
                layer_replacement["child_block_configs"] = [block]
                candidate["layer_replacement"] = layer_replacement

        key = (layer_idx, str(candidate["block_config"]))
        try:
            value = float(get_nested_key(candidate, objective))
        except (KeyError, TypeError, ValueError):
            value = float("-inf") if bigger_is_better else float("inf")
        existing = dedupe.get(key)
        if existing is not None and (
            existing[1] >= value if bigger_is_better else existing[1] <= value
        ):
            continue
        if existing is not None:
            transformed.pop(existing[0], None)
        new_id = f"depth_{layer_idx}_{serial}"
        serial += 1
        transformed[new_id] = candidate
        dedupe[key] = (new_id, value)

    covered = {int(item["parent_layer_indices"][0]) for item in transformed.values()}
    expected = {int(item["parent_layer_indices"][0]) for item in gathered_metrics.values()}
    if covered != expected:
        raise ValueError(f"forced-depth transform lost layers: {sorted(expected - covered)}")
    return transformed


def _load_all_constraints(args, puzzle_profile):
    def parse_constraints(constraints, constraints_type: PuzzleConstraints.Type):
        if OmegaConf.is_config(constraints):
            constraints = OmegaConf.to_container(constraints, resolve=True)
        if isinstance(constraints, (list, ListConfig)):
            return [PuzzleConstraints(type=constraints_type, constraints=c) for c in constraints]
        elif isinstance(constraints, (dict, DictConfig)):
            return [PuzzleConstraints(type=constraints_type, constraints=constraints)]
        raise TypeError(f"Invalid constraints type: {constraints_type}")

    # Constraints can be given explicitely
    mip_constraints = args.get("mip_constraints", None)
    human_constraints = args.get("human_constraints", None)
    if mip_constraints is not None:
        return parse_constraints(mip_constraints, PuzzleConstraints.Type.MIP)

    if human_constraints is not None:
        return parse_constraints(human_constraints, PuzzleConstraints.Type.HUMAN)

    # Or through the puzzle_profile
    if "mip_constraints" in puzzle_profile:
        return parse_constraints(puzzle_profile["mip_constraints"], PuzzleConstraints.Type.MIP)

    if "human_constraints" in puzzle_profile:
        return parse_constraints(puzzle_profile["human_constraints"], PuzzleConstraints.Type.HUMAN)

    raise ValueError(
        "Constraints must be given either explicitely by --mip_constraints or --human_constraints arguments, or through the puzzle_profile."
    )


def _load_all_subblock_stats_args(args, puzzle_profile):
    # If given explicitely in args
    subblock_stats_args = args.get("subblock_stats_args", None)
    if OmegaConf.is_config(subblock_stats_args):
        subblock_stats_args = OmegaConf.to_container(subblock_stats_args, resolve=True)
    if subblock_stats_args is not None:
        if isinstance(subblock_stats_args, dict):
            return [subblock_stats_args]
        else:
            return subblock_stats_args

    # Or can be given inside puzzle_profile
    if "subblock_stats_args" in puzzle_profile:
        return puzzle_profile["subblock_stats_args"]

    raise ValueError(
        "subblock_stats_args must be given either explicitely by the --subblock_stats_args argument, or through the puzzle_profile."
    )


def _override_args_from_profile(args, puzzle_profile):
    for arg_name in vars(args):
        if arg_name in puzzle_profile:
            if arg_name not in ("mip_constraints", "human_constraints", "subblock_stats_args"):
                setattr(args, arg_name, puzzle_profile[arg_name])


def _assert_valid_config(args, puzzle_profile):
    required_args = (
        "subblock_stats_path",
        "objective",
        "output_path",
    )
    missing_args = [
        arg for arg in required_args if not hasattr(args, arg) or getattr(args, arg) is None
    ]
    if missing_args:
        mprint(f"error: The following arguments are required: {', '.join(missing_args)}")
        sys.exit(1)

    # Make sure we have specified subblock_stats_args
    if not hasattr(args, "subblock_stats_args") and "subblock_stats_args" not in puzzle_profile:
        mprint(
            "error: Must specify `subblock_stats_args` in either puzzle_profile or as a commandline arg."
        )
        sys.exit(1)

    # Make sure we have specified constraints
    if (
        not hasattr(args, "mip_constraints")
        and not hasattr(args, "human_constraints")
        and "mip_constraints" not in puzzle_profile
        and "human_constraints" not in puzzle_profile
    ):
        mprint(
            "error: Must specify either `mip_constraints` or `human_constraints` in one of puzzle_profile or as a commandline argument."
        )
        sys.exit(1)


def _get_minimal_unique_names(dicts: list[dict]) -> list[str]:
    if len(dicts) == 1:
        return ["default"]

    def _stable_value(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        return value

    def _safe_name(value):
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        return str(value).replace(".", "_").replace("/", "_")

    all_keys = set(k for d in dicts for k in d.keys())
    all_values = {k: set(_stable_value(d[k]) for d in dicts if k in d) for k in all_keys}
    non_common_keys = [k for k, values in all_values.items() if len(values) > 1]

    return ["-".join(f"{k}_{_safe_name(d[k])}" for k in non_common_keys) for d in dicts]


def run_puzzle(args: DictConfig) -> list[str]:
    # Loads config from args/puzzle_profile
    if args.puzzle_profile is not None:
        with open(args.puzzle_profile) as f:
            puzzle_profile = yaml.safe_load(f)
        _override_args_from_profile(args, puzzle_profile)
        mprint(f"Loaded Puzzle profile from {args.puzzle_profile}")
    else:
        puzzle_profile = {}
    _assert_valid_config(args, puzzle_profile)

    # Read Metrics and Stats
    score_granularity = str(args.get("score_granularity", "block")).lower()
    if score_granularity not in {"block", "subblock"}:
        raise ValueError(f"unsupported MIP score_granularity={score_granularity!r}")
    if args.gathered_metrics_path is not None:
        gathered_metrics = json.loads(args.gathered_metrics_path.read_text())
    elif score_granularity == "subblock":
        canonical_path = args.get("canonical_solutions_path", None)
        if canonical_path is None:
            raise ValueError("MIP score_granularity=subblock requires canonical_solutions_path")
        gathered_metrics = gather_composed_subblock_puzzle_metrics(
            Path(canonical_path),
            args.single_block_replacement_validation_dir,
            exact_block_validation_dir=(
                Path(args.exact_block_replacement_validation_dir)
                if args.get("exact_block_replacement_validation_dir", None)
                else None
            ),
        )
    else:
        gathered_metrics = gather_multi_layer_puzzle_metrics(
            args.single_block_replacement_validation_dir
        )

    metric_overrides = args.get("metric_overrides", None)
    if metric_overrides is not None:
        gathered_metrics = {**gathered_metrics, **metric_overrides}

    subblock_stats = json.loads(args.subblock_stats_path.read_text())

    all_subblock_args = _load_all_subblock_stats_args(args, puzzle_profile)
    all_subblock_output_folders = [
        args.output_path / unique_name
        for unique_name in _get_minimal_unique_names(all_subblock_args)
    ]
    all_constraints = _load_all_constraints(args, puzzle_profile)

    # Running all puzzles
    solution_paths = []
    for subblock_stats_args, subblock_stats_output_folder in zip(
        all_subblock_args, all_subblock_output_folders
    ):
        for constraints in all_constraints:
            output_folder = subblock_stats_output_folder / constraints.name
            _solution_path = run_single_puzzle_config(
                args,
                gathered_metrics,
                subblock_stats,
                subblock_stats_args,
                constraints,
                output_folder,
            )
            solution_paths.append(_solution_path)
    return solution_paths


def gather_puzzle_metrics(
    single_block_replacement_validation_dir: Path,
) -> PuzzleMetrics:
    single_block_metrics = [
        _parse_single_block_replacement_metrics(metrics_path)
        for metrics_path in single_block_replacement_validation_dir.glob("*solution*.json")
    ]
    all_metric_names = tuple(single_block_metrics[0]["metrics"].keys())
    teacher_metrics = _parse_teacher_block_metrics(
        single_block_replacement_validation_dir, all_metric_names
    )

    n_layer = len(teacher_metrics)
    gathered_metrics = {f"block_{block_idx}": dict() for block_idx in range(n_layer)}
    for variant_metrics in single_block_metrics + teacher_metrics:
        block_config = variant_metrics["block_config"]
        block_name = f"block_{variant_metrics['block_idx']}"
        # if we explicitly measure teacher's blocks don't override them
        gathered_metrics[block_name][block_config] = variant_metrics
        # if not gathered_metrics[block_name].get(block_config):
        #     gathered_metrics[block_name][block_config] = variant_metrics
    return gathered_metrics


def gather_multi_layer_puzzle_metrics(
    single_replacement_validation_dir: Path,
) -> MultiLayerPuzzleMetrics:
    single_sequence_metrics = [
        _parse_single_sequence_replacement_metrics(metrics_path)
        for metrics_path in single_replacement_validation_dir.glob("*solution*.json")
    ]
    all_metric_names = tuple(single_sequence_metrics[0]["metrics"].keys())
    teacher_metrics = _parse_teacher_block_metrics(
        single_replacement_validation_dir, all_metric_names
    )

    gathered_metrics = {
        f"replacement_{replacement_id}": replacement_metrics
        for replacement_id, replacement_metrics in enumerate(
            single_sequence_metrics + teacher_metrics
        )
    }

    return gathered_metrics


def gather_composed_subblock_puzzle_metrics(
    canonical_solutions_path: Path,
    subblock_validation_dir: Path,
    *,
    exact_block_validation_dir: Path | None = None,
) -> MultiLayerPuzzleMetrics:
    """Compose atomic subblock scores into the canonical full-block MIP table."""

    result_paths = sorted(subblock_validation_dir.glob("*solution*.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No atomic solution_*.json files found in {subblock_validation_dir}"
        )
    subblock_results = [json.loads(path.read_text()) for path in result_paths]
    sliced_baseline = subblock_results[0].get("sliced_teacher_baseline") or {}
    teacher_baseline = {
        name: float(value["avg"])
        for name, value in sliced_baseline.items()
        if isinstance(value, dict) and isinstance(value.get("avg"), (int, float))
    }
    if not teacher_baseline:
        raise ValueError(
            f"subblock score {result_paths[0]} has no sliced_teacher_baseline averages"
        )
    teacher_records = _parse_teacher_block_metrics(subblock_validation_dir, teacher_baseline.keys())
    teacher_records.sort(key=lambda record: int(record["block_idx"]))
    teacher_blocks = [record["block_config"] for record in teacher_records]

    canonical_solutions = json.loads(canonical_solutions_path.read_text())
    if not isinstance(canonical_solutions, list):
        raise TypeError("canonical_solutions_path must contain a JSON list")
    exact_results = []
    if exact_block_validation_dir is not None and exact_block_validation_dir.is_dir():
        exact_results = [
            json.loads(path.read_text())
            for path in sorted(exact_block_validation_dir.glob("*solution*.json"))
        ]
    table = compose_full_block_metrics(
        canonical_solutions,
        subblock_results,
        teacher_blocks=teacher_blocks,
        teacher_baseline=teacher_baseline,
        exact_results=exact_results,
    )
    return composed_table_to_gathered_metrics(table, canonical_solutions, teacher_records)


def _parse_single_block_replacement_metrics(metrics_path: Path) -> dict:
    raw_metrics = json.loads(metrics_path.read_text())
    single_block_replacement = raw_metrics["puzzle_solution"]["single_block_replacement"]
    variant_metrics = {
        "block_config": BlockConfig(**single_block_replacement["block_config"]),
        "block_idx": single_block_replacement["block_idx"],
        "metrics": _extract_average_metrics(raw_metrics),
    }
    return variant_metrics


def _parse_single_sequence_replacement_metrics(metrics_path: Path) -> dict:
    raw_metrics = json.loads(metrics_path.read_text())
    single_sequence_replacement = raw_metrics["puzzle_solution"]["single_sequence_replacement"]
    if len(single_sequence_replacement["child_block_configs"]) > 1:
        raise NotImplementedError(
            "Currently we only support many-to-1 replacements, but we can support many-to-many! "
        )
    variant_metrics = {
        "block_config": BlockConfig(**single_sequence_replacement["child_block_configs"][0]),
        # is there cases where child_block_configs has more than one entry?
        "parent_layer_indices": single_sequence_replacement["parent_layer_indices"],
        "metrics": _extract_average_metrics(raw_metrics),
        "layer_replacement": parse_layer_replacement(single_sequence_replacement),
        "is_teacher": False,
    }
    return variant_metrics


def _parse_teacher_block_metrics(
    single_block_replacement_validation_dir: Path,
    all_metric_names: Iterable[str] = ("kl_div_loss",),
) -> list[dict]:
    teacher_path = single_block_replacement_validation_dir / "teacher.json"
    if teacher_path.exists():
        raw_metrics = json.loads(teacher_path.read_text())
    else:
        solution_paths = sorted(single_block_replacement_validation_dir.glob("*solution*.json"))
        if not solution_paths:
            raise FileNotFoundError(
                f"No teacher.json or solution_*.json files found in "
                f"{single_block_replacement_validation_dir}"
            )
        first_solution = json.loads(solution_paths[0].read_text())
        raw_metrics = {"args": first_solution["args"]}
        sliced_baseline = first_solution.get("sliced_teacher_baseline") or {}
        for metric_name in all_metric_names:
            if metric_name.startswith("one_minus_"):
                continue
            baseline_value = sliced_baseline.get(metric_name)
            if isinstance(baseline_value, dict):
                baseline_value = baseline_value.get("avg")
            value = (
                baseline_value
                if isinstance(baseline_value, (int, float))
                else 1.0
                if metric_name.startswith("token_accuracy")
                else 0.0
            )
            raw_metrics[metric_name] = {"avg": value, "per_sample": []}
        mprint(
            f"{teacher_path} not found; synthesized teacher metrics from the "
            f"sliced-teacher baseline in {solution_paths[0].name}"
        )
    teacher_checkpoint_dir = Path(raw_metrics["args"]["teacher_dir"]).resolve()
    descriptor_name = raw_metrics["args"]["descriptor"]
    descriptor = ModelDescriptorFactory.get(descriptor_name)
    trust_remote_code = descriptor.requires_trust_remote_code()
    teacher_model_config = load_model_config(
        teacher_checkpoint_dir, trust_remote_code=trust_remote_code
    )

    teacher_replacements = None
    replacement_library_path = raw_metrics["args"].get("replacement_library_path")
    if replacement_library_path is not None:
        teacher_replacements = dict()
        raw_replacement_library = json.loads(Path(replacement_library_path).read_text())
        if (
            isinstance(raw_replacement_library, dict)
            and raw_replacement_library.get("version") == 2
        ):
            all_layer_replacements = raw_replacement_library.get("entries", [])
            for layer_replacement in all_layer_replacements:
                layer_replacement.setdefault("weight_paths", [])
        else:
            all_layer_replacements = raw_replacement_library
        for layer_replacement in all_layer_replacements:
            layer_replacement = parse_layer_replacement(layer_replacement)
            if replacement_is_teacher(
                layer_replacement, teacher_model_config, teacher_checkpoint_dir
            ):
                block_idx = layer_replacement["parent_layer_indices"][0]
                teacher_replacements[block_idx] = layer_replacement

    teacher_metrics = [
        {
            "block_config": block_config,
            "block_idx": block_idx,
            "parent_layer_indices": [block_idx],
            "metrics": {
                **dict.fromkeys(all_metric_names, 0.0),  # default value 0. for teacher
                **_extract_average_metrics(raw_metrics),  # override with real value if exists
            },
            **(
                {"layer_replacement": teacher_replacements[block_idx]}
                if teacher_replacements is not None
                else {}
            ),
            "is_teacher": True,
        }
        for block_idx, block_config in enumerate(teacher_model_config.block_configs)
    ]
    return teacher_metrics


def _extract_average_metrics(raw_metrics: dict[str, dict]) -> dict[str, float]:
    average_metrics = dict()
    for metric_name in raw_metrics:
        metric_dict = raw_metrics[metric_name]
        if isinstance(metric_dict, dict) and ("avg" in metric_dict.keys()):
            metric_value = raw_metrics[metric_name]["avg"]
            average_metrics[metric_name] = metric_value
            average_metrics[f"one_minus_{metric_name}"] = 1 - metric_value
    return average_metrics


def filter_subblock_stats_by_args(
    all_subblock_stats: list[dict],
    subblock_stats_args: dict[str, Any],
    convert_dicts_to_dataclasses: bool = True,
) -> dict[str, dict]:
    subblock_stats_args = _normalize_subblock_stats_args(dict(subblock_stats_args or {}))
    matching_subblock_stats = [
        subblock_stats
        for subblock_stats in all_subblock_stats
        if _dict_is_subset(subblock_stats_args, subblock_stats["args"])
    ]
    if not matching_subblock_stats:
        raise ValueError(
            "No exact subblock statistics identity matches the requested scenario. "
            "Puzzletron will not substitute measurements from a different hidden width, "
            "workload, dtype, or runtime implementation. "
            f"requested={subblock_stats_args}, "
            f"available={[entry.get('args') for entry in all_subblock_stats]}"
        )
    if len(matching_subblock_stats) > 1:
        runtime_matches = [
            stats for stats in matching_subblock_stats if stats["args"].get("runtime_stats", False)
        ]
        block_runtime_matches = [
            stats
            for stats in runtime_matches
            if stats["args"].get("runtime_granularity") == "block"
        ]
        if len(block_runtime_matches) == 1:
            matching_subblock_stats = block_runtime_matches
        elif len(runtime_matches) == 1:
            matching_subblock_stats = runtime_matches
    assert len(matching_subblock_stats) == 1, (
        "The provided subblock_stats_args should match exactly one measurement "
        f"scenario, instead matched {len(matching_subblock_stats)}:\n"
        f"{[m['args'] for m in matching_subblock_stats]}"
    )
    subblock_stats = deepcopy(matching_subblock_stats[0])

    if convert_dicts_to_dataclasses:
        class_name_to_class = {
            klass.__name__: klass for klass in [AttentionConfig, FFNConfig, MambaConfig, MoEConfig]
        }
        subblocks_dict = dict()
        for substats in subblock_stats["subblocks"]:
            subblock_config_class = class_name_to_class[substats.pop("subblock_config_class")]
            subblock_config = subblock_config_class(**substats.pop("subblock_config"))
            dict_key = (subblock_config, None)
            if "parent_layer_index" in substats:
                dict_key = (subblock_config, substats["parent_layer_index"])
            subblocks_dict[dict_key] = substats
        subblock_stats["subblocks"] = subblocks_dict
    return subblock_stats


def _merge_namespaced_workload_stats(
    base_stats: dict[str, Any],
    all_subblock_stats: list[dict[str, Any]],
    workloads: dict[str, dict[str, Any]],
) -> None:
    """Attach multiple workload measurements to one additive stats table."""

    for workload_name, workload_args in workloads.items():
        workload_stats = filter_subblock_stats_by_args(
            all_subblock_stats,
            workload_args,
        )
        for key, value in workload_stats["non_block"].items():
            if value is None or isinstance(value, (int, float)):
                base_stats["non_block"][f"{key}@{workload_name}"] = value
        missing = set(base_stats["subblocks"]) - set(workload_stats["subblocks"])
        if missing:
            raise ValueError(
                f"workload {workload_name!r} is missing {len(missing)} subblock measurements"
            )
        for subblock_key, base_values in base_stats["subblocks"].items():
            for key, value in workload_stats["subblocks"][subblock_key].items():
                if key == "parent_layer_index":
                    continue
                if value is None or isinstance(value, (int, float)):
                    base_values[f"{key}@{workload_name}"] = value


def _normalize_subblock_stats_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize user-facing MIP stats args to the persisted stats schema."""
    if "batch_sizes" in args and "batch_size" not in args:
        batch_sizes = args.pop("batch_sizes")
        if isinstance(batch_sizes, Iterable) and not isinstance(batch_sizes, (str, bytes, dict)):
            batch_sizes = list(batch_sizes)
            if len(batch_sizes) != 1:
                raise ValueError(
                    "MIP subblock_stats_args must select exactly one batch size, got "
                    f"batch_sizes={batch_sizes}"
                )
            args["batch_size"] = batch_sizes[0]
        else:
            args["batch_size"] = batch_sizes
    return args


def _dict_is_subset(dict1: dict, dict2: dict) -> bool:
    return all(item in dict2.items() for item in dict1.items())


def _add_block_stats_to_gathered_metrics(
    gathered_metrics: PuzzleMetrics, subblock_stats: dict
) -> None:
    for block_name, block_variants in gathered_metrics.items():
        parent_layer_index = None
        if "parent_layer_indices" in block_variants:
            parent_layer_index = block_variants["parent_layer_indices"][0]

        if "metrics" in block_variants:
            # this is a sequence stats object for multi-layer puzzle
            block_variants["stats"] = _get_block_stats(
                subblock_stats, block_variants["block_config"], parent_layer_index
            )
        else:
            for block_config, variant_metrics in block_variants.items():
                variant_parent_layer_index = None
                if isinstance(variant_metrics, dict):
                    if "parent_layer_indices" in variant_metrics:
                        variant_parent_layer_index = variant_metrics["parent_layer_indices"][0]
                    elif "block_idx" in variant_metrics:
                        variant_parent_layer_index = variant_metrics["block_idx"]
                variant_metrics["stats"] = _get_block_stats(
                    subblock_stats, block_config, variant_parent_layer_index
                )


def _iter_teacher_stats(gathered_metrics: PuzzleMetrics) -> Iterable[dict[str, float]]:
    for replacement_info in gathered_metrics.values():
        if not isinstance(replacement_info, dict):
            continue
        if "metrics" in replacement_info:
            if replacement_info.get("is_teacher", False):
                yield replacement_info["stats"]
            continue
        for variant_info in replacement_info.values():
            if isinstance(variant_info, dict) and variant_info.get("is_teacher", False):
                yield variant_info["stats"]


def _get_teacher_total_stats(
    gathered_metrics: PuzzleMetrics, non_block_stats: dict[str, float]
) -> dict[str, float]:
    total_stats = {
        "memory_mib": non_block_stats.get("memory_mib", 0),
        "num_params": non_block_stats.get("num_params", 0),
        "runtime_ms": non_block_stats.get("runtime_ms"),
    }
    num_teacher_blocks = 0
    for block_stats in _iter_teacher_stats(gathered_metrics):
        num_teacher_blocks += 1
        for stat_name in total_stats:
            total_stats[stat_name] = _none_add(total_stats[stat_name], block_stats.get(stat_name))

    if num_teacher_blocks == 0:
        raise ValueError(
            "Ratio human_constraints require teacher metrics in the gathered replacement metrics."
        )
    return total_stats


def _get_block_stats(
    subblock_stats: dict,
    block_config: BlockConfig | Literal["non_block"],
    parent_layer_index: int = None,
) -> dict[str, float]:
    if block_config == "non_block":
        return subblock_stats["non_block"]

    subblock_entries = []
    template_stats = next(iter(subblock_stats["subblocks"].values()), {})
    for subblock_ref in block_config.subblocks():
        if subblock_ref.config.no_op:
            stats = {
                key: (None if value is None else 0.0 if isinstance(value, (int, float)) else value)
                for key, value in template_stats.items()
            }
        else:
            stats = _lookup_subblock_stats(
                subblock_stats["subblocks"], subblock_ref.config, parent_layer_index
            )
        subblock_entries.append((subblock_ref.config, stats))
    if not subblock_entries:
        return {"has_attention": 0, "has_mamba": 0, "has_ffn": 0, "has_moe": 0, "not_no_op": 0}

    stat_keys = {
        key
        for key, value in subblock_entries[0][1].items()
        if value is None or isinstance(value, (int, float))
    }
    for _, stats in subblock_entries[1:]:
        numeric_keys = {
            key for key, value in stats.items() if value is None or isinstance(value, (int, float))
        }
        assert numeric_keys == stat_keys

    block_stats = dict()
    for key in stat_keys:
        block_stats[key] = _none_add_list([stats[key] for _, stats in subblock_entries])
        for subblock_config, stats in subblock_entries:
            block_stats[f"{subblock_config.kind}_{key}"] = stats[key]

        attention_like_values = [
            stats[key]
            for subblock_config, stats in subblock_entries
            if subblock_config.kind in _ATTENTION_LIKE_KINDS
        ]
        if attention_like_values:
            block_stats[f"attention_{key}"] = _none_add_list(attention_like_values)

        ffn_like_values = [
            stats[key]
            for subblock_config, stats in subblock_entries
            if subblock_config.kind in _FFN_LIKE_KINDS
        ]
        if ffn_like_values:
            block_stats[f"ffn_{key}"] = _none_add_list(ffn_like_values)

    attention = block_config.get_subblock("attention")
    mamba = block_config.get_subblock("mamba")
    ffn = block_config.get_subblock("ffn")
    moe = block_config.get_subblock("moe")

    def _active(subblock_config) -> bool:
        return subblock_config is not None and not subblock_config.no_op

    block_stats["has_attention"] = int(_active(attention))
    block_stats["has_mamba"] = int(_active(mamba))
    block_stats["has_ffn"] = int(_active(ffn))
    block_stats["has_moe"] = int(_active(moe))
    block_stats["not_no_op"] = int(
        any(not subblock_config.no_op for subblock_config, _ in subblock_entries)
    )
    block_stats["num_kv_heads"] = (
        attention.num_kv_heads if _active(attention) and attention.num_kv_heads is not None else 0
    )
    block_stats["num_query_heads"] = (
        attention.num_query_heads
        if _active(attention) and attention.num_query_heads is not None
        else 0
    )
    block_stats["num_experts"] = (
        moe.num_experts if _active(moe) and moe.num_experts is not None else 0
    )
    block_stats["top_k"] = moe.top_k if _active(moe) and moe.top_k is not None else 0

    return block_stats


def _lookup_subblock_stats(
    subblocks_stats: dict,
    subblock_config,
    parent_layer_index: int | None,
) -> dict[str, float]:
    lookup_keys = []
    if parent_layer_index is not None:
        lookup_keys.append((subblock_config, parent_layer_index))
    lookup_keys.extend([(subblock_config, None), (subblock_config, -1)])
    for key in lookup_keys:
        if key in subblocks_stats:
            return subblocks_stats[key]

    matches = [
        stats
        for (candidate_config, _candidate_parent_layer_index), stats in subblocks_stats.items()
        if candidate_config == subblock_config
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"Could not find unique stats for {subblock_config} at parent_layer_index={parent_layer_index}"
    )


def _none_add(a: float | int | None, b: float | int | None) -> float | int | None:
    if a is None or b is None:
        return None
    return a + b


def _none_max(a: float | int | None, b: float | int | None) -> float | int | None:
    if a is None or b is None:
        return None
    return max(a, b)


def _none_add_list(l) -> float | int | None:
    r = l[0]
    if len(l) == 1:
        return r
    for e in l[1:]:
        r = _none_add(r, e)
    return r


def _none_max_list(l) -> float | int | None:
    r = l[0]
    if len(l) == 1:
        return r
    for e in l[1:]:
        r = _none_max(r, e)
    return r


if __name__ == "__main__":
    args = parse_args()
    run_puzzle(args)
