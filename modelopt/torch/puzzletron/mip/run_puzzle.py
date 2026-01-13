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
from collections.abc import Hashable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
)
from modelopt.torch.puzzletron.mip.mip_with_multi_layer_replacements import (
    run_mip as run_multi_layer_replacement_mip,
)
from modelopt.torch.puzzletron.replacement_library.replacement_utils import (
    extract_block_configs_and_locations,
    parse_layer_replacement,
    replacement_is_teacher,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.robust_json import json_dump
from modelopt.torch.puzzletron.utils.parsing import get_nested_key, parse_json, parse_path
from modelopt.torch.puzzletron.utils.utils import block_config_to_str, solution_to_str

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


@dataclasses.dataclass
class PuzzleConstraints:
    """A set of puzzle constraints can be expressed either directly as the mip constraints (e.g. 'stats.memory_mib') or as human constraints (e.g. 'target_throughput')"""

    class Type(enum.Enum):
        MIP = "mip"
        HUMAN = "human"

    _ALLOWED_HUMAN_CONSTRAINTS = {
        "target_memory",
        "target_throughput",
        "target_latency",
        "target_time_to_first_token",
        "num_params",
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
            "Can't realize human constraints without 'block_size' and 'generation_seq_len' in subblock_stats_args."
        )
        batch_size = subblock_stats_args["batch_size"]
        generation_seq_len = subblock_stats_args["generation_seq_len"]

        mip_constraints = {}

        # Memory constraints
        if "target_memory" in self.constraints:
            mip_constraints["stats.memory_mib"] = self.constraints["target_memory"]

        # Throughput constraints
        throughput_constraints = []
        if "target_throughput" in self.constraints:
            throughput_constraints.append(
                batch_size * generation_seq_len / self.constraints["target_throughput"]
            )
        if "target_latency" in self.constraints:
            throughput_constraints.append(self.constraints["target_latency"])
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


def parse_args() -> argparse.Namespace:
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
    return args


def run_single_puzzle_config(
    args: argparse.Namespace | DictConfig,
    gathered_metrics: dict,
    subblock_stats: dict,
    subblock_stats_args: dict,
    constraints: PuzzleConstraints,
    output_folder,
) -> None:
    # we override the constraints and subblock_stats_args for this run to keep reporting out the same way.
    args = deepcopy(args)

    subblock_stats = filter_subblock_stats_by_args(subblock_stats, subblock_stats_args)
    _add_block_stats_to_gathered_metrics(gathered_metrics, subblock_stats)

    output_folder.mkdir(parents=True, exist_ok=True)
    _dump_gathered_metrics(gathered_metrics, output_folder)

    non_block_stats = {"stats": _get_block_stats(subblock_stats, "non_block")}
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
    for cost in set(args.report_additional_costs) - set(orig_mip_constraints.keys()):
        mip_constraints[cost] = np.inf

    mprint(f"After non-block adjustments: {mip_constraints=}")

    solutions = run_multi_layer_replacement_mip(
        replacements=gathered_metrics,
        objective=args.objective,
        constraints=mip_constraints,
        bigger_is_better=args.bigger_is_better,
        max_seconds_per_solution=args.max_seconds_per_solution,
    )

    for solution in solutions:
        for stat_name in set([*orig_mip_constraints.keys(), *args.report_additional_costs]):
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
    mprint(solutions_file)
    return solutions_file


def _dump_gathered_metrics(gathered_metrics: PuzzleMetrics, output_folder: Path) -> None:
    for replacement_id, replacement_info in gathered_metrics.items():
        replacement_info["block_repr"] = block_config_to_str(replacement_info["block_config"])
    gathered_metrics_for_dump = gathered_metrics

    json_dump(gathered_metrics_for_dump, output_folder / "replacement_metrics_and_stats.json")


def _load_all_constraints(args, puzzle_profile):
    def parse_constraints(constraints, constraints_type: PuzzleConstraints.Type):
        if isinstance(constraints, (list, ListConfig)):
            return [PuzzleConstraints(type=constraints_type, constraints=c) for c in constraints]
        elif isinstance(constraints, (dict, DictConfig)):
            return [PuzzleConstraints(type=constraints_type, constraints=constraints)]
        raise TypeError(f"Invalid constraints type: {constraints_type}")

    # Constraints can be given explicitely
    if args.mip_constraints is not None:
        return parse_constraints(args.mip_constraints, PuzzleConstraints.Type.MIP)

    if args.human_constraints is not None:
        return parse_constraints(args.human_constraints, PuzzleConstraints.Type.HUMAN)

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
    if args.subblock_stats_args is not None:
        if isinstance(args.subblock_stats_args, dict):
            return [args.subblock_stats_args]
        else:
            return args.subblock_stats_args

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
    missing_args = [arg for arg in required_args if arg not in args or getattr(args, arg) is None]
    if missing_args:
        mprint(f"error: The following arguments are required: {', '.join(missing_args)}")
        exit(1)

    # Make sure we have specified subblock_stats_args
    if "subblock_stats_args" not in args and "subblock_stats_args" not in puzzle_profile:
        mprint(
            "error: Must specify `subblock_stats_arrs` in either puzzle_profile or as a commandline arg."
        )
        exit(1)

    # Make sure we have specified constraints
    if (
        "mip_constraints" not in args
        and "human_constraints" not in args
        and "mip_constraints" not in puzzle_profile
        and "human_constraints" not in puzzle_profile
    ):
        mprint(
            "error: Must specify either `mip_constraints` or `human_constraints` in one of puzzle_profile or as a commandline argument."
        )
        exit(1)


def _get_minimal_unique_names(dicts: list[dict]) -> list[str]:
    all_keys = set(k for d in dicts for k in d.keys())
    all_values = {k: set(d[k] for d in dicts if k in d) for k in all_keys}
    non_common_keys = [k for k, values in all_values.items() if len(values) > 1]

    return ["-".join(f"{k}_{d[k]}".replace(".", "_") for k in non_common_keys) for d in dicts]


def run_puzzle(args: argparse.Namespace | DictConfig) -> list[str]:
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
    if args.gathered_metrics_path is not None:
        gathered_metrics = json.loads(args.gathered_metrics_path.read_text())
    else:
        gathered_metrics = gather_multi_layer_puzle_metrics(
            args.single_block_replacement_validation_dir
        )

    if args.metric_overrides is not None:
        gathered_metrics = {**gathered_metrics, **args.metric_overrides}

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


def gather_multi_layer_puzle_metrics(
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
    raw_metrics = json.loads((single_block_replacement_validation_dir / "teacher.json").read_text())
    teacher_checkpoint_dir = Path(raw_metrics["args"]["teacher_dir"]).resolve()
    teacher_model_config = load_model_config(teacher_checkpoint_dir)

    teacher_replacements = None
    replacement_library_path = raw_metrics["args"].get("replacement_library_path")
    if replacement_library_path is not None:
        teacher_replacements = dict()
        all_layer_replacements = json.loads(Path(replacement_library_path).read_text())
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
    matching_subblock_stats = [
        subblock_stats
        for subblock_stats in all_subblock_stats
        if _dict_is_subset(subblock_stats_args, subblock_stats["args"])
    ]
    assert len(matching_subblock_stats) == 1, (
        "The provided subblock_stats_args should match exactly one measurement "
        f"scenario, instead matched {len(matching_subblock_stats)}:\n"
        f"{[m['args'] for m in matching_subblock_stats]}"
    )
    subblock_stats = deepcopy(matching_subblock_stats[0])

    if convert_dicts_to_dataclasses:
        class_name_to_class = {klass.__name__: klass for klass in [AttentionConfig, FFNConfig]}
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
                variant_metrics["stats"] = _get_block_stats(subblock_stats, block_config)


def _get_block_stats(
    subblock_stats: dict,
    block_config: BlockConfig | Literal["non_block"],
    parent_layer_index: int = None,
) -> dict[str, float]:
    if block_config == "non_block":
        return subblock_stats["non_block"]

    if block_config.parallel_blocks is None:
        attention_key = (block_config.attention, parent_layer_index)
        ffn_key = (block_config.ffn, parent_layer_index)
        attention_stats = subblock_stats["subblocks"][attention_key]
        ffn_stats = subblock_stats["subblocks"][ffn_key]
        assert set(attention_stats.keys()) == set(ffn_stats.keys())

        block_stats = dict()
        for k in attention_stats.keys():
            block_stats[k] = _none_add(attention_stats[k], ffn_stats[k])
            block_stats[f"attention_{k}"] = attention_stats[k]
            block_stats[f"ffn_{k}"] = ffn_stats[k]

        block_stats["has_attention"] = int(
            not block_config.attention.no_op and block_config.attention.mamba is None
        )
        block_stats["has_ffn"] = int(not block_config.ffn.no_op)
        block_stats["has_moe"] = int(block_config.ffn.moe is not None)
        block_stats["not_no_op"] = int(
            not (block_config.attention.no_op and block_config.ffn.no_op)
        )
        block_stats["num_kv_heads"] = (
            subblock_stats["args"]["n_head"] // block_config.attention.n_heads_in_group
            if block_stats["has_attention"]
            else 0
        )
        block_stats["num_local_experts"] = (
            block_config.ffn.moe.num_local_experts if block_stats["has_moe"] else 0
        )

        return block_stats

    # this is a parallel block
    ADDITIVE_METRICS = ("memory_mib", "num_params", "kv_cache_memory_mib")
    ADDITIVE_METRICS = [
        f"{prefix}{metric}" for prefix in ("", "attention_", "ffn_") for metric in ADDITIVE_METRICS
    ]
    block_stats = [
        _get_block_stats(subblock_stats, sub_parallel)
        for sub_parallel in block_config.parallel_blocks
    ]
    block_stats = {
        k: _none_add_list([sub_parallel_stat[k] for sub_parallel_stat in block_stats])
        if k in ADDITIVE_METRICS
        else _none_max_list([sub_parallel_stat[k] for sub_parallel_stat in block_stats])
        for k in block_stats[0].keys()
    }

    return block_stats


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
