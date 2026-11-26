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

"""Load deployment cases and resolve explicit run overrides without importing engines."""

import copy
import json
import re
from pathlib import Path

import yaml

__all__ = [
    "BACKENDS",
    "engine_tp_kwargs",
    "load_deploy_cases",
    "load_deploy_config",
    "resolve_deploy_config",
]

BACKENDS = ("vllm", "sglang", "trtllm")
_PARAMETER_MAPS = {"engine_kwargs", "sampling_kwargs", "chat_kwargs"}
_BACKEND_FIELDS = _PARAMETER_MAPS | {"mini_sm", "sm_family", "sm_versions", "engine_version"}
_CASE_FIELDS = {
    "model_id",
    "tensor_parallel_size",
    "mini_sm",
    "base_model",
    "input_config",
    "require_speculation",
    "task",
    "revision",
    "pipeline_class",
    "backends",
}
_RESERVED_ENGINE_FIELDS = {"model", "model_path", "tensor_parallel_size", "tp_size"}


class _LocatedYAMLError(yaml.YAMLError):
    def __init__(self, reason, node):
        super().__init__(reason)
        self.problem_mark = node.start_mark


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_object(self, node, deep=False):
        try:
            return super().construct_object(node, deep=deep)
        except (ValueError, TypeError, LookupError, ArithmeticError, AttributeError):
            # PyYAML's native scalar constructors can echo the value, e.g. invalid !!int.
            raise _LocatedYAMLError("invalid YAML value", node) from None

    def flatten_mapping(self, node):
        # Aliases can revisit a node whose inherited keys have already been flattened.
        if getattr(node, "_modelopt_flattened", False):
            return
        # PyYAML also flattens inline merge sources without calling construct_mapping on them.
        merges = [key for key, _ in node.value if key.tag == "tag:yaml.org,2002:merge"]
        if len(merges) > 1:
            raise _LocatedYAMLError("Duplicate YAML merge key", merges[1])
        keys = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=False)
            try:
                if key in keys:
                    raise _LocatedYAMLError("Duplicate YAML key", key_node)
                keys.add(key)
            except TypeError:
                raise _LocatedYAMLError("Invalid YAML key", key_node) from None
        super().flatten_mapping(node)
        node._modelopt_flattened = True


def _mapping(value, name, allowed=None):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    if allowed is not None and (unknown := value.keys() - allowed):
        raise ValueError(f"Unknown {name} field(s): {sorted(unknown)}")
    return value


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_yaml(path):
    try:
        with Path(path).open(encoding="utf-8") as source:
            data = yaml.load(source, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, _LocatedYAMLError) else "invalid YAML"
        mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ValueError(f"{path}: {reason}{where}") from None
    return _mapping(data, str(path))


def _parameter_maps(config, name):
    for field in _PARAMETER_MAPS & config.keys():
        values = _mapping(config[field], f"{name}.{field}")
        if field == "engine_kwargs" and (reserved := values.keys() & _RESERVED_ENGINE_FIELDS):
            raise ValueError(
                f"Set model_id/tensor_parallel_size outside engine_kwargs: {sorted(reserved)}"
            )
        try:
            json.dumps(values, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{field} must contain JSON-compatible values") from exc


def load_deploy_config(path=None):
    """Read optional run/backend overrides; nested engine fields replace whole fields."""
    config = _read_yaml(path) if path is not None else {}
    _mapping(config, "deploy config", {"run", "backends"})
    run = _mapping(config.get("run", {}), "run", {"backend", "tensor_parallel_size"})
    if "backend" in run and run["backend"] not in BACKENDS:
        raise ValueError(f"run.backend must be one of {BACKENDS}")
    if "tensor_parallel_size" in run:
        _positive_int(run["tensor_parallel_size"], "run.tensor_parallel_size")
    backends = _mapping(config.get("backends", {}), "backends", set(BACKENDS))
    for backend, values in backends.items():
        _mapping(values, f"backends.{backend}", _PARAMETER_MAPS)
        _parameter_maps(values, f"backends.{backend}")
    return {"run": run, "backends": backends}


def load_deploy_cases(directory):
    """Load checkpoint files in path order and reject duplicate group/model/backend cases."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Deployment case directory does not exist: {directory}")
    paths = sorted(
        path for path in directory.rglob("*") if path.suffix in {".yaml", ".yml"} and path.is_file()
    )
    if not paths:
        raise ValueError(f"No deployment YAML files in {directory}")
    groups, sources = {}, {}
    for path in paths:
        try:
            file_groups = _load_case_file(path)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        for group, cases in file_groups.items():
            for case in cases:
                for backend in case["backends"]:
                    identity = (group, case["model_id"], backend)
                    if identity in sources:
                        raise ValueError(
                            f"Duplicate deployment case {identity} in {sources[identity]} and {path}"
                        )
                    sources[identity] = path
            groups.setdefault(group, []).extend(cases)
    return groups


def _load_case_file(path):
    groups = _read_yaml(path)
    if not groups:
        raise ValueError("Deployment case file must not be empty")
    for group, cases in groups.items():
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{group} must contain a nonempty case list")
        for case in cases:
            _mapping(case, group, _CASE_FIELDS)
            model = case.get("model_id")
            if not isinstance(model, str) or not model:
                raise ValueError(f"{group}: model_id must be a nonempty string")
            _positive_int(case.get("tensor_parallel_size", 1), f"{model}.tensor_parallel_size")
            if "mini_sm" in case:
                _positive_int(case["mini_sm"], f"{model}.mini_sm")
            if "base_model" in case and not isinstance(case["base_model"], str):
                raise ValueError(f"{model}.base_model must be a string")
            if "task" in case and case["task"] != "image-generation":
                raise ValueError(f"{model}: task must be image-generation when specified")
            if case.get("task") == "image-generation":
                if not isinstance(case.get("revision"), str) or not re.fullmatch(
                    r"[0-9a-f]{40}", case["revision"]
                ):
                    raise ValueError(f"{model}: image deployment requires a complete HF revision")
                if not isinstance(case.get("pipeline_class"), str) or not case["pipeline_class"]:
                    raise ValueError(f"{model}: image deployment requires pipeline_class")
                if {"base_model", "require_speculation"} & case.keys():
                    raise ValueError(f"{model}: image deployment does not accept a draft model")
            elif {"revision", "pipeline_class"} & case.keys():
                raise ValueError(f"{model}: revision/pipeline_class require task=image-generation")
            if "require_speculation" in case:
                if type(case["require_speculation"]) is not bool:
                    raise ValueError(f"{model}.require_speculation must be a boolean")
                if case["require_speculation"] and not case.get("base_model"):
                    raise ValueError(f"{model}: speculative execution requires base_model")
            if "input_config" in case:
                inputs = _mapping(
                    case["input_config"], f"{model}.input_config", {"prompt", "image"}
                )
                for field in {"prompt"} | inputs.keys():
                    if not isinstance(inputs.get(field), str) or not inputs[field]:
                        raise ValueError(f"{model}.input_config.{field} must be a nonempty string")
                if "image" in inputs:
                    image = (path.parent / inputs["image"]).resolve()
                    if not image.is_file():
                        raise ValueError(f"{model}: input image does not exist: {image}")
                    inputs["image"] = str(image)
            backends = _mapping(case.get("backends"), f"{model}.backends", set(BACKENDS))
            if not backends:
                raise ValueError(f"{model}.backends must not be empty")
            if case.get("task") == "image-generation":
                if set(backends) - {"trtllm", "sglang"}:
                    raise ValueError(f"{model}: image deployment requires trtllm or sglang")
                if (
                    not case.get("input_config", {}).get("prompt")
                    or "image" in case["input_config"]
                ):
                    raise ValueError(f"{model}: image generation requires a text prompt")
            for backend, values in backends.items():
                _mapping(values, f"{model}.{backend}", _BACKEND_FIELDS)
                _parameter_maps(values, f"{model}.{backend}")
                if case.get("task") == "image-generation" and (
                    not values.get("engine_version") or "sm_family" in values
                ):
                    raise ValueError(
                        f"{model}.{backend}: image deployment requires engine_version; "
                        "use sm_versions for candidate GPUs"
                    )
                if "engine_version" in values and (
                    case.get("task") != "image-generation"
                    or not isinstance(values["engine_version"], str)
                    or not values["engine_version"]
                ):
                    raise ValueError(f"{model}.{backend}: engine_version requires an image case")
                for field in ("mini_sm", "sm_family"):
                    if field in values:
                        _positive_int(values[field], f"{model}.{backend}.{field}")
                if "sm_versions" in values:
                    versions = values["sm_versions"]
                    if not isinstance(versions, list) or not versions:
                        raise ValueError(f"{model}.{backend}.sm_versions must be a nonempty list")
                    for version in versions:
                        _positive_int(version, f"{model}.{backend}.sm_versions")
    return groups


def resolve_deploy_config(case, backend, overrides, *, tensor_parallel_size=None):
    """Resolve case defaults, YAML overrides, then explicit CLI TP into constructor args."""
    params = {key: copy.deepcopy(value) for key, value in case.items() if key != "backends"}
    params.update(copy.deepcopy(case["backends"][backend]))
    params["backend"] = backend
    for field in _PARAMETER_MAPS:
        params[field] = {
            **params.get(field, {}),
            **copy.deepcopy(overrides["backends"].get(backend, {}).get(field, {})),
        }
    tp = overrides["run"].get("tensor_parallel_size", params.get("tensor_parallel_size", 1))
    if tensor_parallel_size is not None:
        tp = tensor_parallel_size
    params["tensor_parallel_size"] = _positive_int(tp, "tensor_parallel_size")
    return params


def engine_tp_kwargs(backend, tensor_parallel_size):
    """Map the common run TP to the selected engine's constructor argument."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown deployment backend: {backend}")
    return {"tp_size" if backend == "sglang" else "tensor_parallel_size": tensor_parallel_size}
