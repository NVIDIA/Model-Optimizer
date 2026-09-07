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

"""Run lmms-eval for a local checkpoint with durable attempt artifacts."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import re
import shlex
import signal
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..orchestration.mesh import normalize_vllm_topology

__all__ = [
    "DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS",
    "LmmsEvalTimeoutError",
    "run_lmms_eval_checkpoint",
]

_MODEL_ARG_FIELDS = frozenset(
    {
        "dtype",
        "gdn_prefill_backend",
        "gpu_memory_utilization",
        "attention_config",
        "chat_template",
        "max_model_len",
        "trust_remote_code",
        "tokenizer",
        "tokenizer_mode",
        "enforce_eager",
        "limit_mm_per_prompt",
        "reasoning_parser",
    }
)
_RESERVED_TOPOLOGY_MODEL_ARG_FIELDS = frozenset(
    {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "enable_expert_parallel",
        "distributed_executor_backend",
        "expert_parallel_size",
        "gpu_group_size",
        "tp",
        "pp",
        "dp",
        "prefill_cp",
        "decode_cp",
        "ep",
    }
)
_RESERVED_EXTRA_ARG_FLAGS = frozenset(
    {
        "--batch-size",
        "--batch_size",
        "--model",
        "--model_args",
        "--model-args",
        "--output_path",
        "--output-path",
        "--tasks",
    }
)
_BACKEND_CHECKPOINT_ARGS = {"qwen3_5": "pretrained", "vllm": "model"}
DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS = 3600.0
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 10.0
_PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.1
_TIMEOUT_ERRORS = (TimeoutError, asyncio.TimeoutError)
_PROGRESS_PATH_ENV = "PUZZLETRON_EVALUATION_PROGRESS_PATH"
_PROGRESS_TASKS_ENV = "PUZZLETRON_EVALUATION_PROGRESS_TASKS"
_MODEL_RESPONDING = re.compile(
    r"Model Responding:.*?(?P<current>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[^<\],]+)<(?P<remaining>[^,\]]+),\s*"
    r"(?P<rate>\d+(?:\.\d+)?)(?P<rate_unit>it/s|s/it)\]"
)
_MODEL_RESPONDING_WITHOUT_TOTAL = re.compile(
    r"Model Responding:\s*(?P<current>\d+)it\s*"
    r"\[(?P<elapsed>[^,\]]+),\s*(?P<rate>\d+(?:\.\d+)?)"
    r"(?P<rate_unit>it/s|s/it)\]"
)
_COMPATIBILITY_TASKS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "alias": "modelopt_gsm8k",
        "config": "tasks/gsm8k/gsm8k.yaml",
        "overrides": {
            "dataset_path": "openai/gsm8k",
            "fewshot_config": {"sampler": "default"},
        },
    },
    "ifeval": {
        "alias": "modelopt_ifeval",
        "config": "tasks/ifeval/ifeval.yaml",
        "overrides": {},
    },
    "mmlu_pro_computer_science": {
        "alias": "modelopt_mmlu_pro_computer_science",
        "config": "tasks/mmlu_pro/mmlu_pro_computer_science.yaml",
        "overrides": {},
    },
    "mmlu_pro_history": {
        "alias": "modelopt_mmlu_pro_history",
        "config": "tasks/mmlu_pro/mmlu_pro_history.yaml",
        "overrides": {},
    },
}


class LmmsEvalTimeoutError(TimeoutError):
    """Report a timed-out lmms-eval process with its captured output."""

    def __init__(self, argv: Sequence[str], timeout: float, *, output: str, stderr: str):
        super().__init__(f"lmms-eval exceeded its {timeout:g}-second timeout")
        self.cmd = list(argv)
        self.timeout = timeout
        self.output = output
        self.stderr = stderr


@dataclass(frozen=True)
class _ProcessResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _as_lmms_eval_arg(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _join_cli_values(value: Any, *, path: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{path} must not be empty")
        return text
    if not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a string or sequence")
    values = [str(item).strip() for item in value]
    if not values or any(not item for item in values):
        raise ValueError(f"{path} must contain at least one non-empty value")
    return ",".join(values)


def _model_arg_keys(value: str) -> tuple[str, ...]:
    keys: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False

    def append(segment: str) -> None:
        key, separator, _ = segment.strip().partition("=")
        if separator and key.strip():
            keys.append(key.strip())

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quote:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            append(value[start:index])
            start = index + 1
    append(value[start:])
    return tuple(keys)


def _reject_reserved_model_args(keys: Sequence[Any], reserved_fields: frozenset[str]) -> None:
    reserved = sorted({str(key).strip() for key in keys} & reserved_fields)
    if reserved:
        raise ValueError(
            "evaluation settings.model_args must not set reserved lmms-eval model arguments: "
            f"{', '.join(reserved)}"
        )


def _configured_tasks(settings: Mapping[str, Any]) -> tuple[str, ...]:
    tasks = _join_cli_values(settings.get("tasks"), path="evaluation settings.tasks")
    values = tuple(task.strip() for task in tasks.split(","))
    if not values or any(not task for task in values):
        raise ValueError("evaluation settings.tasks must contain non-empty task names")
    return values


def _model_arg_string(values: Mapping[str, Any]) -> str:
    parts = []
    for key, value in values.items():
        if value is None:
            continue
        key_text = str(key).strip()
        if not key_text or "," in key_text or "=" in key_text:
            raise ValueError(f"invalid lmms-eval model_args key: {key!r}")
        rendered = _as_lmms_eval_arg(value)
        if "," in rendered:
            raise ValueError(
                f"lmms-eval model_args value for {key_text!r} contains a comma; "
                "provide model_args as a preformatted string instead"
            )
        parts.append(f"{key_text}={rendered}")
    if not parts:
        raise ValueError("lmms-eval model_args must contain at least the checkpoint path")
    return ",".join(parts)


def _merge_model_args(settings: Mapping[str, Any], checkpoint: str, *, model: str) -> str:
    raw = settings.get("model_args")
    expected_checkpoint_arg = _BACKEND_CHECKPOINT_ARGS[model]
    checkpoint_arg = str(settings.get("checkpoint_arg", expected_checkpoint_arg))
    if checkpoint_arg != expected_checkpoint_arg:
        raise ValueError(
            f"evaluation settings.checkpoint_arg must be {expected_checkpoint_arg!r} "
            f"for model {model!r}"
        )
    topology = dict(settings.get("topology") or {})
    if topology and model != "vllm":
        raise ValueError("evaluation settings.topology is supported only for model 'vllm'")
    canonical_topology = normalize_vllm_topology(topology) if topology else {}
    reserved_fields = frozenset(
        (*_BACKEND_CHECKPOINT_ARGS.values(), *_RESERVED_TOPOLOGY_MODEL_ARG_FIELDS)
    )
    derived: dict[str, Any] = {checkpoint_arg: checkpoint}
    if canonical_topology:
        derived.update(
            {
                "tensor_parallel_size": canonical_topology["tp"],
                "pipeline_parallel_size": canonical_topology["pp"],
                "data_parallel_size": canonical_topology["dp"],
                "enable_expert_parallel": canonical_topology["enable_expert_parallel"],
                "distributed_executor_backend": canonical_topology["distributed_executor_backend"],
            }
        )
    for key in sorted(_MODEL_ARG_FIELDS):
        if key in settings:
            derived[key] = settings[key]

    if isinstance(raw, str):
        _reject_reserved_model_args(_model_arg_keys(raw), reserved_fields)
        prefix = raw.strip().strip(",")
        suffix = _model_arg_string(derived)
        return ",".join(part for part in (prefix, suffix) if part)
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError("evaluation settings.model_args must be a mapping or string")
    _reject_reserved_model_args(tuple((raw or {}).keys()), reserved_fields)
    merged = dict(raw or {})
    merged.update(derived)
    return _model_arg_string(merged)


def _command_prefix(settings: Mapping[str, Any]) -> list[str]:
    raw = settings.get("command_prefix")
    if raw is None:
        return [sys.executable, "-m", "lmms_eval"]
    if isinstance(raw, str):
        values = shlex.split(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = [str(item) for item in raw]
    else:
        raise TypeError("evaluation settings.command_prefix must be a string or sequence")
    if not values or any(not value for value in values):
        raise ValueError("evaluation settings.command_prefix must not be empty")
    return values


def _extra_args(settings: Mapping[str, Any]) -> list[str]:
    raw = settings.get("extra_args")
    if raw is None:
        return []
    if isinstance(raw, str):
        values = shlex.split(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = [str(item) for item in raw]
    else:
        raise TypeError("evaluation settings.extra_args must be a string or sequence")
    if any(not value for value in values):
        raise ValueError("evaluation settings.extra_args must not contain empty values")
    reserved = sorted(
        {
            value.split("=", 1)[0]
            for value in values
            if value.split("=", 1)[0] in _RESERVED_EXTRA_ARG_FLAGS
        }
    )
    if reserved:
        raise ValueError(
            "evaluation settings.extra_args must not set reserved lmms-eval flags: "
            f"{', '.join(reserved)}"
        )
    return values


def _build_command(
    settings: Mapping[str, Any],
    *,
    checkpoint: str,
    output_path: Path,
) -> tuple[list[str], dict[str, str], float]:
    """Build a deterministic lmms-eval CLI invocation for one local checkpoint."""

    model = str(settings.get("model", "vllm"))
    if model not in _BACKEND_CHECKPOINT_ARGS:
        supported = ", ".join(sorted(_BACKEND_CHECKPOINT_ARGS))
        raise ValueError(f"evaluation settings.model must be one of: {supported}")
    argv = [
        *_command_prefix(settings),
        "--model",
        model,
        "--model_args",
        _merge_model_args(settings, checkpoint, model=model),
        "--tasks",
        ",".join(_configured_tasks(settings)),
        "--batch_size",
        str(settings.get("batch_size", 1)),
        "--output_path",
        str(output_path),
    ]
    optional_fields = {
        "limit": "--limit",
        "num_fewshot": "--num_fewshot",
        "seed": "--seed",
        "verbosity": "--verbosity",
        "device": "--device",
        "use_cache": "--use_cache",
    }
    for key, flag in optional_fields.items():
        value = settings.get(key)
        if value is not None:
            argv.extend([flag, str(value)])
    if settings.get("gen_kwargs") is not None:
        argv.extend(
            [
                "--gen_kwargs",
                (
                    settings["gen_kwargs"]
                    if isinstance(settings["gen_kwargs"], str)
                    else _model_arg_string(dict(settings["gen_kwargs"]))
                ),
            ]
        )
    if bool(settings.get("log_samples", False)):
        argv.append("--log_samples")
    argv.extend(_extra_args(settings))

    env = os.environ.copy()
    env_overrides = dict(settings.get("env") or {})
    for key, value in env_overrides.items():
        if value is not None:
            env[str(key)] = str(value)
    compatibility = Path(__file__).parents[1] / "benchmarks" / "vllm_compat"
    python_paths = [str(compatibility)]
    if inherited_python_path := env.get("PYTHONPATH"):
        python_paths.append(inherited_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if settings.get("cache_dir") is not None and "LMMS_EVAL_HOME" not in env_overrides:
        env["LMMS_EVAL_HOME"] = str(settings["cache_dir"])
    timeout = settings.get("timeout_seconds")
    if timeout is None:
        timeout = settings.get("timeout")
    if timeout is None:
        timeout = DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("lmms-eval timeout must be a finite positive number")
    return argv, env, timeout


def _prepare_compatibility_tasks(output: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve explicitly requested task fixes without modifying the pinned evaluator."""

    prepared = dict(settings)
    raw_requested = prepared.pop("compatibility_tasks", None)
    requested = set(
        (
            task.strip()
            for task in _join_cli_values(
                raw_requested,
                path="evaluation settings.compatibility_tasks",
            ).split(",")
        )
        if raw_requested is not None
        else ()
    )
    raw_revisions = prepared.pop("task_dataset_revisions", None)
    if raw_revisions is None:
        revisions: dict[str, str] = {}
    elif isinstance(raw_revisions, Mapping):
        revisions = {str(task): str(revision) for task, revision in raw_revisions.items()}
    else:
        raise TypeError("evaluation settings.task_dataset_revisions must be a mapping")
    empty_revisions = sorted(task for task, revision in revisions.items() if not revision.strip())
    if empty_revisions:
        raise ValueError(f"empty lmms-eval dataset revisions for tasks: {empty_revisions}")
    requested.update(revisions)
    unknown = sorted(requested - _COMPATIBILITY_TASKS.keys())
    if unknown:
        raise ValueError(f"unknown lmms-eval compatibility tasks: {unknown}")
    if not requested:
        return prepared
    configured_tasks = _configured_tasks(prepared)
    unconfigured = sorted(requested - set(configured_tasks))
    if unconfigured:
        raise ValueError(
            f"lmms-eval compatibility settings reference unconfigured tasks: {unconfigured}"
        )

    spec = importlib.util.find_spec("lmms_eval")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        raise RuntimeError("lmms_eval is not installed")
    package = Path(next(iter(locations)))
    tasks = [
        str(_COMPATIBILITY_TASKS[task]["alias"]) if task in requested else task
        for task in configured_tasks
    ]
    task_root = output / "task_configs"
    task_root.mkdir(parents=True, exist_ok=True)
    for task in sorted(requested):
        task_spec = _COMPATIBILITY_TASKS[task]
        upstream = package / str(task_spec["config"])
        if not upstream.is_file():
            raise RuntimeError(f"installed lmms_eval has no {task} task config: {upstream}")
        task_config = {
            "include": str(upstream.resolve()),
            "task": str(task_spec["alias"]),
            **dict(task_spec["overrides"]),
        }
        if task in revisions:
            dataset_kwargs = dict(task_config.get("dataset_kwargs") or {})
            dataset_kwargs["revision"] = revisions[task]
            task_config["dataset_kwargs"] = dataset_kwargs
        _atomic_json(task_root / f"{task_spec['alias']}.yaml", task_config)
    prepared["tasks"] = tasks
    prepared["extra_args"] = [*_extra_args(prepared), "--include_path", str(task_root)]
    return prepared


def _numeric_metrics(task_payload: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(metric_name): float(value)
        for metric_name, value in task_payload.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }


def _metric_key(value: Any) -> str:
    return (
        str(value).strip().replace(" ", "_").replace(",", "_").replace("/", "_").replace("\\", "_")
    )


def _flatten_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        return {}
    metrics = {}
    for task_name, task_payload in results.items():
        if isinstance(task_payload, Mapping):
            metrics.update(
                {
                    f"{_metric_key(task_name)}.{_metric_key(metric_name)}": value
                    for metric_name, value in _numeric_metrics(task_payload).items()
                }
            )
    return metrics


def _resolved_tasks(payload: Mapping[str, Any], configured_tasks: Sequence[str]) -> tuple[str, ...]:
    group_subtasks = payload.get("group_subtasks")
    if not isinstance(group_subtasks, Mapping):
        group_subtasks = {}

    def expand(task: str, seen: frozenset[str]) -> tuple[str, ...]:
        raw_subtasks = group_subtasks.get(task)
        if (
            isinstance(raw_subtasks, Sequence)
            and not isinstance(raw_subtasks, str)
            and raw_subtasks
            and task not in seen
        ):
            expanded = []
            for raw_subtask in raw_subtasks:
                expanded.extend(expand(str(raw_subtask), seen | {task}))
            return tuple(dict.fromkeys(expanded))
        return (task,)

    resolved = []
    for task in configured_tasks:
        resolved.extend(expand(task, frozenset()))
    return tuple(dict.fromkeys(resolved))


def _sample_count(payload: Mapping[str, Any], task: str) -> float | None:
    samples = payload.get("n-samples", payload.get("n_samples"))
    if not isinstance(samples, Mapping):
        return None
    value = samples.get(task)
    if isinstance(value, Mapping):
        if "effective" in value:
            value = value["effective"]
        elif "original" in value:
            value = value["original"]
        else:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _validate_completion(
    payload: Mapping[str, Any], configured_tasks: Sequence[str]
) -> dict[str, float]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise RuntimeError("lmms-eval result is missing the results mapping")

    expected_tasks = _resolved_tasks(payload, configured_tasks)
    missing_results = [task for task in expected_tasks if task not in results]
    if missing_results:
        raise RuntimeError(
            f"lmms-eval result is missing configured task results: {sorted(missing_results)}"
        )
    missing_metrics = [
        task
        for task in expected_tasks
        if not isinstance(results[task], Mapping) or not _numeric_metrics(results[task])
    ]
    if missing_metrics:
        raise RuntimeError(
            "lmms-eval result has no numeric metrics for configured tasks: "
            f"{sorted(missing_metrics)}"
        )

    sample_counts = {}
    missing_samples = []
    zero_samples = []
    for task in expected_tasks:
        count = _sample_count(payload, task)
        if count is None:
            missing_samples.append(task)
        elif count <= 0:
            zero_samples.append(task)
        else:
            sample_counts[task] = count
    if missing_samples:
        raise RuntimeError(
            "lmms-eval result is missing sample counts for configured tasks: "
            f"{sorted(missing_samples)}"
        )
    if zero_samples:
        raise RuntimeError(
            "lmms-eval result has zero effective samples for configured tasks: "
            f"{sorted(zero_samples)}"
        )
    return sample_counts


def _result_payload(output_path: Path) -> tuple[dict[str, Any], Path]:
    candidates = []
    for path in sorted(output_path.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(payload, Mapping) and isinstance(payload.get("results"), Mapping):
            candidates.append((path.stat().st_mtime_ns, path, dict(payload)))
    if not candidates:
        raise FileNotFoundError(f"lmms-eval wrote no JSON results below {output_path}")
    _mtime, path, payload = max(candidates, key=lambda item: item[0])
    return payload, path


def _write_streams(output_path: Path, result: _ProcessResult) -> dict[str, str]:
    output_path.mkdir(parents=True, exist_ok=True)
    stream_paths = {}
    for stream_name, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        stream_path = output_path / f"{stream_name}.txt"
        stream_path.write_text(text or "")
        stream_paths[f"{stream_name}_path"] = str(stream_path)
    return stream_paths


def _stream_text(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def _live_stream_bytes(stream: Any, data: bytes) -> None:
    try:
        target = getattr(stream, "buffer", None)
        if target is not None:
            target.write(data)
            target.flush()
            return
        stream.write(data.decode(errors="replace"))
        stream.flush()
    except (OSError, ValueError):
        pass


def _sample_rate_per_second(value: str, unit: str) -> float | None:
    rate = float(value)
    if rate <= 0:
        return None
    return rate if unit == "it/s" else 1.0 / rate


def _task_progress(
    current: int,
    total: int | None,
    tasks: object,
) -> dict[str, object] | None:
    if total is None or not isinstance(tasks, list):
        return None
    parsed: list[tuple[str, int]] = []
    for entry in tasks:
        if not isinstance(entry, Mapping):
            return None
        name = entry.get("name")
        task_total = entry.get("total")
        if not isinstance(name, str) or not isinstance(task_total, int) or task_total <= 0:
            return None
        parsed.append((name, task_total))
    if not parsed or sum(task_total for _name, task_total in parsed) != total:
        return None
    offset = current
    for name, task_total in parsed:
        if offset <= task_total:
            return {"name": name, "current": offset, "total": task_total}
        offset -= task_total
    name, task_total = parsed[-1]
    return {"name": name, "current": task_total, "total": task_total}


def _set_evaluation_progress_status(
    path: Path,
    status: str,
    *,
    sample_counts: Mapping[str, float] | None = None,
    tasks: object = None,
) -> None:
    """Best-effort terminal update for the evaluator progress sidecar."""

    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(payload, Mapping):
        return
    updated = dict(payload)
    updated["status"] = status
    updated["updated_at"] = time.time()
    if status == "completed" and sample_counts:
        sample_total = float(sum(sample_counts.values()))
        if math.isfinite(sample_total) and sample_total > 0 and sample_total.is_integer():
            total = int(sample_total)
            updated["current"] = total
            updated["total"] = total
            task = _task_progress(total, total, tasks)
            if task is None:
                updated.pop("task", None)
            else:
                updated["task"] = task
    try:
        _atomic_json(path, updated)
    except OSError:
        pass


def _evaluation_progress_payload(text: str, tasks: object) -> dict[str, object] | None:
    matches = list(_MODEL_RESPONDING.finditer(text))
    total: int | None
    if matches:
        match = matches[-1]
        total = int(match.group("total"))
    else:
        fallback = list(_MODEL_RESPONDING_WITHOUT_TOTAL.finditer(text))
        if not fallback:
            return None
        match = fallback[-1]
        total = None
    current = int(match.group("current"))
    payload: dict[str, object] = {
        "schema": "modelopt.puzzletron.evaluation-progress/v1",
        "status": "running",
        "unit": "samples",
        "current": current,
        "total": total,
        "rate_per_second": _sample_rate_per_second(match.group("rate"), match.group("rate_unit")),
        "updated_at": time.time(),
    }
    task = _task_progress(current, total, tasks)
    if task is not None:
        payload["task"] = task
    return payload


async def _pump_process_stream(
    reader: asyncio.StreamReader,
    capture: Any,
    live_stream: Any,
    *,
    progress_path: Path | None = None,
    progress_tasks: object = None,
) -> None:
    progress_text = ""
    while data := await reader.read(65536):
        capture.write(data)
        capture.flush()
        _live_stream_bytes(live_stream, data)
        if progress_path is None:
            continue
        progress_text = (progress_text + data.decode(errors="replace"))[-131072:]
        payload = _evaluation_progress_payload(progress_text, progress_tasks)
        if payload is not None:
            try:
                _atomic_json(progress_path, payload)
            except OSError:
                pass


async def _drain_process_pumps(
    process: asyncio.subprocess.Process,
    pumps: tuple[asyncio.Task[None], ...],
    *,
    suppress_errors: bool,
) -> None:
    done, pending = await asyncio.wait(pumps, timeout=_PROCESS_CLEANUP_TIMEOUT_SECONDS)
    for pump in pending:
        pump.cancel()
    if pending:
        # asyncio Process has no public stream-close API. Closing its transport
        # releases inherited pipe readers after a descendant outlives the parent.
        process_transport = getattr(process, "_transport", None)
        if process_transport is not None:
            try:
                process_transport.close()
            except (OSError, RuntimeError):
                pass
    results = await asyncio.gather(*pumps, return_exceptions=True)
    if suppress_errors:
        return
    for pump, result in zip(pumps, results):
        if pump in done and isinstance(result, BaseException):
            raise result


def _output_tail(result: _ProcessResult, *, max_lines: int = 20) -> str:
    sections = []
    for stream_name, text in (("stderr", result.stderr), ("stdout", result.stdout)):
        lines = (text or "").strip().splitlines()
        if lines:
            sections.append(f"{stream_name} tail:")
            sections.extend(lines[-max_lines:])
    return "\n".join(sections)


def _signal_process_group(process: asyncio.subprocess.Process, signal_number: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal_number)
        else:
            process.send_signal(signal_number)
    except ProcessLookupError:
        pass


def _process_group_exists(process: asyncio.subprocess.Process) -> bool:
    if os.name != "posix":
        return process.returncode is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_process_group_exit(
    process: asyncio.subprocess.Process, *, deadline: float
) -> None:
    loop = asyncio.get_running_loop()
    while _process_group_exists(process):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(_PROCESS_GROUP_POLL_INTERVAL_SECONDS, remaining))


async def _wait_for_process_returncode(
    process: asyncio.subprocess.Process, *, timeout: float
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while process.returncode is None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError
        await asyncio.sleep(min(_PROCESS_GROUP_POLL_INTERVAL_SECONDS, remaining))


async def _run_process_async(
    argv: list[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout: float,
) -> _ProcessResult:
    # lmms-eval needs process isolation for bounded GPU-worker cleanup. The argument
    # vector is passed directly; no shell interprets checkpoint or configuration values.
    child_env = dict(env)
    raw_progress_path = child_env.pop(_PROGRESS_PATH_ENV, None)
    raw_progress_tasks = child_env.pop(_PROGRESS_TASKS_ENV, None)
    progress_path = Path(raw_progress_path) if raw_progress_path else None
    try:
        progress_tasks = json.loads(raw_progress_tasks) if raw_progress_tasks else None
    except json.JSONDecodeError:
        progress_tasks = None
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        assert process.stdout is not None
        assert process.stderr is not None
        pumps = (
            asyncio.create_task(_pump_process_stream(process.stdout, stdout_file, sys.stdout)),
            asyncio.create_task(
                _pump_process_stream(
                    process.stderr,
                    stderr_file,
                    sys.stderr,
                    progress_path=progress_path,
                    progress_tasks=progress_tasks,
                )
            ),
        )
        try:
            await _wait_for_process_returncode(process, timeout=timeout)
        except _TIMEOUT_ERRORS as error:
            _signal_process_group(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), _PROCESS_CLEANUP_TIMEOUT_SECONDS)
            except _TIMEOUT_ERRORS:
                _signal_process_group(process, signal.SIGKILL)
                try:
                    await asyncio.wait_for(process.wait(), _PROCESS_CLEANUP_TIMEOUT_SECONDS)
                except _TIMEOUT_ERRORS:
                    pass
            if _process_group_exists(process):
                _signal_process_group(process, signal.SIGKILL)
                await _wait_for_process_group_exit(
                    process,
                    deadline=asyncio.get_running_loop().time() + _PROCESS_CLEANUP_TIMEOUT_SECONDS,
                )
            await _drain_process_pumps(process, pumps, suppress_errors=True)
            stdout_file.seek(0)
            stderr_file.seek(0)
            raise LmmsEvalTimeoutError(
                argv,
                timeout,
                output=_stream_text(stdout_file.read()),
                stderr=_stream_text(stderr_file.read()),
            ) from error
        await _drain_process_pumps(process, pumps, suppress_errors=False)
        stdout_file.seek(0)
        stderr_file.seek(0)
        return _ProcessResult(
            args=argv,
            returncode=int(process.returncode or 0),
            stdout=_stream_text(stdout_file.read()),
            stderr=_stream_text(stderr_file.read()),
        )


def _run_process(
    argv: list[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout: float,
) -> _ProcessResult:
    return asyncio.run(_run_process_async(argv, cwd=cwd, env=env, timeout=timeout))


def _annotate_error(
    error: Exception,
    *,
    command_path: Path,
    stream_paths: Mapping[str, str],
) -> None:
    setattr(error, "command_path", str(command_path))
    for name in ("stdout_path", "stderr_path"):
        if name in stream_paths:
            setattr(error, name, stream_paths[name])


def run_lmms_eval_checkpoint(
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one local checkpoint and preserve an isolated lmms-eval attempt.

    Args:
        checkpoint: Local Hugging Face checkpoint directory.
        output_root: Root under which a unique attempt directory is created.
        settings: lmms-eval tasks, backend model arguments, topology, and runtime controls.

    Returns:
        Flattened metrics and paths to the normalized summary, raw result, command,
        stdout, and stderr artifacts.
    """

    checkpoint_path = Path(checkpoint).expanduser().absolute()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"checkpoint is not a local directory: {checkpoint_path}")
    output = Path(output_root).expanduser().absolute() / f"attempt_{uuid.uuid4().hex}"
    output.mkdir(parents=True, exist_ok=True)
    settings = _prepare_compatibility_tasks(output, settings)
    argv, env, timeout = _build_command(
        settings,
        checkpoint=str(checkpoint_path),
        output_path=output,
    )
    command_payload = {
        "argv": argv,
        "env_overrides": sorted(str(key) for key in dict(settings.get("env") or {})),
        "timeout": timeout,
    }
    command_path = _atomic_json(output / "command.json", command_payload)
    progress_tasks = settings.get("progress_tasks")
    if not isinstance(progress_tasks, list) or not all(
        isinstance(task, Mapping)
        and isinstance(task.get("name"), str)
        and isinstance(task.get("total"), int)
        and not isinstance(task.get("total"), bool)
        and int(task["total"]) > 0
        for task in progress_tasks
    ):
        progress_tasks = []
    try:
        progress_tasks_json = json.dumps(progress_tasks, separators=(",", ":"))
    except (TypeError, ValueError):
        progress_tasks = []
        progress_tasks_json = "[]"
    progress_path = output / "progress.json"
    progress_total = sum(int(task["total"]) for task in progress_tasks)
    initial_progress: dict[str, object] = {
        "schema": "modelopt.puzzletron.evaluation-progress/v1",
        "status": "starting",
        "unit": "samples",
        "current": 0,
        "total": progress_total or None,
        "rate_per_second": None,
        "updated_at": time.time(),
    }
    initial_task = _task_progress(0, progress_total or None, progress_tasks)
    if initial_task is not None:
        initial_progress["task"] = initial_task
    _atomic_json(progress_path, initial_progress)
    env[_PROGRESS_PATH_ENV] = str(progress_path)
    env[_PROGRESS_TASKS_ENV] = progress_tasks_json
    try:
        result = _run_process(argv, cwd=str(output), env=env, timeout=timeout)
    except LmmsEvalTimeoutError as error:
        if not command_path.is_file():
            command_path = _atomic_json(output / "command.json", command_payload)
        captured = _ProcessResult(argv, -1, error.output, error.stderr)
        stream_paths = _write_streams(output, captured)
        _annotate_error(error, command_path=command_path, stream_paths=stream_paths)
        _set_evaluation_progress_status(progress_path, "timed_out")
        raise
    except OSError:
        _set_evaluation_progress_status(progress_path, "failed")
        raise

    stream_paths = _write_streams(output, result)
    if result.returncode:
        tail = _output_tail(result)
        failure = RuntimeError(
            f"lmms-eval failed with exit code {result.returncode}" + (f": {tail}" if tail else "")
        )
        _annotate_error(failure, command_path=command_path, stream_paths=stream_paths)
        _set_evaluation_progress_status(progress_path, "failed")
        raise failure

    try:
        payload, result_path = _result_payload(output)
        sample_counts = _validate_completion(payload, _configured_tasks(settings))
        metrics = _flatten_metrics(payload)
        if not metrics:
            raise RuntimeError(f"lmms-eval result has no numeric task metrics: {result_path}")
    except FileNotFoundError as error:
        tail = _output_tail(result)
        failure = FileNotFoundError(f"{error}: {tail}" if tail else str(error))
        _annotate_error(failure, command_path=command_path, stream_paths=stream_paths)
        _set_evaluation_progress_status(progress_path, "failed")
        raise failure from error
    except RuntimeError as error:
        _annotate_error(error, command_path=command_path, stream_paths=stream_paths)
        _set_evaluation_progress_status(progress_path, "failed")
        raise

    summary = {
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "raw_result_path": str(result_path),
        "sample_counts": sample_counts,
    }
    try:
        summary_path = _atomic_json(output / "summary.json", summary)
    except OSError:
        _set_evaluation_progress_status(progress_path, "failed")
        raise
    _set_evaluation_progress_status(
        progress_path,
        "completed",
        sample_counts=sample_counts,
        tasks=progress_tasks,
    )
    return {
        "metrics": metrics,
        "result_path": str(summary_path),
        "raw_result_path": str(result_path),
        "command_path": str(command_path),
        **stream_paths,
    }
