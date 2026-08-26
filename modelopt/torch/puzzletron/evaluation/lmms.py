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
import json
import math
import os
import shlex
import signal
import sys
import tempfile
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
        "gpu_memory_utilization",
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
_BACKEND_CHECKPOINT_ARGS = {
    "qwen3_5": "pretrained",
    "vllm": "model",
}
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
DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS = 3600.0
_PROCESS_CLEANUP_TIMEOUT_SECONDS = 10.0
_PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.1
_TIMEOUT_ERRORS = (TimeoutError, asyncio.TimeoutError)


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


def _backend_contract(settings: Mapping[str, Any]) -> tuple[str, str]:
    backend = str(settings.get("model", "vllm"))
    if backend not in _BACKEND_CHECKPOINT_ARGS:
        supported = ", ".join(sorted(_BACKEND_CHECKPOINT_ARGS))
        raise ValueError(f"evaluation settings.model must be one of: {supported}")
    expected_checkpoint_arg = _BACKEND_CHECKPOINT_ARGS[backend]
    checkpoint_arg = str(settings.get("checkpoint_arg", expected_checkpoint_arg))
    if checkpoint_arg != expected_checkpoint_arg:
        raise ValueError(
            f"evaluation settings.checkpoint_arg for {backend} must be {expected_checkpoint_arg!r}"
        )
    return backend, checkpoint_arg


def _merge_model_args(
    settings: Mapping[str, Any],
    checkpoint: str,
    *,
    backend: str,
    checkpoint_arg: str,
) -> str:
    raw = settings.get("model_args")
    topology = dict(settings.get("topology") or {})
    if topology and backend != "vllm":
        raise ValueError("evaluation settings.topology is supported only for vllm")
    canonical_topology = normalize_vllm_topology(topology) if topology else {}
    reserved_fields = frozenset(
        (
            *_BACKEND_CHECKPOINT_ARGS.values(),
            *_RESERVED_TOPOLOGY_MODEL_ARG_FIELDS,
        )
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
    if backend == "vllm":
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

    model, checkpoint_arg = _backend_contract(settings)
    argv = [
        *_command_prefix(settings),
        "--model",
        model,
        "--model_args",
        _merge_model_args(
            settings,
            checkpoint,
            backend=model,
            checkpoint_arg=checkpoint_arg,
        ),
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
    stream_paths = {}
    for stream_name, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        stream_path = output_path / f"{stream_name}.txt"
        stream_path.write_text(text or "")
        stream_paths[f"{stream_name}_path"] = str(stream_path)
    return stream_paths


def _stream_text(value: str | bytes | None) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


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


async def _run_process_async(
    argv: list[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout: float,
) -> _ProcessResult:
    # lmms-eval needs process isolation for bounded GPU-worker cleanup. The argument
    # vector is passed directly; no shell interprets checkpoint or configuration values.
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=os.name == "posix",
        )
        try:
            await asyncio.wait_for(process.wait(), timeout)
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
            stdout_file.seek(0)
            stderr_file.seek(0)
            raise LmmsEvalTimeoutError(
                argv,
                timeout,
                output=_stream_text(stdout_file.read()),
                stderr=_stream_text(stderr_file.read()),
            ) from error
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
    settings = dict(settings)
    argv, env, timeout = _build_command(
        settings,
        checkpoint=str(checkpoint_path),
        output_path=output,
    )
    output.mkdir(parents=True, exist_ok=True)
    command_path = _atomic_json(
        output / "command.json",
        {
            "argv": argv,
            "env_overrides": sorted(str(key) for key in dict(settings.get("env") or {})),
            "timeout": timeout,
        },
    )
    try:
        result = _run_process(argv, cwd=str(output), env=env, timeout=timeout)
    except LmmsEvalTimeoutError as error:
        captured = _ProcessResult(argv, -1, error.output, error.stderr)
        stream_paths = _write_streams(output, captured)
        _annotate_error(error, command_path=command_path, stream_paths=stream_paths)
        raise

    stream_paths = _write_streams(output, result)
    if result.returncode:
        tail = _output_tail(result)
        failure = RuntimeError(
            f"lmms-eval failed with exit code {result.returncode}" + (f": {tail}" if tail else "")
        )
        _annotate_error(failure, command_path=command_path, stream_paths=stream_paths)
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
        raise failure from error
    except RuntimeError as error:
        _annotate_error(error, command_path=command_path, stream_paths=stream_paths)
        raise

    summary = {
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "raw_result_path": str(result_path),
        "sample_counts": sample_counts,
    }
    summary_path = _atomic_json(output / "summary.json", summary)
    return {
        "metrics": metrics,
        "result_path": str(summary_path),
        "raw_result_path": str(result_path),
        "command_path": str(command_path),
        **stream_paths,
    }
