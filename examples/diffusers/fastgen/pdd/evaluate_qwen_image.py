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

"""Evaluate an authenticated Qwen-Image PDD export over a prompt manifest."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import sys
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.dont_write_bytecode = True

_THIS_DIR = Path(__file__).resolve().parent
_FASTGEN_DIR = _THIS_DIR.parent
_REPO_ROOT = _FASTGEN_DIR.parents[2]
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pdd.artifacts import load_canonical_json, sha256_file, write_canonical_json  # noqa: E402
from pdd.export import PDD_INFERENCE_SCHEDULES  # noqa: E402
from pdd.inference_runtime import (  # noqa: E402
    QwenPDDInferenceRuntime,
    load_qwen_pdd_runtime,
    save_png,
)

_PROMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class PromptPair:
    prompt_id: str
    prompt: str
    seed: int


@dataclass(frozen=True)
class Observation:
    scheduler_calls: int
    transformer_calls: int
    transformer_seconds: float
    end_to_end_seconds: float
    peak_device_memory_bytes: int | None
    image: Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--schedule", choices=tuple(PDD_INFERENCE_SCHEDULES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _prompt_pairs(value: Any) -> list[PromptPair]:
    if not isinstance(value, dict) or set(value) != {"schema_version", "prompts"}:
        raise ValueError("prompt manifest must contain exactly schema_version and prompts.")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("prompt manifest schema_version must be 1.")
    prompts = value["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt manifest prompts must be a non-empty list.")
    pairs: list[PromptPair] = []
    previous_id: str | None = None
    for index, item in enumerate(prompts):
        if not isinstance(item, dict) or set(item) != {"prompt_id", "prompt", "seeds"}:
            raise ValueError(f"prompts[{index}] has incompatible fields.")
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]
        seeds = item["seeds"]
        if not isinstance(prompt_id, str) or _PROMPT_ID.fullmatch(prompt_id) is None:
            raise ValueError(f"prompts[{index}].prompt_id is not a safe path component.")
        if previous_id is not None and prompt_id <= previous_id:
            raise ValueError("prompt IDs must be unique and lexicographically ordered.")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"prompts[{index}].prompt must be non-empty text.")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"prompts[{index}].seeds must be non-empty.")
        if any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in seeds):
            raise ValueError(f"prompts[{index}].seeds contains an invalid seed.")
        if seeds != sorted(set(seeds)):
            raise ValueError(f"prompts[{index}].seeds must be sorted and unique.")
        pairs.extend(PromptPair(prompt_id, prompt, seed) for seed in seeds)
        previous_id = prompt_id
    return pairs


def _reject_symlink_components(path: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("evaluation paths cannot contain parent traversal components.")
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"evaluation path traverses a symlink: {current}.")
    return absolute.resolve(strict=False)


def _resolve_output_paths(output_value: Path, result_value: Path) -> tuple[Path, Path, Path]:
    output = _reject_symlink_components(output_value)
    result = _reject_symlink_components(result_value)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"evaluation output already exists: {output}.")
    if output.name in {"", ".", ".."}:
        raise ValueError("evaluation output directory name is invalid.")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("evaluation output parent must be an existing regular directory.")
    try:
        relative_result = result.relative_to(output)
    except ValueError as error:
        raise ValueError("result JSON must be strictly beneath output_dir.") from error
    if not relative_result.parts or relative_result == Path("."):
        raise ValueError("result JSON must be strictly beneath output_dir.")
    if result.suffix.lower() != ".json":
        raise ValueError("result JSON must use a .json suffix.")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    os.mkdir(staging, mode=0o770)
    os.chmod(staging, 0o770, follow_symlinks=False)
    return output, relative_result, staging


def _restore_instance_method(
    owner: Any, name: str, *, had_instance_value: bool, instance_value: Any
) -> None:
    if had_instance_value:
        setattr(owner, name, instance_value)
    elif name in vars(owner):
        delattr(owner, name)


def _run_repetition(
    runtime: QwenPDDInferenceRuntime,
    prompt: str,
    raw_noise: torch.Tensor,
    max_sequence_length: int,
) -> Observation:
    scheduler_calls = 0
    transformer_calls = 0
    cpu_forward_started: list[float] = []
    cpu_forward_seconds = 0.0
    cuda_event_pairs: list[tuple[Any, Any]] = []
    scheduler = runtime.scheduler
    scheduler_values = vars(scheduler)
    scheduler_had_step = "step" in scheduler_values
    scheduler_instance_step = scheduler_values.get("step")
    original_step = scheduler.step

    def counted_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal scheduler_calls
        scheduler_calls += 1
        return original_step(*args, **kwargs)

    def before_forward(_module: Any, _args: Any, _kwargs: Any) -> None:
        nonlocal transformer_calls
        transformer_calls += 1
        if runtime.device.type == "cuda":
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            cuda_event_pairs.append((started, ended))
        else:
            cpu_forward_started.append(time.perf_counter())

    def after_forward(_module: Any, _args: Any, _kwargs: Any, _output: Any) -> None:
        nonlocal cpu_forward_seconds
        if runtime.device.type == "cuda":
            cuda_event_pairs[-1][1].record()
        else:
            if not cpu_forward_started:
                raise RuntimeError("transformer post-hook ran without its pre-hook.")
            cpu_forward_seconds += time.perf_counter() - cpu_forward_started.pop()

    with ExitStack() as cleanup:
        cleanup.callback(
            _restore_instance_method,
            scheduler,
            "step",
            had_instance_value=scheduler_had_step,
            instance_value=scheduler_instance_step,
        )
        setattr(scheduler, "step", counted_step)
        pre_hook = runtime.student.register_forward_pre_hook(before_forward, with_kwargs=True)
        cleanup.callback(pre_hook.remove)
        post_hook = runtime.student.register_forward_hook(after_forward, with_kwargs=True)
        cleanup.callback(post_hook.remove)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
            torch.cuda.reset_peak_memory_stats(runtime.device)
        started = time.perf_counter()
        condition = runtime.encode_prompt(prompt, max_sequence_length)
        images = runtime.sample_decode(condition, raw_noise)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        end_to_end_seconds = time.perf_counter() - started
        if runtime.device.type == "cuda":
            transformer_seconds = sum(
                start.elapsed_time(end) / 1000.0 for start, end in cuda_event_pairs
            )
            peak_memory = torch.cuda.max_memory_allocated(runtime.device)
        else:
            transformer_seconds = cpu_forward_seconds
            peak_memory = None

    expected = len(runtime.config.inference_blocks)
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError("one evaluation repetition must return exactly one image.")
    if scheduler_calls != 0:
        raise RuntimeError(f"PDD evaluation observed {scheduler_calls} scheduler.step calls.")
    if transformer_calls != expected:
        raise RuntimeError(
            f"PDD evaluation observed {transformer_calls} transformer calls; expected {expected}."
        )
    if cpu_forward_started:
        raise RuntimeError("transformer forward hooks are unbalanced.")
    if (
        not math.isfinite(end_to_end_seconds)
        or not math.isfinite(transformer_seconds)
        or end_to_end_seconds <= 0
        or transformer_seconds <= 0
        or transformer_seconds > end_to_end_seconds + 1e-6
    ):
        raise RuntimeError("evaluation timing invariants failed.")
    if peak_memory is not None and (type(peak_memory) is not int or peak_memory < 0):
        raise RuntimeError("evaluation peak device memory is invalid.")
    return Observation(
        scheduler_calls=scheduler_calls,
        transformer_calls=transformer_calls,
        transformer_seconds=transformer_seconds,
        end_to_end_seconds=end_to_end_seconds,
        peak_device_memory_bytes=peak_memory,
        image=images[0],
    )


def _summary(values: list[float | int]) -> dict[str, float | int]:
    if not values or any(type(value) not in {int, float} for value in values):
        raise ValueError("summary values must be a non-empty numeric list.")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {"median": median, "p95": p95}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_staging(staging: Path, output: Path) -> None:
    """Rename a complete staging tree and roll back a failed parent fsync."""
    staging.rename(output)
    try:
        _fsync_directory(output.parent)
    except BaseException:
        output.rename(staging)
        _fsync_directory(output.parent)
        raise


def _record_for_pair(
    runtime: QwenPDDInferenceRuntime,
    pair: PromptPair,
    *,
    trajectory: dict[str, Any],
    observations: list[Observation],
    image_reference: str,
    image_sha256: str,
) -> dict[str, Any]:
    scheduler = [item.scheduler_calls for item in observations]
    actual = [item.transformer_calls for item in observations]
    transformer = [item.transformer_seconds for item in observations]
    end_to_end = [item.end_to_end_seconds for item in observations]
    throughput = [1.0 / value for value in end_to_end]
    memory = [item.peak_device_memory_bytes for item in observations]
    expected = len(runtime.config.inference_blocks)
    if scheduler != [0] * len(observations) or actual != [expected] * len(observations):
        raise RuntimeError("measured repetition counters are inconsistent.")
    numeric_memory = [value for value in memory if value is not None]
    if numeric_memory and len(numeric_memory) != len(memory):
        raise RuntimeError("peak-memory observations mix CPU and CUDA domains.")
    return {
        "prompt_id": pair.prompt_id,
        "prompt_sha256": hashlib.sha256(pair.prompt.encode("utf-8")).hexdigest(),
        "seed": pair.seed,
        "raw_noise_sha256": trajectory["raw_noise_sha256"],
        "initial_state_sha256": trajectory["initial_state_sha256"],
        "requested_scheduler_steps": expected,
        "logical_pdd_blocks": list(runtime.config.inference_blocks),
        "logical_pdd_block_count": expected,
        "observed_scheduler_step_calls": scheduler,
        "actual_transformer_invocations": actual,
        "batch_normalized_transformer_evaluations": list(actual),
        "transformer_latency_seconds": transformer,
        "end_to_end_latency_seconds": end_to_end,
        "throughput_images_per_second": throughput,
        "peak_device_memory_bytes": memory,
        "summaries": {
            "transformer_latency_seconds": _summary(transformer),
            "end_to_end_latency_seconds": _summary(end_to_end),
            "throughput_images_per_second": _summary(throughput),
            "peak_device_memory_bytes": _summary(numeric_memory) if numeric_memory else None,
        },
        "output": {"path": image_reference, "sha256": image_sha256},
    }


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if type(args.warmup_runs) is not int or args.warmup_runs < 1:
        raise ValueError("warmup_runs must be a positive integer.")
    if type(args.measured_runs) is not int or args.measured_runs < 1:
        raise ValueError("measured_runs must be a positive integer.")
    if args.height < 1 or args.width < 1 or args.max_sequence_length < 1:
        raise ValueError("height, width, and max_sequence_length must be positive.")
    prompt_manifest = load_canonical_json(args.prompts)
    pairs = _prompt_pairs(prompt_manifest)
    prompt_manifest_sha256 = sha256_file(args.prompts)
    output, result_reference, staging = _resolve_output_paths(args.output_dir, args.result_json)
    try:
        runtime = load_qwen_pdd_runtime(args.export_dir, args.schedule, args.device)
        records: list[dict[str, Any]] = []
        grid_identity: dict[str, Any] | None = None
        for pair in pairs:
            raw_noise = runtime.make_raw_noise(seed=pair.seed, height=args.height, width=args.width)
            trajectory = runtime.trajectory_identity(raw_noise)
            if grid_identity is None:
                grid_identity = trajectory
            elif any(
                trajectory[key] != grid_identity[key]
                for key in (
                    "full_time_nodes",
                    "full_time_nodes_sha256",
                    "boundary_indices",
                    "boundary_time_nodes",
                    "boundary_time_nodes_sha256",
                    "first_sigma",
                )
            ):
                raise RuntimeError("PDD trajectory grid changed between prompt/seed pairs.")
            for _ in range(args.warmup_runs):
                _run_repetition(runtime, pair.prompt, raw_noise, args.max_sequence_length)
            observations = [
                _run_repetition(runtime, pair.prompt, raw_noise, args.max_sequence_length)
                for _ in range(args.measured_runs)
            ]
            image_reference = f"images/{pair.prompt_id}/{pair.seed}.png"
            image_path = staging / image_reference
            save_png(image_path, observations[0].image)
            records.append(
                _record_for_pair(
                    runtime,
                    pair,
                    trajectory=trajectory,
                    observations=observations,
                    image_reference=image_reference,
                    image_sha256=sha256_file(image_path),
                )
            )
        if grid_identity is None:
            raise RuntimeError("evaluation produced no trajectory identity.")
        dtype_name = str(runtime.dtype).removeprefix("torch.")
        result = {
            "schema_version": 1,
            "record_type": "pdd_qwen_evaluation",
            "identity": {
                "export_manifest_sha256": sha256_file(runtime.descriptor.root / "manifest.json"),
                "prompt_manifest_sha256": prompt_manifest_sha256,
                "model": dict(runtime.model_identity),
                "schedule": args.schedule,
                "grid": {
                    "grid_size": runtime.config.grid_size,
                    "grid_max_t": runtime.config.grid_max_t,
                    "flow_shift": runtime.config.flow_shift,
                    "full_time_nodes": grid_identity["full_time_nodes"],
                    "full_time_nodes_sha256": grid_identity["full_time_nodes_sha256"],
                    "boundary_indices": grid_identity["boundary_indices"],
                    "boundary_time_nodes": grid_identity["boundary_time_nodes"],
                    "boundary_time_nodes_sha256": grid_identity["boundary_time_nodes_sha256"],
                    "first_sigma": grid_identity["first_sigma"],
                },
            },
            "protocol": {
                "height": args.height,
                "width": args.width,
                "max_sequence_length": args.max_sequence_length,
                "batch_size": 1,
                "warmup_runs": args.warmup_runs,
                "measured_runs": args.measured_runs,
                "device": str(runtime.device),
                "dtype": dtype_name,
                "end_to_end_scope": "prompt_encode_through_vae_postprocess",
                "transformer_scope": "sum_of_root_student_forward_calls",
                "cuda_synchronize": runtime.device.type == "cuda",
                "tensor_hash_schema": "pdd_tensor_sha256_v1",
            },
            "records": records,
        }
        staged_result = staging / result_reference
        staged_result.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(staged_result, result)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        _publish_staging(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output / result_reference)


if __name__ == "__main__":
    main()
