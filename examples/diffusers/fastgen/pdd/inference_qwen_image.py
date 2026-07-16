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

"""Run conditional-only PDD inference from a complete Qwen-Image PDD export."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

sys.dont_write_bytecode = True

_THIS_DIR = Path(__file__).resolve().parent
_FASTGEN_DIR = _THIS_DIR.parent
_REPO_ROOT = _FASTGEN_DIR.parents[2]
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pdd.inference_runtime import (  # noqa: E402
    _model_identity,
    _normalize_prompt_condition,
    _validate_qwen_projection,
    build_pdd_student,
    load_qwen_pdd_runtime,
    save_png,
)

__all__ = [
    "_model_identity",
    "_normalize_prompt_condition",
    "_validate_qwen_projection",
    "build_pdd_student",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--schedule", choices=("pdd-2", "pdd-4", "pdd-8"), default="pdd-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def _resolve_outputs(output_value: Path, result_value: Path) -> tuple[Path, Path, str]:
    if output_value.is_symlink() or result_value.is_symlink():
        raise ValueError("PDD output and result JSON cannot be symlinks.")
    output = output_value.resolve()
    result_json = result_value.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"PDD inference output already exists: {output}.")
    if result_json.exists() or result_json.is_symlink():
        raise FileExistsError(f"PDD result JSON already exists: {result_json}.")
    try:
        reference = output.relative_to(result_json.parent).as_posix()
    except ValueError as error:
        raise ValueError("PDD output must be beneath the result JSON directory.") from error
    return output, result_json, reference


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    from pdd.artifacts import sha256_file, write_canonical_json

    output, result_json, output_reference = _resolve_outputs(args.output, args.result_json)
    if not isinstance(args.prompt_id, str) or not args.prompt_id.strip():
        raise ValueError("prompt_id must be non-empty.")
    if args.seed < 0 or args.seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63).")
    if args.max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive.")

    runtime = load_qwen_pdd_runtime(args.export_dir, args.schedule, args.device)
    condition = runtime.encode_prompt(args.prompt, args.max_sequence_length)
    noise = runtime.make_raw_noise(seed=args.seed, height=args.height, width=args.width)
    transformer_invocations = 0
    scheduler_step_calls = 0

    def count_invocation(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: Mapping[str, Any]
    ) -> None:
        nonlocal transformer_invocations
        transformer_invocations += 1

    scheduler = runtime.scheduler
    scheduler_state = vars(scheduler).get("step")
    scheduler_had_instance_step = "step" in vars(scheduler)
    original_scheduler_step = scheduler.step

    def counted_scheduler_step(*call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal scheduler_step_calls
        scheduler_step_calls += 1
        return original_scheduler_step(*call_args, **call_kwargs)

    def restore_scheduler_step() -> None:
        if scheduler_had_instance_step:
            setattr(scheduler, "step", scheduler_state)
        elif "step" in vars(scheduler):
            delattr(scheduler, "step")

    with ExitStack() as cleanup:
        cleanup.callback(restore_scheduler_step)
        setattr(scheduler, "step", counted_scheduler_step)
        hook = runtime.student.register_forward_pre_hook(count_invocation, with_kwargs=True)
        cleanup.callback(hook.remove)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()
        images = runtime.sample_decode(condition, noise)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
        latency = time.perf_counter() - started

    expected_invocations = len(runtime.config.inference_blocks)
    if transformer_invocations != expected_invocations:
        raise RuntimeError(
            f"PDD sampler made {transformer_invocations} transformer calls; "
            f"expected {expected_invocations}."
        )
    if scheduler_step_calls != 0:
        raise RuntimeError(
            f"PDD sampler unexpectedly called scheduler.step {scheduler_step_calls} times."
        )
    if len(images) != 1:
        raise RuntimeError(f"PDD single-prompt inference returned {len(images)} images.")
    if not math.isfinite(latency) or latency <= 0:
        raise RuntimeError("PDD inference latency measurement is invalid.")
    save_png(output, images[0])

    result_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 2,
        "record_type": "pdd_inference",
        "condition": args.schedule.replace("-", "_"),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "schedule": args.schedule,
        "blocks": list(runtime.config.inference_blocks),
        "height": args.height,
        "width": args.width,
        "export_manifest_sha256": sha256_file(runtime.descriptor.root / "manifest.json"),
        "output": {"path": output_reference, "sha256": sha256_file(output)},
        "scheduler_steps": expected_invocations,
        "observed_scheduler_step_calls": scheduler_step_calls,
        "actual_transformer_invocations": transformer_invocations,
        "batch_normalized_transformer_evaluations": transformer_invocations,
        "latency_seconds": latency,
    }
    write_canonical_json(result_json, result)
    print(result_json)


if __name__ == "__main__":
    main()
