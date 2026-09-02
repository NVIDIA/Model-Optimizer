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

"""Evaluate a local Hugging Face checkpoint with lmms-eval and vLLM."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["main"]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _load_runner() -> Callable[..., dict[str, object]]:
    """Load ModelOpt while keeping the CLI stdout machine-readable."""
    with redirect_stdout(sys.stderr):
        module = importlib.import_module("modelopt.torch.puzzletron.evaluation")
    return cast("Callable[..., dict[str, object]]", module.run_lmms_eval_checkpoint)


run_lmms_eval_checkpoint = _load_runner()

DEFAULT_TASKS = "ifeval,gsm8k"
_TASK_ALIASES = {"gsm8k": "modelopt_gsm8k"}
_QWEN_3_5_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
DEFAULT_SMOKE_TIMEOUT_SECONDS = 3_000.0
DEFAULT_FULL_TIMEOUT_SECONDS = 24 * 60 * 60.0


def _checkpoint_directory(value: str) -> Path:
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_dir():
        raise argparse.ArgumentTypeError(f"checkpoint is not a local directory: {checkpoint}")
    return checkpoint


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _gpu_memory_utilization(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("GPU memory utilization must be in (0, 1]")
    return parsed


def _task_selection(value: str) -> str:
    tasks = [task.strip() for task in value.split(",")]
    if not tasks or any(not task for task in tasks):
        raise argparse.ArgumentTypeError("tasks must be a comma-separated list of names")
    return ",".join(tasks)


def _resolved_tasks(value: str) -> str:
    return ",".join(_TASK_ALIASES.get(task, task) for task in value.split(","))


def _automatic_model_args(checkpoint: Path) -> dict[str, object]:
    """Return narrow compatibility defaults inferred from local checkpoint metadata."""
    try:
        config = json.loads((checkpoint / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(config, dict):
        return {}
    model_types = set()
    if isinstance(config.get("model_type"), str):
        model_types.add(config["model_type"])
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("model_type"), str):
        model_types.add(text_config["model_type"])
    if model_types & _QWEN_3_5_MODEL_TYPES:
        return {"reasoning_parser": "qwen3"}
    return {}


def _lmms_eval_gsm8k_config() -> Path:
    spec = importlib.util.find_spec("lmms_eval")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        raise RuntimeError(
            "lmms_eval is not installed; run this command in the Puzzletron worker image"
        )
    config = Path(next(iter(locations))) / "tasks/gsm8k/gsm8k.yaml"
    if not config.is_file():
        raise RuntimeError(f"installed lmms_eval has no GSM8K task config: {config}")
    return config.resolve()


def _prepare_compatibility_tasks(output_root: Path, tasks: str) -> Path | None:
    """Materialize narrow task overrides while inheriting pinned lmms-eval behavior."""
    if "modelopt_gsm8k" not in tasks.split(","):
        return None
    tasks_root = output_root.expanduser().resolve() / "task_configs"
    tasks_root.mkdir(parents=True, exist_ok=True)
    config = {
        "include": str(_lmms_eval_gsm8k_config()),
        "task": "modelopt_gsm8k",
        "dataset_path": "openai/gsm8k",
        "fewshot_config": {"sampler": "default"},
    }
    (tasks_root / "modelopt_gsm8k.yaml").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return tasks_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="For the full native interface, run: python -m lmms_eval --help",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=_checkpoint_directory,
        help="Local Hugging Face checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory for isolated per-attempt artifacts.",
    )
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS,
        type=_task_selection,
        help=(
            f"Comma-separated text tasks (default: {DEFAULT_TASKS}); gsm8k uses a generated "
            "namespaced compatibility task that inherits the pinned evaluator config."
        ),
    )
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument(
        "--limit",
        type=_positive_int,
        default=8,
        help="Maximum samples per task; the default 8 is a wiring smoke.",
    )
    limit.add_argument(
        "--full",
        dest="limit",
        action="store_const",
        const=None,
        help="Run every sample instead of the default wiring smoke.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=_gpu_memory_utilization,
        default=0.85,
    )
    parser.add_argument("--max-model-len", type=_positive_int, default=8192)
    parser.add_argument(
        "--model-profile",
        choices=("auto", "none"),
        default="auto",
        help=(
            "Apply narrow vLLM compatibility defaults inferred from config.json; "
            "auto currently maps Qwen 3.5 to reasoning_parser=qwen3, while none "
            "leaves all model-specific arguments explicit."
        ),
    )
    parser.add_argument(
        "--reasoning-parser",
        default=None,
        help="Explicit vLLM reasoning parser; overrides the detected model profile.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=None,
        help=(
            "Subprocess timeout; defaults to 3000 seconds for a limited smoke "
            "and 86400 seconds with --full."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow reviewed checkpoint-provided Python code (disabled by default).",
    )
    parser.add_argument(
        "--lmms-eval-args",
        nargs=argparse.REMAINDER,
        default=[],
        metavar="ARG",
        help="Forward remaining arguments to lmms-eval; this option must be last.",
    )
    return parser


def _settings(
    args: argparse.Namespace,
    *,
    compatibility_tasks_root: Path | None = None,
    automatic_model_args: dict[str, object] | None = None,
) -> dict[str, object]:
    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = (
            DEFAULT_FULL_TIMEOUT_SECONDS if args.limit is None else DEFAULT_SMOKE_TIMEOUT_SECONDS
        )
    model_args = {
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_profile == "auto":
        if automatic_model_args is None:
            automatic_model_args = _automatic_model_args(args.checkpoint)
        model_args.update(automatic_model_args)
    if args.reasoning_parser is not None:
        model_args["reasoning_parser"] = args.reasoning_parser
    settings = {
        "tasks": _resolved_tasks(args.tasks),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timeout_seconds": timeout_seconds,
        "topology": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": False,
            "distributed_executor_backend": "mp",
            "gpu_group_size": args.tensor_parallel_size,
        },
        "model_args": model_args,
    }
    extra_args = list(args.lmms_eval_args)
    if compatibility_tasks_root is not None:
        extra_args.extend(["--include_path", str(compatibility_tasks_root)])
    if extra_args:
        settings["extra_args"] = extra_args
    return settings


def main(argv: list[str] | None = None) -> int:
    """Evaluate a checkpoint and print results or failure diagnostics as JSON.

    Args:
        argv: Command-line arguments. Uses the process arguments when omitted.

    Returns:
        Zero on success and one on failure. Results are written to stdout and
        failure diagnostics are written to stderr.
    """
    args = _build_parser().parse_args(argv)
    try:
        tasks = _resolved_tasks(args.tasks)
        compatibility_tasks_root = _prepare_compatibility_tasks(args.output_dir, tasks)
        automatic_model_args = (
            _automatic_model_args(args.checkpoint) if args.model_profile == "auto" else {}
        )
        if args.reasoning_parser is None and automatic_model_args:
            rendered = ",".join(
                f"{key}={value}" for key, value in sorted(automatic_model_args.items())
            )
            print(
                f"Detected Qwen 3.5 checkpoint; applying vLLM model argument {rendered}. "
                "Override with --reasoning-parser or disable with --model-profile none.",
                file=sys.stderr,
            )
        result = run_lmms_eval_checkpoint(
            args.checkpoint,
            output_root=args.output_dir,
            settings=_settings(
                args,
                compatibility_tasks_root=compatibility_tasks_root,
                automatic_model_args=automatic_model_args,
            ),
        )
    except Exception as error:
        payload = {
            "error": type(error).__name__,
            "message": str(error),
            **{
                name: getattr(error, name)
                for name in ("command_path", "stdout_path", "stderr_path")
                if getattr(error, name, None)
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
