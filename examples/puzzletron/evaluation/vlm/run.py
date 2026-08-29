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

"""Evaluate a local VLM checkpoint with a pinned benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["evaluate"]

REPOSITORY_ROOT = Path(__file__).absolute().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.puzzletron.evaluation import checkpoint  # noqa: E402
from examples.puzzletron.evaluation.vlm import suites  # noqa: E402
from examples.puzzletron.evaluation.vlm.evaluator import evaluate  # noqa: E402


def _checkpoint_directory(value: str) -> Path:
    checkpoint_path = Path(value).expanduser().absolute()
    if not checkpoint_path.is_dir():
        raise argparse.ArgumentTypeError(f"checkpoint is not a local directory: {checkpoint_path}")
    return checkpoint_path


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
        "--suite",
        choices=(
            "short",
            "quick",
            "adapter-smoke",
            *suites.SINGLE_TASK_SMOKE_SUITES,
            "full",
        ),
        default="short",
        help="Pinned Qwen 3.5 0.8B VLM benchmark suite.",
    )
    parser.add_argument("--batch-size", type=checkpoint.positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timeout-seconds",
        type=checkpoint.positive_float,
        default=None,
        help=(
            "Subprocess timeout; defaults to 3000 seconds for smoke suites and "
            "86400 seconds otherwise."
        ),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Hugging Face cache root used by the offline VLM benchmark media preflight.",
    )
    parser.add_argument(
        "--quick-manifest",
        type=Path,
        default=None,
        help="Versioned exact-row manifest required by the 344-row quick suite.",
    )
    parser.add_argument(
        "--mmvu-judge-api-type",
        choices=("openai", "azure"),
        default=None,
        help="Explicit MMVU judge provider for the full suite.",
    )
    parser.add_argument(
        "--mmvu-judge-model",
        default=None,
        help="Explicit MMVU judge model for the full suite.",
    )
    parser.add_argument(
        "--allow-judge-calls",
        action="store_true",
        help="Acknowledge that the full suite may call the configured MMVU judge.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate profile, task configs, revisions, credentials, and media roots without "
            "evaluation."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = evaluate(args)
        if args.preflight_only:
            result = {"preflight": result["preflight"]}
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
