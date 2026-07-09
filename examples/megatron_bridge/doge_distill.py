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
"""Tune distillation data-blend weights with DoGE and Megatron-Bridge.

DoGE paper: https://arxiv.org/abs/2310.15393

The planned outputs under ``--output_dir`` are DoGE distillation checkpoints,
``doge_weights.jsonl`` containing the weight trajectory, and
``learned_data_blend.txt`` containing the learned fixed blend.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the DoGE distillation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    model = parser.add_argument_group("model")
    model.add_argument("--student_hf_path", required=True, help="Student Hugging Face model")
    model.add_argument("--teacher_hf_path", required=True, help="Teacher Hugging Face model")
    model.add_argument("--trust_remote_code", action="store_true", help="Trust remote model code")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        metavar="VALUE",
        help=(
            "Tunable training sources as WEIGHT PATH pairs.\n"
            "Each pair is an independently tunable DoGE source; initial weights are normalized.\n"
            "Example:\n"
            "  --data_paths 0.1 /data/wikitext 0.45 /data/math 0.45 /data/stem"
        ),
    )
    data.add_argument(
        "--target_data_paths",
        nargs="+",
        required=True,
        metavar="VALUE",
        help=(
            "Fixed held-out target objective as WEIGHT PATH pairs.\n"
            "Sources may differ from the training sources.\n"
            "Example:\n"
            "  --target_data_paths 0.6 /data/reasoning 0.4 /data/knowledge"
        ),
    )
    data.add_argument("--data_path_to_cache", help="Directory for Megatron dataset indices")

    parallelism = parser.add_argument_group("parallelism")
    parallelism.add_argument(
        "--tp_size", type=_positive_int, default=1, help="Tensor-parallel size"
    )
    parallelism.add_argument(
        "--pp_size", type=_positive_int, default=1, help="Pipeline-parallel size"
    )
    parallelism.add_argument(
        "--cp_size", type=_positive_int, default=1, help="Context-parallel size"
    )
    parallelism.add_argument(
        "--ep_size", type=_positive_int, default=1, help="Expert-parallel size"
    )

    training = parser.add_argument_group("training")
    training.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for checkpoints, logs, weight trajectories, and the learned blend",
    )
    training.add_argument(
        "--seq_length", type=_positive_int, default=4096, help="Tokens per training sequence"
    )
    training.add_argument("--mbs", type=_positive_int, default=1, help="Micro-batch size")
    training.add_argument("--gbs", type=_positive_int, default=768, help="Global batch size")
    training.add_argument(
        "--train_iters",
        type=_positive_int,
        required=True,
        help="DoGE iterations, each updating both the blend weights and the student",
    )
    training.add_argument(
        "--lr", type=_positive_float, default=1e-4, help="Maximum student learning rate"
    )
    training.add_argument(
        "--min_lr", type=_positive_float, default=1e-5, help="Minimum student learning rate"
    )
    training.add_argument(
        "--lr_warmup_iters",
        type=int,
        default=50,
        help="Student learning-rate warm-up iterations",
    )
    training.add_argument(
        "--eval_interval",
        type=_positive_int,
        default=100,
        help="Training iterations between target evaluations",
    )
    training.add_argument(
        "--eval_iters", type=_positive_int, default=32, help="Batches per target evaluation"
    )
    training.add_argument(
        "--log_interval", type=_positive_int, default=10, help="Training iterations between logs"
    )
    training.add_argument(
        "--seed", type=int, default=1234, help="Random seed for model and data operations"
    )

    doge = parser.add_argument_group("DoGE")
    doge.add_argument(
        "--doge_meta_lr",
        type=_positive_float,
        required=True,
        help="Learning rate for exponentiated blend-weight updates",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    """Report that the DoGE distillation workflow has not been implemented yet."""
    raise SystemExit(
        "DoGE data-blend weight tuning is not implemented yet. "
        f"No outputs were written to {args.output_dir}."
    )


if __name__ == "__main__":
    main(get_args())
