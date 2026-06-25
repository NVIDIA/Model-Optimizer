#!/usr/bin/env python3
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

"""Knowledge distillation with heterogeneous AnyModel/Puzzletron models via Megatron Bridge.

Overview
--------
Thin wrapper around ``megatron_bridge/distill.py`` that injects Puzzletron-specific
behavior (per-layer block_configs, provider patches, hybrid MoE aux-loss fix, YAML/Hydra
config overrides) via :class:`~hooks.PuzzletronHooks`.

For a full description of the heterogeneous distillation pipeline, including the two-level
provider patching and block_config sources, see ``hooks.py``.

Supported models (``--student`` / ``--teacher``)
-------------------------------------------------
    gptoss → openai/gpt-oss-20b                          (GPT-OSS, all-MoE)
    nemo2  → nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  (Nemotron-H, Mamba+MoE hybrid)
    llama  → meta-llama/Llama-3.2-3B-Instruct            (dense GPT)
    qwen   → Qwen/Qwen3-8B                               (dense GPT with GQA)

Usage
-----
    # Minimal: Nemotron-H teacher → Llama student (1 GPU)
    torchrun --nproc_per_node=1 distill.py \\
        --student llama --teacher nemo2 \\
        --teacher-checkpoint /path/to/nemotronh

    # From local checkpoints, with custom YAML config:
    torchrun --nproc_per_node=8 distill.py \\
        --student llama --student-checkpoint /path/to/student \\
        --teacher nemo2  --teacher-checkpoint /path/to/teacher \\
        --config-file kd.yaml

    # CLI overrides (Hydra-style), overriding YAML defaults:
    torchrun --nproc_per_node=8 distill.py \\
        --student llama --teacher nemo2 \\
        model.tensor_model_parallel_size=4 \\
        train.train_iters=50000 \\
        optimizer.lr=1e-4

    # Parallelism flags (convenience; same fields can be set via YAML or Hydra dotlist):
    torchrun --nproc_per_node=8 distill.py \\
        --student llama --teacher nemo2 \\
        --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \\
        --expert-model-parallel-size 4 --expert-tensor-parallel-size 1

    # All-heterogeneous: GPT-OSS teacher → Nemotron-H student
    torchrun --nproc_per_node=8 distill.py \\
        --student nemo2  --student-checkpoint /path/to/student \\
        --teacher gptoss --teacher-checkpoint /path/to/teacher \\
        model.tensor_model_parallel_size=4 \\
        model.expert_model_parallel_size=4

Configuration precedence
------------------------
1. Defaults from ``_pretrain_common()`` (set in PuzzletronHooks.build_config).
2. YAML file from ``--config-file`` (defaults to ``kd-container-default.yaml`` if present).
3. Hydra-style CLI overrides (highest priority).

All sections of the YAML (``model``, ``train``, ``checkpoint``, ``dataset``, …) are applied to
the :class:`~megatron.bridge.training.config.ConfigContainer`.

KD-specific settings (in YAML or as CLI ``model.kd_config.*``):
    logit_layers, intermediate_layer_pairs, skip_lm_loss,
    kd_loss_scale, logit_kl_temperature.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from _common import DEFAULT_CONFIG_FILE, MODEL_REGISTRY, configure_logging, run_entrypoint
from hooks import PuzzletronHooks

# Add megatron_bridge to sys.path so we can import the shared main()
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "megatron_bridge"))
from distill import main as mbridge_main  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

if os.environ.get("MBRIDGE_PATCHER_DEBUG", "0").lower() in ("1", "true", "yes", "on"):
    logging.getLogger("layer_patchers").setLevel(logging.DEBUG)
    logging.getLogger("provider_patch").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    _validate_registry_keys(args)

    student_hf_id, student_converter = MODEL_REGISTRY[args.student]
    teacher_hf_id, teacher_converter = MODEL_REGISTRY[args.teacher]
    student_path = args.student_checkpoint or student_hf_id
    teacher_path = args.teacher_checkpoint or teacher_hf_id

    logger.info("=== puzzletron/distillation/distill ===")
    logger.info("  student key:       %s  (converter=%s)", args.student, student_converter)
    logger.info("  student load path: %s", student_path)
    logger.info("  teacher key:       %s  (converter=%s)", args.teacher, teacher_converter)
    logger.info("  teacher load path: %s", teacher_path)
    logger.info("  config file:       %s", args.config_file)
    logger.info(
        "  parallelism:       TP=%s PP=%s EP=%s ETP=%s",
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.expert_tensor_parallel_size,
    )
    logger.info("  CLI overrides:     %s", args.overrides)

    # Map puzzletron arg names to the mbridge-compatible names expected by
    # _configure_provider() in megatron_bridge/distill.py.
    args.student_hf_path = student_path
    args.teacher_hf_path = teacher_path
    args.tp_size = args.tensor_model_parallel_size
    args.pp_size = args.pipeline_model_parallel_size
    args.ep_size = args.expert_model_parallel_size
    args.cp_size = 1  # context parallelism not yet supported for heterogeneous models
    args.seq_length = None  # comes from YAML config, not CLI

    hooks = PuzzletronHooks(
        student_path=student_path,
        student_converter=student_converter,
        teacher_path=teacher_path,
        teacher_converter=teacher_converter,
        config_file=args.config_file,
        overrides=args.overrides,
        trust_remote_code=args.trust_remote_code,
    )
    mbridge_main(args, hooks=hooks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_registry_keys(args: argparse.Namespace) -> None:
    valid = sorted(MODEL_REGISTRY)
    if args.student not in MODEL_REGISTRY:
        raise ValueError(f"Unknown --student '{args.student}'. Valid choices: {valid}")
    if args.teacher not in MODEL_REGISTRY:
        raise ValueError(f"Unknown --teacher '{args.teacher}'. Valid choices: {valid}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Student ---
    parser.add_argument(
        "--student",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help=(
            "Student model key. Determines the HuggingFace model ID (used when "
            "--student-checkpoint is omitted) and the AnyModel converter for block_configs."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        dest="trust_remote_code",
        help="Pass trust_remote_code=True to HuggingFace config/model loading.",
    )
    parser.add_argument(
        "--student-checkpoint",
        default=None,
        metavar="PATH",
        dest="student_checkpoint",
        help=(
            "Local directory of an HF-format student checkpoint. "
            "If omitted, the model is loaded from HuggingFace Hub."
        ),
    )

    # --- Teacher ---
    parser.add_argument(
        "--teacher",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help=(
            "Teacher model key. Determines the HuggingFace model ID (used when "
            "--teacher-checkpoint is omitted) and the AnyModel converter for block_configs."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        metavar="PATH",
        dest="teacher_checkpoint",
        help=(
            "Local directory of an HF-format teacher checkpoint. "
            "If omitted, the model is loaded from HuggingFace Hub."
        ),
    )

    # --- Configuration file ---
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE) if DEFAULT_CONFIG_FILE.exists() else None,
        metavar="YAML",
        dest="config_file",
        help=(
            "Path to a YAML override file (OmegaConf format). "
            "Merged on top of the base config before CLI overrides are applied. "
            "Defaults to kd-container-default.yaml in the script directory (if it exists)."
        ),
    )

    # --- Parallelism ---
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="tensor_model_parallel_size",
        help="Tensor model parallel size (TP). Also enables sequence parallel when > 1.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="pipeline_model_parallel_size",
        help="Pipeline model parallel size (PP).",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="expert_model_parallel_size",
        help="Expert model parallel size (EP) for MoE.",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="expert_tensor_parallel_size",
        help="Expert tensor parallel size (ETP) for MoE.",
    )

    # Hydra-style pass-through overrides (everything that doesn't match a known flag)
    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    return args


if __name__ == "__main__":
    run_entrypoint(main, _parse_args)
