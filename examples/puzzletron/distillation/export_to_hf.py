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

"""Export a Megatron Bridge checkpoint back to HuggingFace format.

Overview
--------
Takes a trained Megatron-Core (MCore) checkpoint produced by ``distill.py`` and converts
it back to a HuggingFace-format checkpoint using ``AutoBridge.export_ckpt()``.

The student model is identified by its HF config (no weights needed for the bridge
construction itself); the MCore checkpoint weights are then mapped back to HF tensors
and written to ``--output-hf-checkpoint``.

Pipeline
--------
1. Load the student HF config (config.json only) to obtain ``block_configs``.
2. Get the AnyModel ModelDescriptor for the student converter.
3. Apply class-level provider patches (same as distill.py).
4. Load the student HF weights into a Bridge object (via ``deci_x_patcher`` if available).
5. Export the MCore checkpoint to HF format via ``bridge.export_ckpt()``.
6. Copy tokenizer / config files from the student HF checkpoint to the export directory.

Usage
-----
    python export_to_hf.py \\
        --student nemo \\
        --student-hf-checkpoint /path/to/student-hf \\
        --student-mcore-checkpoint /path/to/mcore-ckpt \\
        --output-hf-checkpoint /path/to/output-hf

Configuration precedence
------------------------
All remaining arguments are passed through as Hydra-style dotlist overrides
(e.g. ``model.tensor_model_parallel_size=4``).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from _common import (
    MODEL_REGISTRY,
    _get_block_configs,
    _get_model_descriptor,
    _load_bridge,
    _load_hf_config,
    configure_logging,
    run_entrypoint,
)
from megatron.bridge.utils.common_utils import print_rank_0

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    _validate_registry_keys(args)

    student_hf_id, student_converter = MODEL_REGISTRY[args.student]
    student_path = args.student_hf_checkpoint or student_hf_id

    _log_config(args, student_converter)

    # ------------------------------------------------------------------
    # Step 1: Load HF config (no weights, just config.json)
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading HF config")
    student_hf_cfg = _load_hf_config(student_path, args.trust_remote_code)

    # ------------------------------------------------------------------
    # Step 2: Obtain block_configs for the student
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading block_configs")
    student_block_configs = _get_block_configs(student_hf_cfg, student_converter)
    logger.info(
        "  student block_configs: %s",
        f"{len(student_block_configs)} layers" if student_block_configs else "None (homogeneous)",
    )

    # ------------------------------------------------------------------
    # Step 3: Get AnyModel descriptor (needed for deci_x_patcher)
    # ------------------------------------------------------------------
    logger.info("Step 3: Getting AnyModel descriptor")
    student_descriptor = _get_model_descriptor(student_converter)

    # ------------------------------------------------------------------
    # Step 4: Apply class-level provider patches (one-time setup)
    # ------------------------------------------------------------------
    logger.info("Step 4: Applying provider patches")
    from provider_patch import apply_distillation_patch, apply_patch

    apply_patch()
    apply_distillation_patch()

    # ------------------------------------------------------------------
    # Step 5: Load HF model into a Bridge object
    # ------------------------------------------------------------------
    logger.info("Step 5: Loading HF model into Megatron Bridge")
    student_bridge = _load_bridge(student_path, args.trust_remote_code, student_descriptor)
    logger.info("  student bridge: %s", type(student_bridge).__name__)

    # ------------------------------------------------------------------
    # Step 6: Export MCore checkpoint → HuggingFace format
    # ------------------------------------------------------------------
    print_rank_0(f"\n{'=' * 80}")
    print_rank_0("Exporting to HuggingFace format...")
    print_rank_0(f"{'=' * 80}\n")
    print_rank_0(f"  MCore checkpoint : {args.student_mcore_checkpoint}")
    print_rank_0(f"  HF output path   : {args.output_hf_checkpoint}")

    os.makedirs(args.output_hf_checkpoint + "/subblocks_safetensors", exist_ok=True)

    from layer_patchers import mbridge_patcher

    with mbridge_patcher(
        block_configs=student_block_configs,
        num_attention_heads=student_hf_cfg.num_attention_heads,
        hidden_size=student_hf_cfg.hidden_size,
        apply_no_ops=True,
    ):
        student_bridge.export_ckpt(
            megatron_path=args.student_mcore_checkpoint,
            hf_path=args.output_hf_checkpoint,
            show_progress=True,
            strict=True,
        )

    print_rank_0(f"✅ Successfully exported model to: {args.output_hf_checkpoint}")

    # ------------------------------------------------------------------
    # Step 7: Copy tokenizer / config files to the export directory
    # ------------------------------------------------------------------
    print_rank_0(f"  Copying configs from: {student_path}")
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        src = Path(student_path) / fname
        dst = Path(args.output_hf_checkpoint) / fname
        shutil.copy(src, dst)
        print_rank_0(f"  ✅ Copied {fname}")

    print_rank_0(f"\n{'=' * 80}")
    print_rank_0("Export complete!")
    print_rank_0(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_registry_keys(args: argparse.Namespace) -> None:
    valid = sorted(MODEL_REGISTRY)
    if args.student not in MODEL_REGISTRY:
        raise ValueError(f"Unknown --student '{args.student}'. Valid choices: {valid}")


def _log_config(args: argparse.Namespace, student_converter: str) -> None:
    logger.info("=== mbridge_distillation/export ===")
    logger.info("  student key:              %s  (converter=%s)", args.student, student_converter)
    logger.info("  student HF checkpoint:    %s", args.student_hf_checkpoint)
    logger.info("  student MCore checkpoint: %s", args.student_mcore_checkpoint)
    logger.info("  output HF path:           %s", args.output_hf_checkpoint)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --student, --trust-remote-code ---
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
        help="Pass trust_remote_code=True to HuggingFace config/model loading.",
    )

    # --- Export-specific paths ---
    parser.add_argument(
        "--student-hf-checkpoint",
        default=None,
        metavar="PATH",
        dest="student_hf_checkpoint",
        help=(
            "Local directory of the HF-format student checkpoint (config + weights). "
            "If omitted, the model config is fetched from HuggingFace Hub."
        ),
    )
    parser.add_argument(
        "--student-mcore-checkpoint",
        required=True,
        metavar="PATH",
        dest="student_mcore_checkpoint",
        help="Local directory of the trained Megatron-Core checkpoint to export.",
    )
    parser.add_argument(
        "--output-hf-checkpoint",
        required=True,
        metavar="PATH",
        dest="output_hf_checkpoint",
        help="Destination directory for the exported HuggingFace checkpoint.",
    )

    # --- Hydra-style pass-through overrides ---
    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    return args


if __name__ == "__main__":
    run_entrypoint(main, _parse_args)
