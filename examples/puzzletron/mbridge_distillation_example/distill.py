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

"""
Knowledge Distillation Script for Puzzletron Models using Megatron-Bridge.

This script performs knowledge distillation using Megatron-Bridge checkpoints.
It loads both student and teacher models from Megatron-Bridge checkpoint paths
and supports YAML configuration files and CLI overrides.

Examples:
    see distill.sh for examples
"""

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING

import torch
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.recipes.llama import llama32_3b_pretrain_config
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer

logger: logging.Logger = logging.getLogger(__name__)


# Default paths (only for optional arguments)
DEFAULT_OUTPUT_DIR = "./distilled_output"


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation with Megatron-Bridge checkpoints",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to the YAML OmegaConf override file (optional)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--student-ckpt",
        type=str,
        required=True,
        help="Path to student checkpoint (iter_XXXXXXX directory). Required.",
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        required=True,
        help="Path to teacher checkpoint (iter_XXXXXXX directory). Required.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to tokenized dataset (without .bin extension). Required.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for distilled checkpoint. Default: {DEFAULT_OUTPUT_DIR}",
    )

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the knowledge distillation script.

    This function orchestrates the complete configuration workflow:
    1. Loads base configurations for student and teacher (from recipes)
    2. Sets checkpoint paths to load from Megatron-Bridge checkpoints
    3. Sets data paths
    4. Wraps both in a DistillationProvider
    5. Applies YAML overrides from --config-file (if exists)
    6. Applies CLI overrides using Hydra-style syntax
    7. Starts Megatron distillation with the final merged configuration
    """
    args, cli_overrides = parse_cli_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info("Megatron-Bridge Knowledge Distillation Script")
    logger.info("=" * 70)

    # Load base configurations from recipes (without weights)
    # We'll load weights from checkpoints via cfg.checkpoint.load
    logger.info("Loading base student and teacher configurations...")
    cfg: ConfigContainer = llama32_3b_pretrain_config()
    teacher_cfg = llama32_3b_pretrain_config()

    # Set checkpoint paths to load from Megatron-Bridge checkpoints
    cfg.checkpoint.load = args.student_ckpt
    teacher_cfg.checkpoint.load = args.teacher_ckpt

    # Set data paths
    # Format: [[list_of_paths], [list_of_weights]]
    cfg.dataset.blend = [[args.data_path], [1.0]]

    # Set output directory
    cfg.checkpoint.save = args.output_dir

    # Create distillation config
    kd_config = ModelOptDistillConfig(
        logit_layers=["output_layer", "output_layer"],
        intermediate_layer_pairs=[
            ["decoder.layers.3", "decoder.layers.3"],
            ["decoder.layers.6", "decoder.layers.9"],
            ["decoder.layers.11", "decoder.layers.18"],
            ["decoder.final_layernorm", "decoder.final_layernorm"],
        ],
        skip_lm_loss=True,
        kd_loss_scale=1.0,
        logit_kl_temperature=1.0,
    )

    # Convert to distillation provider
    cfg.model = convert_to_distillation_provider(cfg.model, teacher_cfg.model, kd_config)
    logger.info("Converted to DistillationProvider")

    # Print initial configuration on rank 0
    if get_rank_safe() == 0:
        logger.info("--- Initial Configuration ---")
        cfg.print_yaml()

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides if a config file is provided
    if args.config_file:
        logger.info(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.info("YAML overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    # These overrides take precedence over argparse defaults and YAML config
    # Example: train.train_iters=10000 checkpoint.save="/path/to/output"
    if cli_overrides:
        logger.info(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.info("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Display final configuration
    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("=" * 70)

    # Start distillation
    logger.info("Starting distillation...")
    distill(config=cfg)

    # Cleanup process group
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
