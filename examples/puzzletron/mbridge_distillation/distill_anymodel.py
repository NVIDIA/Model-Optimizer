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

"""Knowledge Distillation Script for AnyModel Checkpoints.

This script performs knowledge distillation between student and teacher models
that have been converted to Megatron-Bridge format using import_anymodel_to_mbridge.py.

The distillation uses KL-Divergence loss between student and teacher logits
with temperature scaling (standard knowledge distillation from Hinton et al., 2015).

Usage:
    cd /workspace/Model-Optimizer

    # TODO: remove this once Megatron-Bridge is installed in the environment
    export PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Model-Optimizer:${PYTHONPATH}"

    # Basic usage (uses model's max seq_length, which may be very large)
    torchrun --nproc_per_node=1 examples/puzzletron/mbridge_distillation/distill_anymodel.py \
        --student-mbridge-ckpt /path/to/student/iter_0000000 \
        --teacher-mbridge-ckpt /path/to/teacher/iter_0000000 \
        --data-path /path/to/tokenized/dataset \
        --output-dir ./distilled_output

    # Recommended: Override sequence length and other training params for faster training
    torchrun --nproc_per_node=8 examples/puzzletron/mbridge_distillation/distill_anymodel.py \
        --student-mbridge-ckpt /path/to/student/iter_0000000 \
        --teacher-mbridge-ckpt /path/to/teacher/iter_0000000 \
        --data-path /path/to/tokenized/dataset \
        --output-dir ./distilled_output \
        dataset.sequence_length=8192 \
        model.tensor_model_parallel_size=8 \
        model.teacher.tensor_model_parallel_size=8 \
        train.global_batch_size=4 \
        train.micro_batch_size=1 \
        train.train_iters=5000 \
        logger.log_interval=1
"""

import argparse
import logging
import os
import sys

import torch
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.training.checkpointing import get_checkpoint_run_config_filename
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.distill import distill
from megatron.bridge.training.model_load_save import load_model_config
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from omegaconf import OmegaConf

# Import GenericHeterogeneousProvider so it can be instantiated when loading
# checkpoint configs that reference it (e.g., run_config.yaml with
# _target_: modelopt.torch.puzzletron.export.mbridge.base.GenericHeterogeneousProvider)
import modelopt.torch.puzzletron.export.mbridge  # noqa: F401

logger: logging.Logger = logging.getLogger(__name__)


def create_distillation_config() -> ModelOptDistillConfig:
    """Create KD config with output layer distillation only."""
    return ModelOptDistillConfig(
        logit_layers=["output_layer", "output_layer"],
        intermediate_layer_pairs=[],
        skip_lm_loss=True,
        kd_loss_scale=1.0,
        logit_kl_temperature=1.0,
    )


def create_base_config(
    student_model_provider,
    data_path: str,
    student_ckpt: str,
    output_dir: str,
    use_bf16: bool,
    use_fp16: bool,
) -> ConfigContainer:
    """Create base ConfigContainer with defaults."""
    return ConfigContainer(
        model=student_model_provider,
        train=TrainingConfig(global_batch_size=1, micro_batch_size=1, train_iters=100),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=1e-4,
            min_lr=1e-5,
            weight_decay=0.01,
            bf16=use_bf16,
            fp16=use_fp16,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="linear",
            lr_warmup_iters=0,
            start_weight_decay=0.01,
            end_weight_decay=0.01,
            weight_decay_incr_style="constant",
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            blend=[[data_path], [1.0]],
            split="9999,8,2",
            seq_length=student_model_provider.seq_length,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            dataloader_type="single",
        ),
        checkpoint=CheckpointConfig(load=student_ckpt, save=output_dir),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=None),
        validation=ValidationConfig(eval_interval=500, eval_iters=100),
        ddp=DistributedDataParallelConfig(grad_reduce_in_fp32=True),
        dist=DistributedInitConfig(),
        rng=RNGConfig(),
        rerun_state_machine=RerunStateMachineConfig(),
    )


def merge_checkpoint_configs(
    cfg: ConfigContainer, checkpoint_path: str, use_bf16: bool, use_fp16: bool
) -> None:
    """Merge tokenizer and optimizer configs from checkpoint if available."""
    try:
        run_config_path = get_checkpoint_run_config_filename(checkpoint_path)
        checkpoint_cfg = ConfigContainer.from_yaml(run_config_path)
        if checkpoint_cfg.tokenizer is not None:
            cfg.tokenizer = checkpoint_cfg.tokenizer
        if checkpoint_cfg.optimizer is not None:
            for key, value in checkpoint_cfg.optimizer.__dict__.items():
                if (
                    value is not None
                    and hasattr(cfg.optimizer, key)
                    and key not in ("bf16", "fp16")
                ):
                    setattr(cfg.optimizer, key, value)
        # Ensure bf16/fp16 are set correctly based on model dtype
        cfg.optimizer.bf16 = use_bf16
        cfg.optimizer.fp16 = use_fp16
    except Exception as e:
        logger.warning(f"Could not load additional configs from checkpoint: {e}")


def parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation with AnyModel checkpoints",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--student-mbridge-ckpt",
        type=str,
        required=True,
        help="Path to student checkpoint in MBridge format (must be iter_XXXXXXX directory).",
    )
    parser.add_argument(
        "--teacher-mbridge-ckpt",
        type=str,
        required=True,
        help="Path to teacher checkpoint in MBridge format (must be iter_XXXXXXX directory).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to tokenized dataset (without .bin extension).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./distilled_output",
        help="Output directory for distilled checkpoint.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to YAML OmegaConf override file (optional).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args, cli_overrides = parser.parse_known_args()
    return args, cli_overrides


def main() -> None:
    """Main distillation function."""
    args, cli_overrides = parse_cli_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info("Megatron-Bridge Knowledge Distillation Script (AnyModel)")
    logger.info("=" * 70)

    # Load model configs from checkpoints
    logger.info("Loading model configs from MBridge checkpoints...")
    student_model_provider, _ = load_model_config(args.student_mbridge_ckpt)
    teacher_model_provider, _ = load_model_config(args.teacher_mbridge_ckpt)

    # Detect model dtype for optimizer config
    model_params_dtype = getattr(student_model_provider, "params_dtype", torch.float32)
    use_bf16 = model_params_dtype == torch.bfloat16
    use_fp16 = model_params_dtype == torch.float16

    # Create base config with defaults
    cfg = create_base_config(
        student_model_provider,
        args.data_path,
        args.student_mbridge_ckpt,
        args.output_dir,
        use_bf16,
        use_fp16,
    )

    # Merge tokenizer and optimizer from checkpoint if available
    merge_checkpoint_configs(cfg, args.student_mbridge_ckpt, use_bf16, use_fp16)

    # Create distillation config and convert to DistillationProvider
    kd_config = create_distillation_config()
    cfg.model = convert_to_distillation_provider(
        student_provider=student_model_provider,
        teacher_provider=teacher_model_provider,
        kd_config=kd_config,
    )

    # Apply YAML and CLI overrides
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)
    if args.config_file:
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
    apply_overrides(cfg, OmegaConf.to_container(merged_omega_conf, resolve=True), excluded_fields)

    # Sync model seq_length with dataset sequence_length if they differ
    if (
        hasattr(cfg.dataset, "sequence_length")
        and cfg.dataset.sequence_length != cfg.model.seq_length
    ):
        cfg.model.seq_length = cfg.dataset.sequence_length
        if hasattr(cfg.model, "teacher") and cfg.model.teacher is not None:
            cfg.model.teacher.seq_length = cfg.dataset.sequence_length

    if get_rank_safe() == 0:
        logger.info("--- Final Configuration ---")
        cfg.print_yaml()

    # Run distillation
    logger.info("Starting distillation training...")
    distill(cfg)

    logger.info(f"✓ Distillation complete! Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
