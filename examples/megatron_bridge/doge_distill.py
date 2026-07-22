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
"""

import argparse
import contextlib
import os
from dataclasses import fields
from pathlib import Path

import torch
from distill_callbacks import _TargetValidationCallback
from megatron.bridge import AutoBridge
from megatron.bridge.models.distillation_provider import (
    DistillationProvider,
    convert_to_distillation_provider,
)
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.training.pretrain import pretrain
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.distributed import DistributedDataParallelConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.distill.doge_megatron import DoGEForwardStep

with contextlib.suppress(ModuleNotFoundError):
    import modelopt.torch.puzzletron.plugins.mbridge  # noqa: F401


def _patched_to_cfg_dict(self):
    """Patch DistillationProvider config serialization for heterogeneous students.

    TODO: Remove once Megatron-Bridge serializes heterogeneous distillation providers correctly.
    """
    from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

    result = {"_target_": f"{self._super_class.__module__}.{self._super_class.__qualname__}"}
    excluded_fields = {"teacher", "kd_config"}
    for field in fields(self._super_class):
        if field.name.startswith("_") or field.name in excluded_fields:
            continue
        if hasattr(self, field.name):
            result[field.name] = _ConfigContainerBase._convert_value_to_dict(
                getattr(self, field.name)
            )
    for field in fields(self):
        if field.name.startswith("_") or field.name in excluded_fields:
            continue
        if field.name not in result:
            result[field.name] = _ConfigContainerBase._convert_value_to_dict(
                getattr(self, field.name)
            )
    return result


DistillationProvider.to_cfg_dict = _patched_to_cfg_dict


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


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
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
            "Fixed held-out DoGE target objective as WEIGHT PATH pairs.\n"
            "Used for DoGE weight updates and periodic target validation. Sources may differ "
            "from the training sources.\n"
            "Example:\n"
            "  --target_data_paths 0.6 /data/reasoning 0.4 /data/knowledge"
        ),
    )
    data.add_argument("--data_path_to_cache", help="Directory for Megatron dataset indices")
    data.add_argument("--use_mock_data", action="store_true", help="Use mock data for smoke tests")

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
    doge.add_argument(
        "--doge_min_blend_weight",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "Minimum normalized weight for each DoGE source after every update. Use this to keep "
            "low-weight sources active while alignment scores evolve."
        ),
    )
    doge.add_argument(
        "--doge_freeze_student",
        action="store_true",
        help="Log DoGE score diagnostics without updating student weights",
    )
    doge.add_argument(
        "--doge_freeze_blend",
        action="store_true",
        help="Log candidate DoGE blend weights without applying them",
    )
    doge.add_argument(
        "--doge_train_loss_mode",
        choices=("weighted", "sampled"),
        default="weighted",
        help=(
            "Student-update loss semantics. 'weighted' computes every source each iteration and "
            "combines losses by the current blend weights. 'sampled' samples one source by the "
            "current blend weights and returns that unweighted loss, matching normal sampled "
            "data-blend training more closely."
        ),
    )
    doge.add_argument(
        "--doge_weight_update_strategy",
        choices=("alignment", "kd_gap"),
        default="alignment",
        help=(
            "Signal used to update adaptive blend weights. 'alignment' uses the DoGE "
            "source-to-target gradient-alignment score. 'kd_gap' sets weights proportional to "
            "per-source KD loss as a naive PASER-style baseline."
        ),
    )
    doge.add_argument(
        "--doge_schedule_end_data_paths",
        nargs="+",
        help=(
            "Optional final blend in WEIGHT PATH format. When provided, DoGE linearly "
            "interpolates from --data_paths to this blend over the training run and does not "
            "apply adaptive blend updates."
        ),
    )
    doge.add_argument(
        "--doge_virtual_step_candidate_weights",
        action="append",
        nargs="+",
        type=float,
        metavar="WEIGHT",
        help=(
            "Candidate source-order blend weights for virtual-step diagnostics. Repeat this "
            "argument to evaluate multiple candidates, e.g. "
            "--doge_virtual_step_candidate_weights 95 2.5 2.5 "
            "--doge_virtual_step_candidate_weights 90 5 5. Weights are normalized."
        ),
    )
    doge.add_argument(
        "--doge_virtual_step_lr",
        type=_positive_float,
        help=(
            "Learning rate for virtual selected-parameter diagnostic steps. Defaults to --lr when "
            "--doge_virtual_step_candidate_weights is provided."
        ),
    )
    doge.add_argument(
        "--doge_virtual_step_num_steps",
        type=_positive_int,
        default=1,
        help=(
            "Number of repeated virtual SGD steps per candidate diagnostic. Values above 1 "
            "recompute source gradients on the same sampled source batches after each virtual "
            "parameter update."
        ),
    )
    doge.add_argument(
        "--doge_alignment_param_scope",
        choices=("final_mlp", "all_trainable"),
        default="final_mlp",
        help=(
            "Parameter scope for DoGE gradient scoring and virtual-step diagnostics. "
            "'final_mlp' is the cheap Qwen3-8B PoC probe; 'all_trainable' is expensive and "
            "intended only for diagnostics."
        ),
    )
    return parser.parse_args(argv)


def _build_model_provider(args: argparse.Namespace, hf_path: str):
    bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=args.trust_remote_code)
    provider = bridge.to_megatron_provider(load_weights=True)

    provider.tensor_model_parallel_size = args.tp_size
    provider.sequence_parallel = args.tp_size > 1
    provider.pipeline_model_parallel_size = args.pp_size
    provider.pipeline_dtype = torch.bfloat16
    provider.context_parallel_size = args.cp_size
    provider.expert_model_parallel_size = args.ep_size
    provider.expert_tensor_parallel_size = 1
    provider.seq_length = args.seq_length
    return provider


def _build_config(args: argparse.Namespace) -> ConfigContainer:
    student_provider = _build_model_provider(args, args.student_hf_path)
    teacher_provider = _build_model_provider(args, args.teacher_hf_path)
    distill_provider = convert_to_distillation_provider(
        student_provider,
        teacher_provider,
        ModelOptDistillConfig(skip_lm_loss=True, kd_loss_scale=1.0),
    )
    optimizer_config, scheduler_config = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=args.lr_warmup_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        adam_beta2=0.95,
    )
    dataset_kwargs = {
        "seq_length": args.seq_length,
        "path_to_cache": args.data_path_to_cache,
        "random_seed": args.seed,
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "num_dataset_builder_threads": 1,
        "data_sharding": True,
        "dataloader_type": "single",
        "skip_getting_attention_mask_from_dataset": True,
    }
    if args.use_mock_data:
        dataset_config = MockGPTDatasetConfig(**dataset_kwargs)
    else:
        dataset_config = GPTDatasetConfig(
            blend=get_blend_from_list(args.data_paths),
            split="99,1,0",
            **dataset_kwargs,
        )
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    return ConfigContainer(
        model=distill_provider,
        train=TrainingConfig(
            train_iters=args.train_iters,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            global_batch_size=args.gbs,
            micro_batch_size=args.mbs,
            manual_gc=True,
            manual_gc_interval=100,
        ),
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=dataset_config,
        logger=LoggerConfig(
            log_interval=args.log_interval,
            tensorboard_dir=os.path.join(args.output_dir, "tb_logs"),
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer", vocab_size=distill_provider.vocab_size
        ),
        checkpoint=CheckpointConfig(
            save_interval=args.eval_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            most_recent_k=5,
            ckpt_format="torch_dist",
            async_save=True,
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=args.seed),
        mixed_precision="bf16_mixed",
    )


def main(args: argparse.Namespace) -> None:
    """Build the DoGE forward step and pass it to the Megatron-Bridge training loop."""
    forward_step = DoGEForwardStep(
        data_paths=args.data_paths,
        target_data_paths=args.target_data_paths,
        meta_lr=args.doge_meta_lr,
        output_dir=args.output_dir,
        min_blend_weight=args.doge_min_blend_weight,
        freeze_student=args.doge_freeze_student,
        freeze_blend=args.doge_freeze_blend,
        schedule_end_data_paths=args.doge_schedule_end_data_paths,
        virtual_step_candidate_weights=args.doge_virtual_step_candidate_weights,
        virtual_step_lr=(
            args.doge_virtual_step_lr
            if args.doge_virtual_step_lr is not None
            else args.lr
            if args.doge_virtual_step_candidate_weights
            else None
        ),
        virtual_step_num_steps=args.doge_virtual_step_num_steps,
        alignment_param_scope=args.doge_alignment_param_scope,
        train_loss_mode=args.doge_train_loss_mode,
        weight_update_strategy=args.doge_weight_update_strategy,
        sampling_seed=args.seed,
    )

    print("Initial DoGE blend weights:")
    for path, weight in forward_step.blend_weights.items():
        print(f"  {weight:.6g} {path}")

    forward_step.write_trajectory_record(0)
    callbacks = []
    # Target validation does not work with mock data.
    if not args.use_mock_data:
        callbacks.append(_TargetValidationCallback(get_blend_from_list(args.target_data_paths)))
    pretrain(_build_config(args), forward_step, callbacks=callbacks)


if __name__ == "__main__":
    args = get_args()
    dist.setup()
    try:
        main(args)
    finally:
        dist.cleanup()
