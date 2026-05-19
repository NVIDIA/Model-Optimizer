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
"""Distillation script for Megatron-Bridge.

Loads student and teacher models directly from HuggingFace checkpoints (local or remote) and saves the distilled model
to `<output_dir>/checkpoints` in megatron distributed checkpoint or HuggingFace format.

See `README.md` in this directory for example usage and data preparation instructions.
"""

import argparse
import contextlib
import os
from dataclasses import fields

import torch
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
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.distributed import DistributedDataParallelConfig
from transformers import AutoConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import print_rank_0

with contextlib.suppress(ModuleNotFoundError):
    import modelopt.torch.puzzletron.plugins.mbridge  # noqa: F401

SEED = 1234


def _patched_to_cfg_dict(self):
    """Patched DistillationProvider.to_cfg_dict method for heterogeneous teacher and student models.

    TODO: Upstream this patch to Megatron-Bridge.
    """
    from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

    result = {"_target_": f"{self._super_class.__module__}.{self._super_class.__qualname__}"}
    # Use fields from the actual student provider class, not DistillationProvider.
    # DistillationProvider's __dataclass_fields__ only includes TransformerConfig fields
    # (set at class definition time), missing GPTModelProvider-level fields like
    # vocab_size, share_embeddings_and_output_weights, etc.
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


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Distillation for Megatron-Bridge.")
    # Model arguments (accepts HuggingFace input only at the moment)
    parser.add_argument(
        "--student_hf_path",
        type=str,
        required=True,
        help="HuggingFace model name or path for the student (e.g. Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--teacher_hf_path",
        type=str,
        required=True,
        help="HuggingFace model name or path for the teacher (e.g. Qwen/Qwen3-8B)",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    # Parallelism arguments
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp_size", type=int, default=1, help="Expert tensor parallel size")

    # Dataset arguments
    parser.add_argument(
        "--data_paths",
        nargs="+",
        help="List of tokenized data paths to load from (weight1 path1 weight2 path2 ...)",
    )
    parser.add_argument(
        "--split", type=str, default="99,1,0", help="Train,Val,Test ratios to split data"
    )
    parser.add_argument(
        "--data_path_to_cache", type=str, default=None, help="Path to cache the dataset indices"
    )
    parser.add_argument(
        "--use_mock_data", action="store_true", help="Use mock data instead of --data_paths"
    )
    # Training & Eval arguments
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Folder for logging and checkpoint saving"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=4096,
        help="Number of tokens per input sample. Use 8192 if your dataset has longer sequences.",
    )
    parser.add_argument("--mbs", type=int, default=1, help="Micro-batch Size")
    parser.add_argument("--gbs", type=int, default=768, help="Global Batch Size")
    parser.add_argument(
        "--train_iters", type=int, required=True, help="Number of training iterations"
    )
    parser.add_argument(
        "--no_skip_lm_loss", action="store_true", help="Disable skipping language model loss"
    )
    parser.add_argument("--kd_loss_scale", type=float, default=1.0, help="KD loss weight")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--lr_warmup_iters", type=int, default=50, help="Number of LR warmup steps")
    parser.add_argument(
        "--eval_interval", type=int, default=100, help="Validate + checkpoint every <N> steps"
    )
    parser.add_argument(
        "--eval_iters", type=int, default=32, help="Number of batches per validation stage"
    )
    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=10, help="Write to log every <N> steps")
    parser.add_argument(
        "--wandb_project", type=str, help="Wandb project name (required to enable Wandb logging)"
    )
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity name (optional)")
    parser.add_argument("--wandb_exp_name", type=str, help="Wandb experiment name (optional)")
    # Export arguments
    parser.add_argument(
        "--hf_export_path",
        type=str,
        default=None,
        help=(
            "Path where to save the HuggingFace export. "
            "If provided, exports last iteration checkpoint to HF format after distillation."
        ),
    )
    parser.add_argument(
        "--student_hf_model",
        type=str,
        required=False,
        default=None,
        help="HuggingFace model ID to use as template for export (e.g., Qwen/Qwen3-0.6B). "
        "Should match the base architecture of the student model if --hf_export_path is provided.",
    )
    args = parser.parse_args()

    # Sanity checks
    if not args.use_mock_data and not args.data_paths:
        raise ValueError("Must provide either --data_paths or set --use_mock_data.")

    if args.hf_export_path and not args.student_hf_model:
        raise ValueError("Must provide --student_hf_model if --hf_export_path is provided.")

    print_rank_0("\n==================== Arguments ====================")
    for k, v in args.__dict__.items():
        print_rank_0(f"{k:<35} {v}")
    print_rank_0("===================================================\n")

    return args


def main(args: argparse.Namespace):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    tensorboard_dir = os.path.join(args.output_dir, "tb_logs")

    # Build student and teacher model providers
    def _build_model_provider(hf_path):
        bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=args.trust_remote_code)
        provider = bridge.to_megatron_provider(load_weights=True)

        # Override parallelism / training settings
        provider.tensor_model_parallel_size = args.tp_size
        provider.sequence_parallel = args.tp_size > 1
        provider.pipeline_model_parallel_size = args.pp_size
        provider.pipeline_dtype = torch.bfloat16
        provider.context_parallel_size = args.cp_size
        provider.expert_model_parallel_size = args.ep_size
        provider.expert_tensor_parallel_size = args.etp_size
        provider.seq_length = args.seq_length
        return provider

    # TODO: Support megatron-ckpt as an alternative to HF checkpoints (e.g. /path/to/ckpt/iter_0000000)
    # Still requires an HF model name or path to build provider correctly
    student_provider = _build_model_provider(args.student_hf_path)
    teacher_provider = _build_model_provider(args.teacher_hf_path)

    # Wrap into DistillationProvider
    kd_config = ModelOptDistillConfig(
        skip_lm_loss=not args.no_skip_lm_loss, kd_loss_scale=args.kd_loss_scale
    )
    distill_provider = convert_to_distillation_provider(
        student_provider, teacher_provider, kd_config
    )

    # Build optimizer and scheduler
    optimizer_config, scheduler_config = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=args.lr_warmup_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        adam_beta2=0.95,
    )

    # Build dataset config
    dataset_kwargs = {
        "seq_length": args.seq_length,
        "path_to_cache": args.data_path_to_cache,
        "random_seed": SEED,
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
        # Convert flat CLI list (e.g. ["1.0", "/path/data"]) to Megatron blend format
        blend = get_blend_from_list(args.data_paths)
        dataset_config = GPTDatasetConfig(blend=blend, split=args.split, **dataset_kwargs)

    # Assemble ConfigContainer and run distillation
    config = ConfigContainer(
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
        # TODO: Replace validation args in train with validation config in nemo:26.04
        # validation=ValidationConfig(eval_interval=args.eval_interval, eval_iters=args.eval_iters),
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
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
            # Weights & Biases logging
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,  # optional
            wandb_exp_name=args.wandb_exp_name,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer", vocab_size=distill_provider.vocab_size
        ),
        checkpoint=CheckpointConfig(
            save_interval=args.eval_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,  # Resume from this directory (if exists)
            most_recent_k=5,  # Keeps 5 most recent checkpoints (not metric-based)
            ckpt_format="torch_dist",
            async_save=True,
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=SEED),
        mixed_precision="bf16_mixed",
    )

    print_rank_0("\nStarting distillation...")
    distill(config)
    print_rank_0(
        f"\nDistillation done! Saved checkpoint to {checkpoint_dir}"
        " in megatron distributed checkpoint format.\n"
    )

    if args.hf_export_path:
        print_rank_0(f"Exporting final distilled ckpt to HF format to {args.hf_export_path}")
        # Save rank before destroying process group (dist.rank() won't work after destruction)
        is_rank_0 = dist.rank() == 0

        # Destroy process group on all ranks -- export_ckpt will create its own temporary one.
        # This prevents cleanup from hanging (cleanup tries to barrier, but rank 0 would be gone).
        dist.cleanup()

        if is_rank_0:
            export_bridge = AutoBridge.from_hf_pretrained(
                args.student_hf_model, trust_remote_code=args.trust_remote_code
            )
            # Copy weights and remote code
            export_bridge.export_ckpt(
                megatron_path=f"{checkpoint_dir}/iter_{args.train_iters:07d}",
                hf_path=args.hf_export_path,
                show_progress=True,
                strict=True,
            )
            # Copy config.json from student_hf_path (handles both local paths and HF model IDs)
            AutoConfig.from_pretrained(
                args.student_hf_path, trust_remote_code=args.trust_remote_code
            ).save_pretrained(args.hf_export_path)
            # Repair tokenizer_config.json if the save chain emitted special tokens
            # under the wrong key (transformers v5 quirk that breaks vLLM on reload).
            from modelopt.torch.utils.plugins.vlm import _normalize_tokenizer_special_tokens
            _normalize_tokenizer_special_tokens(args.hf_export_path)

            # If we extracted the student LM from a VLM up front, reinsert the distilled LM
            # back into the original VLM container now. The HF export wrote a plain causal
            # LM at args.hf_export_path; ``_vlm_ctx["final_export_path"]`` is where the
            # caller originally asked the assembled VLM to land.
            _maybe_reinsert_distilled_lm_into_vlm(
                getattr(args, "_vlm_ctx", None), trust_remote_code=args.trust_remote_code
            )


def _maybe_wrap_for_vlm(args: argparse.Namespace) -> dict | None:
    """If student or teacher is a VLM, extract its language tower to a temp HF causal-LM dir.

    Rewrites ``args.student_hf_path`` (and ``args.teacher_hf_path``) so the rest of the
    pipeline operates on plain causal LMs. If ``args.hf_export_path`` is set and the
    student is a VLM, repoints it to a temp dir so the HF export writes a causal LM that
    we can later splice back into the original VLM container. Stashes the context on
    ``args._vlm_ctx`` for the reinsertion step at the end of :func:`main`. Returns the
    same context (or ``None`` if neither input is a VLM) for caller inspection.

    Extraction happens on rank 0 only; other ranks wait on a barrier so the temp dirs are
    visible by the time ``main`` starts loading from them.
    """
    import shutil

    from modelopt.torch.utils.plugins.vlm import (
        extract_text_tower_to_hf_causal_lm,
        is_vlm_checkpoint,
    )

    student_is_vlm = is_vlm_checkpoint(
        args.student_hf_path, trust_remote_code=args.trust_remote_code
    )
    teacher_is_vlm = is_vlm_checkpoint(
        args.teacher_hf_path, trust_remote_code=args.trust_remote_code
    )
    if not student_is_vlm and not teacher_is_vlm:
        args._vlm_ctx = None
        return None

    output_root = args.hf_export_path or args.output_dir
    tmp_student_lm = f"{output_root}_vlm_student_lm"
    tmp_teacher_lm = f"{output_root}_vlm_teacher_lm" if teacher_is_vlm else None
    final_export_path = args.hf_export_path
    tmp_export_lm = f"{output_root}_vlm_distilled_lm" if (student_is_vlm and final_export_path) else None
    original_student_vlm_path = args.student_hf_path if student_is_vlm else None
    student_hf_model = args.student_hf_model

    if dist.rank() == 0:
        if student_is_vlm:
            print_rank_0(
                f"Detected VLM student ({args.student_hf_path}); extracting language tower to "
                f"{tmp_student_lm}."
            )
            shutil.rmtree(tmp_student_lm, ignore_errors=True)
            extract_text_tower_to_hf_causal_lm(
                args.student_hf_path,
                tmp_student_lm,
                trust_remote_code=args.trust_remote_code,
            )
        if teacher_is_vlm:
            print_rank_0(
                f"Detected VLM teacher ({args.teacher_hf_path}); extracting language tower to "
                f"{tmp_teacher_lm}."
            )
            shutil.rmtree(tmp_teacher_lm, ignore_errors=True)
            extract_text_tower_to_hf_causal_lm(
                args.teacher_hf_path,
                tmp_teacher_lm,
                trust_remote_code=args.trust_remote_code,
            )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if student_is_vlm:
        args.student_hf_path = tmp_student_lm
        # ``student_hf_model`` is the architecture template AutoBridge uses for the HF
        # export. After extraction it's a plain causal LM, so repoint the template too.
        if args.student_hf_model is None or args.student_hf_model == original_student_vlm_path:
            args.student_hf_model = tmp_student_lm
        if tmp_export_lm is not None:
            args.hf_export_path = tmp_export_lm
    if teacher_is_vlm:
        args.teacher_hf_path = tmp_teacher_lm

    ctx = {
        "student_is_vlm": student_is_vlm,
        "teacher_is_vlm": teacher_is_vlm,
        "original_student_vlm_path": original_student_vlm_path,
        "final_export_path": final_export_path,
        "tmp_student_lm": tmp_student_lm,
        "tmp_teacher_lm": tmp_teacher_lm,
        "tmp_export_lm": tmp_export_lm,
        "original_student_hf_model": student_hf_model,
    }
    args._vlm_ctx = ctx
    return ctx


def _maybe_reinsert_distilled_lm_into_vlm(
    ctx: dict | None, trust_remote_code: bool
) -> None:
    """Splice the distilled student LM back into the original student VLM container.

    Rank-0-only; the surrounding rank gating lives in :func:`main`. Cleans up the
    extraction / reinsertion temp dirs after the assembled VLM is saved.
    """
    if ctx is None or not ctx.get("student_is_vlm"):
        return
    if not ctx.get("final_export_path") or not ctx.get("tmp_export_lm"):
        return

    import shutil

    from modelopt.torch.utils.plugins.vlm import reinsert_pruned_lm_into_vlm

    print_rank_0(
        f"Reinserting distilled LM from {ctx['tmp_export_lm']} into VLM container; "
        f"writing final VLM to {ctx['final_export_path']}."
    )
    reinsert_pruned_lm_into_vlm(
        pruned_causal_lm_path=ctx["tmp_export_lm"],
        original_vlm_path=ctx["original_student_vlm_path"],
        output_dir=ctx["final_export_path"],
        trust_remote_code=trust_remote_code,
    )
    shutil.rmtree(ctx["tmp_student_lm"], ignore_errors=True)
    if ctx.get("tmp_teacher_lm"):
        shutil.rmtree(ctx["tmp_teacher_lm"], ignore_errors=True)
    shutil.rmtree(ctx["tmp_export_lm"], ignore_errors=True)


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    _maybe_wrap_for_vlm(args)
    try:
        main(args)
    finally:
        dist.cleanup()
