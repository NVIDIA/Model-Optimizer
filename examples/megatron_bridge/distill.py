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

import torch
from export_distilled_megatron_to_hf import export_llm_to_hf, save_vlm_to_hf
from megatron.bridge import AutoBridge
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.utils import unwrap_model
from transformers import AutoConfig, AutoTokenizer

import modelopt.torch.distill as mtd
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import print_args, print_rank_0, warn_rank_0
from modelopt.torch.utils.plugins.mbridge import load_modelopt_megatron_checkpoint

with contextlib.suppress(ModuleNotFoundError):
    import modelopt.torch.puzzletron.plugins.mbridge  # noqa: F401


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Distillation for Megatron-Bridge.")
    # Model arguments (HF teacher, HF or Megatron student)
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
    parser.add_argument(
        "--student_megatron_path",
        type=str,
        default=None,
        help=(
            "Path to a Megatron checkpoint to initialize the student weights from, instead of the HuggingFace "
            "weights of --student_hf_path (which is still required to build the student model structure). "
            "If the checkpoint carries ModelOpt state (i.e. it is quantized), the quantizers are "
            "restored automatically, turning distillation into Quantization Aware Distillation (QAD)."
        ),
    )
    # Parallelism arguments
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")

    # Dataset arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for data shuffling and RNG state",
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        help="List of tokenized data paths to load from (weight1 path1 weight2 path2 ...)",
    )
    parser.add_argument(
        "--data_path_to_cache", type=str, default=None, help="Path to cache the dataset indices"
    )
    parser.add_argument(
        "--use_mock_data", action="store_true", help="Use mock data instead of --data_paths"
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Distill on prompt-completion jsonl from --sft_dataset_root with the loss masked to "
        "the completion, instead of pre-tokenized --data_paths.",
    )
    parser.add_argument(
        "--sft_dataset_root",
        type=str,
        default=None,
        help="Directory holding training.jsonl (and validation.jsonl when --eval_iters > 0) of "
        '{"input": <prompt>, "output": <response>} records (used with --sft). See the README for '
        "how the fields are tokenized and truncated.",
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
        "--recompute_granularity",
        type=str,
        default=None,
        choices=["selective", "full"],
        help="Activation recomputation: omit (off), 'selective' (attn only), 'full' (whole layers)",
    )
    parser.add_argument(
        "--recompute_method",
        type=str,
        default=None,
        choices=["uniform", "block"],
        help="Activation recomputation method (only used when --recompute_granularity=full)",
    )
    parser.add_argument(
        "--recompute_num_layers",
        type=int,
        default=None,
        help="Number of layers per recomputation chunk (only used when --recompute_granularity=full)",
    )
    parser.add_argument(
        "--recompute_modules",
        type=str,
        nargs="+",
        default=None,
        help="Modules to recompute with --recompute_granularity=selective. Defaults to ['core_attn']. "
        "Allowed: core_attn, mlp, moe, moe_act, layernorm, mla_up_proj, shared_experts.",
    )
    parser.add_argument(
        "--eval_interval", type=_positive_int, default=100, help="Validate every <N> steps"
    )
    parser.add_argument(
        "--eval_iters",
        type=_nonnegative_int,
        default=32,
        help="Number of batches per validation stage; set to 0 to disable validation",
    )
    parser.add_argument(
        "--save_interval",
        type=_positive_int,
        default=None,
        help="Checkpoint every <N> steps; defaults to --eval_interval",
    )
    parser.add_argument(
        "--exit_interval",
        type=_positive_int,
        default=None,
        help="Save a checkpoint and exit when the iteration is divisible by this value",
    )
    parser.add_argument(
        "--exit_duration_in_mins",
        type=_positive_int,
        default=None,
        help="Save a checkpoint and exit after this many minutes",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Skip training and run validation at iteration 0.",
    )
    parser.add_argument(
        "--checkpoint_keep_last",
        type=int,
        default=5,
        help=(
            "Keep only the most recent <N> Megatron checkpoints. Set to -1 to disable "
            "checkpoint rotation and keep all validation checkpoints, for example for Hugging Face "
            "export and downstream evaluation."
        ),
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
        help="Reference HF model with a homogeneous architecture, used as the export template for a "
        "heterogeneous (Puzzletron/NAS) student's weights. Defaults to --student_hf_path, which is "
        "correct for homogeneous students; unused for VLMs.",
    )
    args = parser.parse_args()

    # Sanity checks
    if not args.sft and not args.use_mock_data and not args.data_paths:
        raise ValueError("Must provide either --data_paths or set --use_mock_data.")

    if args.student_hf_model is None:
        args.student_hf_model = args.student_hf_path
    if args.checkpoint_keep_last < -1:
        raise ValueError("--checkpoint_keep_last must be >= -1.")
    if args.validate_only and args.eval_iters == 0:
        raise ValueError("--validate_only requires --eval_iters > 0.")

    if args.sft and not args.sft_dataset_root:
        raise ValueError(
            "--sft requires --sft_dataset_root (a directory with training.jsonl, plus "
            "validation.jsonl when --eval_iters > 0)."
        )
    if args.sft and (args.data_paths or args.use_mock_data):
        raise ValueError(
            "--sft is mutually exclusive with --data_paths / --use_mock_data: the SFT branch wins "
            "the dataset selection, so those inputs would be silently ignored."
        )
    if args.sft_dataset_root and not args.sft:
        raise ValueError("--sft_dataset_root requires --sft; without it the SFT path is not used.")
    if args.sft:
        # Fail on a mistyped root here rather than after both checkpoints have loaded onto GPUs.
        required = ["training.jsonl"] + (["validation.jsonl"] if args.eval_iters > 0 else [])
        absent = [f for f in required if not os.path.isfile(os.path.join(args.sft_dataset_root, f))]
        if absent:
            raise ValueError(f"--sft_dataset_root {args.sft_dataset_root} is missing: {absent}.")
        # Decided once here so it reaches print_args and costs a single tokenizer load.
        args.sft_add_bos = _tokenizer_prepends_bos(args)

    _check_shared_vocabulary(args)

    print_args(args)

    return args


def _check_shared_vocabulary(args) -> None:
    """Raise unless teacher and student use the same tokenizer."""
    _tok = {"trust_remote_code": args.trust_remote_code}
    student_vocab = AutoTokenizer.from_pretrained(args.student_hf_path, **_tok).get_vocab()
    teacher_vocab = AutoTokenizer.from_pretrained(args.teacher_hf_path, **_tok).get_vocab()
    if student_vocab != teacher_vocab:
        raise ValueError(
            "Distillation scores the teacher on the student's token ids, so teacher and student "
            "must use the same tokenizer."
        )


def _tokenizer_prepends_bos(args) -> bool:
    """True when the student tokenizer prepends a BOS at inference.

    Probes an encode: fast tokenizers prepend via a post-processor that exposes no attribute.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_hf_path, trust_remote_code=args.trust_remote_code
    )
    if not getattr(tokenizer, "bos_token", None):
        return False
    return tokenizer("x").input_ids[:1] == [tokenizer.bos_token_id]


def main(args: argparse.Namespace):
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    tensorboard_dir = os.path.join(args.output_dir, "tb_logs")

    # Build student and teacher model providers
    def _build_model_provider(hf_path, load_weights=True):
        bridge = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=args.trust_remote_code)
        provider = bridge.to_megatron_provider(load_weights=load_weights)

        # Override parallelism / training settings
        provider.tensor_model_parallel_size = args.tp_size
        provider.sequence_parallel = args.tp_size > 1
        provider.pipeline_model_parallel_size = args.pp_size
        provider.pipeline_dtype = torch.bfloat16
        provider.context_parallel_size = args.cp_size
        provider.expert_model_parallel_size = args.ep_size
        provider.expert_tensor_parallel_size = 1  # Expert tensor parallelism is not supported
        provider.seq_length = args.seq_length
        if args.sft:
            # A response-only loss mask needs per-token reduction to combine across CP ranks.
            # Must stay in sync with ``average_in_collective=not args.sft`` on the DDP config.
            provider.calculate_per_token_loss = True
        if args.recompute_granularity is not None:
            provider.recompute_granularity = args.recompute_granularity
            provider.recompute_method = args.recompute_method
            provider.recompute_num_layers = args.recompute_num_layers
            if args.recompute_modules is not None:
                provider.recompute_modules = args.recompute_modules
        return provider

    # The student structure is always built from --student_hf_path. When --student_megatron_path is
    # given, the HF weights are skipped (they are overwritten by the Megatron checkpoint, loaded into
    # the built student inside the patched provide() below).
    student_has_modelopt_state = args.student_megatron_path is not None and has_modelopt_state(
        args.student_megatron_path
    )
    student_provider = _build_model_provider(
        args.student_hf_path, load_weights=args.student_megatron_path is None
    )
    if student_has_modelopt_state:
        # Gradient accumulation fusion is not supported with ModelOpt quantized models. Disable it
        # before the model is built so the student's linear layers are constructed accordingly.
        student_provider.gradient_accumulation_fusion = False
    teacher_provider = _build_model_provider(args.teacher_hf_path)

    # The KD losses compare logits elementwise over the vocab dim, so both output layers must have
    # the same padded width. A shared tokenizer does not imply it: the HF configs can disagree.
    padded = {
        name: calculate_padded_vocab_size(
            p.vocab_size, p.make_vocab_size_divisible_by, p.tensor_model_parallel_size
        )
        for name, p in (("student", student_provider), ("teacher", teacher_provider))
    }
    if padded["student"] != padded["teacher"]:
        raise ValueError(
            "Distillation needs student and teacher logits of equal width, but their padded vocab "
            f"sizes differ ({padded['student']} vs {padded['teacher']})."
        )

    kd_config = ModelOptDistillConfig(
        skip_lm_loss=not args.no_skip_lm_loss, kd_loss_scale=args.kd_loss_scale
    )

    # HF VLM configs expose ``vision_config``; Megatron-Bridge nests the text model under
    # ``language_model`` (used as ``distill_submodule`` below).
    is_vlm = hasattr(
        AutoConfig.from_pretrained(args.student_hf_path, trust_remote_code=args.trust_remote_code),
        "vision_config",
    )

    if is_vlm:
        warn_rank_0(
            "VLM detected: distilling model.language_model only (vision tower / projector untouched). "
            "To export megatron non-quantized checkpoint, use export_distilled_megatron_to_hf.py"
        )
    distill_provider = convert_to_distillation_provider(
        student_provider,
        teacher_provider,
        kd_config,
        distill_submodule="language_model" if is_vlm else None,
    )

    if args.student_megatron_path:
        # QAD: restore the quantized student weights + ModelOpt state before the KD conversion (a no-op
        # once converted). Prepend so this runs before the provider's KD-conversion pre-wrap hook.
        if student_has_modelopt_state:
            print_rank_0(
                f"Detected ModelOpt state in {args.student_megatron_path}; "
                "restoring quantizers for Quantization Aware Distillation (QAD)."
            )

        def _restore_student_hook(model_chunks):
            print_rank_0(
                f"Loading student weights from Megatron checkpoint {args.student_megatron_path}"
            )
            load_modelopt_megatron_checkpoint(
                [unwrap_model(model_chunks[0])], args.student_megatron_path
            )
            return model_chunks

        distill_provider.register_pre_wrap_hook(_restore_student_hook, prepend=True)

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
        "random_seed": args.seed,
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "num_dataset_builder_threads": 1,
        "data_sharding": True,
        "dataloader_type": "single",
        "skip_getting_attention_mask_from_dataset": True,
    }
    if args.sft:
        # SFT-masked distillation via Bridge's FinetuningDatasetConfig -> NeMo-style GPTSFTDataset,
        # reading {"input", "output"} jsonl. Fields are tokenized as written except that each is
        # ``.strip(" ")``-ed; see --sft_dataset_root help.
        dataset_config = FinetuningDatasetConfig(
            seq_length=args.seq_length,
            dataset_root=args.sft_dataset_root,
            seed=args.seed,
            dataloader_type="batch",
            # Honour --eval_iters 0 so a training-only dataset_root does not have to carry a
            # dummy validation.jsonl just to satisfy the builder.
            do_validation=args.eval_iters > 0,
            do_test=False,
            dataset_kwargs={
                "prompt_template": "{input}{output}",
                "label_key": "output",
                "truncation_field": "input",
                # Drop the oldest context. The default "right" would cut the prompt/answer
                # boundary and then "output" itself, the only span the loss is computed on.
                "truncation_method": "left",
                "answer_only_loss": True,
                # Prepended after truncation, so it survives a record that had to be cut.
                "add_bos": args.sft_add_bos,
                "add_eos": True,
            },
        )
    elif args.use_mock_data:
        dataset_config = MockGPTDatasetConfig(**dataset_kwargs)
    else:
        # Convert flat CLI list (e.g. ["1.0", "/path/data"]) to Megatron blend format
        blend = get_blend_from_list(args.data_paths)
        dataset_config = GPTDatasetConfig(blend=blend, split="99,1,0", **dataset_kwargs)

    # Assemble ConfigContainer and run distillation
    config = ConfigContainer(
        model=distill_provider,
        train=TrainingConfig(
            train_iters=args.train_iters,
            global_batch_size=args.gbs,
            micro_batch_size=args.mbs,
            exit_interval=args.exit_interval,
            exit_duration_in_mins=args.exit_duration_in_mins,
            manual_gc=True,
            manual_gc_interval=100,
        ),
        validation=ValidationConfig(
            eval_iters=args.eval_iters,
            eval_interval=args.eval_interval,
            skip_train=args.validate_only,
        ),
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=not args.sft,  # per-token loss must not be pre-averaged
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
        tokenizer=(
            # SFT reads raw text, so it needs the model's real tokenizer; the pretraining path
            # consumes pre-tokenized data and keeps NullTokenizer.
            TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=args.student_hf_path,
                hf_tokenizer_kwargs={
                    "trust_remote_code": args.trust_remote_code,
                    # Default True would make text_to_ids inject a BOS at the answer boundary,
                    # since "{input}" and "{output}" are tokenized separately. Consumed by Bridge
                    # in training/tokenizers/config.py.
                    "include_special_tokens": False,
                },
            )
            if args.sft
            else TokenizerConfig(
                tokenizer_type="NullTokenizer", vocab_size=distill_provider.vocab_size
            )
        ),
        checkpoint=CheckpointConfig(
            save_interval=(
                args.save_interval if args.save_interval is not None else args.eval_interval
            ),
            save=checkpoint_dir,
            load=checkpoint_dir,  # Resume from this directory (if exists)
            most_recent_k=args.checkpoint_keep_last,  # Keeps most recent checkpoints (-1 keeps all)
            ckpt_format="torch_dist",
            async_save=True,
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=args.seed),
        mixed_precision="bf16_mixed",
    )

    print_rank_0("\nStarting distillation...")
    distill(config)
    if args.validate_only:
        print_rank_0("\nValidation-only run done! Skipped training and checkpoint export.\n")
        return

    print_rank_0(
        f"\nDistillation done! Saved checkpoint to {checkpoint_dir}"
        " in megatron distributed checkpoint format.\n"
    )

    if args.hf_export_path and is_vlm:
        # Only the language model was distilled; export it back into the full VLM.
        print_rank_0(f"Exporting distilled VLM to HF format to {args.hf_export_path}")
        # ``distill`` tore down the model-parallel groups on exit, so rebuild them.
        distill_provider.initialize_model_parallel(seed=args.seed)
        full_student = distill_provider.full_model
        # Strip the distillation wrapper -> plain trained language model (in place; reassign to be safe).
        full_student.language_model = mtd.export(full_student.language_model)
        save_vlm_to_hf(
            full_student,
            args.hf_export_path,
            args.student_hf_path,
            trust_remote_code=args.trust_remote_code,
        )
        print_rank_0(f"Saved distilled VLM to {args.hf_export_path} in HF format")
    elif args.hf_export_path:
        print_rank_0(f"Exporting final distilled ckpt to HF format to {args.hf_export_path}")
        # Save rank before destroying process group (dist.rank() won't work after destruction)
        is_rank_0 = dist.rank() == 0

        # Destroy process group on all ranks -- export_ckpt will create its own temporary one.
        # This prevents cleanup from hanging (cleanup tries to barrier, but rank 0 would be gone).
        dist.cleanup()

        if is_rank_0:
            export_llm_to_hf(
                megatron_path=f"{checkpoint_dir}/iter_{args.train_iters:07d}",
                hf_export_path=args.hf_export_path,
                student_hf_path=args.student_hf_path,
                template_hf=args.student_hf_model,
                trust_remote_code=args.trust_remote_code,
            )


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        main(args)
    except BaseException:
        dist.abort()  # peers may be stuck in a collective this rank will never reach
    finally:
        dist.cleanup()
