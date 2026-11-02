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

"""Run Megatron Bridge QAT or QAD with GDN/KDA recurrent-state fake quantization."""

import argparse
import sys
from pathlib import Path

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.checkpointing import has_modelopt_state
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from transformers import AutoTokenizer

from modelopt.recipe import ModelOptPTQRecipe, load_recipe

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "megatron_bridge"))
from linear_attention import linear_attention_training

sys.path.pop(0)


def run_training(
    config,
    quant_config,
    prefill_tokens=None,
    teacher_provider=None,
    *,
    train_all_weights=False,
    teacher_serving_arithmetic=False,
    recompute_gdn=False,
    logit_kl_topk=None,
):
    """Compatibility API; QAT and QAD share the standard Bridge training hooks."""
    with linear_attention_training(
        config,
        quant_config,
        prefill_tokens,
        teacher_provider,
        train_all_weights=train_all_weights,
        teacher_serving_arithmetic=teacher_serving_arithmetic,
        recompute_gdn=recompute_gdn,
    ) as forward_step:
        if teacher_provider is None:
            pretrain(config, forward_step)
        else:
            config.model = convert_to_distillation_provider(
                config.model,
                teacher_provider,
                ModelOptDistillConfig(kd_loss_alpha=1.0, logit_kl_topk=logit_kl_topk),
            )
            distill(config, forward_step)


def model_provider(path, options, *, load_weights=True):
    """Build a Megatron provider with the same topology options as the Bridge example."""
    bridge = AutoBridge.from_hf_pretrained(str(path), trust_remote_code=options.trust_remote_code)
    provider = bridge.to_megatron_provider(load_weights=load_weights)
    provider.tensor_model_parallel_size = options.tp_size
    provider.pipeline_model_parallel_size = options.pp_size
    provider.pipeline_dtype = torch.bfloat16
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = options.ep_size
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = options.tp_size > 1
    provider.gradient_accumulation_fusion = False
    provider.calculate_per_token_loss = True
    provider.seq_length = options.length
    return provider


def main():
    """Train a local model with QAT, or add a frozen teacher for QAD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, help="Unquantized teacher; enables QAD")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument(
        "--sft-data", type=Path, help="Bridge prompt-completion directory with training.jsonl"
    )
    data.add_argument(
        "--train-data",
        type=Path,
        help="Megatron tokenized dataset prefix (.bin/.idx)",
    )
    parser.add_argument(
        "--recipe",
        default="general/ptq/linear_attention_state_int8_block32_dynamic",
        help="Path to a quantization recipe YAML (built-in or custom)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument(
        "--prefill-tokens", type=int, help="Fixed split override; required for token-only data"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--train-all-weights", action="store_true")
    parser.add_argument("--teacher-serving-arithmetic", action="store_true")
    parser.add_argument(
        "--recompute-gdn", action="store_true", help="Recompute GDN activations during backward"
    )
    parser.add_argument(
        "--logit-kl-topk",
        type=int,
        help="Distill the teacher's top-k logits; default: full vocabulary",
    )
    options = parser.parse_args()
    if options.teacher_serving_arithmetic and options.teacher_model is None:
        parser.error("--teacher-serving-arithmetic requires --teacher-model")
    if options.logit_kl_topk is not None and (
        options.teacher_model is None or options.logit_kl_topk < 1
    ):
        parser.error("--logit-kl-topk requires --teacher-model and a positive value")
    if options.train_steps < 1 or options.length < 2:
        parser.error("train-steps must be positive and length must be at least two")
    if options.train_data and options.prefill_tokens is None:
        parser.error(
            "Token-only --train-data needs --prefill-tokens; use --sft-data for prompt boundaries"
        )
    if options.prefill_tokens is not None and not 0 <= options.prefill_tokens < options.length:
        parser.error("prefill-tokens must leave at least one suffix label")
    if options.sft_data and options.pp_size != 1:
        parser.error("--sft-data state QAT currently requires --pp_size 1")
    if options.global_batch_size < 1:
        parser.error("global-batch-size must be positive")
    if min(options.tp_size, options.pp_size, options.ep_size) < 1:
        parser.error("Parallel sizes must be positive")
    if options.train_data and any(
        not Path(f"{options.train_data}.{suffix}").is_file() for suffix in ("bin", "idx")
    ):
        parser.error("train-data must name an existing Megatron .bin/.idx dataset prefix")

    if options.sft_data and not (options.sft_data / "training.jsonl").is_file():
        parser.error("--sft-data requires training.jsonl with input/output fields")

    recipe = load_recipe(options.recipe)
    if not isinstance(recipe, ModelOptPTQRecipe):
        parser.error("--recipe must select a PTQ quantization recipe")

    checkpoint_dir = str(options.output / "checkpoints")
    resume = has_modelopt_state(checkpoint_dir)
    student = model_provider(options.model, options, load_weights=not resume)
    teacher = None
    if options.teacher_model is not None:
        tokenizer_kwargs = {"trust_remote_code": options.trust_remote_code}
        student_vocab = AutoTokenizer.from_pretrained(options.model, **tokenizer_kwargs).get_vocab()
        teacher_vocab = AutoTokenizer.from_pretrained(
            options.teacher_model, **tokenizer_kwargs
        ).get_vocab()
        if student_vocab != teacher_vocab:
            parser.error("QAD requires student and teacher to use the same tokenizer vocabulary")
        teacher = model_provider(options.teacher_model, options)
        vocab_sizes = [
            calculate_padded_vocab_size(
                p.vocab_size, p.make_vocab_size_divisible_by, p.tensor_model_parallel_size
            )
            for p in (student, teacher)
        ]
        if vocab_sizes[0] != vocab_sizes[1]:
            parser.error("QAD requires matching student and teacher output vocabulary dimensions")

    config = ConfigContainer(
        model=student,
        train=TrainingConfig(
            train_iters=options.train_steps,
            global_batch_size=options.global_batch_size,
            micro_batch_size=1,
        ),
        validation=ValidationConfig(eval_iters=0, eval_interval=options.train_steps),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=options.learning_rate,
            min_lr=0,
            weight_decay=0,
            clip_grad=1.0,
            use_distributed_optimizer=True,
        ),
        scheduler=SchedulerConfig(
            lr_decay_style="constant",
            lr_warmup_iters=0,
            start_weight_decay=0,
            end_weight_decay=0,
            use_checkpoint_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            average_in_collective=False, use_distributed_optimizer=True
        ),
        dataset=(
            FinetuningDatasetConfig(
                seq_length=options.length,
                dataset_root=str(options.sft_data),
                seed=options.seed,
                dataloader_type="batch",
                do_validation=False,
                do_test=False,
                dataset_kwargs={
                    "prompt_template": "{input}{output}",
                    "label_key": "output",
                    "answer_only_loss": True,
                    "add_bos": False,
                    "add_eos": True,
                    "truncation_field": "input",
                    "truncation_method": "left",
                },
            )
            if options.sft_data
            else GPTDatasetConfig(
                seq_length=options.length,
                blend=([str(options.train_data)], None),
                split="100,0,0",
                random_seed=options.seed,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                dataloader_type="single",
                num_workers=0,
            )
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(options.model),
            hf_tokenizer_kwargs={"trust_remote_code": options.trust_remote_code},
        ),
        checkpoint=CheckpointConfig(
            save=checkpoint_dir,
            load=checkpoint_dir,
            save_interval=options.train_steps,
            async_save=False,
            ckpt_format="torch_dist",
        ),
        logger=LoggerConfig(log_interval=1),
        rng=RNGConfig(seed=options.seed),
        mixed_precision="bf16_mixed",
    )
    run_training(
        config,
        recipe.quantize,
        options.prefill_tokens,
        teacher,
        train_all_weights=options.train_all_weights,
        teacher_serving_arithmetic=options.teacher_serving_arithmetic,
        recompute_gdn=options.recompute_gdn,
        logit_kl_topk=options.logit_kl_topk,
    )


if __name__ == "__main__":
    main()
