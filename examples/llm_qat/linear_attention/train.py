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
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
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
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.post_training.checkpointing import (
    has_modelopt_state,
    load_modelopt_state,
)
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.ssm import gated_delta_net
from megatron.core.utils import unwrap_model
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase
from modelopt.torch.quantization.utils import is_quantized


def run_training(config, quant_config, prefill_tokens, teacher_provider=None):
    """Delegate optimization and checkpointing to Bridge, with a fixed phase per sequence."""
    if not 0 <= prefill_tokens < config.dataset.seq_length:
        raise ValueError("prefill-tokens must leave at least one suffix label")
    resume = config.checkpoint.load and has_modelopt_state(config.checkpoint.load)
    with ExitStack() as phases:

        def prepare_student(models):
            # Restore the saved policy before quantization and before the QAD conversion hook.
            if resume:
                load_modelopt_state(models, config.checkpoint.load)
            layer_types = (gated_delta_net.GatedDeltaNet,)
            if hasattr(gated_delta_net, "KimiDeltaAttention"):
                layer_types += (gated_delta_net.KimiDeltaAttention,)
            for student in unwrap_model(models):
                layers = [m for m in student.modules() if isinstance(m, layer_types)]
                if not layers:
                    raise ValueError(
                        "Each local model chunk must contain Megatron GatedDeltaNet or "
                        "KimiDeltaAttention; choose a pipeline layout with linear attention on every stage"
                    )
                student.requires_grad_(False)
                for layer in layers:
                    layer.requires_grad_(True)
                if config.model.recompute_granularity == "full":

                    def require_input_grad(module, args, output):
                        return output.requires_grad_(True) if module.training else output

                    # Reentrant checkpointing needs a differentiable input with frozen embeddings.
                    for module in student.modules():
                        if isinstance(module, LanguageModelEmbedding):
                            handle = module.register_forward_hook(require_input_grad)
                            phases.callback(handle.remove)
                if not is_quantized(student):
                    mtq.quantize(student, quant_config)
                phases.enter_context(
                    linear_attention_training_phase(
                        student, [prefill_tokens] * config.train.micro_batch_size
                    )
                )
            return models

        def masked_batch(data_iterator):
            batch = dict(next(data_iterator))
            batch["loss_mask"] = batch["loss_mask"].clone()
            batch["loss_mask"][..., :prefill_tokens] = 0
            yield batch

        def forward_step(state: GlobalState, data_iterator, model, return_schedule_plan=False):
            # Consume lazily: middle pipeline stages may not need any batch tensors.
            return forward_step_modelopt(
                state, masked_batch(data_iterator), model, return_schedule_plan
            )

        config.model.register_pre_wrap_hook(prepare_student)
        if teacher_provider is not None:
            config.model = convert_to_distillation_provider(
                config.model, teacher_provider, ModelOptDistillConfig(skip_lm_loss=True)
            )
            distill(config, forward_step)
        else:
            pretrain(config, forward_step)


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
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Megatron tokenized dataset prefix (.bin/.idx)",
    )
    parser.add_argument(
        "--quant-config",
        type=Path,
        default=Path(__file__).with_name("configs") / "decode_state_int8.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    options = parser.parse_args()
    if options.train_steps < 1 or options.length < 2:
        parser.error("train-steps must be positive and length must be at least two")
    if not 0 <= options.prefill_tokens < options.length:
        parser.error("prefill-tokens must leave at least one suffix label")
    if options.global_batch_size < 1:
        parser.error("global-batch-size must be positive")
    if min(options.tp_size, options.pp_size, options.ep_size) < 1:
        parser.error("Parallel sizes must be positive")
    if any(not Path(f"{options.train_data}.{suffix}").is_file() for suffix in ("bin", "idx")):
        parser.error("train-data must name an existing Megatron .bin/.idx dataset prefix")

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
        dataset=GPTDatasetConfig(
            seq_length=options.length,
            blend=([str(options.train_data)], None),
            split="100,0,0",
            random_seed=options.seed,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            dataloader_type="single",
            num_workers=0,
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
        config, json.loads(options.quant_config.read_text()), options.prefill_tokens, teacher
    )


if __name__ == "__main__":
    main()
