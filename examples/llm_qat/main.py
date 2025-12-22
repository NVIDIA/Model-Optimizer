# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/3783d18/train.py

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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

import os
from dataclasses import dataclass, field
from warnings import warn

import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    get_lora_config,
    get_metrics_with_perplexity,
    make_supervised_data_module,
    monkey_patch_training_step_to_fix_memory_leak,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.huggingface import LMLogitsLoss
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer, QATTrainer
from modelopt.torch.utils import print_rank_0

# Enable automatic save/load of modelopt state huggingface checkpointing
mto.enable_huggingface_checkpointing()

CUSTOM_QUANT_CFG = {
    "INT4_WEIGHT_INT8_ACTIVATIONS": {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            "*input_quantizer": {"num_bits": 8, "axis": None, "enable": True},
            "*lm_head*": {"enable": False},
            "default": {"enable": False},
        },
        "algorithm": "max",
    }
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-hf")
    teacher_model: str | None = field(
        default=None,
        metadata={"help": ("The name or path of the teacher model to use for distillation.")},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            )
        },
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)
    lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add LoRA (Low-Rank Adaptation) adapter before training. When using real quantization, "
                "the LoRA adapter must be set, as quantized weights will be frozen during training."
            )
        },
    )
    distill: bool = field(
        default=False,
        metadata={"help": "Select if training with distillation."},
    )


@dataclass
class DataArguments:
    dataset: str = field(
        default="Daring-Anteater",
        metadata={"help": "Specify the dataset.", "choices": ["Daring-Anteater"]},
    )
    train_size: int = field(
        default=0,
        metadata={"help": "Number of training samples to use. If `0`, use default training size."},
    )
    eval_size: int = field(
        default=0,
        metadata={
            "help": "Number of evaluation samples to use. If `0`, use default evaluation size."
        },
    )


@dataclass
class QuantizationArguments:
    quant_cfg: str | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format"
            ),
            "choices": mtq.config.choices | CUSTOM_QUANT_CFG.keys(),
        },
    )
    calib_size: int = field(
        default=512,
        metadata={
            "help": (
                "Specify the calibration size for quantization. The calibration dataset is used to"
                " setup the quantization scale parameters for PTQ/QAT."
            )
        },
    )
    compress: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compress the model weights after quantization. "
                "This is useful for reducing the model size."
            )
        },
    )


def _teacher_factory(model_name_or_path, cache_dir=None):
    """Function to create a teacher model."""
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, DataArguments, QuantizationArguments)
    )
    model_args, training_args, data_args, quant_args = parser.parse_args_into_dataclasses()
    print_rank_0(f"arguments: {model_args}, {training_args}, {data_args}, {quant_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.do_sample = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=training_args.model_max_length
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # We set model.config.use_cache to False for training when gradient_checkpointing=False.
    # Currently useful for FSDP2 to allow for setting activation_checkpointing=True in the config file.åå
    model.config.use_cache = False

    print_rank_0("Loading dataset...")
    data_module = make_supervised_data_module(
        dataset=data_args.dataset,
        tokenizer=tokenizer,
        train_size=data_args.train_size,
        eval_size=data_args.eval_size,
    )

    # Ensure calibration size doesn't exceed evaluation dataset size
    eval_dataset_size = len(data_module["eval_dataset"])
    if quant_args.calib_size > eval_dataset_size:
        warn(
            f"{quant_args.calib_size=} is larger than {eval_dataset_size=}. Setting calib_size to {eval_dataset_size}."
        )
        quant_args.calib_size = eval_dataset_size

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None and training_args.lora:
        raise RuntimeError("Does not support LoRA resuming training yet!")

    # Torch >= 2.4 throws an error if `use_reentrant` is not set explicitly
    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if quant_args.quant_cfg is not None:
        quant_args.quant_cfg = (
            CUSTOM_QUANT_CFG[quant_args.quant_cfg]
            if quant_args.quant_cfg in CUSTOM_QUANT_CFG
            else getattr(mtq, quant_args.quant_cfg)
        )
    distill_kwargs = {}
    if training_args.distill:
        assert model_args.teacher_model is not None, "Teacher model is required for distillation."
        distill_config = {
            "teacher_model": (
                _teacher_factory,
                (
                    model_args.teacher_model,
                    training_args.cache_dir,
                ),
                {},
            ),
            "criterion": LMLogitsLoss(),
            "expose_minimal_state_dict": False,  # FSDP forces us to disable this
        }
        distill_kwargs["distill_config"] = distill_config
    trainer_cls = QADTrainer if training_args.distill else QATTrainer

    if training_args.lora:
        training_args.lora_config = get_lora_config()

    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        quant_args=quant_args,
        **distill_kwargs,
        **data_module,
    )

    # There could be GPU memory leak during QAT causing OOM. This is a workaround to fix it.
    monkey_patch_training_step_to_fix_memory_leak(trainer)

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        print_rank_0("Training completed.")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics = get_metrics_with_perplexity(metrics)
        print_rank_0(f"Evaluation results: \n{metrics}")

    if training_args.do_train or quant_args.quant_cfg is not None:
        print_rank_0("Saving the model...")
        trainer.save_state()
        kwargs = {"export_student": True} if training_args.distill else {}
        trainer.save_model(training_args.output_dir, **kwargs)


if __name__ == "__main__":
    train()
