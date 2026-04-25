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

"""QAT/QAD training script for pre-quantized LLMs.

The model should be pre-quantized using quantize.py before running this script.

Usage:
    accelerate launch --config-file configs/accelerate/fsdp2.yaml train.py \
        --config configs/train/qat_nvfp4.yaml
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import transformers
from arguments import DataArguments, ModelArguments, QuantizeArguments, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from utils import get_lora_config, get_metrics_with_perplexity, make_supervised_data_module

import modelopt.torch.opt as mto
from modelopt.torch.distill.plugins.huggingface import DistillArguments
from modelopt.torch.opt.plugins.transformers import ModelOptArgParser
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer, QATTrainer
from modelopt.torch.utils import print_rank_0, warn_rank_0

# Enable automatic save/load of modelopt state huggingface checkpointing
mto.enable_huggingface_checkpointing()


def train():
    parser = ModelOptArgParser(
        (ModelArguments, TrainingArguments, DataArguments, DistillArguments, QuantizeArguments)
    )
    model_args, training_args, data_args, distill_args, quant_args = (
        parser.parse_args_into_dataclasses()
    )

    if quant_args.recipe:
        warn_rank_0(
            "--recipe is ignored by train.py; use quantize.py for the quantization step before QAT/QAD training"
        )

    print_rank_0(f"arguments: {model_args}, {training_args}, {data_args}, {distill_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print_rank_0(f"Last checkpoint detected: {last_checkpoint}")

    model_kwargs = {}
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        **model_kwargs,
    )
    model.generation_config.do_sample = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=model_args.model_max_length
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # We set model.config.use_cache to False for training when gradient_checkpointing=False.
    # Currently useful for FSDP2 to allow for setting activation_checkpointing=True in the config file.
    model.config.use_cache = False

    print_rank_0("Loading dataset...")
    data_module = make_supervised_data_module(data_args, tokenizer)

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None and training_args.lora:
        raise RuntimeError("Does not support LoRA resuming training yet!")

    trainer_cls = QADTrainer if distill_args.distill else QATTrainer

    if training_args.lora:
        training_args.lora_config = get_lora_config()

    distill_config = None
    if distill_args.distill:
        assert distill_args.teacher_model is not None, "Teacher model is required for distillation."
        teacher = transformers.AutoModelForCausalLM.from_pretrained(
            distill_args.teacher_model,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        distill_config = {
            "teacher_model": teacher,
            "temperature": distill_args.temperature,
            "criterion": distill_args.criterion,
            "liger_jsd_beta": distill_args.liger_jsd_beta,
        }

    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **({"distill_args": distill_config} if distill_config is not None else {}),  # type: ignore[arg-type]
        **data_module,
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        print_rank_0("Training completed.")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics = get_metrics_with_perplexity(metrics)
        print_rank_0(f"Evaluation results: \n{metrics}")

    if training_args.do_train:
        print_rank_0("Saving the model...")
        trainer.save_state()
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
