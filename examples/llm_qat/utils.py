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

import gc
import types
from functools import partial

import torch
import transformers
from peft import LoraConfig, TaskType
from transformers import default_data_collator


def make_supervised_data_module(
    data_args,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Make dataset and collator for supervised fine-tuning.

    Loads a dataset blend from a YAML config file, tokenizes with distributed
    rank-sharding, and returns train/eval datasets with a data collator.

    ``data_args.train_samples`` and ``data_args.eval_samples`` are applied as
    runtime caps on the pre-split dataset (0 = use all).
    """
    from dataset_utils import build_blend_dataset, load_blend_config

    config = load_blend_config(data_args.dataset_config)
    max_length = getattr(tokenizer, "model_max_length", 4096)

    ds = build_blend_dataset(
        config,
        tokenizer,
        max_length,
        seed=data_args.dataset_seed,
        cache_dir=data_args.dataset_cache_dir,
        shuffle=data_args.shuffle,
        shuffle_buffer=data_args.shuffle_buffer,
        num_proc=data_args.num_proc,
    )

    train_ds = ds["train"]
    if data_args.train_samples > 0 and data_args.train_samples < len(train_ds):
        train_ds = train_ds.select(range(data_args.train_samples))

    eval_ds = ds["eval"]
    if data_args.eval_samples > 0 and data_args.eval_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(data_args.eval_samples))

    return {
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": default_data_collator,
    }


def get_lora_config():
    return LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )


def monkey_patch_training_step_to_fix_memory_leak(trainer):
    def new_func(original_f_name, trainer, *args, **kwargs):
        gc.collect()
        return getattr(trainer, original_f_name)(*args, **kwargs)

    for f_name in ["training_step", "prediction_step", "_load_best_model"]:
        setattr(trainer, "_original_" + f_name, getattr(trainer, f_name))
        setattr(
            trainer, f_name, types.MethodType(partial(new_func, "_original_" + f_name), trainer)
        )


def get_metrics_with_perplexity(metrics):
    """Add perplexity to the metrics."""
    if "eval_loss" in metrics:
        metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])))
    return metrics
