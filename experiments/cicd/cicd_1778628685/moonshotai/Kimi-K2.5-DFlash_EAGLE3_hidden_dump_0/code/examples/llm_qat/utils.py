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
from contextlib import contextmanager
from functools import partial

import datasets
import torch
import transformers
from peft import LoraConfig, TaskType
from transformers import default_data_collator

IGNORE_INDEX = -100


@contextmanager
def main_process_first():
    """Context manager to run code on the main process first."""
    if not torch.distributed.is_initialized():
        yield
        return

    rank = torch.distributed.get_rank()
    if rank == 0:
        yield
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        yield
    torch.distributed.barrier()


def get_daring_anteater(
    tokenizer: transformers.AutoTokenizer,
    split="train",
    max_length=4096,
    train_size=0,
    eval_size=0,
):
    # sample = {
    #     'system': '{system message}',
    #     'conversations': [
    #         {'from': 'User', 'value': '{turn 1 user message}', 'label': None},
    #         {'from': 'Assistant', 'value': '{turn 1 assistant message}', 'label': '{turn 1 assistant label}'},
    #         {'from': 'User', 'value': '{turn 2 user message}', 'label': None},
    #         {'from': 'Assistant', 'value': '{turn 2 assistant message}', 'label': '{turn 2 assistant label}'},
    #     ],
    #     "mask": "User",
    #     "type": "VALUE_TO_TEXT",
    # }

    def process_and_tokenize(sample):
        conversations = sample["conversations"]
        all_input_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
        all_labels = [IGNORE_INDEX] if tokenizer.bos_token_id else []

        for conversation in conversations:
            role = conversation["from"]
            input_ids = tokenizer.encode(conversation["value"] + "\n", add_special_tokens=False)
            labels = input_ids if role == "Assistant" else [IGNORE_INDEX] * len(input_ids)

            all_input_ids.extend(input_ids)
            all_labels.extend(labels)

            if len(all_input_ids) > max_length:
                break

        all_input_ids.append(tokenizer.eos_token_id)
        all_labels.append(IGNORE_INDEX)
        all_attention_mask = [1] * len(all_input_ids)

        cur_seq_length = len(all_input_ids)
        if cur_seq_length < max_length:
            pad_token = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            all_input_ids += [pad_token] * (max_length - cur_seq_length)
            all_attention_mask += [0] * (max_length - cur_seq_length)
            all_labels += [IGNORE_INDEX] * (max_length - cur_seq_length)

        return {
            "input_ids": all_input_ids[:max_length],
            "attention_mask": all_attention_mask[:max_length],
            "labels": all_labels[:max_length],
        }

    if hasattr(get_daring_anteater, "cached_dataset"):
        dataset = get_daring_anteater.cached_dataset
    else:
        with main_process_first():
            dataset = datasets.load_dataset("nvidia/Daring-Anteater", split="train")
            # Shuffle and subsample the dataset
            eval_size = 2000 if eval_size == 0 else eval_size
            train_size = len(dataset) - eval_size if train_size == 0 else train_size
            assert train_size + eval_size <= len(dataset) and train_size > 0 and eval_size > 0, (
                "not enough data for train-eval split"
            )
            dataset = dataset.shuffle(seed=42).select(range(train_size + eval_size))
            dataset = dataset.map(process_and_tokenize, remove_columns=list(dataset.features))
            dataset = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
        get_daring_anteater.cached_dataset = dataset
    return dataset[split]


def make_supervised_data_module(
    dataset="Daring-Anteater",
    tokenizer: transformers.PreTrainedTokenizer = None,
    train_size: int = 0,
    eval_size: int = 0,
) -> dict:
    """Make dataset and collmtor for supervised fine-tuning."""
    if dataset == "Daring-Anteater":
        train_dataset = get_daring_anteater(
            tokenizer, "train", tokenizer.model_max_length, train_size, eval_size
        )
        val_dataset = get_daring_anteater(
            tokenizer, "test", tokenizer.model_max_length, train_size, eval_size
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
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
