# Adapted from: https://github.com/FasterDecoding/Medusa/blob/e2a5d20/medusa/train/train_legacy.py
# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
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

import json

import torch
import transformers
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import print_rank_0

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def change_format(conversations):
    chat = []
    for conversation in conversations:
        # Detect format: either role/content or from/value
        if "role" in conversation and "content" in conversation:
            role = conversation["role"]
            content = conversation["content"]
        elif "from" in conversation and "value" in conversation:
            role = conversation["from"]
            content = conversation["value"]
        else:
            raise ValueError(f"Unknown conversation format: {conversation}")
        turn = {"role": role.lower(), "content": content.lower()}
        chat.append(turn)
    return chat


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []

    for i, conversation in enumerate(sources):
        chat = change_format(conversation["conversations"])
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(prompt)
        conversations.append(chat)

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(
        zip(conversations, targets, prompts)
    ):
        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"]
                # Unfortunate strip() necessary because chat templates are doing the same.
                start = prompt.index(content.strip())
                stop = start + len(content)
                indices = []
                for tok_index, (tok_start, tok_stop) in enumerate(
                    encoding.offset_mapping[conv_index]
                ):
                    if tok_start >= start and tok_stop <= stop:
                        indices.append(tok_index)
                target[indices] = encoding.input_ids[conv_index][indices]

        # Shift target to the left by 1 token
        targets[conv_index] = torch.cat(
            [target[1:], torch.tensor([IGNORE_TOKEN_ID], dtype=target.dtype)]
        )

    return {
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()

        print_rank_0("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
        }


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        print_rank_0("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = {
            "input_ids": ret["input_ids"][0],
            "labels": ret["labels"][0],
            "attention_mask": ret["attention_mask"][0],
        }
        self.cached_data_dict[i] = ret

        return ret


def make_medusa_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    print_rank_0("Loading data...")

    if data_args.data_path.endswith("jsonl"):
        with open(data_args.data_path) as f:
            data_json = [json.loads(line) for line in f]
    else:
        data_json = json.load(open(data_args.data_path))
    train_dataset = dataset_cls(data_json[: int(len(data_json) * 0.95)], tokenizer=tokenizer)
    eval_dataset = dataset_cls(data_json[int(len(data_json) * 0.95) :], tokenizer=tokenizer)

    return {"train_dataset": train_dataset, "eval_dataset": eval_dataset}
