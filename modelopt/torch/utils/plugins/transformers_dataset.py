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

"""Processing large data to tokenize for pretraining."""

import copy
import itertools

import torch
import transformers
from datasets import load_dataset
from transformers.trainer_pt_utils import LabelSmoother

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def _sharegpt_to_openai_messages(conversations: list[dict]):
    role_mapping = {
        "user": "user",
        "User": "user",
        "human": "user",
        "assistant": "assistant",
        "Assistant": "assistant",
        "gpt": "assistant",
        "system": "system",
        "System": "system",
    }
    messages = []
    for msg in conversations:
        role = role_mapping[msg["from"]]
        content = msg["value"]
        messages.append({"role": role, "content": content})
    return messages


class ShardedDataset(torch.utils.data.Dataset):
    """ShardedDataset is a subclass of torch.utils.data.Dataset that is used to load data from a dataset."""

    def __init__(
        self,
        name: str,
        subset: str | None = None,
        split: str = "train",
        num_shards: int = 1,
        shard_index: int = 0,
        num_streaming_samples: int | None = None,
    ):
        """Initialize the ShardedDataset."""
        self.name = name
        self.subset = subset
        self.split = split
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.num_streaming_samples = num_streaming_samples

        self._load_dataset()

    def __len__(self):
        if self.num_streaming_samples is not None:
            return self.num_streaming_samples
        else:
            return len(self._raw_samples)

    def __getitem__(self, index):
        index = index // self.num_shards

        if self.num_streaming_samples is not None:
            while index >= len(self._raw_samples):
                self._raw_samples.append(next(self._stream_iterator))

        return self._raw_samples[index]

    def _load_dataset(self):
        dataset = load_dataset(
            self.name,
            self.subset,
            split=self.split,
            # num_proc=4,  # TODO: Make this configurable
            streaming=self.num_streaming_samples is not None,
        )

        shard = dataset.shard(num_shards=self.num_shards, index=self.shard_index)

        if self.num_streaming_samples is not None:
            self._raw_samples = []
            self._stream_samples = shard
            self._stream_iterator = itertools.cycle(self._stream_samples)
        else:
            self._raw_samples = shard


class LanguageDataCollator:
    """LanguageDataCollator is a class that is used to collate language data."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: int = 4096,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        answer_only_loss: bool = False,
        json_key: str = "text",
    ):
        """Initialize the LanguageDataset."""
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError(
                "The tokenizer must be a transformers.PreTrainedTokenizerBase but got {}".format(
                    type(tokenizer)
                )
            )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt
        self.answer_only_loss = answer_only_loss
        self.json_key = json_key

        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        else:
            self._post_process_chat_template()

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

    def _post_process_tokenizer(self):
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token == "<|eot_id|>":  # nosec
                self.tokenizer.pad_token = "<|end_of_text|>"  # nosec
            else:
                raise ValueError("The tokenizer has no pad_token!")

    def _post_process_chat_template(self):
        # [WAR]: For DeepSeek-V3/R1 tokenizer, we modify the chat_template such that the <think>
        # tokens are preserved for supervised learning.
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

    def _process_chat_sample(self, examples: list):
        tokenized_examples = self.tokenizer.apply_chat_template(
            examples,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=self.add_generation_prompt,
            return_assistant_tokens_mask=self.answer_only_loss,
        )
        return tokenized_examples

    def _process_text_sample(self, examples: list):
        tokenized_examples = self.tokenizer(
            examples,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return tokenized_examples

    def __call__(self, examples):
        """Call the LanguageDataCollator."""
        batch = []

        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("The sample must be a Dict but got {}".format(type(example)))
            text = example.get(self.json_key, None)
            if isinstance(text, str):
                batch.append(text)
            else:
                messages = example.get("messages", None)
                if messages is None:
                    conversations = example.get("conversations", None)
                    if conversations is None:
                        raise ValueError(
                            "The sample must in either OpenAI messages format or ShareGPT conversations format."
                        )
                    else:
                        messages = _sharegpt_to_openai_messages(conversations)
                batch.append(messages)

        return self._process_chat_sample(batch)


class LanguageDataset(ShardedDataset):
    """LanguageDataset is a subclass of ShardedDataset that is used to load language data."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        name: str,
        subset: str | None = None,
        split: str = "train",
        num_shards: int = 1,
        shard_index: int = 0,
        max_length: int = 4096,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        answer_only_loss: bool = False,
        json_key: str = "text",
    ):
        """Initialize the LanguageDataset."""
        super().__init__(
            name=name,
            subset=subset,
            split=split,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        self.collator = LanguageDataCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            answer_only_loss=answer_only_loss,
            json_key=json_key,
        )

    def __getitem__(self, index):
        """Get the item at the given index."""
        index = index // self.num_shards

        if self.num_streaming_samples is not None:
            while index >= len(self._raw_samples):
                self._raw_samples.append(next(self._stream_iterator))

        return self.collator([self._raw_samples[index]])


class VisionLanguageDataCollator(LanguageDataCollator):
    """VisionLanguageDataCollator is a subclass of LanguageDataCollator that is used to collate vision-language data."""

    def __init__(
        self,
        processor: transformers.ProcessorMixin,
        max_length: int = 8192,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        answer_only_loss: bool = False,
        local_image_path: str | None = None,
    ):
        """Initialize the VisionLanguageDataset."""
        if not isinstance(processor, transformers.ProcessorMixin):
            raise ValueError(
                "The processor must be a transformers.ProcessorMixin but got {}".format(
                    type(processor)
                )
            )

        self.processor = processor
        self.max_length = max_length
        self.chat_template = chat_template
        self.add_generation_prompt = add_generation_prompt
        self.answer_only_loss = answer_only_loss
        self.local_image_path = local_image_path

        super().__init__(
            tokenizer=self.processor.tokenizer,
            max_length=max_length,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            answer_only_loss=answer_only_loss,
        )

    def _process_multimodal_sample(self, examples):
        tokenized_messages = self.processor.apply_chat_template(
            examples,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=self.add_generation_prompt,
            return_assistant_tokens_mask=self.answer_only_loss,
        )
        return tokenized_messages

    def __call__(self, examples):
        """Call the VisionLanguageDataCollator."""
        batch = []

        for example in examples:
            messages = example.get("messages", None)
            if messages is None:
                # print(example)
                conversations = example.get("conversations", None)
                if conversations is None:
                    raise ValueError(
                        "The sample must in either OpenAI messages format or ShareGPT conversations format."
                    )
                else:
                    messages = _sharegpt_to_openai_messages(conversations)

            copy_messages = copy.deepcopy(messages)

            for msg in copy_messages:
                if isinstance(msg["content"], str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]
                for ctn in msg["content"]:
                    if ctn["type"] == "image" and "path" in ctn:
                        ctn["path"] = self.local_image_path + "/" + ctn["path"]

            batch.append(copy_messages)

        return self._process_multimodal_sample(batch)
