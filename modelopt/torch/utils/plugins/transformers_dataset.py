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
import os

import torch
import transformers
from datasets import load_dataset
from transformers.trainer_pt_utils import LabelSmoother

from modelopt.torch.utils import print_rank_0

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def _sharegpt_to_openai_messages(conversations: list[dict]):
    """Optionally align sharedgpt format to openai format."""
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
        role = role_mapping[msg["role"]]
        content = msg["content"]
        messages.append({"role": role, "content": content})
    return messages


class ShardedDataset(torch.utils.data.Dataset):
    """Subclass of torch.utils.data.Dataset to load data from HuggingFace dataset."""

    def __init__(
        self,
        name: str,
        subset: str | None = None,
        data_files: str | None = None,
        split: str = "train",
        num_shards: int = 1,
        shard_index: int = 0,
        num_streaming_samples: int | None = None,
    ):
        """Initialize the ShardedDataset."""
        self.name = name
        self.subset = subset
        self.split = split
        self.data_files = data_files
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
            data_files=self.data_files,
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
    """Data collator for language modeling tasks.

    Accepts samples in OpenAI or ShareGPT formats and returns
    tokenized outputs with padding and truncation, including
    input_ids and attention_mask.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        train_len: int = 4096,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        answer_only_loss: bool = False,
        json_key: str = "text",
        return_labels: bool = False,
    ):
        """Initialize the LanguageDataset."""
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError(
                "The tokenizer must be a transformers.PreTrainedTokenizerBase but got {}".format(
                    type(tokenizer)
                )
            )
        self.tokenizer = tokenizer
        self.train_len = train_len
        self.add_generation_prompt = add_generation_prompt
        self.answer_only_loss = answer_only_loss
        self.json_key = json_key
        self.return_labels = return_labels

        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        else:
            self._post_process_chat_template()

        self._post_process_tokenizer()
        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

        if self.answer_only_loss:
            self._ensure_generation_tags()

    def _post_process_tokenizer(self):
        if self.tokenizer.pad_token_id is None:
            print_rank_0("The tokenizer has no pad_token_id, using eos_token_id instead.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

    # Simplified chat templates with {% generation %} tags for answer_only_loss.
    #
    # PURPOSE:
    #   HuggingFace's return_assistant_tokens_mask requires {% generation %} /
    #   {% endgeneration %} tags in the Jinja chat template to identify which tokens
    #   belong to assistant responses. Many models (Qwen3, Llama3) ship without these
    #   tags. These simplified templates add them so that answer_only_loss works
    #   reliably without regex fallbacks.
    #
    # HOW IT WORKS:
    #   When answer_only_loss=True, _ensure_generation_tags() detects the model's
    #   template style (ChatML, Llama3) and replaces the tokenizer's chat_template
    #   with one of these simplified versions. The {% generation %} tags tell HF
    #   exactly which tokens are assistant content for loss masking.
    #
    # WHAT IS PRESERVED:
    #   - System / user / assistant role formatting (exact token match)
    #   - Multi-turn conversation structure
    #   - <think> block injection on last assistant turn (Qwen3-style, chatml_think)
    #   - Content is output as-is — training data with <think> blocks is handled correctly
    #
    # WHAT IS DROPPED (vs original model templates):
    #   - Tool call formatting (tool_call XML tags, function signatures)
    #   - Multi-step tool response handling
    #   - reasoning_content vs content splitting logic
    #   - enable_thinking parameter support
    #   - VLM/multimodal content handling
    #
    # LIMITATIONS:
    #   - Training data with tool_call messages will not be formatted correctly.
    #     Use the original template with manually added {% generation %} tags for
    #     tool-use training data.
    #   - The chatml_think variant adds <think>\n\n</think>\n\n only to the last
    #     assistant turn (matching Qwen3 behavior). Non-last turns without <think>
    #     in their content will differ from the original template which also
    #     conditionally adds think wrappers based on multi-step reasoning context.
    #   - Only ChatML (<|im_start|>/<|im_end|>) and Llama3
    #     (<|start_header_id|>/<|eot_id|>) styles are supported. Other template
    #     styles fall back to regex-based assistant span detection.
    #
    # TO USE A CUSTOM TEMPLATE INSTEAD:
    #   Pass chat_template= to LanguageDataCollator with your own template that
    #   includes {% generation %}...{% endgeneration %} around assistant content.
    _GENERATION_TEMPLATES = {
        # Basic ChatML without <think> injection (Phi, older Qwen, generic ChatML)
        "chatml": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% generation %}"
            "{{ message['content'] }}"
            "{% endgeneration %}"
            "{{ '<|im_end|>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        ),
        # ChatML with <think> wrapper on last assistant turn (Qwen3-style)
        "chatml_think": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% generation %}"
            "{% if loop.last and not message['content'].startswith('<think>') %}"
            "{{ '<think>\n\n</think>\n\n' }}"
            "{% endif %}"
            "{{ message['content'] }}"
            "{% endgeneration %}"
            "{{ '<|im_end|>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        ),
        "llama3": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% generation %}"
            "{{ message['content'] }}{% endgeneration %}{{ '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        ),
    }

    def _ensure_generation_tags(self):
        """Ensure chat template has {% generation %} tags for answer_only_loss.

        If the template already has generation tags, no action taken.
        Otherwise, detect the template style and replace with a simplified
        version that includes proper generation tags.
        """
        template = self.tokenizer.chat_template
        if template is None:
            return

        if "{% generation %}" in template or "{%generation%}" in template:
            return

        # Detect template style and replace with generation-tagged version
        old_template = template
        if "<|im_start|>" in template and "<|im_end|>" in template:
            # Check if original template injects <think> (Qwen3-style)
            style = "chatml_think" if "<think>" in template else "chatml"
        elif "<|start_header_id|>" in template and "<|eot_id|>" in template:
            style = "llama3"
        else:
            print_rank_0(
                "=== WARNING === Cannot auto-inject {% generation %} tags for this chat "
                "template. answer_only_loss will not work correctly. Provide a template "
                "with {% generation %} tags via the chat_template parameter."
            )
            return

        new_template = self._GENERATION_TEMPLATES[style]
        self.tokenizer.chat_template = new_template

        # Verify
        try:
            test_msgs = [
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            ]
            result = self.tokenizer.apply_chat_template(
                test_msgs,
                return_dict=True,
                return_assistant_tokens_mask=True,
                padding=True,
                return_tensors="pt",
            )
            mask = result.get("assistant_masks", None)
            if mask is not None and mask.any():
                print_rank_0(
                    f"Replaced chat template with {style} generation-tagged version "
                    f"for answer_only_loss."
                )
                return
        except Exception:
            pass

        # Revert on failure
        self.tokenizer.chat_template = old_template
        print_rank_0(
            f"=== WARNING === Failed to apply {style} generation template. "
            "answer_only_loss will not work correctly."
        )

    def _process_chat_sample(self, examples: list):
        tokenized_examples = self.tokenizer.apply_chat_template(
            examples,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            truncation=True,
            max_length=self.train_len,
            add_generation_prompt=self.add_generation_prompt,
            return_assistant_tokens_mask=self.answer_only_loss,
        )
        if self.return_labels:
            input_ids = tokenized_examples["input_ids"]
            labels = input_ids.new_full(input_ids.shape, IGNORE_TOKEN_ID)
            labels[..., :-1] = input_ids[..., 1:]
            if self.answer_only_loss:
                if "assistant_masks" in tokenized_examples:
                    assistant_mask = tokenized_examples["assistant_masks"]
                    if isinstance(assistant_mask, torch.Tensor) and assistant_mask.any():
                        labels[assistant_mask == 0] = IGNORE_TOKEN_ID
                    else:
                        # All assistant content truncated or no assistant in batch — mask all
                        labels[:] = IGNORE_TOKEN_ID
                else:
                    raise ValueError(
                        "answer_only_loss requires {% generation %} tags in the chat "
                        "template but assistant_masks was not returned by the tokenizer. "
                        "Ensure _ensure_generation_tags() ran successfully."
                    )
            tokenized_examples["labels"] = labels
        return tokenized_examples

    def _process_text_sample(self, examples: list):
        tokenized_examples = self.tokenizer(
            examples,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.train_len,
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
                conversations = example.get("conversations", None)
                # Prefer whichever has an assistant turn for training
                if messages and any(m.get("role") == "assistant" for m in messages):
                    batch.append(messages)
                elif conversations:
                    converted = _sharegpt_to_openai_messages(conversations)
                    if not any(m.get("role") == "assistant" for m in converted):
                        print_rank_0(
                            "=== WARNING === Skipping sample with no assistant turn in conversations."
                        )
                        continue
                    batch.append(converted)
                elif messages:
                    if not any(m.get("role") == "assistant" for m in messages):
                        print_rank_0(
                            "=== WARNING === Skipping sample with no assistant turn in messages."
                        )
                        continue
                    batch.append(messages)
                else:
                    raise ValueError(
                        "The sample must in either OpenAI messages format or ShareGPT conversations format."
                    )

        if not batch:
            # All samples skipped — create a dummy batch with all-masked labels
            # so the training step produces zero loss without crashing DDP
            batch = [[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]]  # type: ignore[list-item]

        return self._process_chat_sample(batch)


class VisionLanguageDataCollator(LanguageDataCollator):
    """VisionLanguageDataCollator is a subclass of LanguageDataCollator that is used to collate vision-language data."""

    def __init__(
        self,
        processor: str,
        train_len: int = 8192,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        answer_only_loss: bool = False,
        local_image_path: str = "",
        return_labels: bool = False,
    ):
        """Initialize the VisionLanguageDataset."""
        self.processor = transformers.AutoProcessor.from_pretrained(processor)
        self.chat_template = chat_template
        self.local_image_path = local_image_path

        super().__init__(
            tokenizer=self.processor.tokenizer,
            train_len=train_len,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            answer_only_loss=answer_only_loss,
            return_labels=return_labels,
        )

    def _process_multimodal_sample(self, examples):
        tokenized_messages = self.processor.apply_chat_template(
            examples,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            truncation=True,
            max_length=self.train_len,
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
                    if ctn["type"] == "image" and "image" in ctn:
                        ctn["image"] = os.path.abspath(
                            os.path.join(self.local_image_path, ctn["image"])
                        )
                    # If any value in ctn is None, delete that key
                    # HF dataloader add Nones to align keys. Leads to error in processor.
                    keys_to_delete = [k for k, v in ctn.items() if v is None]
                    for k in keys_to_delete:
                        del ctn[k]

            batch.append(copy_messages)

        return self._process_multimodal_sample(batch)
