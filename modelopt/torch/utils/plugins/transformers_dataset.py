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
import json
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
        # datasets' resolve_pattern only matches entries with type=="file", so passing
        # a bare directory path as data_files results in FileNotFoundError.
        # Use data_dir for directory paths instead.
        data_dir = None
        data_files = self.data_files
        if data_files and os.path.isdir(data_files):
            data_dir = data_files
            data_files = None

        try:
            dataset = load_dataset(
                self.name,
                self.subset,
                data_files=data_files,
                data_dir=data_dir,
                split=self.split,
                # num_proc=4,  # TODO: Make this configurable
                streaming=self.num_streaming_samples is not None,
            )
        except Exception as exc:
            # A merged JSONL may contain valid records with incompatible Arrow
            # schemas.  Preserve those records as Python dicts rather than
            # silently coercing multimodal content to ``None``.  This path is
            # deliberately limited to local, non-streaming JSONL inputs so an
            # unrelated dataset-loading failure remains visible to the caller.
            if (
                self.name != "json"
                or self.num_streaming_samples is not None
                or data_dir is not None
                or data_files is None
            ):
                raise

            files = data_files if isinstance(data_files, list) else [data_files]
            if not all(isinstance(path, str) and os.path.isfile(path) for path in files):
                raise

            print_rank_0(
                f"load_dataset raised {type(exc).__name__}: {exc}\n"
                "Falling back to line-by-line JSONL loading to preserve heterogeneous records."
            )
            rows = []
            for path in files:
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))

            self._raw_samples = rows[self.shard_index :: self.num_shards]
            print_rank_0(
                f"Loaded {len(rows)} JSONL records; {len(self._raw_samples)} assigned to "
                f"shard {self.shard_index}/{self.num_shards}."
            )
            return

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
        shift_labels: bool = True,
        json_key: str = "text",
        return_labels: bool = False,
    ):
        """Initialize the LanguageDataset.

        Args:
            tokenizer: HuggingFace tokenizer.
            train_len: Maximum sequence length for training.
            chat_template: Optional custom chat template override.
            add_generation_prompt: Whether to add generation prompt to chat template.
            answer_only_loss: If True, mask loss on non-assistant tokens using
                ``{% generation %}`` tags in the chat template.
            shift_labels: Label alignment mode.
                If True (default), labels are shifted by 1 for autoregressive training
                (label[i] = input[i+1], used by EAGLE3). The answer_only_loss mask is
                also shifted to align with the target tokens.
                If False, labels are unshifted for diffusion-style training
                (label[i] = input[i], used by DFlash). The answer_only_loss mask is
                applied directly without shifting.
            json_key: Key for plain text samples (non-chat format).
            return_labels: Whether to include labels in the output.
        """
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError(
                f"The tokenizer must be a transformers.PreTrainedTokenizerBase but got {type(tokenizer)}"
            )
        self.tokenizer = tokenizer
        self.train_len = train_len
        self.add_generation_prompt = add_generation_prompt
        self.answer_only_loss = answer_only_loss
        self.shift_labels = shift_labels
        self.json_key = json_key
        self.return_labels = return_labels
        self._conversations_warned = False

        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        else:
            self._post_process_chat_template()

        self._post_process_tokenizer()
        if self.tokenizer.chat_template is None:
            raise ValueError(
                "No valid chat template! The tokenizer for this model has no chat_template, "
                "which is common for base / pretrained checkpoints. Use an instruction-tuned "
                "model (e.g. a '*-Instruct' variant), or pass a custom chat_template via the "
                "training config or LanguageDataCollator(chat_template=...)."
            )

        if self.answer_only_loss:
            self._verify_generation_tags()

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
        if self.tokenizer.chat_template is None:
            return
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

    def _verify_generation_tags(self):
        """Verify the chat template supports answer_only_loss via {% generation %} tags.

        answer_only_loss requires the tokenizer's chat template to include
        {% generation %} / {% endgeneration %} tags around assistant content.
        These tags tell HuggingFace's apply_chat_template() which tokens are
        assistant responses, enabling return_assistant_tokens_mask.

        If the template lacks these tags, this method raises an error with
        instructions for the user to provide a compatible template.

        Per-model chat templates with generation tags should be maintained
        alongside model recipes (e.g., in modelopt_recipes/) where correctness
        can be tested.
        """
        template = self.tokenizer.chat_template
        if template and ("generation" in template and "endgeneration" in template):
            return

        raise ValueError(
            "answer_only_loss requires {% generation %} / {% endgeneration %} tags in the "
            "chat template, but the current template does not have them.\n\n"
            "To fix, provide a chat_template with generation tags:\n"
            "  1. Copy your model's chat_template from tokenizer_config.json\n"
            "  2. Wrap assistant content with {% generation %} / {% endgeneration %}\n"
            "  3. Pass via LanguageDataCollator(chat_template=...) or the training config\n\n"
            "See https://huggingface.co/docs/transformers/en/chat_templating"
            "#train-on-completions-only for the official HuggingFace guide.\n\n"
            "Per-model templates are maintained in modelopt_recipes/ alongside training recipes."
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
            if self.shift_labels:
                # Autoregressive: label[i] = input[i+1]
                labels[..., :-1] = input_ids[..., 1:]
            else:
                # Diffusion: label[i] = input[i]
                labels[:] = input_ids
            if self.answer_only_loss:
                if "assistant_masks" in tokenized_examples:
                    assistant_mask = tokenized_examples["assistant_masks"]
                    if isinstance(assistant_mask, torch.Tensor) and assistant_mask.any():
                        if self.shift_labels:
                            # Shifted labels: mask based on whether the *target* token
                            # (input[i+1]) is assistant content.
                            shifted_mask = assistant_mask[..., 1:]
                            labels[..., :-1][shifted_mask == 0] = IGNORE_TOKEN_ID
                        else:
                            # Unshifted labels: mask based on the input token directly.
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
                raise ValueError(f"The sample must be a Dict but got {type(example)}")
            text = example.get(self.json_key, None)
            if isinstance(text, str):
                batch.append(text)
            else:
                messages = example.get("messages", None)
                if not messages and example.get("conversations", None):
                    messages = example["conversations"]
                    if not self._conversations_warned:
                        print_rank_0(
                            "=== DEPRECATION WARNING === 'conversations' field is deprecated. "
                            "Use 'messages' (OpenAI format) instead."
                        )
                        self._conversations_warned = True
                if not messages:
                    raise ValueError(
                        "Sample must have a 'messages' field in OpenAI format "
                        "(list of {role, content} dicts)."
                    )
                if not any(m.get("role") == "assistant" for m in messages):
                    print_rank_0(
                        "=== WARNING === Skipping sample with no assistant turn in messages."
                    )
                    continue
                batch.append(messages)

        if not batch:
            # All samples skipped — create a dummy batch with all-masked labels
            # so the training step produces zero loss without crashing DDP
            batch = [[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]]  # type: ignore[list-item]

        return self._process_chat_sample(batch)


class VisionLanguageDataCollator(LanguageDataCollator):
    """Collate multimodal conversations for online speculative-decoding training."""

    def __init__(
        self,
        processor: str,
        train_len: int = 8192,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        answer_only_loss: bool = False,
        shift_labels: bool = True,
        local_image_path: str = "",
        return_labels: bool = False,
    ):
        """Initialize the collator and its Hugging Face processor.

        ``VLM_MIN_PIXELS`` and ``VLM_MAX_PIXELS`` allow launchers to bound
        Qwen-VL image token expansion without baking environment-specific
        limits into a recipe.
        """
        processor_kwargs = {}
        for env_name, kwarg_name in (
            ("VLM_MIN_PIXELS", "min_pixels"),
            ("VLM_MAX_PIXELS", "max_pixels"),
        ):
            value = os.environ.get(env_name)
            if value is not None:
                try:
                    processor_kwargs[kwarg_name] = int(value)
                except ValueError as exc:
                    raise ValueError(f"{env_name} must be an integer, got {value!r}") from exc

        self.processor = transformers.AutoProcessor.from_pretrained(processor, **processor_kwargs)
        if processor_kwargs:
            print_rank_0(f"Loaded VLM processor with {processor_kwargs}")
        self.chat_template = chat_template
        self.local_image_path = local_image_path
        self._conversations_warned = False

        super().__init__(
            tokenizer=self.processor.tokenizer,
            train_len=train_len,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            answer_only_loss=answer_only_loss,
            shift_labels=shift_labels,
            return_labels=return_labels,
        )
        # Processor.apply_chat_template() consults the processor-level template,
        # while the base collator applies an optional training override to its
        # tokenizer. Keep the two views in sync.
        self.processor.chat_template = self.tokenizer.chat_template

    def _verify_generation_tags(self):
        """Accept VLM templates whose assistant spans we derive from chat markers.

        Some Qwen-VL processor versions fail while deriving Hugging Face
        generation masks at a truncation boundary.  The VLM path below builds
        the same answer-only mask from the tokenized ChatML/Llama markers, so
        generation tags are not required for those templates.
        """
        if self._assistant_marker_specs():
            return
        super()._verify_generation_tags()

    def _encode_chat_marker(self, marker: str) -> list[int]:
        return list(self.tokenizer(marker, add_special_tokens=False)["input_ids"])

    def _assistant_marker_specs(self) -> list[tuple[list[int], list[list[int]]]]:
        if hasattr(self, "_cached_assistant_marker_specs"):
            return self._cached_assistant_marker_specs

        template = self.tokenizer.chat_template or ""
        specs = []
        if "<|im_start|>" in template and "<|im_end|>" in template:
            specs.append(
                (
                    self._encode_chat_marker("<|im_start|>assistant\n"),
                    [
                        self._encode_chat_marker("<|im_end|>\n"),
                        self._encode_chat_marker("<|im_end|>"),
                    ],
                )
            )
        if "<|start_header_id|>" in template and "<|eot_id|>" in template:
            specs.append(
                (
                    self._encode_chat_marker(
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    ),
                    [self._encode_chat_marker("<|eot_id|>")],
                )
            )

        self._cached_assistant_marker_specs = [
            (start, [end for end in end_markers if end])
            for start, end_markers in specs
            if start and any(end_markers)
        ]
        return self._cached_assistant_marker_specs

    @staticmethod
    def _find_subsequence(
        values: list[int], pattern: list[int], start: int = 0, stop: int | None = None
    ) -> int:
        stop = len(values) if stop is None else stop
        if not pattern or start >= stop:
            return -1
        for index in range(start, stop - len(pattern) + 1):
            if values[index : index + len(pattern)] == pattern:
                return index
        return -1

    def _build_assistant_masks(self, tokenized_messages):
        """Return an assistant-content mask from stable chat boundary tokens."""
        input_ids = tokenized_messages["input_ids"]
        attention_mask = tokenized_messages.get("attention_mask")
        assistant_masks = torch.zeros_like(input_ids)

        for row_index, row in enumerate(input_ids):
            tokens = row.tolist()
            if isinstance(attention_mask, torch.Tensor):
                active = attention_mask[row_index].nonzero(as_tuple=False).flatten()
                if active.numel() == 0:
                    continue
                sequence_start = int(active[0])
                sequence_end = int(active[-1]) + 1
            else:
                sequence_start, sequence_end = 0, len(tokens)

            for start_marker, end_markers in self._assistant_marker_specs():
                search_from = sequence_start
                while search_from < sequence_end:
                    start = self._find_subsequence(
                        tokens, start_marker, search_from, sequence_end
                    )
                    if start == -1:
                        break
                    content_start = start + len(start_marker)
                    end_positions = [
                        position
                        for marker in end_markers
                        if (
                            position := self._find_subsequence(
                                tokens, marker, content_start, sequence_end
                            )
                        )
                        != -1
                    ]
                    content_end = min(end_positions) if end_positions else sequence_end
                    if content_start < content_end:
                        assistant_masks[row_index, content_start:content_end] = 1
                    search_from = max(content_start + 1, content_end + 1)

        return assistant_masks

    def _pad_sequence_tensors(self, tokenized_messages):
        """Pad processor outputs to DFlash's fixed training sequence length.

        Some Transformers 5 VLM processors ignore ``padding='max_length'`` in
        ``apply_chat_template``. DFlash operates on fixed-size blocks, so pad
        the text-side tensors here after processor tokenization. The processor
        remains responsible for truncation, which keeps image-token and pixel
        feature alignment intact.
        """
        if self.train_len is None:
            return tokenized_messages

        input_ids = tokenized_messages.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("VLM processor did not return rank-2 input_ids.")
        if input_ids.shape[1] > self.train_len:
            raise ValueError(
                f"VLM processor returned seq_len {input_ids.shape[1]} above "
                f"training_seq_len {self.train_len}; ensure processor truncation is enabled."
            )

        pad_width = self.train_len - input_ids.shape[1]
        if pad_width == 0:
            return tokenized_messages

        for key, pad_value in (
            ("input_ids", self.tokenizer.pad_token_id),
            ("attention_mask", 0),
        ):
            value = tokenized_messages.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            pad = value.new_full((value.shape[0], pad_width), pad_value)
            tokenized_messages[key] = torch.cat((value, pad), dim=1)

        return tokenized_messages

    def _apply_chat_template(self, examples):
        """Tokenize VLM messages and derive answer-only spans after truncation."""
        processor_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.train_len,
        }
        derive_masks_from_markers = bool(self._assistant_marker_specs())
        common_kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
            "add_generation_prompt": self.add_generation_prompt,
            "return_assistant_tokens_mask": (
                self.answer_only_loss and not derive_masks_from_markers
            ),
        }
        # Transformers 5 Qwen-VL processors consume text-tokenization options
        # directly. Passing them under ``processor_kwargs`` is silently ignored,
        # leaving long samples untruncated.
        tokenized_messages = self.processor.apply_chat_template(
            examples, **processor_kwargs, **common_kwargs
        )
        tokenized_messages = self._pad_sequence_tensors(tokenized_messages)
        if self.answer_only_loss and derive_masks_from_markers:
            tokenized_messages["assistant_masks"] = self._build_assistant_masks(
                tokenized_messages
            )
        return tokenized_messages

    def _process_multimodal_sample(self, examples):
        tokenized_messages = self._apply_chat_template(examples)

        if self.return_labels:
            input_ids = tokenized_messages["input_ids"]
            labels = input_ids.new_full(input_ids.shape, IGNORE_TOKEN_ID)
            if self.shift_labels:
                labels[..., :-1] = input_ids[..., 1:]
            else:
                labels[:] = input_ids

            if self.answer_only_loss:
                assistant_mask = tokenized_messages["assistant_masks"]
                if not assistant_mask.any():
                    labels[:] = IGNORE_TOKEN_ID
                elif self.shift_labels:
                    labels[..., :-1][assistant_mask[..., 1:] == 0] = IGNORE_TOKEN_ID
                else:
                    labels[assistant_mask == 0] = IGNORE_TOKEN_ID
            tokenized_messages["labels"] = labels

        return tokenized_messages

    def __call__(self, examples):
        """Call the VisionLanguageDataCollator."""
        batch = []

        for example in examples:
            messages = example.get("messages", None)
            if not messages and example.get("conversations", None):
                messages = example["conversations"]
                if not self._conversations_warned:
                    print_rank_0(
                        "=== DEPRECATION WARNING === 'conversations' field is deprecated. "
                        "Use 'messages' (OpenAI format) instead."
                    )
                    self._conversations_warned = True
            if messages is None:
                raise ValueError(
                    "Sample must have a 'messages' field in OpenAI format "
                    "(list of {role, content} dicts)."
                )

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
