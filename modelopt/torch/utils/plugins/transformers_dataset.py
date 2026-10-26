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
import time

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

        dataset = load_dataset(
            self.name,
            self.subset,
            data_files=data_files,
            data_dir=data_dir,
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
    """VisionLanguageDataCollator is a subclass of LanguageDataCollator that is used to collate vision-language data."""

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
        """Initialize the collator and its Hugging Face processor."""
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
        self._configure_image_processor(self.processor, processor_kwargs)
        self._configure_video_processor(self.processor, processor_kwargs)
        max_prompt_tokens = os.environ.get("VLM_MAX_PROMPT_TOKENS")
        self.max_prompt_tokens = None
        if max_prompt_tokens is not None:
            try:
                self.max_prompt_tokens = int(max_prompt_tokens)
            except ValueError as exc:
                raise ValueError(
                    f"VLM_MAX_PROMPT_TOKENS must be a positive integer, got {max_prompt_tokens!r}"
                ) from exc
            if self.max_prompt_tokens <= 0:
                raise ValueError("VLM_MAX_PROMPT_TOKENS must be a positive integer.")
        self._prompt_content_truncation_warned = False
        max_assistant_tokens = os.environ.get("VLM_MAX_ASSISTANT_TOKENS")
        self.max_assistant_tokens = None
        if max_assistant_tokens is not None:
            try:
                self.max_assistant_tokens = int(max_assistant_tokens)
            except ValueError as exc:
                raise ValueError(
                    "VLM_MAX_ASSISTANT_TOKENS must be a positive integer, got "
                    f"{max_assistant_tokens!r}"
                ) from exc
            if self.max_assistant_tokens <= 0:
                raise ValueError("VLM_MAX_ASSISTANT_TOKENS must be a positive integer.")
        self._assistant_content_truncation_warned = False
        self._empty_assistant_mask_warned = False
        # Transient media-read retries; see _processor_apply.
        self._media_retries = int(os.environ.get("VLM_MEDIA_LOAD_RETRIES", "4"))
        self._media_retry_sleep = float(os.environ.get("VLM_MEDIA_RETRY_SLEEP", "0.5"))
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
        self.processor.chat_template = self.tokenizer.chat_template

    @staticmethod
    def _configure_image_processor(processor, image_pixel_bounds):
        """Apply visual pixel bounds to the image processor's ``size``.

        ``min_pixels``/``max_pixels`` are accepted as ``from_pretrained`` kwargs, but the
        Qwen3-VL image processor reads its bounds from ``size['shortest_edge']`` /
        ``size['longest_edge']`` and ignores the kwargs. The cap therefore applied to video
        (which is configured through ``size``) and silently did nothing for images -- the
        measured token count per image was identical at max_pixels 3145728, 1048576 and
        524288. Set ``size`` here so the knob binds for both modalities.
        """
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None or not image_pixel_bounds:
            return
        size = dict(getattr(image_processor, "size", None) or {})
        for kwarg_name, size_key in (
            ("min_pixels", "shortest_edge"),
            ("max_pixels", "longest_edge"),
        ):
            if kwarg_name in image_pixel_bounds:
                size[size_key] = image_pixel_bounds[kwarg_name]
        # Only a size that carries both bounds is meaningful to the processor; a partial
        # one would be read with whatever the checkpoint shipped for the other edge.
        if "shortest_edge" not in size or "longest_edge" not in size:
            return
        if size["shortest_edge"] > size["longest_edge"]:
            raise ValueError("VLM_MIN_PIXELS must not exceed VLM_MAX_PIXELS.")
        image_processor.size = size
        print_rank_0(f"Configured VLM image processor with {size}")

    @staticmethod
    def _configure_video_processor(processor, image_pixel_bounds):
        """Apply visual pixel bounds to a separate Qwen video processor."""
        video_processor = getattr(processor, "video_processor", None)
        video_pixel_bounds = {}
        explicit_video_bounds = False
        for env_name, size_key, image_key in (
            ("VLM_VIDEO_MIN_PIXELS", "shortest_edge", "min_pixels"),
            ("VLM_VIDEO_MAX_PIXELS", "longest_edge", "max_pixels"),
        ):
            value = os.environ.get(env_name)
            if value is None:
                value = image_pixel_bounds.get(image_key)
            else:
                explicit_video_bounds = True
                try:
                    value = int(value)
                except ValueError as exc:
                    raise ValueError(f"{env_name} must be an integer, got {value!r}") from exc
            if value is not None:
                video_pixel_bounds[size_key] = value

        if explicit_video_bounds and video_processor is None:
            raise ValueError(
                "VLM_VIDEO_MIN_PIXELS/VLM_VIDEO_MAX_PIXELS were set, but the "
                "configured VLM processor has no video_processor."
            )
        if video_processor is not None and video_pixel_bounds:
            video_size = dict(video_processor.size)
            video_size.update(video_pixel_bounds)
            if video_size["shortest_edge"] > video_size["longest_edge"]:
                raise ValueError("VLM_VIDEO_MIN_PIXELS must not exceed VLM_VIDEO_MAX_PIXELS.")
            video_processor.size = video_size
            print_rank_0(f"Configured VLM video processor with {video_size}")

    def _verify_generation_tags(self):
        """Accept VLM templates whose assistant spans have stable chat markers.

        Cosmos/Qwen ChatML templates do not necessarily use Hugging Face's
        ``{% generation %}`` tags.  For those templates we derive the same
        assistant-only loss mask from the tokenized assistant boundaries.
        """
        if self._assistant_marker_specs():
            return
        super()._verify_generation_tags()

    def _assistant_marker_specs(self):
        """Return tokenized assistant start/end boundaries for supported templates."""
        if hasattr(self, "_cached_assistant_marker_specs"):
            return self._cached_assistant_marker_specs

        template = self.tokenizer.chat_template or ""
        specs = []
        if "<|im_start|>" in template and "<|im_end|>" in template:
            specs.append(
                (
                    self.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)[
                        "input_ids"
                    ],
                    [
                        self.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"],
                        self.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"],
                    ],
                )
            )
        self._cached_assistant_marker_specs = [
            (start, [end for end in ends if end]) for start, ends in specs if start and any(ends)
        ]
        return self._cached_assistant_marker_specs

    @staticmethod
    def _find_subsequence(values, pattern, start=0, stop=None):
        stop = len(values) if stop is None else stop
        if not pattern or start >= stop:
            return -1
        for index in range(start, stop - len(pattern) + 1):
            if values[index : index + len(pattern)] == pattern:
                return index
        return -1

    def _build_assistant_masks(self, tokenized_messages):
        """Build assistant-content masks from ChatML boundaries."""
        input_ids = tokenized_messages["input_ids"]
        attention_mask = tokenized_messages.get("attention_mask")
        assistant_masks = torch.zeros_like(input_ids)

        for row_index, row in enumerate(input_ids):
            tokens = row.tolist()
            if isinstance(attention_mask, torch.Tensor):
                active = attention_mask[row_index].nonzero(as_tuple=False).flatten()
                if active.numel() == 0:
                    continue
                sequence_start, sequence_end = int(active[0]), int(active[-1]) + 1
            else:
                sequence_start, sequence_end = 0, len(tokens)

            for start_marker, end_markers in self._assistant_marker_specs():
                search_from = sequence_start
                while search_from < sequence_end:
                    start = self._find_subsequence(tokens, start_marker, search_from, sequence_end)
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
        """Pad processor outputs to the fixed DFlash training sequence length."""
        if self.train_len is None:
            return tokenized_messages

        input_ids = tokenized_messages.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("VLM processor did not return rank-2 input_ids.")
        if input_ids.shape[1] > self.train_len:
            raise ValueError(
                f"VLM processor returned seq_len {input_ids.shape[1]} above "
                f"training_seq_len {self.train_len}; reduce visual or assistant-text caps."
            )

        pad_width = self.train_len - input_ids.shape[1]
        if pad_width == 0:
            return tokenized_messages

        sequence_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        for key, value in tokenized_messages.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape != (batch_size, sequence_length)
            ):
                continue
            pad_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
            pad = value.new_full((batch_size, pad_width), pad_value)
            tokenized_messages[key] = torch.cat((value, pad), dim=1)

        return tokenized_messages

    def _truncate_assistant_content(self, messages):
        """Limit assistant text before image/video placeholders expand into tokens."""
        if self.max_assistant_tokens is None:
            return

        for message in messages:
            if message.get("role") != "assistant":
                continue
            for content in message.get("content", []):
                if content.get("type") != "text" or not isinstance(content.get("text"), str):
                    continue
                token_ids = self.tokenizer(content["text"], add_special_tokens=False).input_ids
                if len(token_ids) <= self.max_assistant_tokens:
                    continue

                suffix_tokens = min(128, max(1, self.max_assistant_tokens // 4))
                prefix_tokens = self.max_assistant_tokens - suffix_tokens
                content["text"] = self.tokenizer.decode(
                    token_ids[:prefix_tokens] + token_ids[-suffix_tokens:],
                    skip_special_tokens=True,
                )
                if not self._assistant_content_truncation_warned:
                    print_rank_0(
                        "Truncating assistant content to "
                        f"{self.max_assistant_tokens} tokens before VLM tokenization."
                    )
                    self._assistant_content_truncation_warned = True

    def _truncate_prompt_content(self, messages):
        """Limit user/system text before image/video placeholders expand into tokens."""
        if self.max_prompt_tokens is None:
            return

        for message in messages:
            if message.get("role") not in {"system", "user"}:
                continue
            for content in message.get("content", []):
                if content.get("type") != "text" or not isinstance(content.get("text"), str):
                    continue
                token_ids = self.tokenizer(content["text"], add_special_tokens=False).input_ids
                if len(token_ids) <= self.max_prompt_tokens:
                    continue

                suffix_tokens = min(128, max(1, self.max_prompt_tokens // 4))
                prefix_tokens = self.max_prompt_tokens - suffix_tokens
                content["text"] = self.tokenizer.decode(
                    token_ids[:prefix_tokens] + token_ids[-suffix_tokens:],
                    skip_special_tokens=True,
                )
                if not self._prompt_content_truncation_warned:
                    print_rank_0(
                        "Truncating user/system content to "
                        f"{self.max_prompt_tokens} tokens before VLM tokenization."
                    )
                    self._prompt_content_truncation_warned = True

    # Messages the media loaders raise when a file does not resolve. On a shared
    # filesystem these are usually transient, not a statement about the data.
    _TRANSIENT_MEDIA_ERRORS = (
        "incorrect image source",
        "incorrect format used for video",
        "incorrect padding",
    )

    def _processor_apply(self, examples, derive_masks_from_markers):
        """Run the processor, retrying media loads that fail transiently.

        On a parallel filesystem under many ranks, an image or video that exists and is
        perfectly valid can still fail to open for one call. transformers then reports
        ``Incorrect image source ... Got <path>`` or ``Incorrect format used for video``,
        the rank raises, and every other rank in the job dies with it. Verified on the
        failures this hit: all six reported paths existed and `file` identified them as
        valid PNG/JPEG. Over several epochs each file is touched repeatedly, so a
        once-in-a-million read error becomes a recurring job killer.

        Retries only the message shapes above, and only a few times, so genuinely
        undecodable data (a BMP named .jpg, say) still fails loudly instead of being
        retried into a silent skip.
        """
        last_exc = None
        for attempt in range(self._media_retries + 1):
            try:
                return self.processor.apply_chat_template(
                    examples,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=self.add_generation_prompt,
                    return_assistant_tokens_mask=(
                        self.answer_only_loss and not derive_masks_from_markers
                    ),
                )
            except (ValueError, TypeError, OSError) as exc:  # noqa: PERF203
                # PERF203: the try/except IS the retry; the cost is irrelevant next to
                # a processor call that decodes images and video.
                text = str(exc).lower()
                if not any(m in text for m in self._TRANSIENT_MEDIA_ERRORS):
                    raise
                last_exc = exc
                if attempt < self._media_retries:
                    time.sleep(self._media_retry_sleep * (attempt + 1))
        raise RuntimeError(
            f"media load failed {self._media_retries + 1} times: {last_exc}"
        ) from last_exc

    def _apply_chat_template(self, examples):
        """Tokenize VLM messages without passing text-only kwargs to the processor."""
        derive_masks_from_markers = bool(self._assistant_marker_specs())
        tokenized_messages = self._processor_apply(examples, derive_masks_from_markers)
        tokenized_messages = self._pad_sequence_tensors(tokenized_messages)
        if self.answer_only_loss and derive_masks_from_markers:
            tokenized_messages["assistant_masks"] = self._build_assistant_masks(tokenized_messages)

        return tokenized_messages

    def _process_multimodal_sample(self, examples):
        tokenized_messages = self._apply_chat_template(examples)

        if self.return_labels:
            input_ids = tokenized_messages["input_ids"]
            labels = input_ids.new_full(input_ids.shape, IGNORE_TOKEN_ID)
            if self.shift_labels:
                labels[..., :-1] = input_ids[..., 1:]
            else:
                # DFlash predicts the token at the current position rather
                # than the next autoregressive token.
                labels[:] = input_ids

            if self.answer_only_loss:
                if "assistant_masks" not in tokenized_messages:
                    raise ValueError(
                        "answer_only_loss requires assistant_masks from the VLM chat template."
                    )
                assistant_mask = tokenized_messages["assistant_masks"]
                if not isinstance(assistant_mask, torch.Tensor) or not assistant_mask.any():
                    # Every label in the batch becomes -100, so the step contributes no
                    # gradient. Silently that looks like a healthy run with loss 0 and
                    # grad_norm 0, so say it out loud once with the two things to check.
                    if not getattr(self, "_empty_assistant_mask_warned", False):
                        print_rank_0(
                            "answer_only_loss produced an EMPTY assistant mask for an entire "
                            "batch: these samples contribute no gradient. Check that the chat "
                            "template marks assistant spans with {% generation %} tags, and "
                            "that the records actually carry an assistant turn."
                        )
                        self._empty_assistant_mask_warned = True
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
                elif isinstance(msg["content"], dict):
                    # A top-level structured answer can legitimately contain
                    # keys such as "image" or "video" (for example, a URL in
                    # a JSON response).  It is media only when it explicitly
                    # declares a supported multimodal type.
                    if msg["content"].get("type") in {"text", "image", "video"}:
                        msg["content"] = [msg["content"]]
                    else:
                        msg["content"] = [
                            {
                                "type": "text",
                                "text": json.dumps(msg["content"], ensure_ascii=False),
                            }
                        ]
                elif not isinstance(msg["content"], list):
                    # Some synthetic rows use a scalar answer or prompt (for
                    # example, an integer multiple-choice answer). Preserve it
                    # as text rather than discarding an otherwise valid row.
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": json.dumps(msg["content"], ensure_ascii=False),
                        }
                    ]

                for index, ctn in enumerate(msg["content"]):
                    if isinstance(ctn, str):
                        ctn = {"type": "text", "text": ctn}
                        msg["content"][index] = ctn
                    if not isinstance(ctn, dict):
                        ctn = {"type": "text", "text": json.dumps(ctn, ensure_ascii=False)}
                        msg["content"][index] = ctn
                    if not {"type", "text", "image", "video", "fps"}.intersection(ctn):
                        ctn = {"type": "text", "text": json.dumps(ctn, ensure_ascii=False)}
                        msg["content"][index] = ctn
                    # Some JSONL producers use a fixed multimodal-part schema
                    # (text/image/video/fps on every part) so Arrow can load
                    # heterogeneous image and video datasets together.  Drop
                    # the inactive placeholders before handing a part to the
                    # processor, which expects only fields relevant to its type.
                    content_type = ctn.get("type")
                    if content_type not in {"text", "image", "video"}:
                        inferred_types = [
                            part_type
                            for key, part_type in (
                                ("text", "text"),
                                ("image", "image"),
                                ("video", "video"),
                            )
                            if ctn.get(key) not in (None, "")
                        ]
                        if len(inferred_types) != 1:
                            raise ValueError(
                                f"Unable to infer a multimodal content type from {ctn!r}."
                            )
                        content_type = inferred_types[0]
                        ctn["type"] = content_type
                    if content_type != "text" and ctn.get("text") == "":
                        del ctn["text"]
                    if content_type != "image" and ctn.get("image") == "":
                        del ctn["image"]
                    if content_type != "video":
                        if ctn.get("video") == "":
                            del ctn["video"]
                        if ctn.get("fps") == 0:
                            del ctn["fps"]
                    if ctn["type"] == "image" and "image" in ctn:
                        ctn["image"] = os.path.abspath(
                            os.path.join(self.local_image_path, ctn["image"])
                        )
                    # If any value in ctn is None, delete that key
                    # HF dataloader add Nones to align keys. Leads to error in processor.
                    keys_to_delete = [k for k, v in ctn.items() if v is None]
                    for k in keys_to_delete:
                        del ctn[k]

            self._truncate_prompt_content(copy_messages)
            self._truncate_assistant_content(copy_messages)
            batch.append(copy_messages)

        return self._process_multimodal_sample(batch)
