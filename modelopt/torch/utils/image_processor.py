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

# Adapted from tensorrt_llm/quantization/image_processing.py
"""Utility classes for image processing."""

import torch


class BaseImageProcessor:
    """Base class for image processors."""

    def __init__(self, tokenizer, device="cuda"):
        """Constructor."""
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, **kwargs):
        """Call the tokenizer."""
        return self.tokenizer(**kwargs)

    def preprocess_function(self, examples):
        """Preprocess function."""
        raise NotImplementedError("Each image processor must implement its own preprocess method")

    def collate_function(self, examples):
        """Collate function to process images during data loading."""
        raise NotImplementedError("Each image processor must implement its own collate method")


# A light Encapsulation for Huggingface MllamaImageProcessor


class MllamaImageProcessor(BaseImageProcessor):
    """Image processor for Mllama."""

    def preprocess_function(self, examples):
        """Preprocess function."""
        # Prepare prompts in a generic chat format
        question = examples.get("question", "Describe this image.")

        if examples["image"] is not None:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": question}],
                        }
                    ],
                    add_generation_prompt=True,
                )
            else:
                prompt = f"<|image|><|begin_of_text|>{question}"

            # Process images using the processor's image processor
            values = self.tokenizer(text=prompt, images=examples["image"], return_tensors="pt").to(
                self.device
            )
        else:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": question}],
                        }
                    ],
                    add_generation_prompt=True,
                )
            else:
                prompt = question

            values = self.tokenizer(text=prompt, images=None, return_tensors="pt").to(self.device)

            values["pixel_values"] = None
            values["aspect_ratio_ids"] = None
            values["aspect_ratio_mask"] = None
            values["cross_attention_mask"] = None

        return values

    def collate_function(self, batch):
        """Collate function to process images during data loading."""
        batch[0]["input_ids"] = torch.LongTensor(batch[0]["input_ids"]).to(self.device)
        batch[0]["attention_mask"] = torch.LongTensor(batch[0]["attention_mask"]).to(self.device)

        if batch[0]["pixel_values"] is not None:
            batch[0]["pixel_values"] = torch.Tensor(batch[0]["pixel_values"]).to(self.device)
            batch[0]["aspect_ratio_ids"] = torch.LongTensor(batch[0]["aspect_ratio_ids"]).to(
                self.device
            )
            batch[0]["aspect_ratio_mask"] = torch.LongTensor(batch[0]["aspect_ratio_mask"]).to(
                self.device
            )
            batch[0]["cross_attention_mask"] = torch.LongTensor(
                batch[0]["cross_attention_mask"]
            ).to(self.device)

        return batch[0]


class Qwen3OmniTextProcessor(BaseImageProcessor):
    """Text-only processor for Qwen3-Omni that applies proper conversation template.

    This processor wraps raw text in the Qwen3-Omni conversation format and applies
    the chat template before tokenization. Use this for text-only calibration datasets.

    See: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking
    """

    def __init__(self, processor, device="auto", dtype=None):
        """Constructor.

        Args:
            processor: The Qwen3OmniMoeProcessor (from AutoProcessor.from_pretrained).
            device: Device to move tensors to.
            dtype: dtype for float tensors (e.g., torch.bfloat16). If None, uses default.
        """
        super().__init__(processor, device)
        self.dtype = dtype

    def preprocess_function(self, text: str) -> dict:
        """Preprocess a single text sample by applying conversation template.

        Args:
            text: Raw text string from dataset.

        Returns:
            Dictionary with tokenized inputs.
        """
        # Build conversation in Qwen format (text-only)
        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        formatted_text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        # Tokenize with the processor (no multimodal inputs)
        values = self.tokenizer(
            text=formatted_text,
            audio=None,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
        )

        return values

    def collate_function(self, batch):
        """Collate function to process text inputs during data loading."""
        result = {}
        first = batch[0]

        if "input_ids" in first and first["input_ids"] is not None:
            result["input_ids"] = torch.LongTensor(first["input_ids"]).to(self.device)
        if "attention_mask" in first and first["attention_mask"] is not None:
            result["attention_mask"] = torch.LongTensor(first["attention_mask"]).to(self.device)

        return result


class Qwen3OmniImageProcessor(BaseImageProcessor):
    """Image processor for Qwen3-Omni multimodal model."""

    def __init__(self, tokenizer, device="auto", use_audio_in_video=False):
        """Constructor."""
        super().__init__(tokenizer, device)
        self.use_audio_in_video = use_audio_in_video
        # Try to import qwen_omni_utils for multimodal processing
        try:
            from qwen_omni_utils import process_mm_info

            self.process_mm_info = process_mm_info
        except ImportError:
            raise ImportError(
                "qwen_omni_utils is required for Qwen3OmniImageProcessor. "
                "Please install it from https://github.com/QwenLM/Qwen3-Omni"
            )

    def preprocess_function(self, examples):
        """Preprocess function for Qwen3-Omni."""
        question = examples.get("question", "Describe this image.")

        # Build conversation in Qwen format
        content = []
        if examples.get("image") is not None:
            content.append({"type": "image", "image": examples["image"]})
        if examples.get("audio") is not None:
            content.append({"type": "audio", "audio": examples["audio"]})
        if examples.get("video") is not None:
            content.append({"type": "video", "video": examples["video"]})
        content.append({"type": "text", "text": question})

        conversation = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        # Extract multimodal info using qwen_omni_utils
        audios, images, videos = self.process_mm_info(
            conversation, use_audio_in_video=self.use_audio_in_video
        )

        # Process inputs with the processor
        values = self.tokenizer(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        )

        # Define all possible keys to ensure consistent schema for Arrow serialization
        all_keys = [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "audio_features",
            "audio_feature_lens",
            "video_grid_thw",
        ]

        # Convert tensors to lists for Arrow serialization compatibility
        # Tensor conversion back happens in collate_function
        result = dict.fromkeys(all_keys)  # Initialize all keys to None
        for key, val in values.items():
            if val is not None and hasattr(val, "tolist"):
                result[key] = val.tolist()
            elif val is not None:
                result[key] = val

        return result

    def collate_function(self, batch):
        """Collate function to process inputs during data loading."""
        result = {}

        # Take first item from batch (batch_size handling)
        first = batch[0]

        # Convert lists to tensors and move to device
        if "input_ids" in first and first["input_ids"] is not None:
            result["input_ids"] = torch.LongTensor(first["input_ids"]).to(self.device)
        if "attention_mask" in first and first["attention_mask"] is not None:
            result["attention_mask"] = torch.LongTensor(first["attention_mask"]).to(self.device)

        # Handle pixel values for images
        if first.get("pixel_values") is not None:
            result["pixel_values"] = torch.tensor(first["pixel_values"]).to(self.device)

        # Handle image grid thw (tile height width info)
        if first.get("image_grid_thw") is not None:
            result["image_grid_thw"] = torch.LongTensor(first["image_grid_thw"]).to(self.device)

        # Handle audio features if present
        if first.get("audio_feature_lens") is not None:
            result["audio_feature_lens"] = torch.LongTensor(first["audio_feature_lens"]).to(
                self.device
            )
        if first.get("audio_features") is not None:
            result["audio_features"] = torch.tensor(first["audio_features"]).to(self.device)

        # Handle video features if present
        if first.get("video_grid_thw") is not None:
            result["video_grid_thw"] = torch.LongTensor(first["video_grid_thw"]).to(self.device)

        return result
