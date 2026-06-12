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
