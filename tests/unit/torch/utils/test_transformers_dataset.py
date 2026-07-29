# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for Hugging Face dataset collators."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from modelopt.torch.utils.plugins.transformers_dataset import (
    IGNORE_TOKEN_ID,
    VisionLanguageDataCollator,
)


def test_vlm_collator_derives_masks_pads_and_masks_unshifted_labels():
    """VLM processors receive text options directly and output fixed-size DFlash batches."""
    processor = SimpleNamespace(
        apply_chat_template=Mock(
            return_value={
                "input_ids": torch.tensor([[1, 11, 7, 12]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            }
        )
    )
    collator = VisionLanguageDataCollator.__new__(VisionLanguageDataCollator)
    collator.processor = processor
    collator.tokenizer = SimpleNamespace(pad_token_id=0)
    collator.train_len = 6
    collator.add_generation_prompt = False
    collator.answer_only_loss = True
    collator.shift_labels = False
    collator.return_labels = True
    collator._assistant_marker_specs = Mock(return_value=[([11], [[12]])])

    result = collator._process_multimodal_sample([[{"role": "assistant", "content": "x"}]])

    processor.apply_chat_template.assert_called_once_with(
        [[{"role": "assistant", "content": "x"}]],
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=False,
        padding="max_length",
        truncation=True,
        max_length=6,
    )
    torch.testing.assert_close(result["input_ids"], torch.tensor([[1, 11, 7, 12, 0, 0]]))
    torch.testing.assert_close(result["assistant_masks"], torch.tensor([[0, 0, 1, 0, 0, 0]]))
    torch.testing.assert_close(
        result["labels"],
        torch.tensor(
            [
                [
                    IGNORE_TOKEN_ID,
                    IGNORE_TOKEN_ID,
                    7,
                    IGNORE_TOKEN_ID,
                    IGNORE_TOKEN_ID,
                    IGNORE_TOKEN_ID,
                ]
            ]
        ),
    )
