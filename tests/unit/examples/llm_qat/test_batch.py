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

import runpy
from pathlib import Path

import pytest
import torch

batch_helpers = runpy.run_path(
    str(Path(__file__).parents[4] / "examples/llm_qat/linear_attention/batch.py")
)
phase_lengths = batch_helpers["phase_lengths"]
mask_prefix_labels = batch_helpers["mask_prefix_labels"]


def test_prompt_metadata_excludes_padding_and_handles_packing():
    batch = {
        "tokens": torch.zeros(2, 16),
        "context_lengths": torch.tensor([3, 5]),
        "token_count": [9, 13],
        "loss_mask": torch.zeros(2, 16),
    }
    assert phase_lengths(batch) == ([3, 5], [8, 12], None)
    assert phase_lengths(batch, 2) == ([2, 2], [8, 12], None)
    assert phase_lengths(batch, 10) == ([7, 10], [8, 12], None)
    packed = {
        "tokens": torch.zeros(1, 32),
        "prefill_lengths": [3, 5],
        "sequence_lengths": [8, 12],
        "cu_seqlens_q": [0, 8, 20],
        "cu_seqlens_q_padded": [0, 16, 32],
    }
    assert phase_lengths(packed) == ([3, 5], [8, 12], [0, 16, 32])
    del packed["sequence_lengths"]
    with pytest.raises(ValueError, match="per segment"):
        phase_lengths(packed)
    batch["context_lengths"] = [9, 5]
    with pytest.raises(ValueError, match="prefix <= valid"):
        phase_lengths(batch)


def test_prefix_loss_mask_retains_first_shifted_suffix_label():
    dense = {"loss_mask": torch.ones(2, 8)}
    mask_prefix_labels(dense, [3, 1])
    torch.testing.assert_close(
        dense["loss_mask"],
        torch.tensor(
            [
                [0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
        ),
    )

    packed = {"loss_mask": torch.ones(1, 12)}
    mask_prefix_labels(packed, [3, 2], [0, 6, 12])
    torch.testing.assert_close(
        packed["loss_mask"],
        torch.tensor([[0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]], dtype=torch.float32),
    )
