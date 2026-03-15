# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import Mock

import pytest
import torch

from modelopt.torch.utils.dataset_utils import _process_batch, get_dataset_samples


def setup_test_data():
    # Create sample batch data
    batch_data = {
        "input_ids": torch.ones((8, 512), dtype=torch.long),
        "attention_mask": torch.ones((8, 512), dtype=torch.long),
    }

    # Create a mock inference method that raises OOM for batch sizes > 2
    def mock_infer(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        if batch_size > 2:
            raise torch.cuda.OutOfMemoryError

    return batch_data, mock_infer


def test_successful_processing():
    _, mock_infer = setup_test_data()

    # Test with small batch that shouldn't trigger OOM
    small_batch = {
        "input_ids": torch.ones((2, 512), dtype=torch.long),
        "attention_mask": torch.ones((2, 512), dtype=torch.long),
    }

    # Should complete without raising any exceptions
    assert _process_batch(small_batch, mock_infer) == 2


def test_oom_splitting():
    batch_data, mock_infer = setup_test_data()

    # Mock to track calls to the inference method
    mock_infer_spy = Mock(side_effect=mock_infer)

    # Process a batch that will trigger OOM and force splitting
    assert _process_batch(batch_data, mock_infer_spy) == 2

    # The batch should be split multiple times until processable sizes are reached
    # With initial batch size 8, splits occur:
    # 8 -> OOM
    # 4,4 -> OOM for the first 4
    # 2,2,2,2 -> Success
    # Total calls: 1 (initial) + 1 (size 4) + 4 (size 2) = 6
    assert mock_infer_spy.call_count == 6


def test_oom_with_single_sample():
    # Test handling of OOM with batch size 1
    single_batch = {
        "input_ids": torch.ones((1, 512), dtype=torch.long),
        "attention_mask": torch.ones((1, 512), dtype=torch.long),
    }

    def mock_infer_always_oom(**kwargs):
        raise torch.cuda.OutOfMemoryError

    # Should raise assertion error since can't split batch size 1
    with pytest.raises(AssertionError):
        _process_batch(single_batch, mock_infer_always_oom)


def test_batch_contents_preserved():
    # Create batch with distinct values to verify splitting preserves data
    batch_data = {
        "input_ids": torch.arange(4).view(4, 1),
        "attention_mask": torch.ones((4, 1), dtype=torch.long),
    }

    processed_values = []

    def mock_infer_collect(**kwargs):
        if kwargs["input_ids"].shape[0] > 2:
            raise torch.cuda.OutOfMemoryError
        processed_values.extend(kwargs["input_ids"].flatten().tolist())

    _process_batch(batch_data, mock_infer_collect)

    # Verify all values were processed in the correct order
    assert processed_values == [0, 1, 2, 3]


@pytest.mark.parametrize("test_local_path", [True, False])
def test_get_dataset_samples_with_unsupported_minipile_dataset(tmp_path, test_local_path):
    pytest.importorskip("datasets")
    pytest.importorskip("huggingface_hub")

    from huggingface_hub import snapshot_download

    dataset_name = "nanotron/minipile_100_samples"
    if test_local_path:
        local_dir = str(tmp_path / dataset_name)
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=local_dir,
        )
        dataset_name = local_dir

    samples = get_dataset_samples(dataset_name, num_samples=5)

    assert isinstance(samples, list)
    assert len(samples) == 5
    assert all(isinstance(s, str) and len(s) > 0 for s in samples)
