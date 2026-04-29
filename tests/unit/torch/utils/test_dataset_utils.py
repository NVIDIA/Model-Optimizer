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

from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from modelopt.torch.utils.dataset_utils import (
    _disable_use_cache,
    _forward_loop,
    _process_batch,
    get_dataset_samples,
    get_max_batch_size,
)


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


def test_process_batch_allowed_non_tensor_keys_accepted():
    """Non-tensor values under allowed_non_tensor_keys should not raise."""
    batch_data = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "base_model_outputs": [{"hidden_states": torch.zeros(2, 8, 16)}],  # non-tensor
    }

    def mock_infer(**kwargs):
        pass

    # Should not raise
    _process_batch(batch_data, mock_infer, allowed_non_tensor_keys={"base_model_outputs"})


def test_process_batch_non_tensor_without_allowlist_raises():
    """Non-tensor values without allowlist should raise AssertionError."""
    batch_data = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "base_model_outputs": [{"hidden_states": torch.zeros(2, 8, 16)}],
    }

    def mock_infer(**kwargs):
        pass

    with pytest.raises(AssertionError):
        _process_batch(batch_data, mock_infer)


def test_process_batch_other_keys_still_validated():
    """Non-tensor values under non-allowed keys should still raise even with allowlist set."""
    batch_data = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "unexpected_key": "some_string",  # not in allowed list
    }

    def mock_infer(**kwargs):
        pass

    with pytest.raises(AssertionError):
        _process_batch(batch_data, mock_infer, allowed_non_tensor_keys={"base_model_outputs"})


class _Config:
    """Minimal config stand-in; instances start with no `use_cache` attribute."""


def test_disable_use_cache_no_config_attr():
    """Model without a `config` attribute: CM is a no-op and does not raise."""
    model = torch.nn.Linear(4, 4)
    assert not hasattr(model, "config")

    with _disable_use_cache(model):
        assert not hasattr(model, "config")

    assert not hasattr(model, "config")


@pytest.mark.parametrize("prev_value", [True, False])
def test_disable_use_cache_with_existing_attr(prev_value):
    """Config that already has `use_cache`: forced to False inside, restored on exit."""
    model = torch.nn.Linear(4, 4)
    model.config = _Config()
    model.config.use_cache = prev_value

    with _disable_use_cache(model):
        assert model.config.use_cache is False

    assert model.config.use_cache is prev_value


def test_disable_use_cache_without_existing_attr():
    """Config that lacks `use_cache`: set to False inside, attribute removed on exit (no leak)."""
    model = torch.nn.Linear(4, 4)
    model.config = _Config()
    assert not hasattr(model.config, "use_cache")

    with _disable_use_cache(model):
        assert model.config.use_cache is False

    assert not hasattr(model.config, "use_cache")


def test_forward_loop_runs_under_disabled_use_cache():
    """`_forward_loop` runs forward on every batch and restores `use_cache` on exit."""
    seen_use_cache: list[bool] = []

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Config()
            self.config.use_cache = True

        def forward(self, **kwargs):
            seen_use_cache.append(self.config.use_cache)

    model = _Model()

    def _collate(samples):
        return {"input_ids": torch.stack([s["input_ids"] for s in samples])}

    data = [{"input_ids": torch.zeros(8, dtype=torch.long)} for _ in range(3)]
    loader = DataLoader(data, batch_size=1, collate_fn=_collate)

    _forward_loop(model, loader)

    assert seen_use_cache == [False, False, False]
    assert model.config.use_cache is True


def test_disable_use_cache_restores_on_exception():
    """Restore must run even if the with-block raises."""
    model = torch.nn.Linear(4, 4)
    model.config = _Config()
    model.config.use_cache = True

    with pytest.raises(RuntimeError, match="boom"), _disable_use_cache(model):
        assert model.config.use_cache is False
        raise RuntimeError("boom")

    assert model.config.use_cache is True


def test_get_max_batch_size_oom_retry_shrinks_input():
    """On OOM, target_input must be re-expanded to the halved batch size.

    Regression test: previously target_input was built once and never shrunk,
    so OOM retries kept feeding the same too-large tensor to infer_method.
    """
    seq_len = 8
    sample_input = torch.ones((1, seq_len), dtype=torch.int32)

    seen_batch_sizes: list[int] = []

    def fake_forward(x):
        seen_batch_sizes.append(x.shape[0])
        # First call is the single-batch probe — succeeds.
        # Second call is the target-batch attempt — OOMs.
        # Third call (after halving) — succeeds.
        if len(seen_batch_sizes) == 2:
            raise torch.cuda.OutOfMemoryError

    model = Mock(spec=torch.nn.Module)
    model.forward = fake_forward
    model.__class__.__name__ = "DummyModel"  # not enc/dec

    free_before = 1000
    free_after = 900  # 100 bytes per batch -> target = 1000/100 = 10

    device_props = Mock()
    device_props.total_memory = 10**12

    with (
        patch("torch.cuda.empty_cache"),
        patch("torch.cuda.get_device_properties", return_value=device_props),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.mem_get_info", side_effect=[(free_before, 0), (free_after, 0)]),
        patch("torch.cuda.max_memory_allocated", side_effect=[0, 0]),
    ):
        result = get_max_batch_size(
            model,
            max_sample_length=seq_len,
            sample_input_single_batch=sample_input,
        )

    # Forward calls: probe(1), retry-at-target(10), retry-after-halve(5)
    assert seen_batch_sizes == [1, 10, 5]
    # Final batch is 5 -> regulated to 4 (5 // 4 * 4 = 4).
    assert result == 4


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
