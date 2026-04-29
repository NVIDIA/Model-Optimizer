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
from torch.utils.data import DataLoader

from modelopt.torch.utils.dataset_utils import (
    _disable_use_cache,
    _forward_loop,
    _process_batch,
    get_dataset_dataloader,
    get_dataset_samples,
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


# ---------------------------------------------------------------------------
# Local JSONL loading — must flow through the same auto-preprocess path as a
# downloaded HF dataset, so chat / prompt / text columns are all handled.
# ---------------------------------------------------------------------------


def _write_jsonl(path, rows):
    """Write a list of dicts to *path* as JSONL. Returns the path as ``str``."""
    import json

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return str(path)


@pytest.fixture
def chat_tokenizer():
    """Mock tokenizer whose ``apply_chat_template`` joins messages role:content."""
    tok = Mock()
    tok.apply_chat_template = Mock(
        side_effect=lambda msgs, tokenize=False, **kw: " | ".join(
            f"{m['role']}:{m['content']}" for m in msgs
        )
    )
    return tok


class TestLocalJsonlLoading:
    """Local ``.jsonl`` paths route through HF's ``json`` builder + auto-preprocess."""

    def test_text_column(self, tmp_path):
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "plain.jsonl",
            [{"text": f"plain {i}"} for i in range(3)],
        )
        samples = get_dataset_samples(path, num_samples=3)
        assert samples == ["plain 0", "plain 1", "plain 2"]

    def test_messages_column_uses_chat_template(self, tmp_path, chat_tokenizer):
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "chat.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": f"hello {i}"},
                        {"role": "assistant", "content": f"hi {i}"},
                    ]
                }
                for i in range(3)
            ],
        )
        samples = get_dataset_samples(path, num_samples=3, tokenizer=chat_tokenizer)
        assert len(samples) == 3
        assert samples[0] == "user:hello 0 | assistant:hi 0"
        # apply_chat_template must have been invoked once per sample
        assert chat_tokenizer.apply_chat_template.call_count == 3

    def test_conversations_column_uses_chat_template(self, tmp_path, chat_tokenizer):
        """Auto-detect also recognizes ``conversations`` (Magpie-style)."""
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "convs.jsonl",
            [
                {
                    "conversations": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ],
        )
        samples = get_dataset_samples(path, num_samples=1, tokenizer=chat_tokenizer)
        assert samples == ["user:q | assistant:a"]

    def test_prompt_completion_concatenated(self, tmp_path):
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "prompt.jsonl",
            [{"prompt": "Q?", "completion": "A."}],
        )
        samples = get_dataset_samples(path, num_samples=1)
        assert samples == ["Q?\nA."]

    def test_input_output_concatenated(self, tmp_path):
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "io.jsonl",
            [{"input": "in", "output": "out"}],
        )
        samples = get_dataset_samples(path, num_samples=1)
        assert samples == ["in\nout"]

    def test_num_samples_honored(self, tmp_path):
        """Loads only the requested number of rows even when the file is larger."""
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "many.jsonl",
            [{"text": f"row {i}"} for i in range(100)],
        )
        samples = get_dataset_samples(path, num_samples=5)
        assert len(samples) == 5
        assert samples == [f"row {i}" for i in range(5)]

    def test_tools_forwarded_to_chat_template(self, tmp_path, chat_tokenizer):
        """If a row carries a ``tools`` field, it's passed through to apply_chat_template."""
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "tools.jsonl",
            [
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [{"name": "calc"}],
                }
            ],
        )
        get_dataset_samples(path, num_samples=1, tokenizer=chat_tokenizer)
        _, kwargs = chat_tokenizer.apply_chat_template.call_args
        assert kwargs.get("tools") == [{"name": "calc"}]

    def test_unrecognized_columns_raise(self, tmp_path):
        """Auto-detect raises ValueError when no recognized column is present.

        The HF builder loads the rows fine; auto-detect rejects them. There's no
        ``text`` field to fall back to, so the error propagates instead of being
        masked by the legacy fallback.
        """
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "bad.jsonl",
            [{"unrelated_field": "value"}],
        )
        with pytest.raises(ValueError, match="Cannot auto-detect format"):
            get_dataset_samples(path, num_samples=1)

    def test_sparse_recognized_column_falls_through_to_text(self, tmp_path):
        """Sparse ``prompt`` column (None on most rows) must not shadow ``text``.

        HF's schema unification fills missing values with None across heterogeneous
        rows, so a row with only ``text`` ends up exposing ``prompt=None`` in the
        unified schema.  Auto-detect must skip null-valued recognized columns
        rather than crash on ``"\\n".join([None])``.
        """
        pytest.importorskip("datasets")
        rows = [
            {"text": "row a"},
            {"text": "row b", "prompt": "ignored", "completion": "stuff"},
            {"text": "row c"},
        ]
        path = _write_jsonl(tmp_path / "sparse.jsonl", rows)
        samples = get_dataset_samples(path, num_samples=3)
        # text-only rows fall through to ``text``; the prompt-bearing row uses
        # the prompt+completion path.
        assert samples == ["row a", "ignored\nstuff", "row c"]

    def test_legacy_text_fallback_on_hf_builder_failure(self, tmp_path):
        """If the HF json builder raises, fall back to the legacy text-field reader."""
        pytest.importorskip("datasets")
        # Mixed-type ``meta`` field across rows — int vs string — trips PyArrow
        # schema unification in the HF json builder. The rows still carry a
        # ``text`` field, so the legacy reader can recover the samples.
        rows = [
            {"text": "row a", "meta": 1},
            {"text": "row b", "meta": "two"},
            {"text": "row c", "meta": 3},
        ]
        path = _write_jsonl(tmp_path / "mixed.jsonl", rows)
        samples = get_dataset_samples(path, num_samples=3)
        assert samples == ["row a", "row b", "row c"]


# ---------------------------------------------------------------------------
# get_dataset_dataloader — blending across multiple sources
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal callable tokenizer that mimics the HF tokenizer surface used by the dataloader.

    Tokenizes by character ordinal and left-pads to the longest sample (capped at max_length).
    Avoids a hard dependency on ``transformers`` in the test environment.
    """

    padding_side = "left"
    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=16):
        ids = [[ord(c) % 100 + 1 for c in t][:max_length] for t in texts]
        n = max(len(x) for x in ids)
        input_ids = [[self.pad_token_id] * (n - len(x)) + x for x in ids]
        attention = [[0] * (n - len(x)) + [1] * len(x) for x in ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


@pytest.fixture
def pad_tokenizer():
    return _FakeTokenizer()


class TestGetDatasetDataloaderBlending:
    """``get_dataset_dataloader`` accepts a list of sources and concatenates them."""

    def test_single_jsonl(self, tmp_path, pad_tokenizer):
        pytest.importorskip("datasets")
        path = _write_jsonl(
            tmp_path / "single.jsonl",
            [{"text": f"row {i}"} for i in range(4)],
        )
        loader = get_dataset_dataloader(
            dataset_name=path,
            tokenizer=pad_tokenizer,
            batch_size=2,
            num_samples=4,
            max_sample_length=16,
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape[0] == 2
        assert "attention_mask" in batches[0]

    def test_list_of_jsonl_blends(self, tmp_path, pad_tokenizer):
        """Two local JSONL files concatenated into a single dataloader."""
        pytest.importorskip("datasets")
        a = _write_jsonl(tmp_path / "a.jsonl", [{"text": f"a{i}"} for i in range(3)])
        b = _write_jsonl(tmp_path / "b.jsonl", [{"text": f"b{i}"} for i in range(2)])

        loader = get_dataset_dataloader(
            dataset_name=[a, b],
            tokenizer=pad_tokenizer,
            batch_size=5,
            num_samples=[3, 2],
            max_sample_length=16,
        )
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape[0] == 5

    def test_mixed_formats_blended(self, tmp_path, pad_tokenizer):
        """Mixing a text-column JSONL with a prompt/completion JSONL — both should flow."""
        pytest.importorskip("datasets")
        plain = _write_jsonl(tmp_path / "plain.jsonl", [{"text": "hello"}])
        pc = _write_jsonl(tmp_path / "pc.jsonl", [{"prompt": "Q?", "completion": "A."}])

        loader = get_dataset_dataloader(
            dataset_name=[plain, pc],
            tokenizer=pad_tokenizer,
            batch_size=2,
            num_samples=[1, 1],
            max_sample_length=16,
        )
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape[0] == 2

    def test_length_mismatch_raises(self, tmp_path, pad_tokenizer):
        """``dataset_name`` and ``num_samples`` lists must align."""
        pytest.importorskip("datasets")
        a = _write_jsonl(tmp_path / "a.jsonl", [{"text": "x"}])
        b = _write_jsonl(tmp_path / "b.jsonl", [{"text": "y"}])
        with pytest.raises(AssertionError, match="same length"):
            get_dataset_dataloader(
                dataset_name=[a, b],
                tokenizer=pad_tokenizer,
                num_samples=[1],
                max_sample_length=16,
            )


# ---------------------------------------------------------------------------
# Live HF dataset round-trips. ``hf-internal-testing/dataset_with_data_files``
# is a 10-row x {train,test} fixture maintained by HF for their own CI — tiny
# enough to download in a unit test and stable across releases.
# ---------------------------------------------------------------------------

_HF_TINY = "hf-internal-testing/dataset_with_data_files"  # train, test splits, ``text`` col


def _hf_dump_to_jsonl(name: str, split: str, path) -> str:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    ds.to_json(str(path), lines=True)
    return str(path)


class TestHfTinyDataset:
    """End-to-end coverage with a real (tiny) HF dataset."""

    def test_load_single_split_directly(self):
        pytest.importorskip("datasets")
        samples = get_dataset_samples(_HF_TINY, num_samples=4, split="train")
        assert len(samples) == 4
        assert all(isinstance(s, str) and s for s in samples)

    def test_load_multiple_splits_directly(self):
        """``split=["train", "test"]`` divides ``num_samples`` across both splits."""
        pytest.importorskip("datasets")
        samples = get_dataset_samples(_HF_TINY, num_samples=6, split=["train", "test"])
        assert len(samples) == 6
        # Default per-split is num_samples // n + remainder; for 6/2 → 3 from each.
        # We can't assert exact origin without re-reading, but both splits should
        # contribute, which we'll confirm by comparing against direct loads below.
        train_only = set(get_dataset_samples(_HF_TINY, num_samples=10, split="train"))
        test_only = set(get_dataset_samples(_HF_TINY, num_samples=10, split="test"))
        assert any(s in train_only for s in samples)
        assert any(s in test_only for s in samples)

    def test_default_split_is_train(self):
        pytest.importorskip("datasets")
        default_samples = get_dataset_samples(_HF_TINY, num_samples=4)
        train_samples = get_dataset_samples(_HF_TINY, num_samples=4, split="train")
        assert default_samples == train_samples

    def test_download_to_jsonl_then_load(self, tmp_path):
        """Dump the HF dataset to JSONL, then reload it via the local-jsonl path."""
        pytest.importorskip("datasets")
        jsonl_path = _hf_dump_to_jsonl(_HF_TINY, "train", tmp_path / "train.jsonl")
        from_jsonl = get_dataset_samples(jsonl_path, num_samples=10)
        from_hf = get_dataset_samples(_HF_TINY, num_samples=10, split="train")
        assert from_jsonl == from_hf

    def test_dataloader_blending_two_hf_datasets(self, pad_tokenizer):
        """Two HF datasets concatenated via ``get_dataset_dataloader``."""
        pytest.importorskip("datasets")
        loader = get_dataset_dataloader(
            dataset_name=[_HF_TINY, "hf-internal-testing/multi_dir_dataset"],
            tokenizer=pad_tokenizer,
            batch_size=4,
            num_samples=[3, 1],
            max_sample_length=16,
        )
        batches = list(loader)
        assert sum(b["input_ids"].shape[0] for b in batches) == 4

    def test_dataloader_mixing_hf_and_local_jsonl(self, tmp_path, pad_tokenizer):
        """Live HF dataset blended with a local synthetic JSONL file."""
        pytest.importorskip("datasets")
        local = _write_jsonl(tmp_path / "local.jsonl", [{"text": f"local {i}"} for i in range(2)])
        loader = get_dataset_dataloader(
            dataset_name=[_HF_TINY, local],
            tokenizer=pad_tokenizer,
            batch_size=5,
            num_samples=[3, 2],
            max_sample_length=16,
        )
        batches = list(loader)
        assert sum(b["input_ids"].shape[0] for b in batches) == 5
