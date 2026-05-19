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

"""Unit tests for bypass-distillation dataloader utilities.

Covers the pure-Python branches of ``utils/data/dataloaders.py`` that don't
need a real tokenizer / GPU / distributed init: the validation-split
auto-detect rules, the ``num_workers`` guard rail, the dataset-loader
delegators, the ``Printer`` fake accelerator, and the small numeric helpers
(``create_padded_tensor``, ``realize_dataset_in_memory``, ``collate_none_fn``).
"""

import datasets
import pytest
import torch
from datasets import Dataset, DatasetDict

import modelopt.torch.puzzletron.utils.data.dataset as dataset_module
import modelopt.torch.puzzletron.utils.data.dataloaders as dl
from modelopt.torch.puzzletron.utils.data.dataloaders import (
    Printer,
    collate_fn_with_none_support,
    collate_none_fn,
    create_padded_tensor,
    create_train_dataloader,
    create_validation_dataloader,
    load_from_disk_fn,
    load_streaming_fn,
    realize_dataset_in_memory,
)
from modelopt.torch.puzzletron.utils.data.dataset import ConstantLengthDataset

# ---------------------------------------------------------------------------
# realize_dataset_in_memory: pure list materialisation with optional cap
# ---------------------------------------------------------------------------


def test_realize_dataset_in_memory_full():
    items = [{"a": 1}, {"a": 2}, {"a": 3}]
    out = realize_dataset_in_memory(iter(items), eval_samples=None)
    assert out == items


def test_realize_dataset_in_memory_capped():
    items = [{"a": 1}, {"a": 2}, {"a": 3}]
    out = realize_dataset_in_memory(iter(items), eval_samples=2)
    assert out == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# create_padded_tensor: identity, 1D pad, 2D pad with non-zero pad value
# ---------------------------------------------------------------------------


def test_create_padded_tensor_identity():
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = create_padded_tensor(t, desired_shape=(2, 3))
    assert out is t  # short-circuit, no copy


def test_create_padded_tensor_pads_1d_with_default_zero():
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    out = create_padded_tensor(t, desired_shape=(5,))
    assert out.tolist() == [1, 2, 3, 0, 0]
    assert out.dtype == torch.int32


def test_create_padded_tensor_pads_2d_with_custom_value():
    t = torch.tensor([[1.0, 2.0]])
    out = create_padded_tensor(t, desired_shape=(2, 3), padding_value=-100.0)
    assert out.tolist() == [[1.0, 2.0, -100.0], [-100.0, -100.0, -100.0]]


# ---------------------------------------------------------------------------
# Collate helpers: None-aware default collator
# ---------------------------------------------------------------------------


def test_collate_none_fn_returns_none():
    assert collate_none_fn([None, None]) is None
    assert collate_none_fn([1, 2, 3]) is None  # unconditional


def test_collate_fn_with_none_support_passes_none_through():
    """A label tensor of None should not be coerced to ``[None, None]`` — the
    bypass val loop expects a single ``None`` so it can short-circuit loss
    computation. This pins the ``type(None) -> collate_none_fn`` registration."""
    batch = [{"x": torch.tensor([1.0]), "y": None}, {"x": torch.tensor([2.0]), "y": None}]
    out = collate_fn_with_none_support(batch)
    assert out["y"] is None
    assert torch.equal(out["x"], torch.tensor([[1.0], [2.0]]))


# ---------------------------------------------------------------------------
# Printer: degenerate "main process" stand-in for Accelerator
# ---------------------------------------------------------------------------


def test_printer_attributes_match_main_process_contract():
    assert Printer.is_main_process is True
    assert Printer.process_index is None
    Printer.print("hello world")  # must not raise


# ---------------------------------------------------------------------------
# load_from_disk_fn / load_streaming_fn: thin wrappers around datasets.*
# ---------------------------------------------------------------------------


def test_load_from_disk_fn_delegates_to_datasets(monkeypatch):
    captured = {}

    def fake_load_from_disk(path, keep_in_memory=False):
        captured["path"] = path
        captured["keep_in_memory"] = keep_in_memory
        return "sentinel"

    monkeypatch.setattr(datasets, "load_from_disk", fake_load_from_disk)
    out = load_from_disk_fn("/some/path", content_field="conversation", keep_in_memory=True)
    assert out == "sentinel"
    assert captured == {"path": "/some/path", "keep_in_memory": True}


def test_load_streaming_fn_uses_streaming_with_features(monkeypatch):
    """``load_streaming_fn`` must request streaming and pin the content field's
    feature schema — without ``features=`` HuggingFace would auto-infer types
    per-shard, which has caused bypass jobs to crash on schema drift in the past.
    """
    captured = {}

    def fake_load_dataset(path, streaming, features, keep_in_memory):
        captured["path"] = path
        captured["streaming"] = streaming
        captured["features"] = features
        captured["keep_in_memory"] = keep_in_memory
        return "stream-sentinel"

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    out = load_streaming_fn("hf-org/dataset", content_field="text", keep_in_memory=False)
    assert out == "stream-sentinel"
    assert captured["path"] == "hf-org/dataset"
    assert captured["streaming"] is True
    assert captured["keep_in_memory"] is False
    # features must be a Features object keyed by the requested content_field
    # with a string Value — schema-drift protection is the whole point of this fn.
    assert isinstance(captured["features"], datasets.Features)
    assert "text" in captured["features"]
    assert captured["features"]["text"].dtype == "string"


# ---------------------------------------------------------------------------
# create_train_dataloader: ``num_workers > 0`` is a configuration error
# ---------------------------------------------------------------------------


def test_create_train_dataloader_rejects_num_workers_gt_zero():
    """ConstantLengthDataset doesn't shard work via ``get_worker_info`` — every
    worker would emit the same samples. The guard fires before tokenizer or
    dataset are touched, so bare-bones args are enough."""
    with pytest.raises(ValueError, match="num_workers"):
        create_train_dataloader(
            seed=0,
            tokenizer=None,
            block_size=8,
            dataset_path={"train": []},
            content_field="text",
            fim_rate=0.0,
            fim_spm_rate=0.0,
            micro_batch_size=1,
            num_workers=2,
        )


class _NoChatTemplateTokenizer:
    eos_token_id = 1
    bos_token_id = None

    def __init__(self):
        self.seen_texts = None
        self.vocab = {}

    def __call__(self, texts, truncation=False):
        self.seen_texts = texts
        return {"input_ids": [[0] for _ in texts]}


class _ConversationDataset:
    column_names = ("text",)

    def __iter__(self):
        yield {
            "text": [
                {"role": "user", "content": {"text": "hello"}},
                {"role": "assistant", "content": "world"},
            ]
        }


class _StructuredContentDataset:
    column_names = ("text",)

    def __iter__(self):
        yield {
            "text": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": {"value": 3}},
            ]
        }


def test_constant_length_dataset_no_chat_template_adds_role_tags_and_warns_once(monkeypatch):
    monkeypatch.setattr(dataset_module, "_CHAT_TEMPLATE_FALLBACK_WARNING_EMITTED", False)
    tokenizer = _NoChatTemplateTokenizer()
    dataset = ConstantLengthDataset(
        tokenizer,
        _ConversationDataset(),
        infinite=False,
        seq_length=2,
        num_of_sequences=1,
        chars_per_token=100,
        content_field="text",
        fim_rate=0.0,
        fim_spm_rate=0.0,
        label_shift=False,
    )

    with pytest.warns(UserWarning, match="no chat_template"):
        realized = list(dataset)

    assert tokenizer.seen_texts == ["user: hello\nassistant: world"]
    assert len(realized) == 1
    assert torch.equal(realized[0]["input_ids"], torch.tensor([0, 1]))
    assert torch.equal(realized[0]["targets"], torch.tensor([0, 1]))


def test_constant_length_dataset_no_chat_template_rejects_unknown_structured_content(monkeypatch):
    monkeypatch.setattr(dataset_module, "_CHAT_TEMPLATE_FALLBACK_WARNING_EMITTED", False)
    dataset = ConstantLengthDataset(
        _NoChatTemplateTokenizer(),
        _StructuredContentDataset(),
        infinite=False,
        seq_length=2,
        num_of_sequences=1,
        chars_per_token=100,
        content_field="text",
        fim_rate=0.0,
        fim_spm_rate=0.0,
        label_shift=False,
    )

    with pytest.raises(ValueError, match="Unsupported structured message content"):
        list(dataset)


# ---------------------------------------------------------------------------
# create_validation_dataloader: split auto-detect + explicit override
# ---------------------------------------------------------------------------


class _FakeConstantLengthDataset:
    """Stub for ``ConstantLengthDataset`` that records its ``dataset`` arg.

    Yields one trivial item so ``realize_dataset_in_memory`` can iterate over
    it without touching a tokenizer.
    """

    last_dataset = None  # class-level capture so tests can read after construction

    def __init__(self, tokenizer, dataset, **kwargs):
        type(self).last_dataset = dataset
        self._dataset = dataset

    def __iter__(self):
        yield {"input_ids": torch.tensor([0])}


@pytest.fixture
def patched_dataloader(monkeypatch):
    """Replace the heavy bits inside ``create_validation_dataloader`` so the
    function exercises only its pure split-selection logic + DataLoader build."""
    monkeypatch.setattr(dl, "ConstantLengthDataset", _FakeConstantLengthDataset)
    # Force a tiny in-memory list so we don't drain a real iterable.
    monkeypatch.setattr(
        dl,
        "realize_dataset_in_memory",
        lambda dataset, eval_samples: [{"input_ids": torch.tensor([0])}],
    )
    _FakeConstantLengthDataset.last_dataset = None
    return _FakeConstantLengthDataset


def _make_dict_dataset(splits: dict[str, list]) -> DatasetDict:
    return DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()})


def _kwargs():
    return {
        "accelerator": None,  # → Printer (single-process path)
        "seed": 0,
        "tokenizer": None,
        "block_size": 4,
        "content_field": "text",
        "fim_rate": 0.0,
        "fim_spm_rate": 0.0,
        "micro_batch_size": 1,
    }


def test_validation_split_auto_picks_validation_when_present(patched_dataloader):
    dd = _make_dict_dataset({"train": [{"text": "t"}], "validation": [{"text": "v"}]})
    create_validation_dataloader(dataset=dd, dataset_name="__auto__", **_kwargs())
    # The "validation" split must have been the one passed to ConstantLengthDataset.
    assert patched_dataloader.last_dataset is dd["validation"]


def test_validation_split_auto_falls_back_to_test_when_no_val(patched_dataloader):
    dd = _make_dict_dataset({"train": [{"text": "t"}], "test": [{"text": "te"}]})
    create_validation_dataloader(dataset=dd, dataset_name="__auto__", **_kwargs())
    assert patched_dataloader.last_dataset is dd["test"]


def test_validation_split_auto_prefers_val_over_test(patched_dataloader):
    """If both ``validation`` and ``test`` exist, the val* prefix must win —
    bypass relies on this to score against held-out data, not test data."""
    dd = _make_dict_dataset(
        {"train": [{"text": "t"}], "validation": [{"text": "v"}], "test": [{"text": "te"}]}
    )
    create_validation_dataloader(dataset=dd, dataset_name="__auto__", **_kwargs())
    assert patched_dataloader.last_dataset is dd["validation"]


def test_validation_split_auto_assertion_on_multiple_val_options(patched_dataloader):
    """Ambiguity must fail loudly — silently picking one would be a footgun."""
    dd = _make_dict_dataset({"validation": [{"text": "a"}], "valtest": [{"text": "b"}]})
    with pytest.raises(AssertionError, match="exactly one validation split"):
        create_validation_dataloader(dataset=dd, dataset_name="__auto__", **_kwargs())


def test_validation_split_auto_assertion_on_no_val_or_test(patched_dataloader):
    dd = _make_dict_dataset({"train": [{"text": "t"}], "extra": [{"text": "e"}]})
    with pytest.raises(AssertionError, match="exactly one validation split"):
        create_validation_dataloader(dataset=dd, dataset_name="__auto__", **_kwargs())


def test_validation_split_explicit_override_bypasses_auto(patched_dataloader):
    """Explicit ``dataset_name`` must skip the auto-detect, even when the
    chosen name doesn't match val* / test* prefixes."""
    dd = _make_dict_dataset({"my_eval": [{"text": "x"}]})
    create_validation_dataloader(dataset=dd, dataset_name="my_eval", **_kwargs())
    assert patched_dataloader.last_dataset is dd["my_eval"]
