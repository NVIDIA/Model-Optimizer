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

"""Tests for bypass-distillation dataloader behavior added by this PR."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.puzzletron.utils.data.dataloaders as dl
import modelopt.torch.puzzletron.utils.data.dataset as dataset_module
from modelopt.torch.puzzletron.utils.data.dataloaders import create_train_dataloader
from modelopt.torch.puzzletron.utils.data.dataset import ConstantLengthDataset


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


class _FakeTrainConstantLengthDataset:
    last_args = None
    last_kwargs = None

    def __init__(self, *args, **kwargs):
        type(self).last_args = args
        type(self).last_kwargs = kwargs


class _FakeTrainSplit:
    def __init__(self):
        self.shuffle_calls = []

    def shuffle(self, **kwargs):
        self.shuffle_calls.append(kwargs)
        return self


@pytest.fixture
def patched_train_dataloader(monkeypatch):
    captured = {}

    def fake_dataloader(dataset, batch_size, pin_memory, num_workers):
        captured["dataset"] = dataset
        captured["batch_size"] = batch_size
        captured["pin_memory"] = pin_memory
        captured["num_workers"] = num_workers
        return SimpleNamespace(dataset=dataset)

    _FakeTrainConstantLengthDataset.last_args = None
    _FakeTrainConstantLengthDataset.last_kwargs = None
    monkeypatch.setattr(dl, "ConstantLengthDataset", _FakeTrainConstantLengthDataset)
    monkeypatch.setattr(dl, "DataLoader", fake_dataloader)
    return captured


def test_create_train_dataloader_builds_constant_length_dataset_from_loaded_split(
    patched_train_dataloader,
):
    train_split = _FakeTrainSplit()
    load_calls = []

    def fake_load_dataset(dataset_path, content_field, keep_in_memory):
        load_calls.append((dataset_path, content_field, keep_in_memory))
        return {"custom_train": train_split}

    tokenizer = object()
    out = create_train_dataloader(
        seed=7,
        tokenizer=tokenizer,
        block_size=16,
        dataset_path="/tmp/train",
        content_field="conversation",
        fim_rate=0.25,
        fim_spm_rate=0.75,
        micro_batch_size=3,
        load_dataset_fn=fake_load_dataset,
        dataset_name="custom_train",
        keep_in_memory=True,
        shuffle_seed=123,
        source_datasets_to_discard=("bad-source",),
        bos_rate=0.5,
    )

    assert out.dataset is patched_train_dataloader["dataset"]
    assert load_calls == [("/tmp/train", "conversation", True)]
    assert train_split.shuffle_calls == [{"seed": 123, "keep_in_memory": True}]
    assert _FakeTrainConstantLengthDataset.last_args == (tokenizer, train_split)
    assert _FakeTrainConstantLengthDataset.last_kwargs == {
        "infinite": True,
        "seq_length": 16,
        "content_field": "conversation",
        "fim_rate": 0.25,
        "fim_spm_rate": 0.75,
        "seed": 7,
        "source_datasets_to_discard": ("bad-source",),
        "bos_rate": 0.5,
    }
    assert isinstance(patched_train_dataloader["dataset"], _FakeTrainConstantLengthDataset)
    assert patched_train_dataloader["batch_size"] == 3
    assert patched_train_dataloader["pin_memory"] is True
    assert patched_train_dataloader["num_workers"] == 0


def test_create_train_dataloader_streaming_shuffle_omits_keep_in_memory(
    monkeypatch,
    patched_train_dataloader,
):
    class FakeStreamingDataset:
        def __init__(self):
            self.shuffle_seed = None

        def shuffle(self, seed):
            self.shuffle_seed = seed
            return self

    monkeypatch.setattr(dl.datasets, "IterableDataset", FakeStreamingDataset)
    train_split = FakeStreamingDataset()

    create_train_dataloader(
        seed=0,
        tokenizer=object(),
        block_size=8,
        dataset_path={"train": train_split},
        content_field="text",
        fim_rate=0.0,
        fim_spm_rate=0.0,
        micro_batch_size=1,
        load_dataset_fn=lambda *args, **kwargs: pytest.fail("dataset mapping should not load"),
        shuffle_seed=99,
        keep_in_memory=True,
    )

    assert train_split.shuffle_seed == 99
    assert _FakeTrainConstantLengthDataset.last_args[1] is train_split
    assert isinstance(patched_train_dataloader["dataset"], _FakeTrainConstantLengthDataset)


class _NoChatTemplateTokenizer:
    eos_token_id = 1
    bos_token_id = None

    def __init__(self):
        self.seen_texts = None
        self.vocab = {}  # Required by ConstantLengthDataset.get_fim_token_ids.

    def __call__(self, texts, truncation=False):
        self.seen_texts = texts
        return {"input_ids": [[0] for _ in texts]}


class _ChatTemplateTokenizer(_NoChatTemplateTokenizer):
    chat_template = "template"

    def __init__(self):
        super().__init__()
        self.template_messages = None

    def apply_chat_template(self, messages, tokenize=False):
        self.template_messages = messages
        return "templated chat"


class _ConversationDataset:
    column_names = ("text",)

    def __iter__(self):
        yield {
            "text": [
                {"role": "user", "content": {"text": "hello"}},
                {"role": "assistant", "content": "world"},
            ]
        }


class _EmptyConversationDataset:
    column_names = ("text",)

    def __iter__(self):
        yield {"text": []}


def test_constant_length_dataset_formats_conversation_messages(monkeypatch):
    expected_messages = [
        {"role": "user", "content": {"text": "hello"}},
        {"role": "assistant", "content": "world"},
    ]
    for tokenizer, raw_dataset, expected_texts, warning_context in [
        (
            _NoChatTemplateTokenizer(),
            _ConversationDataset(),
            ["user: hello\nassistant: world"],
            pytest.warns(UserWarning, match="no chat_template"),
        ),
        (_ChatTemplateTokenizer(), _ConversationDataset(), ["templated chat"], nullcontext()),
        (_NoChatTemplateTokenizer(), _EmptyConversationDataset(), [""], nullcontext()),
    ]:
        monkeypatch.setattr(dataset_module, "_CHAT_TEMPLATE_FALLBACK_WARNING_EMITTED", False)
        dataset = ConstantLengthDataset(
            tokenizer,
            raw_dataset,
            infinite=False,
            seq_length=2,
            num_of_sequences=1,
            chars_per_token=100,
            content_field="text",
            fim_rate=0.0,
            fim_spm_rate=0.0,
            label_shift=False,
        )

        with warning_context:
            realized = list(dataset)

        assert tokenizer.seen_texts == expected_texts
        assert len(realized) == 1
        if isinstance(tokenizer, _ChatTemplateTokenizer):
            assert tokenizer.template_messages == expected_messages
        else:
            assert torch.equal(realized[0]["input_ids"], torch.tensor([0, 1]))
            assert torch.equal(realized[0]["targets"], torch.tensor([0, 1]))
