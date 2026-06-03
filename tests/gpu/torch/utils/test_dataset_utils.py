# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dataset tests that reach the HuggingFace Hub.

These live under ``tests/gpu`` (networked infra) rather than ``tests/unit``,
which is kept hermetic with toy local fixtures.
"""

import json

import pytest
from datasets import load_dataset
from huggingface_hub import get_token

from modelopt.torch.utils.dataset_utils import (
    SUPPORTED_DATASET_CONFIG,
    get_dataset_dataloader,
    get_dataset_samples,
)


def _write_jsonl(path, rows):
    """Write a list of dicts to *path* as JSONL. Returns the path as ``str``."""
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return str(path)


_NEW_NEMOTRON_KEYS = [
    "nemotron-sft-instruction-following-chat-v2",
    "nemotron-science-v1",
    "nemotron-competitive-programming-v1",
    "nemotron-sft-agentic-v2",
    "nemotron-math-v2",
    "nemotron-sft-swe-v2",
    "nemotron-sft-multilingual-v1",
]


@pytest.mark.parametrize("dataset_key", _NEW_NEMOTRON_KEYS)
def test_new_nemotron_registry_shape(dataset_key):
    """Shape check on the 7 newly registered nvidia/Nemotron-* entries.

    Complements the gated smoke test below — catches typos in dataset paths or
    split names even when the runner has no HF credentials.
    """
    assert dataset_key in SUPPORTED_DATASET_CONFIG
    entry = SUPPORTED_DATASET_CONFIG[dataset_key]
    config = entry["config"]
    assert config["path"].startswith("nvidia/Nemotron-")
    splits = config["split"]
    assert isinstance(splits, list) and splits
    assert all(isinstance(s, str) and s for s in splits)
    assert len(set(splits)) == len(splits)
    assert callable(entry["preprocess"])
    assert entry["chat_key"] == "messages"


@pytest.mark.parametrize(
    "dataset_key",
    [
        # nvidia/Nemotron-SFT-Agentic-v2's ``search`` split currently fails the HF datasets
        # JSON schema cast (upstream column drift), so skip its live download smoke; the
        # registry shape check above still validates its configuration.
        pytest.param(k, marks=pytest.mark.skip(reason="upstream search-split schema cast failure"))
        if k == "nemotron-sft-agentic-v2"
        else k
        for k in _NEW_NEMOTRON_KEYS
    ],
)
def test_get_dataset_samples_new_nemotron(dataset_key):
    """Smoke-test the 7 newly registered nvidia/Nemotron-* calibration datasets.

    Skipped when no HF token is available because these datasets live behind the HF Hub.
    ``huggingface_hub.get_token()`` covers both the ``HF_TOKEN`` env var and tokens
    cached by ``hf auth login``.
    """
    if not get_token():
        pytest.skip(
            "No HF token (env HF_TOKEN or `hf auth login`); skipping gated Nemotron smoke test"
        )

    samples = get_dataset_samples(dataset_key, num_samples=2)
    assert isinstance(samples, list)
    assert len(samples) == 2
    assert all(isinstance(s, str) and len(s) > 0 for s in samples)


# Live HF dataset round-trips. ``hf-internal-testing/dataset_with_data_files`` is a
# 10-row x {train,test} fixture maintained by HF for their own CI — tiny enough to
# download in a test and stable across releases, and ungated (no HF token needed).
_HF_TINY = "hf-internal-testing/dataset_with_data_files"  # train, test splits, ``text`` col


def _hf_dump_to_jsonl(name: str, split: str, path) -> str:
    ds = load_dataset(name, split=split)
    ds.to_json(str(path), lines=True)
    return str(path)


class TestHfTinyDataset:
    """End-to-end coverage of the Hub-download branch with a real (tiny) HF dataset."""

    def test_load_single_split_directly(self):
        samples = get_dataset_samples(_HF_TINY, num_samples=4, split="train")
        assert len(samples) == 4
        assert all(isinstance(s, str) and s for s in samples)

    def test_load_multiple_splits_directly(self):
        """``split=["train", "test"]`` divides ``num_samples`` across both splits."""
        samples = get_dataset_samples(_HF_TINY, num_samples=6, split=["train", "test"])
        assert len(samples) == 6
        # Both splits should contribute; confirm by comparing against direct loads.
        train_only = set(get_dataset_samples(_HF_TINY, num_samples=10, split="train"))
        test_only = set(get_dataset_samples(_HF_TINY, num_samples=10, split="test"))
        assert any(s in train_only for s in samples)
        assert any(s in test_only for s in samples)

    def test_default_split_is_train(self):
        default_samples = get_dataset_samples(_HF_TINY, num_samples=4)
        train_samples = get_dataset_samples(_HF_TINY, num_samples=4, split="train")
        assert default_samples == train_samples

    def test_download_to_jsonl_then_load(self, tmp_path):
        """Dump the HF dataset to JSONL, then reload it via the local-jsonl path."""
        jsonl_path = _hf_dump_to_jsonl(_HF_TINY, "train", tmp_path / "train.jsonl")
        from_jsonl = get_dataset_samples(jsonl_path, num_samples=10)
        from_hf = get_dataset_samples(_HF_TINY, num_samples=10, split="train")
        assert from_jsonl == from_hf

    def test_dataloader_blending_two_hf_datasets(self, tiny_tokenizer):
        """Two HF datasets concatenated via ``get_dataset_dataloader``."""
        loader = get_dataset_dataloader(
            dataset_name=[_HF_TINY, "hf-internal-testing/multi_dir_dataset"],
            tokenizer=tiny_tokenizer,
            batch_size=4,
            num_samples=[3, 1],
            max_sample_length=16,
        )
        batches = list(loader)
        assert sum(b["input_ids"].shape[0] for b in batches) == 4

    def test_dataloader_mixing_hf_and_local_jsonl(self, tmp_path, tiny_tokenizer):
        """Live HF dataset blended with a local synthetic JSONL file."""
        local = _write_jsonl(tmp_path / "local.jsonl", [{"text": f"local {i}"} for i in range(2)])
        loader = get_dataset_dataloader(
            dataset_name=[_HF_TINY, local],
            tokenizer=tiny_tokenizer,
            batch_size=5,
            num_samples=[3, 2],
            max_sample_length=16,
        )
        batches = list(loader)
        assert sum(b["input_ids"].shape[0] for b in batches) == 5
