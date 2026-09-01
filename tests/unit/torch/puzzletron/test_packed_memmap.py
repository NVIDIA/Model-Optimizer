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

import json

import numpy as np

from modelopt.torch.puzzletron.distillation.dataset import make_puzzletron_llm_dataset
from modelopt.torch.puzzletron.utils.data.dataloaders import create_train_dataloader
from modelopt.torch.puzzletron.utils.data.packed_memmap import PackedTokenMemmapDataset


def _write_cache(path, *, samples: int = 2, sequence_length: int = 8):
    tokens = np.arange(samples * (sequence_length + 1), dtype=np.uint32).reshape(
        samples, sequence_length + 1
    )
    tokens.tofile(path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(
            {
                "status": "complete",
                "num_samples": samples,
                "seq_length": sequence_length,
            }
        )
    )
    return tokens


def test_packed_cache_can_return_a_configured_prefix_length(tmp_path):
    path = tmp_path / "tokens.bin"
    tokens = _write_cache(path)

    dataset = PackedTokenMemmapDataset(path, sequence_length=4)

    sample = dataset[0]
    assert sample["input_ids"].tolist() == tokens[0, :4].tolist()
    assert sample["targets"].tolist() == tokens[0, 1:5].tolist()


def test_train_dataloader_applies_block_size_to_packed_cache(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_cache(path)

    dataloader = create_train_dataloader(
        seed=1,
        tokenizer=None,
        block_size=4,
        dataset_path="unused",
        content_field="unused",
        fim_rate=0.0,
        fim_spm_rate=0.0,
        micro_batch_size=2,
        packed_token_cache_path=path,
    )

    batch = next(iter(dataloader))
    assert batch["input_ids"].shape == (2, 4)
    assert batch["targets"].shape == (2, 4)


def test_global_kd_dataset_applies_sequence_length_to_packed_cache(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_cache(path)

    dataset = make_puzzletron_llm_dataset(
        tokenizer=None,
        dataset_path="unused",
        num_samples=2,
        seq_length=4,
        packed_token_cache_path=str(path),
    )

    sample = next(iter(dataset))
    assert sample["input_ids"].shape == (4,)
    assert sample["labels"].shape == (4,)


def test_global_kd_packed_dataset_shuffle_is_seeded(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_cache(path, samples=8)

    def order(seed, *, shuffle=True):
        dataset = make_puzzletron_llm_dataset(
            tokenizer=None,
            dataset_path="unused",
            num_samples=8,
            seq_length=8,
            seed=seed,
            shuffle=shuffle,
            packed_token_cache_path=str(path),
        )
        return [sample["input_ids"][0].item() for sample in dataset]

    assert order(2222, shuffle=False) == sorted(order(2222, shuffle=False))
    assert order(2222) == order(2222)
    assert order(2222) != order(3333)


def test_global_kd_packed_dataset_shuffles_by_default(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_cache(path, samples=8)

    default_dataset = make_puzzletron_llm_dataset(
        tokenizer=None,
        dataset_path="unused",
        num_samples=8,
        seq_length=8,
        seed=2222,
        packed_token_cache_path=str(path),
    )
    ordered_dataset = make_puzzletron_llm_dataset(
        tokenizer=None,
        dataset_path="unused",
        num_samples=8,
        seq_length=8,
        seed=2222,
        shuffle=False,
        packed_token_cache_path=str(path),
    )

    default_order = [sample["input_ids"][0].item() for sample in default_dataset]
    ordered = [sample["input_ids"][0].item() for sample in ordered_dataset]
    assert default_order != ordered


def test_global_kd_packed_dataset_shards_seeded_order_without_overlap(tmp_path):
    path = tmp_path / "tokens.bin"
    _write_cache(path, samples=8)
    dataset = make_puzzletron_llm_dataset(
        tokenizer=None,
        dataset_path="unused",
        num_samples=8,
        seq_length=8,
        seed=2222,
        shuffle=True,
        packed_token_cache_path=str(path),
    )

    global_order = [sample["input_ids"][0].item() for sample in dataset]
    shard_orders = [
        [sample["input_ids"][0].item() for sample in dataset.shard(2, index)] for index in range(2)
    ]

    assert set(shard_orders[0]).isdisjoint(shard_orders[1])
    assert [value for pair in zip(*shard_orders, strict=True) for value in pair] == global_order


def test_global_kd_non_cache_dataset_respects_shuffle(monkeypatch):
    class FakeDataset:
        def __init__(self):
            self.shuffle_seeds = []

        def shuffle(self, *, seed):
            self.shuffle_seeds.append(seed)
            return self

    loaded = FakeDataset()
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distillation.dataset.load_from_disk",
        lambda _: loaded,
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.distillation.dataset.ConstantLengthDataset",
        lambda **_: [],
    )

    make_puzzletron_llm_dataset(
        tokenizer=None,
        dataset_path="unused",
        num_samples=2,
        seq_length=4,
        seed=2222,
        shuffle=False,
    )
    assert loaded.shuffle_seeds == []

    make_puzzletron_llm_dataset(
        tokenizer=None,
        dataset_path="unused",
        num_samples=2,
        seq_length=4,
        seed=2222,
        shuffle=True,
    )
    assert loaded.shuffle_seeds == [2222]
