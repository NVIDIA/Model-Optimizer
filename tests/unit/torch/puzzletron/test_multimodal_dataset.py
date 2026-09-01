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

from __future__ import annotations

import pytest
import torch
from PIL import Image

from modelopt.torch.puzzletron.dataset.batch import DataLayout, Modality
from modelopt.torch.puzzletron.dataset.multimodal import (
    INTERSYN_MULTI_DATASET,
    INTERSYN_MULTI_REVISION,
    INTERSYN_SINGLE_DATASET,
    INTERSYN_SINGLE_REVISION,
    batch_from_automodel,
    load_materialized_conversation_dataset,
    load_materialized_intersyn_subset,
    materialize_intersyn_subset,
    materialize_normalized_intersyn_samples,
    normalize_intersyn_multi,
    normalize_intersyn_single,
)


def test_single_turn_adapter_uses_real_image_as_input_and_text_as_target():
    row = {
        "id": "0001",
        "topic": "lake",
        "human": "What is shown?",
        "gpt": "A glowing lake.",
        "caption": "Blue light on water.",
        "image": "image-1",
    }

    sample = normalize_intersyn_single(row)

    assert sample["source"]["revision"] == INTERSYN_SINGLE_REVISION
    assert sample["source"]["row_id"] == "0001"
    assert sample["image_count"] == 1
    assert sample["conversation"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image-1"},
                {"type": "text", "text": "What is shown?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "A glowing lake."}],
        },
    ]


def test_multi_turn_adapter_preserves_five_turn_and_image_order():
    row = {"id": "multi-7", "topic": "journey"}
    for turn in range(1, 6):
        row[f"human{turn}"] = f"question-{turn}"
        row[f"gpt{turn}"] = f"answer-{turn}"
        row[f"caption{turn}"] = f"caption-{turn}"
        row[f"image{turn}"] = f"image-{turn}"

    sample = normalize_intersyn_multi(row)

    assert sample["source"]["revision"] == INTERSYN_MULTI_REVISION
    assert sample["image_count"] == 5
    assert [
        message["content"][0]["image"]
        for message in sample["conversation"]
        if message["role"] == "user"
    ] == [f"image-{turn}" for turn in range(1, 6)]
    assert [message["role"] for message in sample["conversation"]] == [
        "user",
        "assistant",
    ] * 5


def test_batch_from_automodel_preserves_packing_mrope_and_media_metadata():
    input_ids = torch.arange(12, dtype=torch.long).reshape(1, 12)
    labels = input_ids.clone()
    labels[:, :3] = -100
    collated = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.stack((input_ids, input_ids + 10, input_ids + 20)),
        "cu_seqlens": torch.tensor([0, 5, 12], dtype=torch.int32),
        "seq_idx": torch.tensor([[0] * 5 + [1] * 7], dtype=torch.int32),
        "pixel_values": torch.ones(3, 8),
        "image_grid_thw": torch.tensor([[1, 1, 1]] * 3),
        "n_images_per_sample": torch.tensor([3], dtype=torch.int32),
    }

    batch = batch_from_automodel(
        collated,
        sample_ids=("pack-0",),
        source_metadata={"dataset": "mixed", "revision": "pinned"},
        layout=DataLayout.PACKED_VARLEN,
    )

    assert batch.modality is Modality.MULTIMODAL
    assert batch.model_kwargs["position_ids"].shape == (3, 1, 12)
    assert batch.sequence.global_cu_seqlens.tolist() == [0, 5, 12]
    assert batch.sequence.max_seqlen == 7
    assert batch.sequence.media_counts.tolist() == [3]
    assert batch.ce_mask.tolist() == [[False] * 3 + [True] * 9]
    assert torch.equal(batch.kd_mask, batch.ce_mask)
    assert batch.hidden_mask.all()
    assert "labels" not in batch.model_kwargs
    assert "cu_seqlens" in batch.model_kwargs


def test_batch_from_automodel_derives_sequence_ids_from_legacy_cu_seqlens():
    input_ids = torch.arange(12, dtype=torch.long).reshape(1, 12)

    batch = batch_from_automodel(
        {
            "input_ids": input_ids,
            "targets": input_ids.clone(),
            "cu_seqlens": torch.tensor([[0, 3, 7, 12]], dtype=torch.int32),
        },
        sample_ids=("pack-0",),
        source_metadata={"dataset": "fixture", "revision": "1"},
        layout=DataLayout.PACKED_VARLEN,
    )

    assert batch.sequence.global_cu_seqlens.tolist() == [0, 3, 7, 12]
    assert batch.sequence.seq_ids.tolist() == [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
    assert batch.sequence.sample_offsets == ((0, 3), (3, 7), (7, 12))


def test_batch_from_automodel_recovers_boundaries_from_neat_packing_ids():
    input_ids = torch.arange(8, dtype=torch.long).reshape(1, 8)
    batch = batch_from_automodel(
        {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0]]),
            "_packed_seq_ids": torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0]]),
            "position_ids": torch.arange(8).reshape(1, 8),
        },
        sample_ids=("pack",),
        source_metadata={"dataset": "fixture", "revision": "1"},
        layout=DataLayout.PACKED_VARLEN,
    )

    assert batch.sequence.global_cu_seqlens.tolist() == [0, 3, 6]
    assert batch.sequence.sample_offsets == ((0, 3), (3, 6))
    assert batch.sequence.seq_ids.tolist() == [[0, 0, 0, 1, 1, 1, -1, -1]]
    assert batch.hidden_mask.tolist() == [[True] * 6 + [False, False]]
    assert "_packed_seq_ids" not in batch.model_kwargs


def test_batch_from_automodel_excludes_padding_from_every_canonical_mask():
    input_ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
    labels = input_ids.clone()
    loss_mask = torch.ones_like(input_ids)

    batch = batch_from_automodel(
        {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        },
        sample_ids=("row-0", "row-1"),
        source_metadata={"dataset": "fixture", "revision": "1"},
        layout=DataLayout.PADDED_VARLEN,
    )

    expected = torch.tensor([[True, True, True, False], [True, True, False, False]])
    assert torch.equal(batch.hidden_mask, expected)
    assert torch.equal(batch.ce_mask, expected)
    assert torch.equal(batch.kd_mask, expected)
    assert batch.labels.tolist() == [[0, 1, 2, -100], [4, 5, -100, -100]]


def test_batch_from_automodel_honors_padding_mask_without_attention_mask():
    input_ids = torch.arange(4, dtype=torch.long).reshape(1, 4)

    batch = batch_from_automodel(
        {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "padding_mask": torch.tensor([[False, False, True, True]]),
        },
        sample_ids=("row",),
        source_metadata={"dataset": "fixture", "revision": "1"},
        layout=DataLayout.PADDED_VARLEN,
    )

    assert batch.hidden_mask.tolist() == [[True, True, False, False]]
    assert batch.labels.tolist() == [[0, 1, -100, -100]]


def test_batch_from_automodel_globalizes_multiple_packed_rows_and_slices_them():
    input_ids = torch.arange(16, dtype=torch.long).reshape(2, 8)
    batch = batch_from_automodel(
        {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]),
            "_packed_seq_ids": torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]),
            "position_ids": torch.arange(8).repeat(2, 1),
        },
        sample_ids=("pack-0", "pack-1"),
        source_metadata={"dataset": "fixture", "revision": "1"},
        layout=DataLayout.PACKED_VARLEN,
    )

    assert batch.sequence.global_cu_seqlens.tolist() == [0, 3, 6, 10]
    assert batch.sequence.seq_ids.tolist() == [
        [0, 0, 0, 1, 1, 1, -1, -1],
        [2, 2, 2, 2, -1, -1, -1, -1],
    ]
    assert batch.sequence.max_seqlen == 4

    second_row = batch.dp_slice(dp_rank=1, dp_size=2)
    assert second_row.sequence.global_cu_seqlens.tolist() == [0, 4]
    assert second_row.sequence.seq_ids.tolist() == [[0, 0, 0, 0, -1, -1, -1, -1]]

    local = batch.cp_partition(torch.tensor([0, 1, 3, 4]), cp_rank=0, cp_size=2)
    assert local.sequence.local_cu_seqlens.tolist() == [0, 2, 4, 7]
    assert local.sequence.max_seqlen == 3


def test_multi_turn_adapter_rejects_rows_without_two_images():
    row = {
        "id": "bad",
        "human1": "one",
        "gpt1": "answer",
        "image1": "image-1",
    }

    with pytest.raises(ValueError, match="at least two images"):
        normalize_intersyn_multi(row)


def test_materialized_subset_is_offline_reusable_and_hashes_images(tmp_path):
    single = normalize_intersyn_single(
        {
            "id": "single",
            "human": "question",
            "gpt": "answer",
            "image": Image.new("RGB", (4, 4), color=(255, 0, 0)),
        }
    )
    multi_row = {"id": "multi"}
    for turn in range(1, 3):
        multi_row[f"human{turn}"] = f"question-{turn}"
        multi_row[f"gpt{turn}"] = f"answer-{turn}"
        multi_row[f"image{turn}"] = Image.new("RGB", (4, 4), color=(0, turn, 0))
    multi = normalize_intersyn_multi(multi_row)

    manifest = materialize_normalized_intersyn_samples([single, multi], tmp_path)
    loaded = load_materialized_intersyn_subset(tmp_path)

    assert manifest["sample_count"] == 2
    assert manifest["image_count"] == 3
    assert len(manifest["images"]) == 3
    assert all(len(image["sha256"]) == 64 for image in manifest["images"])
    assert len(loaded) == 2
    image_items = [
        item
        for sample in loaded
        for message in sample["conversation"]
        for item in message["content"]
        if item["type"] == "image"
    ]
    assert len(image_items) == 3
    assert all(Image.open(item["image"]).size == (4, 4) for item in image_items)


def test_subset_materializer_pins_sources_and_skips_invalid_rows(tmp_path):
    rows = {
        "finyorko/single_turn": [
            {"id": "bad", "human": "q", "gpt": "a", "image": None},
            {
                "id": "single-good",
                "human": "q",
                "gpt": "a",
                "image": Image.new("RGB", (2, 2)),
            },
        ],
        "finyorko/multi-turn": [
            {"id": "bad", "human1": "q", "gpt1": "a", "image1": None},
            {
                "id": "multi-good",
                "human1": "q1",
                "gpt1": "a1",
                "image1": Image.new("RGB", (2, 2)),
                "human2": "q2",
                "gpt2": "a2",
                "image2": Image.new("RGB", (2, 2)),
            },
        ],
    }
    calls = []

    def loader(dataset, *, split, revision, streaming):
        calls.append((dataset, split, revision, streaming))
        return rows[dataset]

    manifest = materialize_intersyn_subset(
        tmp_path,
        rows_per_source=1,
        dataset_loader=loader,
    )
    samples = load_materialized_intersyn_subset(tmp_path)

    assert manifest["sample_count"] == 2
    assert [sample["source"]["row_id"] for sample in samples] == [
        "single-good",
        "multi-good",
    ]
    assert calls == [
        ("finyorko/single_turn", "train", INTERSYN_SINGLE_REVISION, True),
        ("finyorko/multi-turn", "multi", INTERSYN_MULTI_REVISION, True),
    ]


def test_materialized_dataset_factory_accepts_automodel_controls_and_rejects_unknowns(tmp_path):
    samples = [
        normalize_intersyn_single(
            {
                "id": f"sample-{index}",
                "human": "question",
                "gpt": "answer",
                "image": Image.new("RGB", (2, 2)),
            }
        )
        for index in range(2)
    ]
    materialize_normalized_intersyn_samples(samples, tmp_path)

    dataset = load_materialized_conversation_dataset(
        tmp_path,
        num_samples=1,
        seq_length=1024,
        pretokenize=True,
        truncate=False,
        inject_fake_images=False,
        max_length=1536,
    )
    assert len(dataset) == 1
    assert dataset[0]["source"]["row_id"] == "sample-0"

    with pytest.raises(TypeError, match="misspelled_option"):
        load_materialized_conversation_dataset(tmp_path, misspelled_option=True)


def test_materialized_dataset_factory_balances_sources_before_prefix_selection(tmp_path):
    single = [
        normalize_intersyn_single(
            {
                "id": f"single-{index}",
                "human": "question",
                "gpt": "answer",
                "image": Image.new("RGB", (2, 2)),
            }
        )
        for index in range(2)
    ]
    multi = [
        normalize_intersyn_multi(
            {
                "id": f"multi-{index}",
                "human1": "question-1",
                "gpt1": "answer-1",
                "image1": Image.new("RGB", (2, 2)),
                "human2": "question-2",
                "gpt2": "answer-2",
                "image2": Image.new("RGB", (2, 2)),
            }
        )
        for index in range(2)
    ]
    materialize_normalized_intersyn_samples([*single, *multi], tmp_path)

    dataset = load_materialized_conversation_dataset(tmp_path, num_samples=2)

    assert [dataset[index]["source"]["dataset"] for index in range(2)] == [
        INTERSYN_SINGLE_DATASET,
        INTERSYN_MULTI_DATASET,
    ]


def test_materialized_dataset_factory_shuffle_is_seeded(tmp_path):
    samples = [
        normalize_intersyn_single(
            {
                "id": f"sample-{index}",
                "human": "question",
                "gpt": "answer",
                "image": Image.new("RGB", (2, 2)),
            }
        )
        for index in range(8)
    ]
    materialize_normalized_intersyn_samples(samples, tmp_path)

    def order(seed):
        dataset = load_materialized_conversation_dataset(tmp_path, seed=seed, shuffle=True)
        return [dataset[index]["source"]["row_id"] for index in range(len(dataset))]

    assert order(2222) == order(2222)
    assert order(2222) != order(3333)
