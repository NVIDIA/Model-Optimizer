# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from modelopt.torch.puzzletron.dataset.batch import (
    DataLayout,
    Modality,
    PackedSequenceMetadata,
    PuzzletronBatch,
)


def _packed_batch(*, revision: str = "rev-a") -> PuzzletronBatch:
    input_ids = torch.arange(16, dtype=torch.long).reshape(1, 16)
    position_ids = torch.stack(
        (
            input_ids,
            input_ids + 100,
            input_ids + 200,
        ),
        dim=0,
    )
    return PuzzletronBatch(
        model_kwargs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": position_ids,
        },
        labels=input_ids.clone(),
        ce_mask=torch.tensor([[0] * 4 + [1] * 4 + [0] * 4 + [1] * 4], dtype=torch.bool),
        kd_mask=torch.tensor([[0] * 4 + [1] * 4 + [0] * 4 + [1] * 4], dtype=torch.bool),
        hidden_mask=torch.ones_like(input_ids, dtype=torch.bool),
        sequence=PackedSequenceMetadata(
            global_cu_seqlens=torch.tensor([0, 4, 10, 16], dtype=torch.int32),
            max_seqlen=6,
            seq_ids=torch.tensor([[0] * 4 + [1] * 6 + [2] * 6], dtype=torch.int32),
            sample_offsets=((0, 4), (4, 10), (10, 16)),
            media_counts=torch.tensor([3], dtype=torch.int32),
            media_offsets=torch.tensor([0, 3], dtype=torch.int32),
        ),
        sample_ids=("pack-0",),
        source_metadata={
            "dataset": "finyorko/multi-turn",
            "revision": revision,
            "row_ids": ["10", "11", "12"],
            "processor": "qwen3.5",
        },
        modality=Modality.MULTIMODAL,
        layout=DataLayout.PACKED_VARLEN,
    )


def test_fingerprint_covers_source_revision_and_tensor_content():
    first = _packed_batch(revision="rev-a")
    same = _packed_batch(revision="rev-a")
    changed_revision = _packed_batch(revision="rev-b")
    changed_tokens = _packed_batch(revision="rev-a").replace_model_kwargs(
        input_ids=torch.full((1, 16), 7, dtype=torch.long)
    )

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != changed_revision.fingerprint
    assert first.fingerprint != changed_tokens.fingerprint


def test_cp_partition_uses_one_token_index_for_tokens_masks_and_mrope():
    batch = _packed_batch()
    token_indices = torch.tensor([0, 1, 4, 5, 10, 11, 12, 13], dtype=torch.long)

    local = batch.cp_partition(token_indices, cp_rank=0, cp_size=2)

    assert local.model_kwargs["input_ids"].tolist() == [[0, 1, 4, 5, 10, 11, 12, 13]]
    assert local.model_kwargs["position_ids"].shape == (3, 1, 8)
    assert local.model_kwargs["position_ids"][1, 0].tolist() == [100, 101, 104, 105, 110, 111, 112, 113]
    assert local.labels.tolist() == [[0, 1, 4, 5, 10, 11, 12, 13]]
    assert local.hidden_mask.shape == (1, 8)
    assert torch.equal(local.kd_mask, local.ce_mask)
    assert local.sequence.global_cu_seqlens.tolist() == [0, 4, 10, 16]
    assert local.sequence.local_cu_seqlens.tolist() == [0, 2, 4, 8]
    assert local.sequence.max_seqlen == 4
    assert local.sequence.cp_rank == 0
    assert local.sequence.cp_size == 2


def _media_batch() -> PuzzletronBatch:
    input_ids = torch.arange(24, dtype=torch.long).reshape(3, 8)
    return PuzzletronBatch(
        model_kwargs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * 5, dtype=torch.long),
        },
        labels=input_ids.clone(),
        ce_mask=torch.ones_like(input_ids, dtype=torch.bool),
        kd_mask=torch.ones_like(input_ids, dtype=torch.bool),
        hidden_mask=torch.ones_like(input_ids, dtype=torch.bool),
        sequence=PackedSequenceMetadata(
            media_counts=torch.tensor([2, 1, 2], dtype=torch.int32),
            media_offsets=torch.tensor([0, 2, 3, 5], dtype=torch.int32),
        ),
        sample_ids=("a", "b", "c"),
        source_metadata={"dataset": "fixture", "revision": "1"},
        modality=Modality.MULTIMODAL,
        layout=DataLayout.PADDED_VARLEN,
    )


def test_pp_microbatches_split_flat_media_by_per_sample_counts():
    batch = _media_batch()

    first, second = batch.pp_microbatches(2)

    assert first.sample_ids == ("a", "b")
    assert first.model_kwargs["pixel_values"].shape[0] == 3
    assert first.sequence.media_counts.tolist() == [2, 1]
    assert second.sample_ids == ("c",)
    assert second.model_kwargs["pixel_values"].shape[0] == 2
    assert second.sequence.media_counts.tolist() == [2]


def test_pp_padding_adds_masked_rows_without_copying_media():
    padded = _media_batch().pad_batch_to_multiple(2)

    assert padded.batch_size == 4
    assert padded.model_kwargs["pixel_values"].shape[0] == 5
    assert padded.model_kwargs["image_grid_thw"].shape[0] == 5
    assert padded.sequence.media_counts.tolist() == [2, 1, 2, 0]
    assert padded.sequence.media_offsets.tolist() == [0, 2, 3, 5, 5]
    assert padded.model_kwargs["attention_mask"][-1].count_nonzero() == 0
    assert padded.labels[-1].eq(-100).all()
    assert not padded.hidden_mask[-1].any()


def test_pp_padding_preserves_packed_boundaries_and_mrope_layout():
    batch = _packed_batch()

    padded = batch.pad_batch_to_multiple(2)

    assert padded.batch_size == 2
    assert padded.model_kwargs["position_ids"].shape == (3, 2, 16)
    assert padded.sequence.global_cu_seqlens.tolist() == [0, 4, 10, 16]
    assert padded.sequence.seq_ids[-1].eq(-1).all()
    assert not padded.hidden_mask[-1].any()


def test_batch_rejects_supervision_on_padding():
    input_ids = torch.arange(4).reshape(1, 4)

    with pytest.raises(ValueError, match="ce_mask.*subset"):
        PuzzletronBatch(
            model_kwargs={"input_ids": input_ids},
            labels=torch.tensor([[0, 1, -100, -100]]),
            ce_mask=torch.ones_like(input_ids, dtype=torch.bool),
            kd_mask=torch.tensor([[True, True, False, False]]),
            hidden_mask=torch.tensor([[True, True, False, False]]),
            layout=DataLayout.PADDED_VARLEN,
        )


def test_dp_slice_returns_new_batch_with_aligned_media_and_sample_identity():
    batch = _media_batch()

    first = batch.dp_slice(dp_rank=0, dp_size=3)
    second = batch.dp_slice(dp_rank=1, dp_size=3)

    assert first.sample_ids == ("a",)
    assert second.sample_ids == ("b",)
    assert first.sequence.media_counts.tolist() == [2]
    assert second.sequence.media_counts.tolist() == [1]
    assert first.model_kwargs["image_grid_thw"].shape[0] == 2
    assert second.model_kwargs["image_grid_thw"].shape[0] == 1


def test_dp_slice_partitions_whole_samples_from_one_packed_text_row():
    input_ids = torch.arange(12, dtype=torch.long).reshape(1, 12)
    batch = PuzzletronBatch(
        model_kwargs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "cu_seqlens": torch.tensor([0, 3, 6, 8, 12], dtype=torch.int32),
        },
        labels=input_ids.clone(),
        ce_mask=torch.ones_like(input_ids, dtype=torch.bool),
        kd_mask=torch.ones_like(input_ids, dtype=torch.bool),
        hidden_mask=torch.ones_like(input_ids, dtype=torch.bool),
        sequence=PackedSequenceMetadata(
            global_cu_seqlens=torch.tensor([0, 3, 6, 8, 12], dtype=torch.int32),
            max_seqlen=4,
            seq_ids=torch.tensor([[0] * 3 + [1] * 3 + [2] * 2 + [3] * 4]),
            sample_offsets=((0, 3), (3, 6), (6, 8), (8, 12)),
        ),
        sample_ids=("pack-0",),
        source_metadata={"dataset": "fixture", "revision": "1"},
        modality=Modality.TEXT,
        layout=DataLayout.PACKED_VARLEN,
    )

    first = batch.dp_slice(dp_rank=0, dp_size=2)
    second = batch.dp_slice(dp_rank=1, dp_size=2)

    assert first.input_ids.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert second.input_ids.tolist() == [[6, 7, 8, 9, 10, 11]]
    assert first.sequence.seq_ids.tolist() == [[0, 0, 0, 1, 1, 1]]
    assert second.sequence.seq_ids.tolist() == [[0, 0, 1, 1, 1, 1]]
    assert first.sequence.global_cu_seqlens.tolist() == [0, 3, 6]
    assert second.sequence.global_cu_seqlens.tolist() == [0, 2, 6]
    assert first.model_kwargs["cu_seqlens"].tolist() == [0, 3, 6]
    assert second.model_kwargs["cu_seqlens"].tolist() == [0, 2, 6]
    assert first.sequence.sample_offsets == ((0, 3), (3, 6))
    assert second.sequence.sample_offsets == ((0, 2), (2, 6))


def test_invalid_packed_offsets_fail_before_forward():
    batch = _packed_batch()
    bad = PackedSequenceMetadata(
        global_cu_seqlens=torch.tensor([0, 4, 9, 16], dtype=torch.int32),
        max_seqlen=7,
        sample_offsets=batch.sequence.sample_offsets,
    )

    with pytest.raises(ValueError, match="sample_offsets"):
        PuzzletronBatch(
            model_kwargs=batch.model_kwargs,
            labels=batch.labels,
            ce_mask=batch.ce_mask,
            kd_mask=batch.kd_mask,
            hidden_mask=batch.hidden_mask,
            sequence=bad,
            sample_ids=batch.sample_ids,
            source_metadata=batch.source_metadata,
            modality=batch.modality,
            layout=batch.layout,
        )
