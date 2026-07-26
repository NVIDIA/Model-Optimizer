# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from modelopt.torch.puzzletron.dataset.batch import DataLayout, Modality, PuzzletronBatch
from modelopt.torch.puzzletron.plugins.automodel.batch_adapter import (
    VisionForwardMonitor,
    canonicalize_position_ids,
    prepare_native_cp_inputs,
    validate_native_feature_config,
    validated_forward_kwargs,
)


class _StrictVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Identity()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        cu_seqlens=None,
    ):
        return input_ids


class _NativePreEmbed(torch.nn.Module):
    def prepare_model_inputs_for_cp(self, input_ids, attention_mask=None, position_ids=None):
        assert attention_mask is not None
        assert attention_mask.bool().any(dim=1).all()
        return {
            "inputs_embeds": input_ids.unsqueeze(-1).float(),
            "position_ids": torch.arange(input_ids.shape[1]).expand(3, input_ids.shape[0], -1),
        }

    def forward(self, input_ids, attention_mask=None, position_ids=None, _pre_embed_only=False):
        if _pre_embed_only:
            return self.prepare_model_inputs_for_cp(input_ids, attention_mask, position_ids)
        return input_ids


def _batch(**extra):
    kwargs = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        **extra,
    }
    return PuzzletronBatch(
        model_kwargs=kwargs,
        sample_ids=("sample",),
        source_metadata={"dataset": "fixture", "revision": "1"},
        modality=Modality.MULTIMODAL,
        layout=DataLayout.PADDED_VARLEN,
    )


def test_validated_forward_kwargs_preserves_vlm_and_packing_fields():
    batch = _batch(
        position_ids=torch.arange(4).reshape(1, 4),
        pixel_values=torch.ones(1, 3, 2, 2),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
    )

    kwargs = validated_forward_kwargs(_StrictVLM(), batch)

    assert tuple(kwargs) == (
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "image_grid_thw",
        "cu_seqlens",
    )


def test_native_pre_embed_handles_fully_masked_pp_padding_rows():
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])

    prepared = prepare_native_cp_inputs(
        _NativePreEmbed(),
        {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": attention_mask,
        },
    )

    assert torch.equal(prepared["attention_mask"], attention_mask)
    assert prepared["inputs_embeds"].shape == (2, 4, 1)


def test_validated_forward_kwargs_rejects_silent_field_dropping():
    batch = _batch(unsupported_media_tensor=torch.ones(1))

    try:
        validated_forward_kwargs(_StrictVLM(), batch)
    except TypeError as exc:
        assert "unsupported_media_tensor" in str(exc)
    else:
        raise AssertionError("unsupported VLM field was silently dropped")


def test_native_feature_validation_rejects_force_hf_for_packed_or_embedding_modes():
    for config in (
        {
            "model": {"force_hf": True},
            "data": {"layout": "packed_varlen", "modality": "text"},
        },
        {
            "model": {"force_hf": True},
            "data": {"layout": "fixed", "modality": "text"},
            "embedding_pruning": {"enabled": True},
        },
    ):
        try:
            validate_native_feature_config(config)
        except ValueError as exc:
            assert "force_hf=False" in str(exc)
        else:
            raise AssertionError("force_hf=True accepted for a native-only feature")


def test_vision_monitor_records_calls_and_tensor_checksum():
    model = _StrictVLM()
    monitor = VisionForwardMonitor(model.visual)

    with monitor:
        model.visual(torch.arange(4, dtype=torch.float32).reshape(1, 4))
        model.visual(torch.ones(1, 4))

    assert monitor.forward_count == 2
    assert len(monitor.output_checksums) == 2
    assert monitor.output_checksums[0] != monitor.output_checksums[1]


def test_descriptor_expands_text_positions_before_distributed_sharding():
    class _MRoPEDescriptor:
        @staticmethod
        def position_id_axes(_config):
            return 3

    positions = torch.arange(4).reshape(1, 4)
    batch = _batch(position_ids=positions)

    normalized = canonicalize_position_ids(
        batch,
        descriptor=_MRoPEDescriptor(),
        config=object(),
    )

    assert batch.model_kwargs["position_ids"].shape == (1, 4)
    assert normalized.model_kwargs["position_ids"].shape == (3, 1, 4)
    assert torch.equal(normalized.model_kwargs["position_ids"][0], positions)
    assert torch.equal(normalized.model_kwargs["position_ids"][2], positions)


def test_descriptor_preserves_precomputed_vlm_mrope_positions():
    class _MRoPEDescriptor:
        @staticmethod
        def position_id_axes(_config):
            return 3

    positions = torch.arange(12).reshape(3, 1, 4)
    batch = _batch(position_ids=positions)

    normalized = canonicalize_position_ids(
        batch,
        descriptor=_MRoPEDescriptor(),
        config=object(),
    )

    assert normalized is batch
    assert normalized.model_kwargs["position_ids"] is positions
