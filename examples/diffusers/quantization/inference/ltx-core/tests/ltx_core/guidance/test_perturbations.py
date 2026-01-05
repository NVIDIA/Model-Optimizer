# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Andrew Kvochko

import pytest
import torch

import ltx_core.guidance.perturbations as ptb


@pytest.fixture
def positive_config() -> ptb.PerturbationConfig:
    return ptb.PerturbationConfig(None)


@pytest.fixture
def modality_config() -> ptb.PerturbationConfig:
    return ptb.PerturbationConfig(
        [
            ptb.Perturbation(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, None),
            ptb.Perturbation(ptb.PerturbationType.SKIP_V2A_CROSS_ATTN, None),
        ]
    )


@pytest.fixture
def stg_config() -> ptb.PerturbationConfig:
    return ptb.PerturbationConfig(
        [
            ptb.Perturbation(ptb.PerturbationType.SKIP_VIDEO_SELF_ATTN, [0, 1, 2]),
            ptb.Perturbation(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, [0, 1]),
        ]
    )


def test_perturbation() -> None:
    perturbation = ptb.Perturbation(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, [0, 1, 2])
    assert perturbation.is_perturbed(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
    assert not perturbation.is_perturbed(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 3)
    assert not perturbation.is_perturbed(ptb.PerturbationType.SKIP_V2A_CROSS_ATTN, 0)

    all_blocks = ptb.Perturbation(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, None)
    assert all_blocks.is_perturbed(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0)


def test_perturbation_config(
    positive_config: ptb.PerturbationConfig, modality_config: ptb.PerturbationConfig, stg_config: ptb.PerturbationConfig
) -> None:
    assert modality_config.is_perturbed(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
    assert modality_config.is_perturbed(ptb.PerturbationType.SKIP_V2A_CROSS_ATTN, 0)
    assert stg_config.is_perturbed(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, 0)
    assert stg_config.is_perturbed(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, 1)
    assert stg_config.is_perturbed(ptb.PerturbationType.SKIP_VIDEO_SELF_ATTN, 0)
    assert not stg_config.is_perturbed(ptb.PerturbationType.SKIP_VIDEO_SELF_ATTN, 3)
    assert not stg_config.is_perturbed(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 3)
    assert not stg_config.is_perturbed(ptb.PerturbationType.SKIP_V2A_CROSS_ATTN, 3)

    assert not positive_config.is_perturbed(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
    assert not positive_config.is_perturbed(ptb.PerturbationType.SKIP_V2A_CROSS_ATTN, 0)
    assert not positive_config.is_perturbed(ptb.PerturbationType.SKIP_VIDEO_SELF_ATTN, 0)
    assert not positive_config.is_perturbed(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, 0)


def test_batched_perturbation_config(
    positive_config: ptb.PerturbationConfig, modality_config: ptb.PerturbationConfig, stg_config: ptb.PerturbationConfig
) -> None:
    batched_ptb = ptb.BatchedPerturbationConfig([positive_config, modality_config, stg_config])
    assert batched_ptb.any_in_batch(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
    assert batched_ptb.any_in_batch(ptb.PerturbationType.SKIP_V2A_CROSS_ATTN, 0)
    assert batched_ptb.any_in_batch(ptb.PerturbationType.SKIP_VIDEO_SELF_ATTN, 0)
    assert batched_ptb.any_in_batch(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, 0)
    assert not batched_ptb.any_in_batch(ptb.PerturbationType.SKIP_VIDEO_SELF_ATTN, 3)
    assert not batched_ptb.any_in_batch(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, 2)
    assert not batched_ptb.all_in_batch(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
    assert not batched_ptb.all_in_batch(ptb.PerturbationType.SKIP_V2A_CROSS_ATTN, 0)

    mask = batched_ptb.mask(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0, torch.device("cpu"), torch.float32)
    assert mask.tolist() == [1, 0, 1]

    mask = batched_ptb.mask_like(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, 2, mask)
    assert mask.tolist() == [1, 1, 1]

    mask = batched_ptb.mask_like(ptb.PerturbationType.SKIP_AUDIO_SELF_ATTN, 1, mask)
    assert mask.tolist() == [1, 1, 0]


def test_empty_batched_perturbation_config() -> None:
    batched_ptb = ptb.BatchedPerturbationConfig.empty(2)
    mask = batched_ptb.mask(ptb.PerturbationType.SKIP_A2V_CROSS_ATTN, 0, torch.device("cpu"), torch.float32)
    assert mask.tolist() == [1, 1]
