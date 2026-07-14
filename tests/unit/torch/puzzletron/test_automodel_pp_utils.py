# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch.distributed.pipelining.microbatch import TensorChunkSpec, _Replicate

from modelopt.torch.puzzletron.plugins.automodel.pp_utils import set_pp_vlm_chunk_specs


def test_mrope_position_ids_chunk_on_batch_dimension():
    schedule = SimpleNamespace(_kwargs_chunk_spec=None)
    set_pp_vlm_chunk_specs(
        schedule,
        {
            "position_ids": torch.zeros(3, 2, 1024, dtype=torch.long),
            "attention_mask": torch.ones(2, 1024, dtype=torch.long),
            "metadata": "replicated",
        },
    )

    assert isinstance(schedule._kwargs_chunk_spec["position_ids"], TensorChunkSpec)
    assert schedule._kwargs_chunk_spec["position_ids"].split_dim == 1
    assert schedule._kwargs_chunk_spec["attention_mask"].split_dim == 0
    assert isinstance(schedule._kwargs_chunk_spec["metadata"], _Replicate)


def test_non_mrope_batches_use_default_pipeline_chunking():
    schedule = SimpleNamespace(_kwargs_chunk_spec={"stale": object()})
    set_pp_vlm_chunk_specs(
        schedule,
        {"position_ids": torch.zeros(2, 1024, dtype=torch.long)},
    )
    assert schedule._kwargs_chunk_spec is None
