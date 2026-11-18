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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pretokenized calibration preserves tokens, masks and DP sample assignment."""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.utils.plugins import megatron_calibration as calibration

generation = importlib.import_module("modelopt.torch.utils.plugins.megatron_generate")


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("single_batch", [False, True])
def test_pretokenized_calibration_preserves_ids_and_masks(
    tmp_path, monkeypatch, padding_side, single_batch
):
    path = tmp_path / "tokens.jsonl"
    rows = [{"input_ids": [11, 12]}, {"input_ids": [21, 22, 23], "attention_mask": [1, 1, 0]}]
    written = (
        [{key: [value] for key, value in row.items()} for row in rows] if single_batch else rows
    )
    path.write_text("\n".join(map(json.dumps, written)))
    monkeypatch.setattr(calibration.mpu, "get_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(calibration.mpu, "get_data_parallel_rank", lambda: 0)
    tokenizer = SimpleNamespace(pad_token="pad", pad_token_id=0, padding_side=padding_side)
    loader = calibration.get_megatron_calibration_dataloader(
        tokenizer,
        dataset_name=str(path),
        num_samples=2,
        batch_size=2,
        seq_length=3,
        device="cpu",
        pretokenized=True,
    )
    batch = next(iter(loader))
    assert batch["input_ids"].tolist() == (
        [[0, 11, 12], [21, 22, 23]] if padding_side == "left" else [[11, 12, 0], [21, 22, 23]]
    )
    assert batch["attention_mask"].tolist() == (
        [[False, True, True], [True, True, False]]
        if padding_side == "left"
        else [[True, True, False], [True, True, False]]
    )
    tokenizer.padding_side = "right"
    for rank in (0, 1):
        monkeypatch.setattr(calibration.mpu, "get_data_parallel_world_size", lambda: 2)
        monkeypatch.setattr(calibration.mpu, "get_data_parallel_rank", lambda rank=rank: rank)
        local = calibration.get_megatron_calibration_dataloader(
            tokenizer,
            dataset_name=str(path),
            num_samples=2,
            batch_size=1,
            seq_length=3,
            device="cpu",
            pretokenized=True,
        )
        assert next(iter(local))["input_ids"].tolist() == [rows[rank]["input_ids"]]


def test_prefill_disables_auxiliary_training_head_and_applies_padding(monkeypatch):
    for name in ("is_pipeline_first_stage", "is_pipeline_last_stage"):
        monkeypatch.setattr(generation.mpu, name, lambda: True)
    monkeypatch.setattr(generation.mpu, "get_context_parallel_world_size", lambda: 1)
    config = SimpleNamespace(pipeline_dtype=torch.float32, sequence_parallel=False)

    class Model(generation.MegatronModule):
        def forward(self, tokens, positions, mask, *, runtime_gather_output, compute_mtp_loss=True):
            assert not compute_mtp_loss
            assert runtime_gather_output
            assert mask[0, 0].tolist() == [
                [True, True, True],
                [True, False, True],
                [True, False, False],
            ]

    model = Model(config)
    generation.megatron_prefill(
        model,
        torch.tensor([[0, 11, 12]]),
        skip_return_logits=True,
        padding_mask=torch.tensor([[0, 1, 1]]),
    )
