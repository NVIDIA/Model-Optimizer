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

"""Megatron prefill preserves output precision and skips auxiliary training heads."""

import importlib
from types import SimpleNamespace

import pytest
import torch

generation = importlib.import_module("modelopt.torch.utils.plugins.megatron_generate")


def test_prefill_disables_auxiliary_training_head(monkeypatch):
    for name in ("is_pipeline_first_stage", "is_pipeline_last_stage"):
        monkeypatch.setattr(generation.mpu, name, lambda: True)
    monkeypatch.setattr(generation.mpu, "get_context_parallel_world_size", lambda: 1)
    config = SimpleNamespace(pipeline_dtype=torch.float32, sequence_parallel=False)

    class Model(generation.MegatronModule):
        def forward(self, tokens, positions, mask, *, runtime_gather_output, compute_mtp_loss=True):
            assert not compute_mtp_loss
            assert runtime_gather_output

    model = Model(config)
    generation.megatron_prefill(
        model,
        torch.tensor([[11, 12]]),
        skip_return_logits=True,
    )


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("skip_return_logits", [False, True])
def test_prefill_preserves_explicit_fp32_output(monkeypatch, pipeline, skip_return_logits):
    monkeypatch.setattr(generation.mpu, "is_pipeline_first_stage", lambda: not pipeline)
    monkeypatch.setattr(generation.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(generation.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(generation, "recv_from_prev_pipeline_rank_", lambda tensor: None)
    received = []

    def broadcast(shape, dtype, tensor):
        assert dtype == tensor.dtype == torch.float32
        received.append(tensor)
        return tensor

    monkeypatch.setattr(generation, "broadcast_from_last_pipeline_stage", broadcast)
    expected = torch.full((1, 2, 3), 1 + 2**-14, dtype=torch.float32)

    class Model(generation.MegatronModule):
        vocab_size = 3

        def set_input_tensor(self, tensor):
            self.input_tensor = tensor

        def forward(self, *args, **kwargs):
            return expected

    model = Model(
        SimpleNamespace(
            pipeline_dtype=torch.bfloat16,
            sequence_parallel=False,
            hidden_size=3,
            bf16=True,
            fp16=False,
            logit_dtype=torch.float32,
        )
    )
    actual = generation.megatron_prefill(
        model, torch.tensor([[1, 2]]), skip_return_logits=skip_return_logits
    )
    if skip_return_logits:
        assert actual is None and not received
    else:
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        assert len(received) == int(pipeline)
