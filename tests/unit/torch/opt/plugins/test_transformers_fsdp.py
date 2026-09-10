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

"""Tests for Hugging Face Trainer FSDP integration."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from modelopt.torch.opt.plugins.transformers import _fully_shard_tied_embeddings


class _TiedModel(nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _accelerator(is_fsdp2=True):
    plugin = SimpleNamespace(
        reshard_after_forward=True,
        cpu_offload=None,
        mixed_precision_policy=None,
    )
    return SimpleNamespace(
        is_fsdp2=is_fsdp2,
        state=SimpleNamespace(fsdp_plugin=plugin),
        torch_device_mesh=None,
    )


def test_fully_shard_tied_embeddings_as_one_group(monkeypatch):
    model = _TiedModel()
    calls = []

    def _record_fully_shard(modules, **kwargs):
        calls.append((modules, kwargs))

    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard", _record_fully_shard)

    _fully_shard_tied_embeddings(model, _accelerator())

    assert len(calls) == 1
    modules, kwargs = calls[0]
    assert modules == [model.embed_tokens, model.lm_head]
    assert kwargs["reshard_after_forward"] is True
    assert kwargs["offload_policy"] is None
    assert isinstance(kwargs["mp_policy"], torch.distributed.fsdp.MixedPrecisionPolicy)
    assert kwargs["mesh"] is None


@pytest.mark.parametrize(("tied", "is_fsdp2"), [(False, True), (True, False)])
def test_fully_shard_tied_embeddings_skips_unsupported_cases(monkeypatch, tied, is_fsdp2):
    model = _TiedModel(tied=tied)

    def _unexpected_fully_shard(*args, **kwargs):
        pytest.fail("fully_shard should not be called")

    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard", _unexpected_fully_shard)

    _fully_shard_tied_embeddings(model, _accelerator(is_fsdp2=is_fsdp2))
