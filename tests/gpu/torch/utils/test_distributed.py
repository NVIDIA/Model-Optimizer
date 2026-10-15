# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU/distributed tests for ``modelopt.torch.utils.distributed``."""

from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.torch.transformers_models import get_tiny_llama
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor

from modelopt.torch.utils.distributed import fsdp2_wrap

VOCAB_SIZE = 32
N_EXPERTS = 4


class _Fp32Router(nn.Module):
    """An MoE router gate pinned to fp32, as Nemotron-3-Nano's modeling code declares it."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(N_EXPERTS, hidden_size, dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states):
        return F.linear(hidden_states.float(), self.weight.float())


class _RoutedMLP(nn.Module):
    """Fronts a bf16 MLP with the fp32 router, so one decoder layer holds both dtypes."""

    def __init__(self, mlp: nn.Module, hidden_size: int):
        super().__init__()
        self.mlp = mlp
        self.gate = _Fp32Router(hidden_size)

    def forward(self, hidden_states):
        scale = self.gate(hidden_states).softmax(-1)[..., :1].to(hidden_states.dtype)
        return self.mlp(hidden_states) * scale


def _mixed_dtype_model(device):
    model = get_tiny_llama(vocab_size=VOCAB_SIZE).to(device)
    for layer in model.model.layers:
        layer.mlp = _RoutedMLP(layer.mlp, model.config.hidden_size).to(device)
    return model.eval()


def _test_fsdp2_wrap_mixed_dtypes(rank, size):
    """A model with a few fp32 params must still wrap, forward, and load state dicts."""
    device = torch.device(f"cuda:{rank}")
    model = _mixed_dtype_model(device)
    assert {p.dtype for p in model.model.layers[0].parameters()} == {
        torch.bfloat16,
        torch.float32,
    }

    fsdp2_wrap(model)

    # Raised "FSDP expects uniform original parameter dtype" before the ignored-param fix.
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8), device=device)
    with torch.no_grad():
        assert model(input_ids=input_ids).logits.shape == (1, 8, VOCAB_SIZE)

    # bf16 weights are sharded; the fp32 router is left replicated in its original dtype.
    gate_weight = model.model.layers[0].mlp.gate.weight
    sharded_weight = model.model.layers[0].mlp.mlp.up_proj.weight
    assert isinstance(sharded_weight, DTensor)
    assert not isinstance(gate_weight, DTensor)
    assert gate_weight.dtype == torch.float32
    # Left out of the wrap, it still has to sit on the compute device alongside the shards.
    assert gate_weight.device == sharded_weight.to_local().device

    # The FSDP2 loader pushes full tensors into each decoder layer; that must still reach the
    # replicated fp32 param as well as the sharded bf16 ones.
    layer = model.model.layers[0]
    hidden_size = model.config.hidden_size
    set_model_state_dict(
        layer,
        {"mlp.gate.weight": torch.full((N_EXPERTS, hidden_size), 3.0, device=device)},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False, strict=False),
    )
    assert torch.equal(
        layer.mlp.gate.weight, torch.full((N_EXPERTS, hidden_size), 3.0, device=device)
    )


def test_fsdp2_wrap_mixed_dtypes(dist_workers):
    dist_workers.run(_test_fsdp2_wrap_mixed_dtypes)


def _test_fsdp2_wrap_moves_ignored_params_to_device(rank, size, cpu_offload):
    """A CPU-resident model must end up computing on GPU: fully_shard skips the params it ignores."""
    model = _mixed_dtype_model(torch.device("cpu"))
    assert model.model.layers[0].mlp.gate.weight.device.type == "cpu"

    fsdp2_wrap(model, cpu_offload=cpu_offload)

    # Under cpu_offload the shard rests on CPU, but compute — and so the ignored params — is
    # still on GPU, which is why the device is taken from the mesh and not from the local shard.
    sharded_weight = model.model.layers[0].mlp.mlp.up_proj.weight
    assert sharded_weight.to_local().device.type == ("cpu" if cpu_offload else "cuda")
    assert model.model.layers[0].mlp.gate.weight.device.type == "cuda"

    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8), device=torch.device(f"cuda:{rank}"))
    with torch.no_grad():
        assert model(input_ids=input_ids).logits.shape == (1, 8, VOCAB_SIZE)


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_fsdp2_wrap_moves_ignored_params_to_device(dist_workers, cpu_offload):
    dist_workers.run(
        partial(_test_fsdp2_wrap_moves_ignored_params_to_device, cpu_offload=cpu_offload)
    )
