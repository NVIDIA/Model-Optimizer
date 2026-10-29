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

"""The no-gather writer's transport and shard writing over NCCL, on every visible GPU."""

import os
import threading

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from modelopt.torch.export.distributed_writer import _materialize_owned


def _raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()


def _full_tensors(world: int) -> dict:
    """What an export moves: BF16 and FP8 weights and packed NVFP4 bytes, sharded unevenly over the
    world, one with a single row and one with none, a replicated scale and a plain buffer."""
    torch.manual_seed(0)
    rows = 2 * world + 1
    return {
        "bf16.weight": torch.randn(rows, 64, dtype=torch.bfloat16),
        "fp8.weight": torch.randn(rows, 32).to(torch.float8_e4m3fn),
        "nvfp4.weight": torch.randint(0, 256, (rows, 16), dtype=torch.uint8),
        "one_row.weight": torch.randn(1, 8, dtype=torch.bfloat16),
        "no_rows.weight": torch.empty(0, 8, dtype=torch.bfloat16),
        **{f"layers.{i}.weight": torch.randn(4, 16, dtype=torch.bfloat16) for i in range(8)},
        "scale": torch.tensor(0.5),
        "buffer": torch.arange(6.0),
    }


def _sharded(full: dict, rank: int, world: int) -> dict:
    """``full`` as this rank holds it: dim-0 shards in ``torch.chunk``'s layout, as FSDP2 cuts them,
    built locally so the setup itself moves nothing; the scale replicated, the buffer plain."""
    mesh = init_device_mesh("cuda", (world,))
    sd = {}
    for key, tensor in full.items():
        tensor = tensor.cuda()
        if key == "buffer":
            sd[key] = tensor
        elif key == "scale":
            sd[key] = DTensor.from_local(tensor, mesh, [Replicate()], run_check=False)
        else:
            chunks = tensor.chunk(world) if tensor.shape[0] else ()
            local = chunks[rank] if rank < len(chunks) else tensor[:0]
            sd[key] = DTensor.from_local(
                local.contiguous(),
                mesh,
                [Shard(0)],
                run_check=False,
                shape=tensor.shape,
                stride=tensor.stride(),
            )
    return sd


def _materialize_job(rank, world_size):
    full = _full_tensors(world_size)
    sd = _sharded(full, rank, world_size)

    _materialize_owned(sd, set(), rank, world_size)

    for key, tensor in sd.items():
        assert not isinstance(tensor, DTensor) and tensor.is_cuda, key
        assert (tensor.dtype, tuple(tensor.shape)) == (full[key].dtype, tuple(full[key].shape)), key
        assert torch.equal(_raw_bytes(tensor), _raw_bytes(full[key])), key
    owned: list = [None] * world_size
    torch.distributed.all_gather_object(owned, sorted(sd))
    assert sorted(key for keys in owned for key in keys) == sorted(full)


def test_materialize_owned_moves_uneven_and_empty_shards_over_nccl(need_2_gpus, dist_workers):
    """Every key ends whole, byte for byte, on exactly one rank: BF16, FP8 and uint8 shards of
    uneven extent, empty ones included, a tensor with no rows, a replicated and a plain one."""
    dist_workers.run(_materialize_job)


def _sparse_first_transfer_job(rank, world_size):
    # A rank stuck in NCCL's initialization exits, so the test fails instead of hanging.
    watchdog = threading.Timer(60, os._exit, args=(3,))
    watchdog.start()
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        full = torch.arange(16, dtype=torch.bfloat16, device="cuda").reshape(2, 8)
        # Rows on ranks 0 and 1 only: the other ranks have nothing to send or receive.
        local = full[rank : rank + 1] if rank < 2 else full[:0]
        sd = {
            "w": DTensor.from_local(
                local.contiguous(),
                mesh,
                [Shard(0)],
                run_check=False,
                shape=full.shape,
                stride=full.stride(),
            )
        }

        _materialize_owned(sd, set(), rank, world_size)
        torch.cuda.synchronize()

        assert torch.equal(sd["w"], full) if rank == 0 else not sd, sorted(sd)
    finally:
        watchdog.cancel()


@pytest.mark.timeout(300)
def test_a_first_transfer_most_ranks_sit_out_finishes_on_a_fresh_group():
    """On a fresh ``cpu:gloo,cuda:nccl`` group, with no collective run on it before, a transfer that
    only two of three or more ranks take part in finishes: NCCL's first batched P2P on a group needs
    every rank of it, and the metadata collective runs on Gloo."""
    world_size = torch.cuda.device_count()
    if world_size < 3:
        pytest.skip("needs 3 or more GPUs, so that a rank holds no shard")
    spawn_multiprocess_job(world_size, _sparse_first_transfer_job, backend="nccl")
