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

"""Amax synchronization across a data-parallel calibration group.

Under data-parallel calibration a module can be exercised on some ranks and not others -- a sparsely
routed MoE expert, or any module a given rank's batch never reaches. Every rank must still join the
collective or the participant set mismatches and NCCL hangs, so these tests pin down what each rank
contributes when it has no local amax, and that the failure modes are symmetric across ranks.
"""

import pytest
import torch
import torch.distributed as dist

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.utils.distributed import DistributedProcessGroup


def _quantizer(cfg, amax=None, rank_device=None):
    q = TensorQuantizer(cfg).to(rank_device)
    if amax is not None:
        q._set_buffer("_amax", torch.as_tensor(amax, dtype=torch.float32, device=rank_device))
    return q


def _whole_world():
    return DistributedProcessGroup(dist.group.WORLD)


# --------------------------------------------------------------------------------- per-tensor amax


def _per_tensor_all_ranks_present(rank, size):
    """Every rank calibrated it: the group agrees on the max."""
    device = torch.device(f"cuda:{rank}")
    q = _quantizer(QuantizerAttributeConfig(num_bits=8), amax=float(rank + 1), rank_device=device)
    q.sync_amax_across_distributed_group(_whole_world())
    assert q.amax.item() == pytest.approx(float(size))


def test_per_tensor_amax_takes_the_group_max(dist_workers):
    dist_workers.run(_per_tensor_all_ranks_present)


def _per_tensor_some_ranks_missing(rank, size):
    """A have-not rank contributes a locally-known scalar zero -- the identity for MAX."""
    device = torch.device(f"cuda:{rank}")
    amax = 7.0 if rank == 0 else None
    q = _quantizer(QuantizerAttributeConfig(num_bits=8), amax=amax, rank_device=device)
    q.sync_amax_across_distributed_group(_whole_world())
    # Both ranks end up with rank 0's value; the have-not must NOT be left at None or zero.
    assert q.amax.item() == pytest.approx(7.0)


def test_per_tensor_amax_survives_uneven_calibration(dist_workers):
    """This is the common case -- a module only some ranks' batches reached."""
    dist_workers.run(_per_tensor_some_ranks_missing)


def _per_tensor_nobody_calibrated(rank, size):
    device = torch.device(f"cuda:{rank}")
    q = _quantizer(QuantizerAttributeConfig(num_bits=8), amax=None, rank_device=device)
    q.sync_amax_across_distributed_group(_whole_world())
    assert q.amax is None, "an all-zero sync must leave the quantizer uncalibrated, not amax==0"


def test_uncalibrated_everywhere_stays_uncalibrated(dist_workers):
    """amax==0 would quantize everything to zero; downstream expects None for 'never calibrated'."""
    dist_workers.run(_per_tensor_nobody_calibrated)


# --------------------------------------------------------------- static non-per-tensor (per-channel)


def _per_channel_all_present(rank, size):
    device = torch.device(f"cuda:{rank}")
    amax = torch.full((4,), float(rank + 1))
    q = _quantizer(QuantizerAttributeConfig(num_bits=8, axis=0), amax=amax, rank_device=device)
    q.sync_amax_across_distributed_group(_whole_world())
    assert torch.allclose(q.amax.cpu(), torch.full((4,), float(size)))


def test_per_channel_amax_takes_the_elementwise_group_max(dist_workers):
    dist_workers.run(_per_channel_all_present)


def _per_channel_unevenly_calibrated(rank, size):
    """Have-nots cannot invent the right shape, so every rank must raise -- together."""
    device = torch.device(f"cuda:{rank}")
    amax = torch.full((4,), 3.0) if rank == 0 else None
    q = _quantizer(QuantizerAttributeConfig(num_bits=8, axis=0), amax=amax, rank_device=device)
    with pytest.raises(RuntimeError, match="unevenly-calibrated"):
        q.sync_amax_across_distributed_group(_whole_world())


def test_uneven_per_channel_calibration_fails_on_every_rank(dist_workers):
    """A one-sided raise would leave the other rank blocked in the collective forever.

    The rank that HAS the amax cannot know the have-nots exist without asking, so the agreement is a
    single MAX all-reduce of a packed presence flag that every rank joins before anyone diverges.
    """
    dist_workers.run(_per_channel_unevenly_calibrated)


def _per_channel_nobody_calibrated(rank, size):
    device = torch.device(f"cuda:{rank}")
    q = _quantizer(QuantizerAttributeConfig(num_bits=8, axis=0), amax=None, rank_device=device)
    q.sync_amax_across_distributed_group(_whole_world())
    assert q.amax is None


def test_per_channel_uncalibrated_everywhere_is_not_an_error(dist_workers):
    """A module no rank's calibration data exercised is expected, not a failure."""
    dist_workers.run(_per_channel_nobody_calibrated)


# ------------------------------------------------------------------------- dynamic block (NVFP4 act)


def _dynamic_block_some_ranks_missing(rank, size):
    """A dynamic-block quantizer stores a SCALAR amax, so the per-tensor path applies."""
    device = torch.device(f"cuda:{rank}")
    cfg = QuantizerAttributeConfig(
        num_bits=(2, 1), block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    )
    amax = 5.0 if rank == 0 else None
    q = _quantizer(cfg, amax=amax, rank_device=device)
    q.sync_amax_across_distributed_group(_whole_world())
    assert q.amax.item() == pytest.approx(5.0)


def test_dynamic_block_amax_is_treated_as_per_tensor(dist_workers):
    """Its block_sizes drive runtime scales; only the per-tensor global scale is stored.

    Classifying it by ``block_sizes is not None`` would send an NVFP4 activation quantizer down the
    static path and raise 'unevenly-calibrated' for the ordinary case of a module some ranks missed.
    """
    dist_workers.run(_dynamic_block_some_ranks_missing)
