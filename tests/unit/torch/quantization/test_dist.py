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

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.quantization.models import SimpleLinear
from torch.nn.parallel import DistributedDataParallel

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import max_calibrate
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import unwrap_model
from modelopt.torch.utils.distributed import DistributedProcessGroup, ParallelState


def _test_data_parallel_helper(rank, size):
    model = DistributedDataParallel(SimpleLinear())
    unwrapped_model = unwrap_model(model)
    # Every pool job starts from the same seed: draw different data per rank, as data
    # parallelism does, so the ranks' amaxes differ unless calibration syncs them.
    torch.manual_seed(rank)
    calib_data = [unwrapped_model.get_input() for _ in range(2)]

    def forward_loop(model):
        for data in calib_data:
            model(data)

    unwrapped_model = mtq.quantize(unwrapped_model, mtq.INT8_DEFAULT_CFG, forward_loop)

    # Check after every rank has joined every all_reduce: a rank that stopped at a mismatch would
    # leave its peer waiting in the next one.
    unsynced = []
    for name, module in unwrapped_model.named_modules():
        if isinstance(module, TensorQuantizer) and "output_quantizer" not in name:
            amax = module.amax.clone()
            dist.all_reduce(amax, op=dist.ReduceOp.MAX)
            if not torch.allclose(amax, module.amax):
                unsynced.append(name)
    assert not unsynced, f"rank {rank}: amax not synced across data-parallel ranks: {unsynced}"


def test_data_parallel(cpu_dist_workers):
    cpu_dist_workers.run(_test_data_parallel_helper)


def _sync(quantizer):
    quantizer.sync_amax_across_distributed_group(DistributedProcessGroup(dist.group.WORLD))


# An NVFP4 activation quantizer's block sizes drive its runtime scales: it stores only a per-tensor
# amax, the global scale, so it syncs as a per-tensor quantizer does.
_PER_TENSOR_CFGS = {
    "int8": {"num_bits": 8},
    "nvfp4-dynamic": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    },
}


def _calibrators_dtype_helper(rank, size, per_tensor_cfg):
    cfg = QuantizerAttributeConfig(**_PER_TENSOR_CFGS[per_tensor_cfg])
    one_calibrated, both_calibrated = TensorQuantizer(cfg), TensorQuantizer(cfg)
    if rank == 0:
        one_calibrated._set_buffer("_amax", torch.tensor(3.0, dtype=torch.bfloat16))
    both_calibrated._set_buffer("_amax", torch.tensor(2.0 + rank, dtype=torch.bfloat16))
    per_channel = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=0))
    per_channel._set_buffer("_amax", torch.tensor([1.0, 2.0 + rank], dtype=torch.bfloat16))
    for quantizer in (one_calibrated, both_calibrated, per_channel):
        _sync(quantizer)
        assert quantizer.amax.dtype == torch.bfloat16
    assert one_calibrated.amax.item() == 3.0
    assert both_calibrated.amax.item() == 3.0
    assert per_channel.amax.tolist() == [1.0, 3.0]


@pytest.mark.parametrize("per_tensor_cfg", list(_PER_TENSOR_CFGS))
def test_the_sync_leaves_every_rank_the_calibrators_dtype(cpu_dist_workers, per_tensor_cfg):
    """And the group's max, including on a rank that never calibrated the quantizer and so had no
    amax of its own. Synced as a per-block quantizer, an NVFP4 activation quantizer's uncalibrated
    rank would raise instead."""
    cpu_dist_workers.run(_calibrators_dtype_helper, per_tensor_cfg=per_tensor_cfg)


def _zero_or_uncalibrated_helper(rank, size, case):
    quantizer = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=8, axis=0 if "channel" in case else None)
    )
    if case == "calibrated-zero":
        quantizer._set_buffer("_amax", torch.tensor(0.0))
    _sync(quantizer)
    if case == "calibrated-zero":
        assert quantizer.amax is not None
        assert quantizer.amax.item() == 0.0
    else:
        assert getattr(quantizer, "_amax", None) is None


@pytest.mark.parametrize("case", ["calibrated-zero", "uncalibrated", "uncalibrated-per-channel"])
def test_the_sync_keeps_a_calibrated_zero_apart_from_uncalibrated(cpu_dist_workers, case):
    """An all-zero weight calibrates to amax 0, which is not "never calibrated"; a quantizer no
    rank's calibration data reached, which is expected rather than an error, stays uncalibrated."""
    cpu_dist_workers.run(_zero_or_uncalibrated_helper, case=case)


def _uneven_per_channel_helper(rank, size):
    # A rank left waiting in a collective its peer never joins gives up in seconds on this group.
    group = DistributedProcessGroup(dist.new_group(timeout=timedelta(seconds=10)))
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=0))
    if rank == 0:
        quantizer._set_buffer("_amax", torch.full((4,), 3.0))
    try:
        quantizer.sync_amax_across_distributed_group(group)
    except RuntimeError as error:
        raised = str(error)
    else:
        raised = "no error"
    assert "unevenly-calibrated" in raised, raised


def test_uneven_per_channel_calibration_fails_on_every_rank(cpu_dist_workers):
    """A rank without the amax cannot rebuild its per-channel shape, so the sync fails, on every
    rank together: one rank raising alone would leave the other waiting in the collective."""
    cpu_dist_workers.run(_uneven_per_channel_helper)


def _expert_input_amax_helper(rank, size, share):
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=None)
    expert = model[0]
    expert._parallel_state = ParallelState(
        data_parallel_group=-1, expert_model_parallel_group=dist.group.WORLD
    )
    if share:
        expert._share_input_amax_across_ep = True

    def forward_loop(m):  # only rank 0's experts receive tokens
        if rank == 0:
            m(torch.ones(2, 16))

    if share:
        max_calibrate(model, forward_loop)
        assert expert.input_quantizer.amax.item() == 1.0
    else:
        with pytest.raises(RuntimeError, match="MoE calibration incomplete"):
            max_calibrate(model, forward_loop)


@pytest.mark.parametrize("share", [False, True], ids=["megatron", "opted-in"])
def test_only_modules_that_opt_in_share_expert_input_amax_before_the_check(cpu_dist_workers, share):
    """Megatron's experts keep main's check-then-sync; HF fused experts share the input amax first."""
    cpu_dist_workers.run(_expert_input_amax_helper, share=share)


def _tensor_parallel_sync_helper(rank, size, kv_cache):
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=None)
    linear = model[0]
    if kv_cache:
        linear.k_bmm_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
        linear.v_bmm_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
        synced = [linear.k_bmm_quantizer, linear.v_bmm_quantizer]
    else:
        linear._is_column_parallel = True
        synced = [linear.input_quantizer]
    # A rank left waiting for a peer that skips the sync gives up in seconds on this group.
    tp_group = dist.new_group(timeout=timedelta(seconds=10))
    linear._parallel_state = ParallelState(data_parallel_group=-1, tensor_parallel_group=tp_group)

    def forward_loop(m):  # only rank 0 calibrates the synced quantizers
        if rank == 0:
            for quantizer in synced:
                quantizer(torch.ones(2, 16))

    max_calibrate(model, forward_loop)
    for quantizer in synced:
        assert quantizer.amax is not None
        assert quantizer.amax.item() == 1.0


@pytest.mark.parametrize("kv_cache", [False, True], ids=["column-parallel", "kv-cache"])
def test_a_rank_without_amax_still_joins_the_tensor_parallel_sync(cpu_dist_workers, kv_cache):
    """Rather than leave the rank that has one waiting in the sync for it."""
    cpu_dist_workers.run(_tensor_parallel_sync_helper, kv_cache=kv_cache)
