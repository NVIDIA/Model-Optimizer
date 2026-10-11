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

from functools import partial

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.quantization.models import SimpleLinear
from torch.nn.parallel import DistributedDataParallel

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import unwrap_model
from modelopt.torch.utils.distributed import DistributedProcessGroup


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


def _test_uneven_amax_sync_helper(rank, size):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    if rank == 0:
        quantizer._set_buffer("_amax", torch.tensor(7.0))
    quantizer.sync_amax_across_distributed_group(DistributedProcessGroup(dist.group.WORLD))
    assert quantizer.amax.item() == 7.0
    dist.destroy_process_group()


def test_amax_sync_with_a_rank_that_never_calibrated(skip_on_windows):
    """The rank without an amax still joins the sync, contributing a zero on the group's device."""
    spawn_multiprocess_job(2, _test_uneven_amax_sync_helper, backend="gloo")


def _sync(quantizer):
    quantizer.sync_amax_across_distributed_group(DistributedProcessGroup(dist.group.WORLD))


def _calibrated_zero_amax_helper(rank, size):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    quantizer._set_buffer("_amax", torch.tensor(0.0))
    _sync(quantizer)
    assert quantizer.amax is not None
    assert quantizer.amax.item() == 0.0
    dist.destroy_process_group()


def test_a_calibrated_zero_amax_survives_the_sync(skip_on_windows):
    """An all-zero weight calibrates to amax 0, which is not "never calibrated"."""
    spawn_multiprocess_job(2, _calibrated_zero_amax_helper, backend="gloo")


def _calibrators_dtype_helper(rank, size):
    per_tensor = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    if rank == 0:
        per_tensor._set_buffer("_amax", torch.tensor(3.0, dtype=torch.bfloat16))
    per_channel = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=0))
    per_channel._set_buffer("_amax", torch.tensor([1.0, 2.0 + rank], dtype=torch.bfloat16))
    _sync(per_tensor)
    _sync(per_channel)
    assert per_tensor.amax.dtype == torch.bfloat16
    assert per_tensor.amax.item() == 3.0
    assert per_channel.amax.dtype == torch.bfloat16
    assert per_channel.amax.tolist() == [1.0, 3.0]
    dist.destroy_process_group()


def test_the_sync_leaves_every_rank_the_calibrators_dtype(skip_on_windows):
    """Including a rank that never calibrated the quantizer and so had no amax of its own."""
    spawn_multiprocess_job(2, _calibrators_dtype_helper, backend="gloo")


def _never_calibrated_helper(rank, size):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    _sync(quantizer)
    assert getattr(quantizer, "_amax", None) is None
    dist.destroy_process_group()


def test_a_quantizer_no_rank_calibrated_stays_uncalibrated(skip_on_windows):
    spawn_multiprocess_job(2, _never_calibrated_helper, backend="gloo")


def _expert_input_amax_helper(rank, size, share):
    from modelopt.torch.quantization.model_calib import max_calibrate
    from modelopt.torch.utils.distributed import ParallelState

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
    dist.destroy_process_group()


@pytest.mark.parametrize("share", [False, True], ids=["megatron", "opted-in"])
def test_only_modules_that_opt_in_share_expert_input_amax_before_the_check(skip_on_windows, share):
    """Megatron's experts keep main's check-then-sync; HF fused experts share the input amax first."""
    spawn_multiprocess_job(2, partial(_expert_input_amax_helper, share=share), backend="gloo")
