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
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import _quantizer_state_device, max_calibrate
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.nn.modules.tensor_quantizer import _collective_device
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


def _cold_rank_device_helper(rank, size, case):
    # meta stands in for a second device: these CPU ranks' collective device is the CPU.
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    if rank == 0:
        quantizer._set_buffer("_amax", torch.tensor(3.0))
    elif case == "a-tensor-it-holds":
        quantizer._set_buffer("_pre_quant_scale", torch.ones(1, device="meta"))
    hint = {"the-callers-device": torch.device("meta"), "a-tensor-it-holds": torch.device("cpu")}
    quantizer.sync_amax_across_distributed_group(
        DistributedProcessGroup(dist.group.WORLD), hint.get(case)
    )
    expected = "cpu" if rank == 0 or case == "the-collectives-device" else "meta"
    assert quantizer.amax.device.type == expected, f"rank {rank}: {quantizer.amax.device}"
    assert quantizer.amax._base is None, "a view of the collective's buffer"


@pytest.mark.parametrize(
    "case", ["a-tensor-it-holds", "the-callers-device", "the-collectives-device"]
)
def test_a_rank_that_never_calibrated_adopts_the_amax_where_it_keeps_its_state(
    cpu_dist_workers, case
):
    """A calibrated rank keeps its amax where it is. One that never calibrated the quantizer puts
    a copy of the group's amax on a tensor the quantizer holds, else on the device its caller keeps
    the module's quantizer state on, else on the collective's device."""
    cpu_dist_workers.run(_cold_rank_device_helper, case=case)


def test_a_group_initialized_without_a_backend_collects_on_cuda(monkeypatch):
    """``init_process_group()`` without a backend names its group ``undefined`` and, in a CUDA
    build of torch, configures it ``cpu:gloo,cuda:nccl``. CI's CPU build configures no NCCL."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(dist, "get_backend", lambda group=None: "undefined")
    monkeypatch.setattr(dist, "get_backend_config", lambda group=None: "cpu:gloo,cuda:nccl")
    assert _collective_device(None) == torch.device("cuda", 0)


@pytest.mark.parametrize("layout", ["a-linear", "fused-experts", "no-weight-quantizer"])
def test_the_state_device_calibration_passes_is_where_a_weight_quantizers_amax_is(layout):
    """Every rank calibrates a weight quantizer from the weight before any sync, so its amax shows
    where the module keeps quantizer state on a rank that saw no tokens; the module's parameters do
    not, resting on the CPU under offload."""
    module = nn.Module()
    module.weight = nn.Parameter(torch.zeros(2, 2))
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    quantizer._set_buffer("_amax", torch.tensor(1.0, device="meta"))
    if layout == "a-linear":
        module.weight_quantizer = quantizer
    elif layout == "fused-experts":
        module.down_proj_weight_quantizers = nn.ModuleList([quantizer])
    module.input_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    expected = None if layout == "no-weight-quantizer" else torch.device("meta")
    assert _quantizer_state_device(module) == expected


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


# Per-channel amaxes the ranks cannot reduce together: rank 1 holds none, or one of another size,
# shape or dtype, or the two hold unlisted dtypes. The forward turns per-token block sizes into a
# per-channel axis, which a rank that never ran it does not see.
_UNEVEN_CASES = {
    "rank-0-only": {"axis": 0},
    "per-token-block-sizes": {"block_sizes": {-1: None, "type": "dynamic"}},
    "different-sizes": {"axis": 0},
    "different-shapes": {"axis": 0},
    "different-dtypes": {"axis": 0},
    "unlisted-dtypes": {"axis": 0},
}


def _uneven_per_channel_helper(rank, size, case):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, **_UNEVEN_CASES[case]))
    if rank == 0 or case.startswith("different-"):  # calibrates it
        quantizer.enable_calib()
        quantizer.disable_quant()
        quantizer(torch.ones(5 if case == "different-sizes" and rank == 1 else 4, 8))
        quantizer.load_calib_amax()
    if case == "different-dtypes" and rank == 1:
        quantizer._set_buffer("_amax", quantizer.amax.bfloat16())
    elif case == "different-shapes":  # of one size
        quantizer._set_buffer("_amax", torch.ones((4, 2) if rank == 0 else (2, 4)))
    elif case == "unlisted-dtypes":
        quantizer._set_buffer("_amax", torch.ones(4, dtype=(torch.int32, torch.int64)[rank]))
    # A rank left waiting in a collective its peer never joins gives up in seconds on this group.
    group = dist.new_group(timeout=timedelta(seconds=10))
    try:
        quantizer.sync_amax_across_distributed_group(DistributedProcessGroup(group))
    except RuntimeError as error:
        raised = str(error)
    else:
        raised = "no error"
    finally:
        dist.destroy_process_group(group)
    assert "unevenly-calibrated" in raised, raised


@pytest.mark.parametrize("case", list(_UNEVEN_CASES))
def test_uneven_per_channel_calibration_fails_on_every_rank(cpu_dist_workers, case):
    """A per-channel amax is reduced only when every rank holds one of the same shape and listed
    dtype: a rank without it cannot rebuild its shape, and NCCL hangs on mismatched ones. Otherwise the sync
    fails, on every rank together: one rank raising alone would leave the other waiting in the
    collective. Also when only the calibrated rank's layout reads per-channel."""
    cpu_dist_workers.run(_uneven_per_channel_helper, case=case)


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


# Block sizes the tensor-parallel sync skips. The forward turns all but static blocks into an axis, a
# tuple or, for block sizes spanning every dimension, None.
_TP_UNSYNCED_BLOCK_SIZES = {
    "static-blocks": {-1: 4},
    "per-token-block-sizes": {-1: None, "type": "dynamic"},
    "every-dim-block-sizes": {-1: None, -2: None},
}
# Per-token block sizes reconfigured after the forward into a layout the column-parallel sync takes:
# per-tensor through the config or the setter, or the per-channel axis -1.
_RECONFIGURED = (
    "reconfigured-to-per-tensor",
    "assigned-per-tensor",
    "axis-configured-after-the-forward",
)


def _tensor_parallel_sync_helper(rank, size, case):
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=None)
    linear = model[0]
    if case == "kv-cache":
        linear.k_bmm_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
        linear.v_bmm_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
        synced = [linear.k_bmm_quantizer, linear.v_bmm_quantizer]
    else:
        linear._is_column_parallel = True
        if case in _TP_UNSYNCED_BLOCK_SIZES:
            linear.input_quantizer.block_sizes = _TP_UNSYNCED_BLOCK_SIZES[case]
        elif case in (*_RECONFIGURED, "axis-none-configured-after-the-forward"):
            linear.input_quantizer.block_sizes = _TP_UNSYNCED_BLOCK_SIZES["per-token-block-sizes"]
        synced = [linear.input_quantizer]
    # A rank left waiting for a peer that skips the sync gives up in seconds on this group.
    tp_group = dist.new_group(timeout=timedelta(seconds=10))
    linear._parallel_state = ParallelState(data_parallel_group=-1, tensor_parallel_group=tp_group)

    def forward_loop(m):  # only rank 0 calibrates the synced quantizers
        if rank == 0:
            for quantizer in synced:
                # One token, so that a per-token amax is a single value.
                quantizer(torch.ones(1 if case in _RECONFIGURED else 2, 16))
        # On every rank, after rank 0's forward turned the per-token block sizes into axis (0,).
        if case == "reconfigured-to-per-tensor":
            linear.input_quantizer.set_from_attribute_config({"block_sizes": None})
        elif case == "assigned-per-tensor":
            linear.input_quantizer.block_sizes = None
        elif case == "axis-configured-after-the-forward":  # replaces the per-token block sizes
            linear.input_quantizer.set_from_attribute_config({"axis": -1})
        elif case == "axis-none-configured-after-the-forward":  # keeps them
            linear.input_quantizer.set_from_attribute_config({"axis": None})

    try:
        max_calibrate(model, forward_loop)
    finally:
        dist.destroy_process_group(tp_group)
    for quantizer in synced:
        if case in _TP_UNSYNCED_BLOCK_SIZES or case == "axis-none-configured-after-the-forward":
            assert (getattr(quantizer, "_amax", None) is not None) == (rank == 0)
        else:
            assert quantizer.amax is not None
            assert quantizer.amax.item() == 1.0


@pytest.mark.parametrize("case", ["column-parallel", "kv-cache", *_RECONFIGURED])
def test_a_rank_without_amax_still_joins_the_tensor_parallel_sync(cpu_dist_workers, case):
    """Rather than leave the rank that has one waiting in the sync for it. Also once a quantizer
    whose forward turned block sizes into an axis is configured, or assigned, per-tensor, or is
    configured with an axis the sync takes."""
    cpu_dist_workers.run(_tensor_parallel_sync_helper, case=case)


@pytest.mark.parametrize(
    "case", [*_TP_UNSYNCED_BLOCK_SIZES, "axis-none-configured-after-the-forward"]
)
def test_a_rank_without_amax_skips_the_tensor_parallel_sync_its_peers_skip(cpu_dist_workers, case):
    """Block quantizers are not synced across tensor parallelism, as judged alike by a rank whose
    forward turned the block sizes into an axis and by one that never ran it, also after a config
    that leaves the block sizes in place."""
    cpu_dist_workers.run(_tensor_parallel_sync_helper, case=case)
