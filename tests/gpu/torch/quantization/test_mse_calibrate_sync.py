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

import copy
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from _test_utils.torch.distributed.utils import get_device_counts, spawn_multiprocess_job

import modelopt.torch.quantization as mtq


def _test_mse_calibrate_sync(distributed_sync: bool, rank: int, size: int) -> None:
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16)).cuda()

    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    config["algorithm"] = {
        "method": "mse",
        "num_steps": 16,
        "start_multiplier": 0.001,
        "stop_multiplier": 4.0,
        "distributed_sync": distributed_sync,
    }

    def forward_loop(model):
        torch.manual_seed(1234 + rank)
        scale = 1.0 if rank == 0 else 100.0
        for _ in range(4):
            model(torch.randn(64, 16, device="cuda") * scale)

    model = mtq.quantize(model, config, forward_loop)

    target = next(module for module in model.modules() if hasattr(module, "input_quantizer"))
    amax_val = target.input_quantizer.amax.detach().float().max()

    gathered = [torch.zeros_like(amax_val) for _ in range(size)]
    dist.all_gather(gathered, amax_val)

    if size < 2 or rank != 0:
        return

    values = torch.stack(gathered)
    if distributed_sync:
        assert torch.allclose(values, values[0], rtol=0, atol=0), (
            "Expected amax values to be synchronized across ranks, but got "
            f"{values.tolist()}"
        )
    else:
        assert (values.max() - values.min()) > 10.0, (
            "Expected amax values to differ across ranks when sync is disabled, but got "
            f"{values.tolist()}"
        )


def _test_mse_calibrate_bias_sync(distributed_sync: bool, rank: int, size: int) -> None:
    for bias_method in ["mean", "max_min"]:
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16)).cuda()

        config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        config["quant_cfg"]["*input_quantizer"]["bias"] = {
            0: None,
            "type": "static",
            "method": bias_method,
        }
        config["algorithm"] = {
            "method": "mse",
            "num_steps": 16,
            "start_multiplier": 0.001,
            "stop_multiplier": 4.0,
            "distributed_sync": distributed_sync,
        }

        def forward_loop(model):
            torch.manual_seed(4321 + rank)
            offset = 0.0 if rank == 0 else 10.0
            for _ in range(4):
                model(torch.randn(64, 16, device="cuda") * 0.1 + offset)

        model = mtq.quantize(model, config, forward_loop)

        target = next(module for module in model.modules() if hasattr(module, "input_quantizer"))
        bias_val = target.input_quantizer.bias_value.detach().float().mean()

        gathered = [torch.zeros_like(bias_val) for _ in range(size)]
        dist.all_gather(gathered, bias_val)

        if size < 2 or rank != 0:
            continue

        values = torch.stack(gathered)
        if distributed_sync:
            assert torch.allclose(values, values[0], rtol=0, atol=0), (
                f"Expected bias values to be synchronized across ranks for {bias_method}, but got "
                f"{values.tolist()}"
            )
        else:
            assert (values.max() - values.min()) > 5.0, (
                f"Expected bias values to differ across ranks for {bias_method} when sync is disabled, "
                f"but got {values.tolist()}"
            )


def _test_max_calibrate_bias_sync(distributed_sync: bool, rank: int, size: int) -> None:
    for bias_method in ["mean", "max_min"]:
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16)).cuda()

        config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        config["quant_cfg"]["*input_quantizer"]["bias"] = {
            0: None,
            "type": "static",
            "method": bias_method,
        }
        config["algorithm"] = {"method": "max", "distributed_sync": distributed_sync}

        def forward_loop(model):
            torch.manual_seed(9876 + rank)
            offset = 0.0 if rank == 0 else 10.0
            for _ in range(4):
                model(torch.randn(64, 16, device="cuda") * 0.1 + offset)

        model = mtq.quantize(model, config, forward_loop)

        target = next(module for module in model.modules() if hasattr(module, "input_quantizer"))
        bias_val = target.input_quantizer.bias_value.detach().float().mean()

        gathered = [torch.zeros_like(bias_val) for _ in range(size)]
        dist.all_gather(gathered, bias_val)

        if size < 2 or rank != 0:
            continue

        values = torch.stack(gathered)
        if distributed_sync:
            assert torch.allclose(values, values[0], rtol=0, atol=0), (
                f"Expected bias values to be synchronized across ranks for {bias_method}, but got "
                f"{values.tolist()}"
            )
        else:
            assert (values.max() - values.min()) > 5.0, (
                f"Expected bias values to differ across ranks for {bias_method} when sync is disabled, "
                f"but got {values.tolist()}"
            )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_mse_calibrate_with_sync(device_count):
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_mse_calibrate_sync, True), backend="nccl"
    )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_mse_calibrate_without_sync(device_count):
    if device_count < 2:
        pytest.skip("need 2 GPUs")
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_mse_calibrate_sync, False), backend="nccl"
    )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_mse_calibrate_bias_with_sync(device_count):
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_mse_calibrate_bias_sync, True), backend="nccl"
    )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_mse_calibrate_bias_without_sync(device_count):
    if device_count < 2:
        pytest.skip("need 2 GPUs")
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_mse_calibrate_bias_sync, False), backend="nccl"
    )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_max_calibrate_bias_with_sync(device_count):
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_max_calibrate_bias_sync, True), backend="nccl"
    )


@pytest.mark.parametrize("device_count", get_device_counts())
def test_max_calibrate_bias_without_sync(device_count):
    if device_count < 2:
        pytest.skip("need 2 GPUs")
    spawn_multiprocess_job(
        size=device_count, job=partial(_test_max_calibrate_bias_sync, False), backend="nccl"
    )
