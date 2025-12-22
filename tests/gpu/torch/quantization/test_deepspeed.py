# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test of quantization with DeepSpeed."""

import argparse
import copy
import os
from functools import partial

import pytest

pytest.importorskip("deepspeed")
pytest.importorskip("accelerate")

import deepspeed
import torch
import torch.nn as nn
from _test_utils.torch.distributed.utils import spawn_multiprocess_job, synchronize_state_dict
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.dynamic import _pytorch_managed


def get_ds_config(zero_stage: int = 3):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": zero_stage},  # Restore Stage 3
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
    }


def _test_deepspeed_simple_linear(zero_stage, rank, size):
    deepspeed.init_distributed()

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)

    dim = 32
    model = nn.Linear(dim, dim).cuda(rank)
    inputs = torch.randn(2, 2, dim).cuda(rank)

    synchronize_state_dict(model)
    deepspeed_model_after = copy.deepcopy(model)

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))

    manager = model._get_dm_attribute_manager()
    assert "weight" in manager.da_keys()
    assert model._get_dm_attribute_manager().get_da_value("weight") is _pytorch_managed

    out_ref = model(inputs)

    # Create cmd_args namespace for DeepSpeed initialization
    cmd_args = argparse.Namespace()

    cmd_args.deepspeed_config = get_ds_config(zero_stage)
    cmd_args.local_rank = rank
    cmd_args.world_size = size

    # Initialize DeepSpeed for the test model
    optimizer_test = torch.optim.Adam(model.parameters(), lr=0.1)
    deepspeed_model, _, _, _ = deepspeed.initialize(
        args=cmd_args, model=model, optimizer=optimizer_test
    )

    assert "weight" in manager.da_keys()
    out_test = deepspeed_model(inputs)
    assert torch.allclose(out_ref, out_test)

    # Test quantization after DeepSpeed initialization
    optimizer_after_test = torch.optim.Adam(deepspeed_model_after.parameters(), lr=0.1)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=get_ds_config(zero_stage))
    )

    deepspeed_model_after, _ = accelerator.prepare(deepspeed_model_after, optimizer_after_test)
    deepspeed_unwrapped = accelerator.unwrap_model(deepspeed_model_after)
    mtq.quantize(deepspeed_unwrapped, mtq.INT8_DEFAULT_CFG, lambda model: model(inputs))

    out_deepspeed_model_after = deepspeed_model_after(inputs)

    assert torch.allclose(out_ref, out_deepspeed_model_after)


def _test_nested_deepspeed_backward(zero_stage, rank, size, quant_cfg):
    # Set required environment variables for DeepSpeed
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)

    dim = 32
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim)),
        nn.Linear(dim, dim),
    ).cuda(rank)
    inputs = torch.randn(2, 2, dim).cuda(rank)
    inputss = inputs.detach().clone()

    # test for quantization after DeepSpeed
    deepspeed_model_quant_after = copy.deepcopy(model)

    model = mtq.quantize(model, quant_cfg, lambda model: model(inputs))
    deepspeed_model = copy.deepcopy(model)

    optimizer_ref = torch.optim.Adam(model.parameters(), lr=0.1)
    out_ref = model(inputs)
    out_ref.sum().backward()

    # Initialize DeepSpeed for the test model
    cmd_args = argparse.Namespace()

    cmd_args.deepspeed_config = get_ds_config(zero_stage)
    cmd_args.local_rank = rank
    cmd_args.world_size = size

    # Create optimizer for DeepSpeed
    optimizer_test = torch.optim.Adam(deepspeed_model.parameters(), lr=0.1)
    deepspeed_model, optimizer_test, _, _ = deepspeed.initialize(
        args=cmd_args, model=deepspeed_model, optimizer=optimizer_test
    )
    out_test = deepspeed_model(inputs)
    deepspeed_model.backward(out_test.sum())

    assert torch.allclose(out_ref, out_test)
    optimizer_ref.step()
    optimizer_ref.zero_grad()

    optimizer_test.step()
    optimizer_test.zero_grad()

    out_ref_1 = model(inputss)
    out_test_1 = deepspeed_model(inputss)
    assert torch.allclose(out_ref_1, out_test_1, rtol=1e-4)

    # Initialize DeepSpeed for quantization after DeepSpeed
    optimizer_quant_after = torch.optim.Adam(deepspeed_model_quant_after.parameters(), lr=0.1)

    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=get_ds_config(zero_stage))
    )

    deepspeed_model_quant_after, optimizer_quant_after = accelerator.prepare(
        deepspeed_model_quant_after, optimizer_quant_after
    )
    deepspeed_unwrapped = accelerator.unwrap_model(deepspeed_model_quant_after)
    mtq.quantize(deepspeed_unwrapped, quant_cfg, lambda model: model(inputs))
    out_quant_after = deepspeed_model_quant_after(inputs)
    accelerator.backward(out_quant_after.sum())

    assert torch.allclose(out_ref, out_quant_after)

    out_quant_after_1 = deepspeed_model_quant_after(inputss)

    assert torch.allclose(out_ref_1, out_quant_after_1, rtol=1e-4)


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
def test_deepspeed_simple_linear(zero_stage):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_deepspeed_simple_linear, zero_stage),
        backend="nccl",
    )


@pytest.mark.parametrize("quant_cfg", [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG])
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
def test_nested_deepspeed_backward(quant_cfg, zero_stage):
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_test_nested_deepspeed_backward, zero_stage, quant_cfg=quant_cfg),
        backend="nccl",
    )
