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
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp import fully_shard

from modelopt.torch.opt.dynamic import DynamicModule, _pytorch_managed


def _run_optimizer_step(model: nn.Module, input: torch.Tensor):
    # Always init the optimizer before the first forward.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    out = model(input)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    return out


def run_fsdp_test(
    get_test_case: Callable[[], tuple[nn.Module, Any]],
    dm_key: str,
    sample_subnet: Callable[[DynamicModule], None],
    rank: int,
    world_size: int,
    fsdp_kwargs: dict[str, Any] | None = None,
):
    # init test case
    model, dummy_input = get_test_case()

    # remember raw weight
    weight_raw = model.get_submodule(dm_key)._parameters["weight"].detach().clone()

    # sample subnet according to provided function
    sample_subnet(model)

    # remember dynamic weight
    weight_dynamic = model.get_submodule(dm_key).weight.detach().clone()

    # check that raw weight and dynamic weight differ, otherwise the test is pointless
    assert not torch.equal(weight_raw, weight_dynamic)

    # get a model copy
    model_copy = copy.deepcopy(model)

    # run a forward and a backward pass
    out_model = _run_optimizer_step(model, dummy_input)

    # store this for later...
    with torch.no_grad():
        out2_model = model(dummy_input)

    # check the output is different (non-trivial test...)
    assert not torch.allclose(out_model, out2_model)

    # create FSDP model
    fsdp_model = FSDP(model_copy, **(fsdp_kwargs or {}))
    use_orig_params = fsdp_model._use_orig_params

    # retrieve the dynamic module
    dm_mod = fsdp_model.get_submodule(dm_key)
    assert isinstance(dm_mod, DynamicModule)

    # some checks
    def _check_weight_and_output(is_summoned):
        assert ("weight" in dm_mod._parameters) == (is_summoned or use_orig_params)
        assert "weight" not in dm_mod.__dict__
        assert "weight" in dm_mod._dm_attribute_manager.da_keys()

    # at this point the weight should not be a parameter anymore. FSDP will have moved the weight
    # into __dict__ and we will remove it from __dict__ and manage it as dynamic attribute.
    _check_weight_and_output(is_summoned=False)

    # try summoning the full parameters
    with FSDP.summon_full_params(fsdp_model):
        # at this point the weight will be a regular parameter again
        _check_weight_and_output(is_summoned=True)

        # parameter comparison
        raw_val = dm_mod._dm_attribute_manager.get_da_value("weight")
        assert raw_val is _pytorch_managed
        if raw_val is _pytorch_managed:
            raw_val = dm_mod._parameters["weight"]
        assert torch.equal(raw_val, weight_raw)
        assert torch.equal(dm_mod.weight, weight_dynamic)

    # here we back to the fsdp+dm managed weight
    _check_weight_and_output(is_summoned=False)

    # forward and backward with FSDP and comparison
    out_fsdp: torch.Tensor = _run_optimizer_step(fsdp_model, dummy_input)
    assert torch.allclose(out_model, out_fsdp)

    # check the 2nd output now after the optimizer step
    out2_fsdp = fsdp_model(dummy_input)
    assert torch.allclose(out2_model, out2_fsdp)


def run_fsdp2_test(
    get_test_case: Callable[[], tuple[nn.Module, Any]],
    dm_key: str,
    sample_subnet: Callable[[DynamicModule], None],
    rank: int,
    world_size: int,
):
    # init test case
    model, dummy_input = get_test_case()

    # remember raw weight
    weight_raw = model.get_submodule(dm_key)._parameters["weight"].detach().clone()

    # sample subnet according to provided function
    sample_subnet(model)
    model_copy = copy.deepcopy(model)

    # remember dynamic weight
    weight_dynamic = model.get_submodule(dm_key).weight.detach().clone()

    # check that raw weight and dynamic weight differ, otherwise the test is pointless
    assert not torch.equal(weight_raw, weight_dynamic)

    # run a optimizer step with the gathered input to simulate FSDP
    out_model = _run_optimizer_step(model, dummy_input)

    # store this for later...
    with torch.no_grad():
        out2_model = model(dummy_input)

    # check the output is different (non-trivial test...)
    assert not torch.allclose(out_model, out2_model)

    # create FSDP model
    fsdp_model = fully_shard(model_copy)

    # # retrieve the dynamic module
    dm_mod = fsdp_model.get_submodule(dm_key)
    assert isinstance(dm_mod, DynamicModule)

    # parameter comparison, fsdp2 use DTensor.full_tensor to get original tensor
    raw_val = dm_mod._dm_attribute_manager.get_da_value("weight")
    assert raw_val is _pytorch_managed
    if raw_val is _pytorch_managed:
        raw_val = dm_mod._parameters["weight"]
    assert torch.equal(raw_val.full_tensor(), weight_raw)
    assert torch.equal(dm_mod.weight.full_tensor(), weight_dynamic)

    # forward with FSDP and comparison
    out_fsdp = _run_optimizer_step(fsdp_model, dummy_input)
    assert torch.allclose(out_model, out_fsdp)

    # check the 2nd output now after the optimizer step
    out2_fsdp = fsdp_model(dummy_input)

    assert torch.allclose(out2_model, out2_fsdp)
