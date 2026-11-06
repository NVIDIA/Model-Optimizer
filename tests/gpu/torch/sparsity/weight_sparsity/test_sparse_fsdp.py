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
import torch.nn as nn
from _test_utils.torch.distributed.fsdp_test import run_fsdp_test
from torch.distributed.fsdp import fully_shard

from modelopt.torch.nas.search_space import SearchSpace
from modelopt.torch.opt.conversion import apply_mode
from modelopt.torch.sparsity import export


def _get_test_case():
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32))
    model = apply_mode(model, "sparse_magnitude")
    model = model.cuda()
    dummy_input = torch.rand(1, 1, 32, device="cuda")
    return model, dummy_input


def _sample_subnet(model):
    model[0].set_mask(torch.rand_like(model[0].weight) > 0.5)
    SearchSpace(model).sample(sample_func=min)


@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fsdp(dist_workers, use_orig_params):
    dist_workers.run(
        partial(
            run_fsdp_test,
            _get_test_case,
            "0",
            _sample_subnet,
            fsdp_kwargs={"use_orig_params": use_orig_params},
        ),
    )


def _run_fsdp2_mask_updates(dtype, initial_mask, rank, world_size):
    model, _ = _get_test_case()
    model.to(dtype=dtype)
    raw_weight = model[0]._parameters["weight"].detach().clone()
    mask = torch.ones_like(raw_weight, dtype=torch.bool)
    mask[:, ::2] = False
    if initial_mask:
        model[0].set_mask(mask)
    model = fully_shard(model)

    for new_mask in [~mask, mask, None, torch.ones_like(mask), ~mask]:
        model[0].set_mask(new_mask)
        expected = raw_weight if new_mask is None else raw_weight * new_mask
        # Reading a sharded dynamic weight must reflect the most recent mask.
        torch.testing.assert_close(model[0].weight.full_tensor(), expected, atol=0, rtol=0)

    # Export materializes the dynamic weight, so a stale cache would bake the
    # previous mask into the checkpoint even though the mask buffer was updated.
    exported = export(model)
    torch.testing.assert_close(
        exported.state_dict()["0.weight"].full_tensor(), expected, atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("initial_mask", [False, True])
def test_fsdp2_mask_updates(dist_workers, dtype, initial_mask):
    dist_workers.run(partial(_run_fsdp2_mask_updates, dtype, initial_mask))
