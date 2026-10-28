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

"""Packing weights that are still sharded, as the no-gather export does (CPU, one gloo rank)."""

import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from modelopt.torch.quantization.qtensor import NVFP4QTensor


def _raw_bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def test_nvfp4_packs_a_sharded_weight_into_the_same_codes(gloo_world_of_one):
    """``DTensor << int`` is a silent no-op, so the nibble pack must not rely on the operator.

    The block scales come out sharded like the weight, an all-zero block included: filling it by
    boolean-mask assignment lowers to ``index_put_``, which DTensor cannot dispatch.
    """
    torch.manual_seed(0)
    weight = torch.randn(4, 32, dtype=torch.bfloat16)
    weight[1, :16] = 0.0  # an all-zero block
    scale_2 = (weight.abs().amax().float() / (6.0 * 448.0)).reshape(())
    scale = NVFP4QTensor.get_weights_scaling_factor(weight, 16, scale_2)[0]
    want = NVFP4QTensor.quantize(weight, 16, scale, scale_2)[0]._quantized_data

    sharded = distribute_tensor(weight, gloo_world_of_one, [Shard(0)])
    sharded_scale = NVFP4QTensor.get_weights_scaling_factor(sharded, 16, scale_2)[0]
    packed = NVFP4QTensor.quantize(sharded, 16, sharded_scale, scale_2)[0]._quantized_data

    assert isinstance(sharded_scale, DTensor) and sharded_scale.placements == (Shard(0),)
    assert torch.equal(_raw_bytes(sharded_scale.to_local()), _raw_bytes(scale))
    assert torch.equal(packed.full_tensor(), want)
    # The second-level scale read off the sharded weight itself, as without one handed in.
    assert torch.equal(
        NVFP4QTensor.get_weights_scaling_factor(sharded, 16)[1],
        NVFP4QTensor.get_weights_scaling_factor(weight, 16)[1],
    )


def test_nvfp4_cast_keeps_the_global_shape_of_an_uneven_shard(gloo_world_of_one):
    """``from_local`` would infer local x world rows; an uneven shard must keep the real total.

    Two rows held of a three-row tensor is what one rank holds when three rows split over two.
    """
    local = torch.randn(2, 32) * 4
    sharded = DTensor.from_local(
        local.clone(),
        gloo_world_of_one,
        [Shard(0)],
        run_check=False,
        shape=torch.Size([3, 32]),
        stride=(32, 1),
    )

    packed = NVFP4QTensor._cast_fp4(sharded)

    assert isinstance(packed, DTensor) and packed.placements == (Shard(0),)
    assert packed.shape == (3, 32)
    assert torch.equal(packed.to_local(), NVFP4QTensor._cast_fp4(local.clone()))
