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

"""Amax synchronization over NCCL.

What the sync decides is tested on gloo, in tests/unit/torch/quantization/test_dist.py. What only
NCCL shows is where a rank with no amax puts the zero it contributes: on the collective's CUDA
device.
"""

import pytest
import torch
import torch.distributed as dist

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.utils.distributed import DistributedProcessGroup


def _per_tensor_job(rank, size, calibrated):
    device = torch.device(f"cuda:{rank}")
    q = TensorQuantizer(QuantizerAttributeConfig(num_bits=8)).to(device)
    if calibrated and rank == 0:
        q._set_buffer("_amax", torch.tensor(7.0, device=device))
    q.sync_amax_across_distributed_group(DistributedProcessGroup(dist.group.WORLD))
    if calibrated:
        assert q.amax.item() == pytest.approx(7.0)
    else:
        assert q.amax is None, "an all-zero sync must leave the quantizer uncalibrated, not amax==0"


@pytest.mark.parametrize("calibrated", [True, False], ids=["rank_0_only", "nobody"])
def test_per_tensor_amax_survives_uneven_calibration(need_2_gpus, dist_workers, calibrated):
    """A module only some ranks' batches reached. With one GPU no rank lacks the amax."""
    dist_workers.run(_per_tensor_job, calibrated=calibrated)
