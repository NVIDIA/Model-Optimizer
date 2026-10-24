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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.tensor_quant import dynamic_block_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for FP4 QDQ"
)


@torch.no_grad()
def test_nvfp4_error_weighted_selection():
    torch.manual_seed(17)
    model = nn.Linear(64, 17, device="cuda")
    batches = [torch.randn(size, 64, device="cuda") for size in (2, 7)]
    error_sq = torch.zeros(64, device="cuda")
    amax = None
    for inputs in batches:
        q = dynamic_block_quant(
            inputs, 16, inputs.abs().amax() if amax is None else amax, None, (2, 1), (4, 3)
        )
        error_sq += (inputs - q).square().sum(dim=0)
        amax = inputs.abs().amax() if amax is None else torch.maximum(amax, inputs.abs().amax())
    score = error_sq.sqrt() * model.weight.detach().square().sum(0).sqrt()
    selected = score.view(-1, 8).topk(2, dim=1).indices
    expected = (selected + torch.arange(8, device="cuda")[:, None] * 8).flatten().sort().values
    cfg = deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "residue_max", "salience_metric": "nvfp4_error_weighted"}
    mtq.quantize(model, cfg, lambda m: [m(inputs) for inputs in batches])
    assert torch.equal(model.input_quantizer.salient_indices, expected)
    assert torch.isfinite(model(batches[-1])).all()
