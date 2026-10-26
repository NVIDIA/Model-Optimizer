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

import pytest
import torch
from transformers import TrainerControl

from modelopt.torch.speculative.plugins.master_weight_adamw import VerifyMasterWeightsCallback


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_callback_checks_first_materialized_optimizer_state(dtype):
    parameter = torch.nn.Parameter(torch.ones(2, dtype=dtype))
    optimizer = torch.optim.AdamW([parameter])
    callback = VerifyMasterWeightsCallback()
    control = TrainerControl()
    assert callback.on_step_end(None, None, control, optimizer=None) is control
    assert callback.on_step_end(None, None, control, optimizer=optimizer) is control
    assert not callback._checked
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    if dtype == torch.bfloat16:
        with pytest.raises(RuntimeError, match="rather than fp32"):
            callback.on_step_end(None, None, control, optimizer=optimizer)
    else:
        assert callback.on_step_end(None, None, control, optimizer=optimizer) is control
        assert callback._checked
        assert callback.on_step_end(None, None, control, optimizer=optimizer) is control
