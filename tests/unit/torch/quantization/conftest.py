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
from _test_utils.torch.quantization.models import ToyFusedExperts, ToyMoE

from modelopt.torch.quantization.nn import QuantModuleRegistry


@pytest.fixture
def moe():
    """A one-layer toy MoE with HF fused experts; unregisters its quantized expert class after."""
    pytest.importorskip("transformers")  # the fused-experts quant wrapper is the HF plugin's
    torch.manual_seed(0)
    model = ToyMoE()
    yield model
    if QuantModuleRegistry.get(ToyFusedExperts) is not None:
        QuantModuleRegistry.unregister(ToyFusedExperts)
