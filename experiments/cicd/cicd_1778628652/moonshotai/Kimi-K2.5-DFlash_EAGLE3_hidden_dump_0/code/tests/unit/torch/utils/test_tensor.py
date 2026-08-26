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

import numpy as np
import torch

from modelopt.torch.utils import torch_to_numpy


def test_convert_input_to_numpy():
    list_input = [torch.rand(1, 3, 224, 224), torch.rand(1, 3, 224, 224)]
    outputs = torch_to_numpy(list_input)

    # Number of elements in the list
    assert len(outputs) == 2
    assert len(outputs[0]) == len(outputs[1]) == 1

    # Type check
    for output in outputs:
        for element in output:
            assert isinstance(element, np.ndarray)
