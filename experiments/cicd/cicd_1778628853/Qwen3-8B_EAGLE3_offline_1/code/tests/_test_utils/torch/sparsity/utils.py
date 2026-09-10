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

import torch

from modelopt.torch.nas.search_space import SearchSpace
from modelopt.torch.opt.dynamic import DynamicModule


def sample_subnet_with_sparsity(model, sample_func=min):
    for module in model.modules():
        if isinstance(model, DynamicModule):
            if hasattr(module, "set_mask"):
                module.set_mask(torch.rand_like(module.weight) > 0.5)
            SearchSpace(module).sample(sample_func)
