# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import modelopt.torch.nas as mtn
import modelopt.torch.opt as mto
from modelopt.torch.opt.utils import search_space_size


def apply_mode_with_sampling(model, mode):
    for i, m in enumerate(mode):
        model = mto.apply_mode(model, mode=m, init_state=i == 0)
        config = mtn.get_subnet_config(model)
        ss_size = search_space_size(model)
        if m in ["fastnas", "autonas", "gradnas"]:
            while config == mtn.get_subnet_config(model) and ss_size > 1:
                mtn.sample(model)
    return model
