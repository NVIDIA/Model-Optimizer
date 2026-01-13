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


def infer_weights_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype:
    weights_dtype = [p.dtype for p in state_dict.values() if torch.is_floating_point(p)]
    weights_dtype = weights_dtype[0] if len(weights_dtype) > 0 else torch.get_default_dtype()
    return weights_dtype
