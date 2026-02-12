#!/usr/bin/env python
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

"""Triton quantization kernels."""

import torch

from modelopt.torch.utils import import_plugin

IS_AVAILABLE = False

if torch.cuda.is_available():
    with import_plugin(
        "triton",
        msg_if_missing=(
            "Your device is potentially capable to use triton kernel to speed up "
            "quantization simulations. Try to install triton with `pip install triton`."
        ),
    ):
        # fp4_kernel works on any CUDA GPU with triton
        from .fp4_kernel import *

        # fp4_kernel_hopper requires compute >= 8.9 (uses tl.float8e4nv)
        if torch.cuda.get_device_capability() >= (8, 9):
            from .fp4_kernel_hopper import *

        IS_AVAILABLE = True
