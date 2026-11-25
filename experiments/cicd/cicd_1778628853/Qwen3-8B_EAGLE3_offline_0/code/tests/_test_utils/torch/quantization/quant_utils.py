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


def quant(x, amax, num_bits=8, fake=False, narrow_range=True):
    """Quantize x using torch."""
    intmax = 2.0 ** (num_bits - 1) - 1.0
    intmin = -intmax if narrow_range else -intmax - 1
    scale = intmax / amax
    # x_q = torch.round(torch.clamp(x * scale, intmin, intmax))
    x_q = torch.clamp((x * scale).round_(), intmin, intmax)

    if fake:
        x_q /= scale

    return x_q


def get_model_size(model):
    return sum([p.element_size() * p.nelement() for p in model.parameters()])
