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

"""A deterministic mixed-IQ problem shared by real Bypass and resume tests."""

from collections import OrderedDict

import torch
from torch import nn

import modelopt.torch.quantization as mtq


class BypassNetwork(nn.Sequential):
    def forward(self, hidden_states, attention_mask=None):
        return super().forward(hidden_states)


def bypass_problem():
    torch.manual_seed(37)
    student = BypassNetwork(
        OrderedDict(
            [
                ("mlp", nn.Linear(256, 256, bias=False)),
                ("activation", nn.SiLU()),
                ("moe", nn.Linear(256, 256, bias=False)),
                ("head", nn.Linear(256, 4, bias=False)),
            ]
        )
    )
    batches = [torch.randn(2, 3, 256) for _ in range(8)]
    mtq.quantize(
        student,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {
                    "quantizer_name": "mlp.weight_quantizer",
                    "cfg": {"num_bits": "iq1_m", "backend": "ggml"},
                },
                {
                    "quantizer_name": "moe.weight_quantizer",
                    "cfg": {"num_bits": "iq2_s", "backend": "ggml"},
                },
            ],
            "algorithm": "max",
        },
        lambda m: m(batches[0]),
    )
    teacher = BypassNetwork(
        OrderedDict(
            [
                ("mlp", nn.Linear(256, 256, bias=False)),
                ("activation", nn.SiLU()),
                ("moe", nn.Linear(256, 256, bias=False)),
                ("head", nn.Linear(256, 4, bias=False)),
            ]
        )
    )
    with torch.no_grad():
        for name in ("mlp", "moe"):
            layer = student.get_submodule(name)
            teacher.get_submodule(name).weight.copy_(layer.weight_quantizer(layer.weight) * 1.02)
        teacher.head.weight.copy_(student.head.weight)
    teacher.head.eval()
    return student, teacher, batches
