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
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.quantization.nn import QuantConv2d, QuantLinear


class QuantConvLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QuantConv2d(1, 8, kernel_size=3)
        self.conv2 = QuantConv2d(8, 4, kernel_size=3)
        self.fc1 = QuantLinear(64, 8)
        self.fc2 = QuantLinear(8, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_input(self):
        return torch.randn(2, 1, 8, 8)


class SimpleLinear(nn.Module):
    """Test Linear model for ONNX export."""

    def __init__(self, bias=True, dtype=torch.float32, add_linear=False):
        super().__init__()
        self.add_linear = add_linear
        self.net = nn.Sequential(
            nn.Linear(16, 32, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(32, 64, bias=bias, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 16, bias=bias, dtype=dtype),
        )
        if add_linear:
            self.linear1 = nn.Linear(16, 16, bias=bias, dtype=dtype)

    def forward(self, x):
        x = self.net(x)
        if self.add_linear:
            x = self.linear1(x)
        return x

    @classmethod
    def get_input(cls):
        return torch.randn(
            2,
            16,
        )


class SimpleConv(nn.Module):
    """Test Conv model for ONNX export."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

    @classmethod
    def get_input(cls):
        return torch.randn(2, 16, 4, 4)


class SimpleConvLinear(nn.Module):
    """Test ConvLinear model for ONNX export."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

    @classmethod
    def get_input(cls):
        return torch.randn(4, 16, 4, 4)


class SDPAAttention(nn.Module):
    def forward(self, qkv: tuple, extra_arg=None):
        # NOTE: Add unused argument y with default value to test that replaced attention retain original defaults
        q, k, v = qkv
        return nn.functional.scaled_dot_product_attention(q, k, v)

    @classmethod
    def get_input(cls, device: str = "cpu"):
        q = torch.randn(1, 4, 8, device=device)
        k = torch.randn(1, 4, 8, device=device)
        v = torch.randn(1, 4, 8, device=device)
        return (q, k, v)


# NOTE: This should match configuration of _test_utils.torch.megatron.models.MegatronModel
class RegularQuantModelForTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = QuantLinear(32, 64)
        self.activation = nn.ReLU()
        self.fc2 = QuantLinear(64, 32)

    def forward(self, x):
        for block in [self.fc1, self.activation, self.fc2]:
            x = block(x)
            if isinstance(x, tuple):
                x = x[0]
        return x


class OneLayerLinear(torch.nn.Module):
    """Linear model for testing."""

    def __init__(self, in_features=32, out_features=64, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Sequential(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x):
        return self.net(x)

    def get_input(self, batch_size=2):
        return torch.randn(batch_size, self.in_features)


TOY_MOE_EXPERTS = 4
TOY_MOE_HIDDEN = 256
TOY_MOE_INTERMEDIATE = 256


class ToyFusedExperts(nn.Module):
    """HF transformers 5 fused-experts layout: one ``F.linear`` pair per routed expert."""

    def __init__(self):
        super().__init__()
        self.num_experts = TOY_MOE_EXPERTS
        self.gate_up_proj = nn.Parameter(
            torch.randn(TOY_MOE_EXPERTS, 2 * TOY_MOE_INTERMEDIATE, TOY_MOE_HIDDEN) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(TOY_MOE_EXPERTS, TOY_MOE_HIDDEN, TOY_MOE_INTERMEDIATE) * 0.02
        )

    def forward(self, hidden_states, expert_of_token):
        output = torch.zeros_like(hidden_states)
        for idx in range(self.num_experts):
            tokens = (expert_of_token == idx).nonzero().squeeze(-1)
            if len(tokens) == 0:
                continue
            gate, up = F.linear(hidden_states[tokens], self.gate_up_proj[idx]).chunk(2, dim=-1)
            output[tokens] = F.linear(F.silu(gate) * up, self.down_proj[idx])
        return output


class ToyMoEBlock(nn.Module):
    """Routes token ``t`` to expert ``t % 3``, so the last expert receives no token."""

    def __init__(self):
        super().__init__()
        self.experts = ToyFusedExperts()
        self.shared_expert = nn.Linear(TOY_MOE_HIDDEN, TOY_MOE_HIDDEN, bias=False)

    def forward(self, hidden_states):
        routed = self.experts(hidden_states, torch.arange(len(hidden_states)) % 3)
        return routed + self.shared_expert(hidden_states)


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = ToyMoEBlock()

    def forward(self, hidden_states):
        return self.mlp(hidden_states)
