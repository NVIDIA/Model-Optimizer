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

"""Lion on bounded FP32 native-field chunks and a direct peak learning rate."""

import math

import torch
from torch.optim import Optimizer

_LION_CHUNK_ELEMENTS = 1 << 18


class Lion(Optimizer):
    """Sign-momentum Lion with FP32 state, betas (0.9, 0.99), and zero decay."""

    def __init__(self, params, lr=0.001):
        """Configure native-field groups without a study-specific LR multiplier."""
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("Lion learning rate must be finite and nonnegative")
        super().__init__(params, {"lr": lr, "betas": (0.9, 0.99), "weight_decay": 0.0})
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.dtype != torch.float32 or not parameter.is_contiguous():
                    raise ValueError("Bypass Lion requires contiguous FP32 native-field masters")

    @torch.no_grad()
    def step(self, closure=None):
        """Apply Lion's direction and moment coefficients without modifying gradients."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and (p.grad.is_sparse or not p.grad.is_contiguous()):
                    raise ValueError("Bypass Lion requires contiguous dense gradients")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                value, gradient, momentum = (
                    parameter.view(-1),
                    parameter.grad.view(-1),
                    state["exp_avg"].view(-1),
                )
                for start in range(0, parameter.numel(), _LION_CHUNK_ELEMENTS):
                    chunk = slice(start, start + _LION_CHUNK_ELEMENTS)
                    direction = momentum[chunk].mul(beta1).add_(gradient[chunk], alpha=1 - beta1)
                    value[chunk].add_(direction.sign_(), alpha=-group["lr"])
                    momentum[chunk].mul_(beta2).add_(gradient[chunk], alpha=1 - beta2)
        return loss


def learning_rate(config, step):
    """One-based update LR: ceil(fraction * budget) warmup, then the chosen decay."""
    warmup = min(config.num_steps, math.ceil(config.num_steps * config.warmup_fraction))
    if step <= warmup:
        return config.max_lr * step / max(1, warmup)
    fraction = (step - warmup) / max(1, config.num_steps - warmup)
    factor = (
        (1 + math.cos(math.pi * fraction)) / 2
        if config.scheduler == "cosine"
        else 1 - fraction
        if config.scheduler == "linear"
        else 1.0
    )
    return config.max_lr * factor
