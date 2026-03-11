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

from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from typing_extensions import override


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_load_state_dict_post_hook(self.load_state_dict_post_hook)

    @staticmethod
    def load_state_dict_post_hook(
        module: torch.nn.Module,
        incompatible_keys: torch.nn.modules.module._IncompatibleKeys,
    ) -> None:
        incompatible_keys.missing_keys.clear()
        incompatible_keys.unexpected_keys.clear()


class DummyBlock(DummyModule):
    def __init__(self, block_index: int):
        super().__init__()
        self.block_index = block_index

    @override
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        return x


class DummyWTE(DummyModule):
    def __init__(self, hidden_size: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.n_embd = hidden_size
        self.dtype = dtype

    @override
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        result = torch.ones((B, T, self.n_embd), dtype=self.dtype, device=input_ids.device)
        return result


class DummyLMHead(DummyModule):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.vocab_size = config.vocab_size

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        result = torch.ones((B, T, self.vocab_size), dtype=x.dtype, device=x.device)
        return result
