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

from torch import nn


class CachedModule(nn.Module):
    def __init__(self, block, select_cache_step_func) -> None:
        super().__init__()
        self.block = block
        self.select_cache_step_func = select_cache_step_func
        self.cur_step = 0
        self.cached_results = None
        self.enabled = True

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def if_cache(self):
        return self.select_cache_step_func(self.cur_step) and self.enabled

    def enable_cache(self):
        self.enabled = True

    def disable_cache(self):
        self.enabled = False
        self.cur_step = 0

    def forward(self, *args, **kwargs):
        if not self.if_cache():
            self.cached_results = self.block(*args, **kwargs)
        if self.enabled:
            self.cur_step += 1
        return self.cached_results
