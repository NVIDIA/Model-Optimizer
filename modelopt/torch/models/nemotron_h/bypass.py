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

"""Nemotron mixer roles; attention and Mamba mixers remain frozen boundaries."""

from modelopt.torch.distill._bypass_capture import ModuleBypassAdapter


class NemotronHBypassAdapter(ModuleBypassAdapter):
    """Map model-owned block types to their feed-forward mixer child."""

    def unit_attributes(self, block):
        """Only dense and routed feed-forward blocks define default repair losses."""
        return ("mixer",) if getattr(block, "block_type", None) in ("mlp", "moe") else ()
