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

"""Native deepseek_v4 repair descriptor."""

from ..runners import native_fast_replay
from .base import NativeFamilyAdapter
from .registry import register

__all__ = []


class _DeepSeekV4Family(NativeFamilyAdapter):
    def snapshot_unit_state(self, module):
        gate = module._modules.get("gate")
        if gate is None or not hasattr(gate, "_pending_input_ids"):
            return None
        return {
            name: getattr(gate, name)
            for name in ("_pending_input_ids", "_pending_vision_token_types")
            if hasattr(gate, name)
        }

    def replay_unit(self, module, args, kwargs, replay_state=None):
        if replay_state is None:
            return super().replay_unit(module, args, kwargs)
        gate = module.gate
        previous = {name: getattr(gate, name) for name in replay_state}
        try:
            for name, value in replay_state.items():
                setattr(gate, name, value)
            return super().replay_unit(module, args, kwargs)
        finally:
            for name, value in previous.items():
                setattr(gate, name, value)


register(_DeepSeekV4Family("deepseek_v4", unit_runner=native_fast_replay))
