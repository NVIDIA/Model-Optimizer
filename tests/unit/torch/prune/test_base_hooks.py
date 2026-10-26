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

import torch

from modelopt.torch.prune.importance_hooks.base_hooks import ForwardHook, L2NormHook


def test_save_hook_states_returns_the_written_checkpoint(tmp_path):
    hook = L2NormHook()
    hook(torch.nn.Identity(), (torch.ones(1, 2, 3),), None)
    path = ForwardHook.save_hook_states({"layer": hook}, tmp_path)
    state = torch.load(path, weights_only=True)
    assert path.parent == tmp_path
    assert state.keys() == {"layer"}
    for name, value in hook.state_dict().items():
        torch.testing.assert_close(state["layer"][name], value)
