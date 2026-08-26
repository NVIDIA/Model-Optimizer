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

"""Hparam for Depth."""

from collections.abc import Callable

from modelopt.torch.trace import Symbol, SymDepth

from ..traced_hp import TracedHp, TracedHpRegistry

__all__ = ["DepthHparam"]


@TracedHpRegistry.register(SymDepth)
class DepthHparam(TracedHp):
    """Hparam describing depth."""

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], TracedHp]
    ) -> dict[Symbol, TracedHp]:
        assert isinstance(sym, SymDepth), f"Unexpected type {type(sym)} for {sym}!"
        assert not sym._dependencies, "Depth should not have any dependencies!"
        assert not sym._parent, "Depth should not have any parents!"

        # we can only skip layers when all layers after are skippable
        choices = {sym.max_depth}
        for i in range(sym.max_depth - 1, -1, -1):
            if not sym.is_skippable(i):
                break
            choices.add(i)

        # record choices and skippable layers
        self.choices = list(choices & set(self.choices))

        return super()._resolve_dependencies(sym, get_hp)
