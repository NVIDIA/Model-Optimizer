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

"""``get_live_pool`` hands back a live pool and replaces a killed one."""

from types import SimpleNamespace

from _test_utils.torch.distributed.utils import get_live_pool


def test_a_killed_pool_is_replaced_and_a_live_one_is_reused():
    pools = {}
    made = []

    def make():
        made.append(SimpleNamespace(broken=False))
        return made[-1]

    first = get_live_pool(pools, 2, make)
    assert get_live_pool(pools, 2, make) is first
    first.broken = True
    second = get_live_pool(pools, 2, make)
    assert second is not first and len(made) == 2
