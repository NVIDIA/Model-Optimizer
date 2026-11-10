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

import torch.nn as nn

from modelopt.torch.puzzletron.sewing_kit.passage import Passage


def test_passages_do_not_share_outputs_cache():
    first = Passage.create(nn.Linear(2, 2))
    first.outputs_cache["layer"] = "cached"
    assert Passage.create(nn.Linear(2, 2)).outputs_cache == {}
