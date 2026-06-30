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

from _test_utils.torch.nas_prune.models import DepthModel1

import modelopt.torch.nas as mtn
from modelopt.torch.nas.search_space import SearchSpace


def test_fastnas_search_space(use_channel_div_4):
    model = DepthModel1()
    model = mtn.convert(model, mode="fastnas")

    ss = SearchSpace(model)
    hps = list(ss.named_hparams(configurable=True))
    assert len(hps) == 1
    assert hps[0][0].rpartition(".")[2] == "out_channels"  # depth is not configurable for FastNAS
