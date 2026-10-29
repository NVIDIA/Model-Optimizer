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

import pytest

from modelopt.torch.export.plugins import mcore_custom


@pytest.mark.parametrize(
    "mapping",
    [
        mcore_custom.CustomModuleMapping,
        mcore_custom.NameRemapping,
        mcore_custom.QKVMerging,
        mcore_custom.GroupedMLPMerging,
        mcore_custom.GroupedMLPSlicing,
        mcore_custom.GroupedGatedMLPSlicing,
        mcore_custom.GatedMLPMerging,
        mcore_custom.QKVSlicing,
        mcore_custom.SelfAttentionScaling,
        mcore_custom.GatedMLPSlicing,
        mcore_custom.GatedDeltaNetSlicing,
        mcore_custom.PackNameRemapping,
        mcore_custom.UnpackNameRemapping,
        mcore_custom.PackNameRemappingGPT,
        mcore_custom.UnpackNameRemappingGPT,
    ],
)
def test_mapping_default_kwargs_are_isolated(mapping):
    first = mapping()
    original = dict(first.func_kwargs)
    first.func_kwargs["custom_option"] = True
    assert mapping().func_kwargs == original
