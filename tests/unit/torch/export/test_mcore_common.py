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
"""The model packages' Megatron-Core mappings reach the registries ``mcore_common`` exposes."""

from pathlib import Path

import pytest

import modelopt.torch.models as models
from modelopt.torch.export.plugins import mcore_custom
from modelopt.torch.export.plugins.mcore_common import MCORE_MODEL_TYPES


def test_every_model_mcore_module_is_imported():
    # A model's mbridge_hf_mapping.py registers its mappings only if mcore_common imports it; a
    # forgotten one would surface as a KeyError at export time on a GPU.
    on_disk = {
        path.parent.name for path in Path(models.__file__).parent.glob("*/mbridge_hf_mapping.py")
    }
    assert on_disk == set(MCORE_MODEL_TYPES)


def test_an_architecture_registers_once(monkeypatch):
    monkeypatch.setitem(mcore_custom.all_mcore_hf_export_mapping, "TestArchForCausalLM", {})
    with pytest.raises(ValueError):
        mcore_custom.register_mcore_hf_mapping("TestArchForCausalLM", export_mapping={})
