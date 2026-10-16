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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The standalone NVFP4 probe must initialize attributes omitted by the preset."""

import importlib.util
import pathlib


def test_nvfp4_probe_expands_quantizer_defaults():
    script = (
        pathlib.Path(__file__).resolve().parents[3]
        / "plugins/modelopt/skills/ptq/scripts/verify_environment.py"
    )
    spec = importlib.util.spec_from_file_location("environment_probe", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    quantizer = module.nvfp4_quantizer()
    assert isinstance(quantizer.rotate_back_is_enabled, bool)
    assert quantizer.fake_quant
