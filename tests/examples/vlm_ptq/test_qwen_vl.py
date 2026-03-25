# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.examples.models import QWEN_VL_PATH
from _test_utils.examples.run_command import run_vlm_ptq_command


@pytest.mark.parametrize("quant", ["fp8", "int8_sq", "nvfp4"])
def test_qwen_vl(quant):
    run_vlm_ptq_command(model=QWEN_VL_PATH, quant=quant)
