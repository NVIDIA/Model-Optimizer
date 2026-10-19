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

import pytest

pytest.importorskip("transformers")

from _test_utils.torch.transformers_models import create_tiny_qwen3_dir
from transformers import AutoModelForCausalLM

import modelopt.torch.puzzletron as mtpz


def test_convert_anymodel(tmp_path):
    input_dir = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    output_dir = tmp_path / "qwen3-0.6b-anymodel"
    mtpz.anymodel.convert_model(input_dir, output_dir, converter="qwen3")

    descriptor = mtpz.anymodel.ModelDescriptorFactory.get("qwen3")
    with mtpz.anymodel.deci_x_patcher(descriptor):
        _ = AutoModelForCausalLM.from_pretrained(output_dir)
