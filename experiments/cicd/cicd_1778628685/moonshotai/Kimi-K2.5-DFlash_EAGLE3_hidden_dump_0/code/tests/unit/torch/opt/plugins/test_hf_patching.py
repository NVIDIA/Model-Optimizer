# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    get_tiny_qwen3,
    tf_output_tester,
)
from transformers import AutoModelForCausalLM

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto


@pytest.mark.parametrize(
    ("model_cls", "teacher_model_type"),
    [
        (AutoModelForCausalLM, "llama"),
        (AutoModelForCausalLM, "qwen3"),
    ],
)
def test_nested_model_save_restore(tmp_path, model_cls, teacher_model_type):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)

    model_ref = model_cls.from_pretrained(tiny_llama_dir)

    if teacher_model_type == "qwen3":
        teacher_model = get_tiny_qwen3()
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)

    kd_config = {
        "teacher_model": teacher_model,
        "criterion": mtd.LogitsDistillationLoss(),
    }
    model = mtd.convert(model_ref, mode=[("kd_loss", kd_config)])
    model.save_pretrained(tiny_llama_dir / "modelopt_model")

    model_test = model_cls.from_pretrained(tiny_llama_dir / "modelopt_model")

    tf_output_tester(model, model_test)
    # KD state is not saved and it should be empty
    assert not mto.ModeloptStateManager(model_test).has_state
