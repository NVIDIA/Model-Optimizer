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
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    tf_modelopt_state_and_output_tester,
)

pytest.importorskip("peft")
from _test_utils.torch.opt.utils import apply_mode_with_sampling
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def test_peft_save_restore(tmp_path):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    model_ref = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        target_modules=["q_proj", "v_proj", "k_proj"],
    )
    model_ref = get_peft_model(model_ref, peft_config)
    model_ref = apply_mode_with_sampling(
        model_ref, ["sparse_magnitude", "export_sparse", "quantize"]
    )
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_peft_model")

    model_test = AutoPeftModelForCausalLM.from_pretrained(tiny_llama_dir / "modelopt_peft_model")
    tf_modelopt_state_and_output_tester(model_ref, model_test)
