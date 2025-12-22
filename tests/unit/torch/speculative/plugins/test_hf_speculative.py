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

import os
from copy import deepcopy

import pytest
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    get_tiny_llama,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForCausalLM, LlamaForCausalLM

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE1_DEFAULT_CFG, EAGLE3_DEFAULT_CFG


def test_medusa_model_convert_save_and_restore(tmp_path):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    model_ref = LlamaForCausalLM.from_pretrained(tiny_llama_dir)

    config = {
        "medusa_num_heads": 2,
        "medusa_num_layers": 1,
    }
    mtsp.convert(model_ref, mode=[("medusa", config)])
    assert isinstance(model_ref, mtsp.plugins.HFMedusaModel)

    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")
    assert os.path.exists(tiny_llama_dir / "modelopt_model/modelopt_state.pth")

    model_test = AutoModelForCausalLM.from_pretrained(tiny_llama_dir / "modelopt_model")
    assert isinstance(model_test, mtsp.plugins.HFMedusaModel)
    tf_modelopt_state_and_output_tester(model_ref, model_test)


@pytest.mark.parametrize("eagle_config", [EAGLE1_DEFAULT_CFG, EAGLE3_DEFAULT_CFG])
def test_eagle_model_convert_save_and_restore(tmp_path, eagle_config):
    model_ref = get_tiny_llama(num_hidden_layers=8)

    config = deepcopy(eagle_config["config"])
    config["eagle_architecture_config"].update(
        {
            "draft_vocab_size": model_ref.config.vocab_size,
            "hidden_size": model_ref.config.hidden_size,
        }
    )

    mtsp.convert(model_ref, mode=[("eagle", config)])
    assert isinstance(model_ref, mtsp.plugins.HFEagleModel)

    model_ref.save_pretrained(tmp_path / "modelopt_model")
    assert os.path.exists(tmp_path / "modelopt_model/modelopt_state.pth")

    model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
    assert isinstance(model_test, mtsp.plugins.HFEagleModel)
    tf_modelopt_state_and_output_tester(model_ref, model_test)
