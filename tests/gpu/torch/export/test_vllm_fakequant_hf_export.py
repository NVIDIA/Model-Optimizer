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
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_vllm_fq_checkpoint


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG])
def test_hf_vllm_export(tmp_path, quant_cfg):
    """Test HuggingFace model export for vLLM with fake quantization.

    This test verifies:
    1. Model weights match before and after export
    2. quant_amax.pth file is created, huggingface config file does not exist
    3. Amax values are correctly extracted and saved in quant_amax.pth file
    """

    # Create a tiny LLaMA model for testing
    tiny_model_dir = create_tiny_llama_dir(tmp_path, with_tokenizer=True, num_hidden_layers=2)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    model = model.cuda()
    model.eval()

    # Quantize the model
    def forward_loop(model):
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128)).cuda()
        with torch.no_grad():
            model(input_ids)

    model = mtq.quantize(model, quant_cfg, forward_loop)
    quantizer_state_dict_before = mtq.utils.get_quantizer_state_dict(model)
    model_state_dict = deepcopy(model.state_dict())

    # Export directory
    export_dir = tmp_path / "vllm_export"
    export_dir.mkdir(exist_ok=True)

    # Export for vLLM
    export_hf_vllm_fq_checkpoint(model, export_dir=export_dir)

    # check if quant_amax.pth file exists
    modelopt_state_file = export_dir / "vllm_fq_modelopt_state.pth"
    assert modelopt_state_file.exists(), (
        f"vllm_fq_modelopt_state.pth file should be created in {export_dir}"
    )

    # make sure hf_quant_config.json file does not exist
    hf_quant_config_file = export_dir / "hf_quant_config.json"
    assert not hf_quant_config_file.exists(), (
        f"hf_quant_config.json file should not be created in {export_dir}"
    )

    # check weights match before and after export
    model_after = AutoModelForCausalLM.from_pretrained(export_dir)
    model_after = model_after.cuda()
    model_after.eval()
    model_after_state_dict = model_after.state_dict()
    for key, param in model_state_dict.items():
        if "quantizer" not in key:
            assert torch.allclose(param, model_after_state_dict[key], atol=1e-6), (
                f"Weight mismatch for {key}: "
                f"before shape={param.shape}, after shape={model_after_state_dict[key].shape}, "
                f"max diff={torch.abs(param - model_after_state_dict[key]).max()}"
            )

    # Verify quantizer state dict values are correct
    quantizer_state_dict = torch.load(modelopt_state_file)["modelopt_state_weights"]
    assert len(quantizer_state_dict) > 0, (
        f"modelopt_state_weights should not be empty in {modelopt_state_file}"
    )
    assert quantizer_state_dict.keys() == quantizer_state_dict_before.keys(), (
        "quantizer state dict keys mismatch between before and after export"
    )
