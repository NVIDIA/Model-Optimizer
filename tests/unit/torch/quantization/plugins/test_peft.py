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
import torch
from _test_utils.torch.transformers_models import get_tiny_gpt_oss, get_tiny_llama, tf_output_tester
from packaging.version import Version

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear

import modelopt.torch.quantization as mtq


def test_convert_loralinear():
    lora_config = LoraConfig(
        r=1,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_ref = get_tiny_llama()
    model_ref = get_peft_model(model_ref, lora_config)
    model_test = get_tiny_llama()
    model_test = get_peft_model(model_test, lora_config)
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for _, module in model_test.named_modules():
        if isinstance(module, LoraLinear):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    tf_output_tester(model_ref, model_test)


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.55"), reason="transformers < 4.55"
)
def test_peft_flow(tmp_path):
    model_original = get_tiny_gpt_oss(num_hidden_layers=1)

    model_full = get_tiny_gpt_oss(num_hidden_layers=1)
    model_full.load_state_dict(model_original.state_dict())

    model_base_lora = get_tiny_gpt_oss(num_hidden_layers=1)
    model_base_lora.load_state_dict(model_original.state_dict())

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=[
            "0.mlp.experts.gate_up_proj",
            "0.mlp.experts.down_proj",
        ],
    )

    peft_model = get_peft_model(model_base_lora, peft_config)

    input_ids = torch.randint(0, model_original.config.vocab_size, (1, 4))

    def forward_loop(model):
        return model(input_ids)

    mtq.quantize(peft_model, mtq.INT8_DEFAULT_CFG, forward_loop)
    mtq.quantize(model_full, mtq.INT8_DEFAULT_CFG, forward_loop)

    outputs_peft = peft_model(input_ids).logits
    outputs_full = model_full(input_ids).logits
    assert torch.allclose(outputs_peft, outputs_full)

    outputs_peft.sum().backward()
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None

    from peft.tuners.lora.layer import Linear as LoraLinear

    for name, module in peft_model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        with torch.no_grad():
            # Lora_B weights are initialized to 0, lets change it
            module.lora_B["default"].weight.copy_(torch.randn_like(module.lora_B["default"].weight))
            lora_total = (
                module.scaling["default"] * module.lora_B["default"].weight
            ) @ module.lora_A["default"].weight.data
            module_full = model_full.get_submodule(name.replace("base_model.model.", ""))
            module_full.weight.data += lora_total

    outputs_peft = peft_model(input_ids).logits
    outputs_full = model_full(input_ids).logits
    assert torch.allclose(outputs_peft, outputs_full)

    peft_model.save_pretrained(tmp_path / "peft_model")

    peft_loaded = PeftModel.from_pretrained(model_original, tmp_path / "peft_model")
    outputs_peft_loaded = peft_loaded(input_ids).logits
    assert torch.allclose(outputs_peft_loaded, outputs_full)

    model_after_peft_merge = peft_loaded.merge_and_unload()
    outputs_after_peft_merge = model_after_peft_merge(input_ids).logits
    assert torch.allclose(outputs_after_peft_merge, outputs_full)
