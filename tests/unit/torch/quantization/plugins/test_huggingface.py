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
import warnings
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.misc import set_seed
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    get_tiny_llama,
    get_tiny_qwen3_moe,
    tf_modelopt_state_and_output_tester,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantLinear, QuantModuleRegistry

pytest.importorskip("transformers")

import transformers
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.dbrx.configuration_dbrx import DbrxConfig, DbrxFFNConfig
from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU, DbrxExperts, DbrxFFN


class HFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # initialization is (out_features, in_features) instead of (in_features, out_features)
            transformers.pytorch_utils.Conv1D(5, 3),
            nn.ReLU(),
            transformers.pytorch_utils.Conv1D(5, 5),
        )

    def forward(self, x):
        return self.net(x)


class PytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            QuantLinear(3, 5),
            nn.ReLU(),
            QuantLinear(5, 5),
        )

    def forward(self, x):
        return self.net(x)


def test_convert_conv1d():
    set_seed()
    assert transformers.pytorch_utils.Conv1D in QuantModuleRegistry

    model_ref = HFModel()
    model_test = HFModel()
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for name, module in model_test.named_modules():
        if isinstance(module, transformers.pytorch_utils.Conv1D):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = torch.randn(2, 3)
    out_1 = model_ref(x)
    out_2 = model_test(x)

    assert torch.allclose(out_1, out_2)

    mtq.set_quantizer_attribute(model_test, "*input_quantizer", {"enable": True})
    mtq.set_quantizer_attribute(model_test, "*weight_quantizer", {"enable": True})
    model_ref = PytorchModel()
    model_ref.load_state_dict(model_test.state_dict())

    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)


def test_dbrx():
    assert DbrxExperts in QuantModuleRegistry
    assert DbrxExpertGLU in QuantModuleRegistry

    config = DbrxConfig(
        ffn_config=DbrxFFNConfig(ffn_hidden_size=8, moe_num_experts=2), hidden_size=32
    )

    model_ref = DbrxFFN(config)
    model_test = DbrxFFN(config)
    with torch.no_grad():
        model_ref.experts.mlp.w1.copy_(torch.randn(16, 32))
        model_ref.experts.mlp.v1.copy_(torch.randn(16, 32))
        model_ref.experts.mlp.w2.copy_(torch.randn(16, 32))

    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)

    expertglu_ref = model_ref.experts.mlp
    expertglu_test = model_test.experts.mlp

    assert hasattr(expertglu_test, "w1_linear") and not hasattr(expertglu_test, "w1")
    assert hasattr(expertglu_test, "v1_linear") and not hasattr(expertglu_test, "v1")
    assert hasattr(expertglu_test, "w2_linear") and not hasattr(expertglu_test, "w2")

    assert torch.allclose(
        torch.concat(list(expertglu_test.w1_linear.parameters()), dim=0),
        expertglu_ref.w1,
    )

    mtq.set_quantizer_attribute(model_test, "*", {"enable": False})

    x = torch.randn(1, 4, 32)
    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1[0], out_2[0])


@pytest.mark.parametrize("method", ["gradient", "kl_div"])
@pytest.mark.parametrize("model_provider", [get_tiny_llama, get_tiny_qwen3_moe])
def test_autoquantize_huggingface(model_provider, method):
    model = model_provider()
    input_ids = model.dummy_inputs["input_ids"]

    def forward_step(model, batch):
        return model(**batch) if method == "gradient" else model(**batch).logits

    warnings.filterwarnings(
        "error", message="AutoQuantize: Error enabling gradient checkpointing for huggingface model"
    )

    # Gradient checkpointing warning should only appear for gradient-based method
    context = (
        pytest.warns(
            UserWarning,
            match="AutoQuantize: Huggingface model detected - Enabling gradient checkpointing. ",
        )
        if method == "gradient"
        else nullcontext()
    )

    with context:
        best_model, search_history = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 11.0},
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            data_loader=[{"input_ids": input_ids, "labels": input_ids} for _ in range(2)],
            forward_step=forward_step,
            loss_func=lambda output, data: output.loss,
            num_calib_steps=2,
            num_score_steps=2,
            verbose=True,
            method=method,
        )


@pytest.mark.parametrize(
    ("model_cls", "quant_config"),
    [
        (LlamaForCausalLM, mtq.INT4_AWQ_CFG),
        (AutoModelForCausalLM, mtq.INT4_AWQ_CFG),
    ],
)
def test_quantized_transformers_save_restore(tmp_path, model_cls, quant_config):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)
    # update config to fit test cases
    if quant_config == mtq.INT4_AWQ_CFG:
        quant_config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: 16}
    else:
        raise ValueError(f"Unsupported quant_config: {quant_config}")

    model_ref = model_cls.from_pretrained(tiny_llama_dir)
    mtq.quantize(model_ref, quant_config, lambda model: model(**model.dummy_inputs))
    mtq.compress(model_ref)
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")
    assert os.path.exists(tiny_llama_dir / "modelopt_model/modelopt_state.pth")

    model_test = model_cls.from_pretrained(tiny_llama_dir / "modelopt_model")
    tf_modelopt_state_and_output_tester(model_ref, model_test)
