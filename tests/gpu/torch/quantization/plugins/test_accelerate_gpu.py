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
import torch
from _test_utils.torch.quantization.quantize_common import INT4_AWQ_CLIP_CFG
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import (
    enable_weight_access_and_writeback,
    is_quantized_linear,
)


@pytest.mark.parametrize(
    "quant_cfg",
    [
        mtq.INT4_AWQ_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        INT4_AWQ_CLIP_CFG,
        mtq.NVFP4_SVDQUANT_DEFAULT_CFG,
        mtq.INT8_DEFAULT_CFG,
    ],
)
def test_cpu_offloaded_tinyllama(tmp_path, quant_cfg):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_hidden_layers=2)

    config = AutoConfig.from_pretrained(tiny_llama_dir)

    model_ref = AutoModelForCausalLM.from_pretrained(
        tiny_llama_dir, torch_dtype=config.torch_dtype
    ).cuda()
    inputs = torch.randint(0, model_ref.config.vocab_size, (1, 4)).cuda()

    mtq.quantize(model_ref, quant_cfg, lambda model: model(inputs))
    output_ref = model_ref(inputs)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = {
        n: 0
        for n, m in model.named_modules()
        if "layers" not in n or n.split("layers.")[-1].isdigit()
    }
    device_map["model.layers.0"] = "cpu"

    model = load_checkpoint_and_dispatch(model, tiny_llama_dir, device_map=device_map)

    assert all(p.device == torch.device("meta") for p in model.model.layers[0].parameters())

    mtq.quantize(model, quant_cfg, lambda model: model(inputs))
    output_test = model(inputs)

    for name, module in model.named_modules():
        if is_quantized_linear(module):
            with enable_weight_access_and_writeback(module, model):
                assert torch.allclose(module.weight, model_ref.get_submodule(name).weight)

    assert torch.allclose(output_ref.logits, output_test.logits)
