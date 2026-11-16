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
"""A tiny LLaMA with GGML-quantized MLPs, for the ``convert_ggml_to_int8.py`` example tests."""

import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM

import modelopt.torch.quantization as mtq

IQ1_S = {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"}
IQ2_XS = {"num_bits": "iq2_xs", "block_sizes": {-1: 256}, "backend": "ggml"}


def quantize_tiny_llama(tmp_path, layer_cfgs=(IQ1_S, IQ2_XS)):
    """Return a two-layer tiny LLaMA whose layer ``i`` MLP takes ``layer_cfgs[i]``, and its tokens.

    The default formats have 32- and 16-weight sub-blocks, so the INT8 checkpoint needs a config
    group for each. The shapes suit vLLM's kernels, and fake quant runs on CPU.
    """
    source = create_tiny_llama_dir(
        tmp_path,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        vocab_size=128,
        initializer_range=0.05,
    )
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            *(
                {"quantizer_name": f"*layers.{i}.mlp*weight_quantizer", "cfg": cfg}
                for i, cfg in enumerate(layer_cfgs)
            ),
        ],
        "algorithm": "max",
    }
    token_ids = torch.randint(0, 128, (1, 48), generator=torch.Generator().manual_seed(0))
    mtq.quantize(model, quant_cfg, lambda m: m(token_ids))
    return model, token_ids
