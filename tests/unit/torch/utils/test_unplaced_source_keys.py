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

"""``unplaced_source_keys`` must count the tensors a dequantizing load folded into the weights.

These drive a real ``from_pretrained`` on CPU, because what is under test is which checkpoint
tensors the Transformers loader consumed: a dequantizing quantizer consumes scale and block tensors
the model's stock conversion mapping never mentions, and the export carries whatever this calls
unplaced. Each checkpoint also holds one genuinely stray tensor, a layer past
``num_hidden_layers``, which must still be reported.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM

from modelopt.torch.utils.plugins.model_load_utils import unplaced_source_keys

STRAY = "model.layers.1.input_layernorm.weight"


def _save(path, config, state, quantization_config):
    state[STRAY] = torch.ones(config.hidden_size, dtype=torch.bfloat16)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    config_json = config.to_dict()
    config_json["quantization_config"] = quantization_config
    (path / "config.json").write_text(json.dumps(config_json))


def _load(path, quantization_config):
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=quantization_config, dtype=torch.bfloat16
    )
    if getattr(model, "_weight_conversions", None) is None:
        pytest.skip("this transformers does not record the conversions from_pretrained ran")
    return model


def test_fp8_block_scales_folded_in_at_load_are_placed(tmp_path):
    """DeepSeek-V3 style FP8: dequantizing folds every ``weight_scale_inv`` into its weight."""
    from transformers import FineGrainedFP8Config, LlamaConfig, LlamaForCausalLM

    # Every projection a whole number of 128x128 blocks: transformers before 5.10.1 cannot
    # dequantize a partial block (a 2-head k_proj would be 64x128).
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        tie_word_embeddings=False,
    )
    state = {}
    for name, tensor in LlamaForCausalLM(config).to(torch.bfloat16).state_dict().items():
        if name.endswith("proj.weight"):
            # One 128x128 block per 128-wide tile; the values only need to be valid FP8.
            blocks = (-(-tensor.shape[0] // 128), -(-tensor.shape[1] // 128))
            state[name] = tensor.to(torch.float8_e4m3fn)
            state[name.removesuffix("weight") + "weight_scale_inv"] = torch.ones(blocks)
        else:
            state[name] = tensor
    fp8 = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}
    _save(tmp_path, config, state, {**fp8, "weight_block_size": [128, 128]})

    model = _load(tmp_path, FineGrainedFP8Config(dequantize=True))

    assert unplaced_source_keys(model, str(tmp_path)) == [STRAY]


def test_mxfp4_expert_blocks_folded_in_at_load_are_placed(tmp_path):
    """gpt-oss, loaded as examples/hf_ptq loads it: ``Mxfp4Config(dequantize=True)`` folds each
    expert's ``*_blocks`` and ``*_scales`` into a bf16 ``gate_up_proj`` / ``down_proj``."""
    from transformers import GptOssConfig, GptOssForCausalLM, Mxfp4Config

    hidden, inter, experts = 64, 64, 4
    config = GptOssConfig(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=experts,
        num_experts_per_tok=2,
        layer_types=["full_attention"],
        tie_word_embeddings=False,
    )
    packed = {"gate_up_proj": (2 * inter, hidden), "down_proj": (hidden, inter)}
    state = {}
    for name, tensor in GptOssForCausalLM(config).to(torch.bfloat16).state_dict().items():
        proj = name.rsplit(".", 1)[-1]
        if ".experts." in name and proj in packed:
            rows, cols = packed[proj]
            grid = (experts, rows, cols // 32)  # one E8M0 scale per 32-value block
            state[f"{name}_blocks"] = torch.zeros(*grid, 16, dtype=torch.uint8)
            state[f"{name}_scales"] = torch.full(grid, 127, dtype=torch.uint8)
        else:
            state[name] = tensor
    # The modules gpt-oss ships unquantized.
    bf16 = [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ]
    _save(tmp_path, config, state, {"quant_method": "mxfp4", "modules_to_not_convert": bf16})

    model = _load(tmp_path, Mxfp4Config(dequantize=True))

    assert unplaced_source_keys(model, str(tmp_path)) == [STRAY]
