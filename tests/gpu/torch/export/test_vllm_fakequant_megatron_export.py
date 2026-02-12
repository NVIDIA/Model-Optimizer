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

import json
from functools import partial

import pytest
import torch
from _test_utils.import_helper import skip_if_no_megatron
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_mcore_gpt_to_hf_vllm_fq

skip_if_no_megatron(apex_or_te_required=True)


def _test_mcore_vllm_export(tmp_path, quant_cfg, rank, size):
    """Test megatron-core model export for vLLM with fake quantization."""
    # Create a tiny mcore GPT model
    num_layers = 2
    hidden_size = 64
    num_attention_heads = 8
    num_query_groups = size
    ffn_hidden_size = 128
    max_sequence_length = 32
    vocab_size = 64

    model = get_mcore_gpt_model(
        tensor_model_parallel_size=size,
        pipeline_model_parallel_size=1,
        initialize_megatron=True,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        activation_func="swiglu",
        normalization="RMSNorm",
        transformer_impl="modelopt",
    ).cuda()
    model.eval()

    # Quantize the model
    def forward_loop(model):
        batch_size = 1
        seq_len = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).cuda()
        # Create causal attention mask
        attention_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len))).cuda()
        attention_mask = attention_mask < 0.5  # Convert to boolean mask
        with torch.no_grad():
            model(input_ids, position_ids, attention_mask)

    model = mtq.quantize(model, quant_cfg, forward_loop)
    # Create HF config for export
    pretrained_config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden_size,
        "max_position_embeddings": max_sequence_length,
        "model_type": "llama",
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_query_groups,
        "torch_dtype": "bfloat16",
    }

    with open(tmp_path / "config.json", "w") as f:
        json.dump(pretrained_config, f)

    # Export directory
    export_dir = tmp_path / "vllm_export"
    export_dir.mkdir(exist_ok=True)

    # Export for vLLM
    export_mcore_gpt_to_hf_vllm_fq(
        model,
        pretrained_model_name_or_path=tmp_path,
        dtype=torch.bfloat16,
        export_dir=str(export_dir),
    )

    # check if quant_amax.pth file exists
    quant_amax_file = export_dir / "quantizer_state.pth"
    assert quant_amax_file.exists(), f"quantizer_state.pth file should be created in {export_dir}"

    # make sure hf_quant_config.json file does not exist
    hf_quant_config_file = export_dir / "hf_quant_config.json"
    assert not hf_quant_config_file.exists(), (
        f"hf_quant_config.json file should not be created in {export_dir}"
    )


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG])
def test_mcore_vllm_export(tmp_path, quant_cfg):
    """Wrapper test function for mcore vLLM export."""
    spawn_multiprocess_job(
        size=1,
        job=partial(_test_mcore_vllm_export, tmp_path, quant_cfg),
        backend="nccl",
    )
