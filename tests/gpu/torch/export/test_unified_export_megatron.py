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

import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import pytest
import torch
import transformers
from _test_utils.import_helper import skip_if_no_megatron
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_gpt_model
from _test_utils.torch.megatron.utils import get_forward
from _test_utils.torch.transformers_models import create_tiny_llama_dir

skip_if_no_megatron(apex_or_te_required=True)

import modelopt.torch.quantization as mtq
import modelopt.torch.speculative as mtsp
from modelopt.torch.export import KV_CACHE_FP8, export_mcore_gpt_to_hf, import_mcore_gpt_from_hf
from modelopt.torch.speculative.eagle.default_config import default_eagle_config
from modelopt.torch.speculative.plugins.megatron_eagle import _DynamicEagleGPTModel
from modelopt.torch.speculative.plugins.megatron_medusa import _DynamicMedusaGPTModel


def _verify_model_quant_config(
    export_dir: Path, quant_config: str | None = None, kv_cache_quant_cfg: str | None = None
):
    """Verify config.json and hf_quant_config.json"""
    config_dict = json.load(open(export_dir / "config.json"))
    hf_quant_config_dict = json.load(open(export_dir / "hf_quant_config.json"))
    # Make sure config.json and hf_quant_config.json are consistent
    assert (
        config_dict["quantization_config"]["quant_algo"]
        == hf_quant_config_dict["quantization"]["quant_algo"]
    )
    assert (
        config_dict["quantization_config"]["ignore"]
        == hf_quant_config_dict["quantization"]["exclude_modules"]
    )

    # Verify config.json
    if kv_cache_quant_cfg:
        assert config_dict["quantization_config"]["kv_cache_scheme"]["num_bits"] == 8

    # Verify hf_quant_config.json
    if quant_config:
        quant_config_dict = hf_quant_config_dict["quantization"]
        quant_type = quant_config_dict["quant_algo"]
        assert (
            quant_type in quant_config
        )  # quant config str is subset of quant config e.g. NVFP4 -> NVFP4_DEFAULT_CFG
        assert len(quant_config_dict["exclude_modules"]) > 1  # Dynamically added exclude modules
        if quant_type == "NVFP4":
            assert quant_config_dict["group_size"] == 16

        if kv_cache_quant_cfg:
            assert quant_config_dict["kv_cache_quant_algo"] == KV_CACHE_FP8


def _test_unified_export_megatron(
    tmp_path, model_type, arch, extra_module, quant_config, kv_cache_quant_cfg, rank, size
):
    num_layers = 2
    hidden_size = 64
    num_attention_heads = 8
    num_query_groups = size
    ffn_hidden_size = 128
    max_sequence_length = 32
    vocab_size = 64

    arch = "NemotronForCausalLM" if model_type == "nemotron" else "LlamaForCausalLM"
    activation_func = "squared_relu" if model_type == "nemotron" else "swiglu"
    normalization = "LayerNorm" if model_type == "nemotron" else "RMSNorm"

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
        activation_func=activation_func,
        normalization=normalization,
        transformer_impl="modelopt",
    ).cuda()

    if quant_config:
        quant_config_dict = getattr(mtq, quant_config)
        if kv_cache_quant_cfg:
            kv_quant_cfg = getattr(mtq, kv_cache_quant_cfg)["quant_cfg"]
            quant_config_dict = mtq.utils.update_quant_cfg_with_kv_cache_quant(
                quant_config_dict, kv_quant_cfg
            )
        forward = get_forward(model)
        model = mtq.quantize(model, quant_config_dict, forward)

    if extra_module == "medusa":
        config = {
            "medusa_num_heads": 1,
            "medusa_num_layers": 1,
        }
        model = mtsp.convert(model, [("medusa", config)])
        assert isinstance(model, _DynamicMedusaGPTModel)
    elif extra_module == "eagle":
        config = {"eagle_architecture_config": deepcopy(default_eagle_config)}
        model = mtsp.convert(model, [("eagle", config)])
        assert isinstance(model, _DynamicEagleGPTModel)

    pretrained_config = {
        "architectures": [arch],
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

    tmp_export_dir = tmp_path / "export"
    export_mcore_gpt_to_hf(
        model,
        tmp_path if arch is not None else None,
        dtype=torch.bfloat16,
        export_dir=str(tmp_export_dir),
    )

    if quant_config:
        _verify_model_quant_config(tmp_export_dir, quant_config, kv_cache_quant_cfg)


@pytest.mark.parametrize(
    ("model_type", "arch", "extra_module", "quant_config", "kv_cache_quant_cfg"),
    [
        ("nemotron", "NemotronForCausalLM", None, None, None),
        ("nemotron", "NemotronForCausalLM", None, "NVFP4_DEFAULT_CFG", None),
        ("nemotron", "NemotronForCausalLM", None, "NVFP4_DEFAULT_CFG", "FP8_KV_CFG"),
        ("nemotron", "NemotronForCausalLM", "eagle", None, None),
        ("nemotron", "NemotronForCausalLM", "medusa", None, None),
        ("llama", "LlamaForCausalLM", None, None, None),
        ("llama", "LlamaForCausalLM", None, "FP8_DEFAULT_CFG", None),
        ("llama", "LlamaForCausalLM", None, "FP8_DEFAULT_CFG", "FP8_KV_CFG"),
        ("llama", "LlamaForCausalLM", "eagle", None, None),
        ("llama", "LlamaForCausalLM", "medusa", None, None),
    ],
)
def test_unified_export_megatron(
    tmp_path, model_type, arch, extra_module, quant_config, kv_cache_quant_cfg
):
    # TODO: Fix TP>1 failures
    spawn_multiprocess_job(
        size=1,  # torch.cuda.device_count(),
        job=partial(
            _test_unified_export_megatron,
            tmp_path,
            model_type,
            arch,
            extra_module,
            quant_config,
            kv_cache_quant_cfg,
        ),
        backend="nccl",
    )


def _test_unified_import_megatron(tiny_llama_dir, rank, size):
    config = transformers.AutoConfig.from_pretrained(tiny_llama_dir)

    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_query_groups = config.num_key_value_heads
    ffn_hidden_size = config.intermediate_size
    max_sequence_length = config.max_position_embeddings
    vocab_size = config.vocab_size
    activation_func = "swiglu"
    normalization = "RMSNorm"

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
        activation_func=activation_func,
        normalization=normalization,
    ).cuda()

    import_mcore_gpt_from_hf(model, tiny_llama_dir)


def test_unified_import_megatron(tmp_path):
    num_gpus = torch.cuda.device_count()
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, num_key_value_heads=num_gpus)
    spawn_multiprocess_job(
        size=num_gpus,
        job=partial(_test_unified_import_megatron, tiny_llama_dir),
        backend="nccl",
    )
