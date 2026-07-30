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


import onnx
import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    create_tiny_llama_seq_cls_dir,
)

# Tiny stand-ins for the target architectures: a plain encoder for the embedding
# path and a sequence-classification model (with a `score` head, kept unquantized
# by the recipe) for the reranking path.
_MODEL_FACTORIES = {
    "embedding": create_tiny_llama_dir,
    "reranking": create_tiny_llama_seq_cls_dir,
}

# Every projection output extent must be divisible by the NVFP4 block size (16)
# for the recipe's output quantizers: kv output = 2 kv-heads * head_dim 16 = 32.
_TINY_CONFIG = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
}


@pytest.mark.parametrize(
    ("model_kind", "recipe", "expected_op"),
    [
        (
            "embedding",
            "huggingface/nemotron_llama/ptq/nvfp4_output_quant_proj",
            "TRT_FP4DynamicQuantize",
        ),
        (
            "reranking",
            "huggingface/nemotron_llama/ptq/fp8_output_quant_proj",
            "QuantizeLinear",
        ),
    ],
)
def test_hf_embedding_quant_to_onnx(tmp_path, model_kind, recipe, expected_op):
    model_dir = _MODEL_FACTORIES[model_kind](tmp_path, with_tokenizer=True, **_TINY_CONFIG)
    onnx_save_path = tmp_path / f"{model_kind}_nvfp4.onnx"

    cmd_parts = extend_cmd_parts(
        ["python", "hf_embedding_quant_to_onnx.py"],
        model_path=str(model_dir),
        recipe=recipe,
        onnx_save_path=str(onnx_save_path),
        calibration_data_size="2",
        batch_size="2",
    )
    run_example_command(cmd_parts, "torch_onnx")

    assert onnx_save_path.exists()
    op_types = {node.op_type for node in onnx.load(onnx_save_path).graph.node}
    assert expected_op in op_types
    if expected_op == "QuantizeLinear":
        assert "TRT_FP4DynamicQuantize" not in op_types
