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
import os

import pytest
import torch
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.transformers_models import create_tiny_llama_dir, create_tiny_t5_dir
from safetensors import safe_open


# Here we map each qformat -> the suffix we expect in the generated safetensors directory
@pytest.mark.parametrize(
    (
        "qformat",
        "expected_suffix",
        "fuse_input_scale",
        "fuse_weight_scale",
        "fuse_weight_scale_2",
        "fuse_pre_quant_scale",
        "fuse_svdquant_lora_a",
    ),
    [
        ("fp8", "tiny_llama-fp8", True, False, True, True, False),
        ("nvfp4", "tiny_llama-nvfp4", True, False, True, True, False),
        ("nvfp4_awq", "tiny_llama-nvfp4-awq", True, False, True, True, False),
        ("int4_awq", "tiny_llama-int4-awq", True, False, True, True, False),
        ("w4a8_awq", "tiny_llama-w4a8-awq", True, False, True, True, False),
        ("int8_wo", "tiny_llama-int8-wo", False, False, False, False, False),
        ("nvfp4_svdquant", "tiny_llama-nvfp4-svdquant", True, False, True, True, True),
    ],
)
def test_unified_hf_export_and_check_safetensors(
    tmp_path,
    qformat,
    expected_suffix,
    fuse_input_scale,
    fuse_weight_scale,
    fuse_weight_scale_2,
    fuse_pre_quant_scale,
    fuse_svdquant_lora_a,
):
    """
    1) Generates a .safetensors file by running hf_ptq.py with each --qformat.
    2) Checks the generated directory for the expected .safetensors file:
       model.safetensors
    3) Optionally, loads the file via safe_open to verify presence of data.

    :param tmp_path: pytest fixture for a temporary directory.
    :param qformat: the quantization format to test (e.g., 'fp8', 'nvfp4', etc.).
    :param expected_suffix: the directory suffix where the .safetensors is expected.
    """
    if expected_suffix.startswith("t5_tiny"):
        tiny_model_dir = create_tiny_t5_dir(
            tmp_path, with_tokenizer=True, num_hidden_layers=1, use_cache=False
        )
    else:
        tiny_model_dir = create_tiny_llama_dir(tmp_path, with_tokenizer=True, num_hidden_layers=1)

    # Create an output directory in tmp_path
    # We'll replicate the naming convention, e.g. "tiny_llama-fp8"
    output_dir = tmp_path / expected_suffix

    # Command to generate .safetensors
    cmd_parts = extend_cmd_parts(
        ["python", "hf_ptq.py"],
        pyt_ckpt_path=tiny_model_dir,
        qformat=qformat,
        export_path=output_dir,
    )

    # Run the command
    # current transformers T5 model seem to be incompatible with multiple GPUs
    # https://github.com/huggingface/transformers/issues/30280
    env = os.environ.copy()
    if expected_suffix.startswith("t5_tiny"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    run_example_command(cmd_parts, "llm_ptq", env=env)

    # Now we expect a file named model.safetensors in output_dir
    generated_file = output_dir / "model.safetensors"
    assert generated_file.exists(), (
        f"Expected .safetensors file not found for qformat={qformat}: {generated_file}"
    )

    # Map scale types to their conditions
    scale_types = [
        ("input_scale", fuse_input_scale),
        ("weight_scale", fuse_weight_scale),
        ("weight_scale_2", fuse_weight_scale_2),
        ("pre_quant_scale", fuse_pre_quant_scale),
        ("weight_quantizer._svdquant_lora_a", fuse_svdquant_lora_a),
    ]

    # Projection pairs to check for equality
    proj_pairs = [("gate_proj", "up_proj"), ("q_proj", "k_proj"), ("q_proj", "v_proj")]

    def _same_scale(name, key1, key2, f):
        if key1 in name:
            tensor1 = f.get_tensor(name)
            tensor2 = f.get_tensor(name.replace(key1, key2))
            assert torch.allclose(tensor1, tensor2)

    # Load the safetensors to do further checks
    with safe_open(generated_file, framework="pt") as f:
        tensor_names = list(f.keys())
        assert len(tensor_names) > 0, f"No tensors found in {generated_file} for qformat={qformat}!"
        for name in tensor_names:
            tensor = f.get_tensor(name)
            # Basic sanity checks
            assert tensor.shape is not None, f"Tensor '{name}' shape is None!"
            assert tensor.dtype is not None, f"Tensor '{name}' dtype is None!"

            # Check each scale type if its condition is met
            for scale_suffix, condition in scale_types:
                if name.endswith(scale_suffix) and condition:
                    # Check each projection pair
                    for proj1, proj2 in proj_pairs:
                        _same_scale(name, proj1, proj2, f)

    # TODO: Load a pre-dumped log to compare textually or use pre-defined dict for sanity checks
