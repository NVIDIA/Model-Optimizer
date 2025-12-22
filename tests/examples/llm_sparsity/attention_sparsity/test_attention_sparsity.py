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

"""Test attention sparsity example script."""

import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command


def run_attention_sparsity_command(*, model: str, method: str = "skip_softmax", **kwargs):
    """Run attention sparsity example script.

    Args:
        model: Path to model
        method: Sparse attention method (corresponds to --sparse_attn arg)
        **kwargs: Additional arguments to pass to the script
    """
    kwargs.update(
        {
            "pyt_ckpt_path": model,
            "sparse_attn": method,
        }
    )
    kwargs.setdefault("seq_len", 128)
    kwargs.setdefault("num_samples", 1)
    kwargs.setdefault("max_new_tokens", 16)

    cmd_parts = extend_cmd_parts(["python", "hf_sa.py"], **kwargs)
    run_example_command(cmd_parts, "llm_sparsity/attention_sparsity")


@pytest.mark.parametrize("method", ["skip_softmax"])
def test_attention_sparsity(tiny_llama_path, tmp_path, method):
    """Test sparse attention with TinyLlama."""
    run_attention_sparsity_command(
        model=tiny_llama_path,
        method=method,
    )
