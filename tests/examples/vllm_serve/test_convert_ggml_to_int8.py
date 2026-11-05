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

import gc
import sys

import pytest
import torch
from _test_utils.examples.ggml_export import quantize_tiny_llama
from _test_utils.examples.run_command import run_example_command
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt

from modelopt.torch.export import export_hf_checkpoint


@pytest.fixture(scope="module")
def int8_export_in_vllm(tmp_path_factory):
    """vLLM serving the INT8 conversion of a GGML export, the model's own logprobs and its tokens."""
    tmp_path = tmp_path_factory.mktemp("ggml_int8")
    # Group 32 (IQ1_S) and group 16 (IQ2_XS) layers take different vLLM kernels.
    model, token_ids = quantize_tiny_llama(tmp_path)
    with torch.no_grad():
        logits = model(token_ids).logits[0, :-1].float()
    expected = logits.log_softmax(-1).gather(-1, token_ids[0, 1:, None])[:, 0]

    export_hf_checkpoint(model, export_dir=tmp_path / "ggml")
    # The vLLM image has no `python` binary.
    run_example_command(
        [sys.executable, "convert_ggml_to_int8.py", str(tmp_path / "ggml"), str(tmp_path / "int8")],
        "hf_ptq",
    )

    llm = LLM(
        model=str(tmp_path / "int8"),
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=64,
        max_num_seqs=1,
        dtype="bfloat16",
        skip_tokenizer_init=True,
    )
    yield llm, token_ids[0], expected
    del llm
    gc.collect()
    cleanup_dist_env_and_memory(shutdown_ray=False)


def test_vllm_serves_the_int8_export_like_the_fake_quantized_model(int8_export_in_vllm):
    llm, token_ids, expected = int8_export_in_vllm
    (output,) = llm.generate(
        TokensPrompt(prompt_token_ids=token_ids.tolist()),
        SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False),
    )
    served = torch.tensor(
        [lp[t].logprob for t, lp in zip(token_ids[1:].tolist(), output.prompt_logprobs[1:])]
    )

    # vLLM's kernels round differently from the fake-quant reference by up to ~0.02; giving each
    # group its neighbour's scale moves these logprobs by over 1.
    torch.testing.assert_close(served, expected, atol=0.05, rtol=0)
