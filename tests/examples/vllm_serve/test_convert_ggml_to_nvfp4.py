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
from _test_utils.examples.ggml_export import IQ1_S, IQ2_XS, quantize_tiny_llama
from _test_utils.examples.run_command import run_example_command
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt

from modelopt.torch.export import export_hf_checkpoint

NVFP4_INPUT = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}


@pytest.mark.parametrize(
    "with_input_scales",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                torch.cuda.get_device_capability() < (10, 0), reason="NVFP4 W4A4 needs Blackwell"
            ),
        ),
    ],
)
def test_vllm_serves_the_nvfp4_conversion_like_the_fake_quantized_model(
    tmp_path, with_input_scales
):
    transcode = {"backend_extra_args": {"transcode": "nvfp4"}}
    model, token_ids = quantize_tiny_llama(
        tmp_path,
        ({**IQ1_S, **transcode}, {**IQ2_XS, **transcode}),
        NVFP4_INPUT if with_input_scales else None,
    )
    with torch.no_grad():
        logits = model(token_ids).logits[0, :-1].float().cpu()
    token_ids = token_ids.cpu()
    expected = logits.log_softmax(-1).gather(-1, token_ids[0, 1:, None])[:, 0]

    export_hf_checkpoint(model, export_dir=tmp_path / "ggml")
    # The vLLM image has no `python` binary.
    run_example_command(
        [
            sys.executable,
            "convert_ggml_to_nvfp4.py",
            str(tmp_path / "ggml"),
            str(tmp_path / "nvfp4"),
        ],
        "hf_ptq",
    )
    llm = LLM(
        model=str(tmp_path / "nvfp4"),
        enforce_eager=True,
        gpu_memory_utilization=0.2,
        max_model_len=64,
        max_num_seqs=1,
        dtype="bfloat16",
        skip_tokenizer_init=True,
    )
    try:
        (output,) = llm.generate(
            TokensPrompt(prompt_token_ids=token_ids[0].tolist()),
            SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False),
        )
    finally:
        del llm
        gc.collect()
        cleanup_dist_env_and_memory(shutdown_ray=False)
    served = torch.tensor(
        [lp[t].logprob for t, lp in zip(token_ids[0, 1:].tolist(), output.prompt_logprobs[1:])]
    )

    # vLLM's kernels round differently from the fake-quant reference by up to ~0.02 with BF16
    # activations. FP4 activations amplify that: a native ModelOpt NVFP4 W4A4 export of this model
    # sits up to 0.36 from its own fake quant. A wrong scale moves these logprobs by over 1.
    torch.testing.assert_close(served, expected, atol=0.4 if with_input_scales else 0.05, rtol=0)
