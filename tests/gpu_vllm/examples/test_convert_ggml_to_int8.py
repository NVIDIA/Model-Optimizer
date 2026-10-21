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
import importlib.util
from pathlib import Path

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

_SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq" / "convert_ggml_to_int8.py"
_SPEC = importlib.util.spec_from_file_location("convert_ggml_to_int8", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
convert_ggml_to_int8 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(convert_ggml_to_int8)


@pytest.fixture(scope="module")
def int8_export_in_vllm(tmp_path_factory):
    """vLLM serving the INT8 export of a GGML model, the model's own logprobs and their tokens."""
    tmp_path = tmp_path_factory.mktemp("ggml_int8")
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
    # Fake quant runs on CPU: the reference needs no GPU, and it spares the GGML CUDA extension build.
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    # Group 32 (IQ1_S) and group 16 (IQ2_XS) layers take different vLLM kernels.
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            *(
                {
                    "quantizer_name": f"*layers.{i}.mlp*weight_quantizer",
                    "cfg": {"num_bits": num_bits, "block_sizes": {-1: 256}, "backend": "ggml"},
                }
                for i, num_bits in enumerate(("iq1_s", "iq2_xs"))
            ),
        ],
        "algorithm": "max",
    }
    token_ids = torch.randint(0, 128, (1, 48), generator=torch.Generator().manual_seed(0))
    mtq.quantize(model, quant_cfg, lambda m: m(token_ids))
    with torch.no_grad():
        logits = model(token_ids).logits[0, :-1].float()
    expected = logits.log_softmax(-1).gather(-1, token_ids[0, 1:, None])[:, 0]

    export_hf_checkpoint(model, export_dir=tmp_path / "ggml")
    export_dir = tmp_path / "int8"
    convert_ggml_to_int8.convert_ggml_export_to_int8(tmp_path / "ggml", export_dir)

    llm = LLM(
        model=str(export_dir),
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
