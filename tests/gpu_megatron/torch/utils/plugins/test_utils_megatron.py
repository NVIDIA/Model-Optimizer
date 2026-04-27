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

import pytest
from _test_utils.torch.megatron.models import get_mcore_qwen3_600m
from _test_utils.torch.megatron.utils import initialize_for_megatron
from transformers import AutoTokenizer

from modelopt.torch.utils.plugins import megatron_generate, megatron_mmlu

SEED = 1234

# TODO: move to regression test folder


def _test_megatron_generate_and_mmlu(rank, size, parallelism):
    if parallelism == "tp":
        initialize_for_megatron(tensor_model_parallel_size=size, seed=SEED)
        model = get_mcore_qwen3_600m(tensor_model_parallel_size=size).cuda().eval()
    elif parallelism == "pp":
        initialize_for_megatron(pipeline_model_parallel_size=size, seed=SEED)
        model = get_mcore_qwen3_600m(pipeline_model_parallel_size=size).cuda().eval()
    else:
        raise ValueError(f"Invalid parallelism: {parallelism}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    messages = [
        {"role": "user", "content": "Give me a short introduction to large language model."}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device="cuda")
    output_ids = megatron_generate(model, model_inputs["input_ids"])
    output_text = tokenizer.batch_decode(output_ids)
    print(rank, output_text)

    assert 0.36 < megatron_mmlu(model, tokenizer, fraction=0.1, batch_size=16) < 0.39


@pytest.mark.parametrize("parallelism", ["tp", "pp"])
def test_megatron_generate_and_mmlu(dist_workers, parallelism, num_gpus):
    if num_gpus == 1 and parallelism == "pp":
        pytest.skip("Skipping as redundant test on 1 GPU")
    dist_workers.run(_test_megatron_generate_and_mmlu, parallelism=parallelism)
