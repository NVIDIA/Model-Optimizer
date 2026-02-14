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


import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.megatron.models import get_mcore_qwen3_600m
from _test_utils.torch.megatron.utils import initialize_for_megatron
from transformers import AutoTokenizer

from modelopt.torch.utils.plugins import megatron_generate, megatron_mmlu

SEED = 1234


def _test_megatron_generate(rank, size):
    initialize_for_megatron(tensor_model_parallel_size=size, seed=SEED)

    model = get_mcore_qwen3_600m(tensor_model_parallel_size=size).cuda().eval()

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
    print(output_text)

    assert megatron_mmlu(model, tokenizer) > 0.24


def test_megatron_generate():
    size = torch.cuda.device_count()

    spawn_multiprocess_job(
        size=size,
        job=_test_megatron_generate,
        backend="nccl",
    )
