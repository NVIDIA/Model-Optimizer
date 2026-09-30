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
import torch
from _test_utils.torch.megatron.models import get_mcore_qwen3_600m
from _test_utils.torch.megatron.utils import initialize_for_megatron
from megatron.core.transformer import MegatronModule, TransformerConfig
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
    elif parallelism == "cp":
        initialize_for_megatron(context_parallel_size=size, seed=SEED)
        model = get_mcore_qwen3_600m(context_parallel_size=size).cuda().eval()
    elif parallelism == "dp":
        # Data parallel is implicit: with all model-parallel sizes 1, DP == world size.
        initialize_for_megatron(seed=SEED)
        model = get_mcore_qwen3_600m().cuda().eval()
    else:
        raise ValueError(f"Invalid parallelism: {parallelism}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # megatron_generate does not support CP (autoregressive decode is not sequence-partitioned).
    if parallelism != "cp":
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


@pytest.mark.parametrize("parallelism", ["tp", "pp", "cp", "dp"])
def test_megatron_generate_and_mmlu(dist_workers, parallelism, num_gpus):
    if num_gpus == 1 and parallelism != "tp":
        pytest.skip("Skipping as redundant test on 1 GPU")
    dist_workers.run(_test_megatron_generate_and_mmlu, parallelism=parallelism)


class _VisionArgRecorder(MegatronModule):
    """VLM stand-in recording the sequence length and vision kwargs of every forward call."""

    vocab_size = 8

    def __init__(self):
        super().__init__(TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1))
        self.calls = []

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        *,
        pixel_values=None,
        image_grid_thw=None,
        image_sizes=None,
        inference_context=None,
        runtime_gather_output=None,
    ):
        self.calls.append(
            {
                "seq_len": input_ids.shape[-1],
                "vision": tuple(x is not None for x in (pixel_values, image_grid_thw, image_sizes)),
            }
        )
        return torch.zeros(*input_ids.shape, self.vocab_size, device=input_ids.device)

    def pop_calls(self) -> tuple[list, list]:
        calls, self.calls = self.calls, []
        return [c["seq_len"] for c in calls], [c["vision"] for c in calls]


def _test_megatron_generate_vlm_vision_inputs(rank, size):
    initialize_for_megatron(seed=SEED)
    model = _VisionArgRecorder().cuda()
    input_ids = torch.zeros((1, 4), dtype=torch.long, device="cuda")
    vision_inputs = {
        "pixel_values": torch.zeros((4, 8), device="cuda"),
        "image_grid_thw": torch.tensor([[1, 2, 2]], device="cuda"),
        "image_sizes": torch.tensor([[8, 8]], device="cuda"),
    }
    osl = 3
    all_vision, no_vision = (True, True, True), (False, False, False)

    # KV-cache decoding: the prefill step consumes the vision inputs, decode steps feed one token.
    megatron_generate(model, input_ids, osl=osl, enable_kv_cache=True, **vision_inputs)
    assert model.pop_calls() == ([4, 1, 1], [all_vision, no_vision, no_vision])

    # No KV cache: every step recomputes the full prefix and must replay the vision inputs,
    # otherwise the image placeholder tokens stay unreplaced and generation loses the image.
    megatron_generate(model, input_ids, osl=osl, enable_kv_cache=False, **vision_inputs)
    assert model.pop_calls() == ([4, 5, 6], [all_vision] * osl)

    # Sequence parallelism silently falls back to no-cache decoding, so it must replay too.
    model.config.sequence_parallel = True
    megatron_generate(model, input_ids, osl=osl, enable_kv_cache=True, **vision_inputs)
    assert model.pop_calls() == ([4, 5, 6], [all_vision] * osl)


def test_megatron_generate_vlm_vision_inputs(dist_workers):
    dist_workers.run(_test_megatron_generate_vlm_vision_inputs)
