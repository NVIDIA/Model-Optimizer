# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import itertools
import time
from typing import Any

try:
    from tensorrt_llm._torch.auto_deploy.llm import LLM
    from tensorrt_llm.llmapi import DraftTargetDecodingConfig
    from tensorrt_llm.sampling_params import SamplingParams
except ImportError:
    print("tensorrt_llm._torch.auto_deploy is not installed.")
    LLM = None

from .base import Model


class AutoDeployModel(Model):
    def __init__(self, model_path, max_concurrent_requests, sampling_kwargs, **kwargs):
        self.model = create_auto_deploy_model(model_path, max_concurrent_requests, kwargs)
        self.sampling_kwargs = sampling_kwargs

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        output_dict = {}
        sampling_config = check_sampling_config(self.sampling_kwargs, max_length, end_id)
        outputs = []
        timing = [time.perf_counter()]
        beam_lens = [[] for _ in range(self.sampling_kwargs.get("beam_width", 1))]

        async for output in self.model.generate_async(
            prompt_ids,
            streaming=not sampling_config.use_beam_search,
            sampling_params=sampling_config,
        ):
            for beam in output.outputs:
                beam_lens[beam.index].append(len(beam.token_ids))
            outputs.append(output.outputs)
            timing.append(time.perf_counter())

        reformatted_output_ids = [[] for _ in range(self.sampling_kwargs.get("beam_width", 1))]
        for beam_idx, beam_len in enumerate(beam_lens):
            response = outputs[-1][beam_idx]
            if beam_len[0] != 0:
                reformatted_output_ids[beam_idx].append(response.token_ids[: beam_len[0]])
            for s, e in itertools.pairwise(beam_len):
                reformatted_output_ids[beam_idx].append(response.token_ids[s:e])
            if len(response.token_ids) > beam_len[-1]:
                reformatted_output_ids[beam_idx].append(response.token_ids[beam_len[-1] :])

        output_dict["output_ids"] = reformatted_output_ids
        output_dict["output_logits"] = None
        output_dict["token_times"] = timing
        return output_dict

    def stop(self):
        """Stop and cleanup the model."""
        if hasattr(self, "model") and self.model is not None:
            with contextlib.suppress(Exception):
                del self.model


def create_auto_deploy_model(model_path: str, max_concurrent_requests: int, kwargs: dict[str, Any]):
    world_size = kwargs.get("world_size", kwargs.get("tensor_parallel_size", 1))

    max_seq_len = kwargs.get("max_seq_len", 8192)

    kv_cache_config = {
        "enable_block_reuse": kwargs.get("prefix_cache", False),
        "free_gpu_memory_fraction": kwargs.get("free_gpu_memory_fraction", 0.75),
    }

    specdec = None
    speculative_algorithm = kwargs.get("speculative_algorithm")

    if speculative_algorithm == "DRAFT_TARGET":
        specdec = DraftTargetDecodingConfig(
            max_draft_len=kwargs.get("speculative_num_steps", 3),
            speculative_model_dir=kwargs.get("draft_model_dir"),
        )
    elif speculative_algorithm == "NONE":
        specdec = None

    max_num_tokens = kwargs.get("max_num_tokens", 8192)

    llm_kwargs = {
        "model": model_path,
        "world_size": world_size,
        "max_batch_size": max_concurrent_requests,
        "max_seq_len": max_seq_len,
        "max_num_tokens": max_num_tokens,
        "skip_tokenizer_init": kwargs.get("skip_tokenizer_init", True),
        "kv_cache_config": kv_cache_config,
        "runtime": "trtllm",
        "disable_overlap_scheduler": kwargs.get("disable_overlap_scheduler", True),
        "speculative_config": specdec,
    }

    if kwargs.get("attn_backend"):
        llm_kwargs["attn_backend"] = kwargs["attn_backend"]

    if kwargs.get("compile_backend"):
        llm_kwargs["compile_backend"] = kwargs["compile_backend"]

    # Optimization mode: "graph" uses full torch.export, "transformers" is simpler
    # Default to "transformers" to avoid torch.export dimension specialization issues
    llm_kwargs["mode"] = kwargs.get("mode", "transformers")

    if kwargs.get("cuda_graph_batch_sizes"):
        llm_kwargs["cuda_graph_batch_sizes"] = kwargs["cuda_graph_batch_sizes"]

    model = LLM(**llm_kwargs)
    return model


def check_sampling_config(sampling_config: dict[str, Any], max_length: int, end_id: int):
    return SamplingParams(
        use_beam_search=sampling_config.get("beam_width", 1) > 1,
        n=sampling_config.get("beam_width", 1),
        top_k=sampling_config.get("top_k"),
        top_p=sampling_config.get("top_p"),
        seed=sampling_config.get("seed"),
        temperature=sampling_config.get("temperature", 1.0),
        max_tokens=max_length,
        end_id=end_id,
        detokenize=False,
    )
