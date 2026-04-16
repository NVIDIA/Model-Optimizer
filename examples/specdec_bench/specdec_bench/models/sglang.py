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

import time

from .base import CumulativeTokenCollector, Model, build_model_output

try:
    import sglang as sgl
except ImportError:
    print("sglang is not installed.")
    sglang = None


class SGLANGModel(Model):
    def __init__(self, model_dir, max_concurrent_requests, sampling_kwargs, **kwargs):
        kwargs.pop("parallel_drafting", None)
        speculative_algorithm = kwargs.pop("speculative_algorithm", None)
        if speculative_algorithm == "MTP":
            speculative_algorithm = "EAGLE"
        elif speculative_algorithm == "DRAFT_TARGET":
            speculative_algorithm = "STANDALONE"
        elif speculative_algorithm == "NGRAM":
            speculative_algorithm = "LOOKAHEAD"
            raise NotImplementedError("Needs more work")
        elif speculative_algorithm == "NONE":
            speculative_algorithm = None
        self.model = sgl.Engine(
            model_path=model_dir,
            skip_tokenizer_init=True,
            trust_remote_code=kwargs.pop("trust_remote_code", False),
            mem_fraction_static=0.8,
            disable_overlap_schedule=kwargs.pop("disable_overlap_schedule", False),
            disable_radix_cache=not kwargs.pop("prefix_cache", False),
            tp_size=kwargs.pop("tensor_parallel_size", 1),
            ep_size=kwargs.pop("moe_expert_parallel_size", 1),
            speculative_algorithm=speculative_algorithm,
            speculative_num_steps=kwargs.pop("speculative_num_steps", 3),
            speculative_eagle_topk=kwargs.pop("speculative_eagle_topk", 1),
            speculative_num_draft_tokens=kwargs.pop("speculative_num_draft_tokens", 4),
            speculative_draft_model_path=kwargs.pop("draft_model_dir", None),
            torch_compile_max_bs=max_concurrent_requests,
            max_running_requests=max_concurrent_requests,
            attention_backend=kwargs.pop("attention_backend", None),
            enable_torch_compile=kwargs.pop("enable_torch_compile", False),
            cuda_graph_max_bs=max_concurrent_requests,
            disable_cuda_graph=kwargs.pop("disable_cuda_graph", False),
            **kwargs,
        )

        self.sampling_kwargs = dict(sampling_kwargs)

    async def run(self, prompt_ids, max_length, end_id, request_id=None, turn_id=None):
        sampling_config = dict(self.sampling_kwargs)
        sampling_config["max_new_tokens"] = max_length
        if end_id != -1:
            sampling_config["stop_token_ids"] = [end_id]
        if sampling_config.get("beam_width", 1) != 1:
            raise ValueError("SGLANGModel only supports beam_width=1")

        collector = CumulativeTokenCollector()
        result = await self.model.async_generate(
            sampling_params=sampling_config,
            input_ids=prompt_ids,
            stream=True,
        )
        async for chunk in result:
            chunk_time = time.perf_counter()
            collector.add_update(0, chunk["output_ids"], chunk_time)

        return build_model_output(
            collector.output_ids(), collector.token_times, collector.chunk_lengths()
        )
