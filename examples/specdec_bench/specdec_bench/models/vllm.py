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

import asyncio
import time

from .base import Model

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM
except ImportError:
    print("vllm is not installed.")
    vllm = None


class VLLMModel(Model):
    def __init__(self, model_dir, max_concurrent_requests, sampling_kwargs, **kwargs):
        specdec = None
        if kwargs.get("speculative_algorithm") == "EAGLE3":
            specdec = {
                "method": "eagle3",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
        elif kwargs.get("speculative_algorithm") == "EAGLE":
            specdec = {
                "method": "eagle",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
        elif kwargs.get("speculative_algorithm") == "NGRAM":
            specdec = {
                "method": "ngram",
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
                "prompt_lookup_max": kwargs.get("max_matching_ngram_size", 3),  # No idea here
            }
        elif kwargs.get("speculative_algorithm") == "DRAFT_TARGET":
            specdec = {
                "method": "draft_target",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
        elif kwargs.get("speculative_algorithm") == "MTP":
            specdec = {
                "method": "mtp",
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
        elif kwargs.get("speculative_algorithm") == "NONE":
            specdec = None
        engine_args = AsyncEngineArgs(
            model=model_dir,
            trust_remote_code=True,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            enable_expert_parallel=kwargs.get("moe_expert_parallel_size", 1) > 1,
            enable_prefix_caching=kwargs.get("prefix_cache", False),
            speculative_config=specdec,
            max_num_seqs=max_concurrent_requests,
            skip_tokenizer_init=False,
        )
        self.model = AsyncLLM.from_engine_args(engine_args)
        self.sampling_kwargs = sampling_kwargs
        # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        self.sampling_config = SamplingParams(
            detokenize=False,
            temperature=sampling_kwargs.get("temperature", 1.0),
            top_p=sampling_kwargs.get("top_p", 1.0),
            top_k=sampling_kwargs.get("top_k", 0),
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        output_dict = {}
        self.sampling_config.max_tokens = max_length
        self.sampling_config.stop_token_ids = [end_id]

        outputs, timing, full_tokens = await self.generate(prompt_ids, request_id, turn_id)

        reformatted_output_ids = [[] for _ in range(self.sampling_kwargs.get("beam_width", 1))]
        start = 0
        timing_to_strip = []
        for i in range(len(outputs)):
            if outputs[i] == start:
                timing_to_strip.append(i)
                continue
            if i == len(outputs) - 1:
                if full_tokens[-1] == end_id:
                    if outputs[i] - start == 1:
                        timing_to_strip.append(i)
                    else:
                        reformatted_output_ids[0].append(full_tokens[start : outputs[i] - 1])
                    break
            reformatted_output_ids[0].append(full_tokens[start : outputs[i]])
            start = outputs[i]
        output_dict["output_ids"] = reformatted_output_ids
        output_dict["output_logits"] = None
        output_dict["token_times"] = [
            timing[i] for i in range(len(timing)) if i not in timing_to_strip
        ]
        return output_dict

    async def generate(self, prompt_ids, request_id, turn_id):
        timing = []
        timing.append(time.perf_counter())
        outputs = []
        full_tokens = []
        async for output in self.model.generate(
            request_id=f"{request_id}.{turn_id}",
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=self.sampling_config,
        ):
            for completion in output.outputs:
                outputs.append(len(completion.token_ids))
                timing.append(time.perf_counter())
                full_tokens = completion.token_ids
            if output.finished:
                break
        return outputs, timing, full_tokens

    def stop(self):
        try:
            self.loop.run_until_complete(self.model.shutdown())
            self.loop.close()
        except Exception:
            pass
