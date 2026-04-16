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

from .base import CumulativeTokenCollector, Model, build_model_output

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
        speculative_algorithm = kwargs.pop("speculative_algorithm", None)
        num_speculative_tokens = kwargs.pop("speculative_num_steps", 3)
        draft_model_dir = kwargs.pop("draft_model_dir", None)
        if speculative_algorithm == "EAGLE3":
            specdec = {
                "method": "eagle3",
                "model": draft_model_dir,
                "num_speculative_tokens": num_speculative_tokens,
            }
        elif speculative_algorithm == "EAGLE":
            specdec = {
                "method": "eagle",
                "model": draft_model_dir,
                "num_speculative_tokens": num_speculative_tokens,
            }
        elif speculative_algorithm == "NGRAM":
            specdec = {
                "method": "ngram",
                "num_speculative_tokens": num_speculative_tokens,
                "prompt_lookup_max": kwargs.pop("max_matching_ngram_size", 3),
            }
        elif speculative_algorithm == "DRAFT_TARGET":
            specdec = {
                "method": "draft_model",
                "model": draft_model_dir,
                "num_speculative_tokens": num_speculative_tokens,
            }
            parallel_draft_block_sizes = kwargs.pop("parallel_draft_block_sizes", None)
            if parallel_draft_block_sizes is not None:
                specdec["disable_padded_drafter_batch"] = True
                specdec["parallel_draft_block_sizes"] = parallel_draft_block_sizes
        elif speculative_algorithm == "MTP":
            specdec = {
                "method": "mtp",
                "num_speculative_tokens": num_speculative_tokens,
            }
        elif speculative_algorithm == "NONE":
            specdec = None

        if kwargs.pop("parallel_drafting", False) and specdec is not None:
            specdec["parallel_drafting"] = True

        engine_args = AsyncEngineArgs(
            model=model_dir,
            trust_remote_code=kwargs.pop("trust_remote_code", False),
            tensor_parallel_size=kwargs.pop("tensor_parallel_size", 1),
            enable_expert_parallel=kwargs.pop("moe_expert_parallel_size", 1) > 1,
            enable_prefix_caching=kwargs.pop("prefix_cache", False),
            speculative_config=specdec,
            max_num_seqs=max_concurrent_requests,
            skip_tokenizer_init=False,
            async_scheduling=kwargs.pop("async_scheduling", True),
            enforce_eager=kwargs.pop("enforce_eager", False),
            **kwargs,
        )
        self.model = AsyncLLM.from_engine_args(engine_args)
        self.sampling_kwargs = dict(sampling_kwargs)

    def _build_sampling_config(self, max_length, end_id):
        sampling_config = SamplingParams(
            detokenize=False,
            temperature=self.sampling_kwargs.get("temperature", 1.0),
            top_p=self.sampling_kwargs.get("top_p", 1.0),
            top_k=self.sampling_kwargs.get("top_k", 0),
            max_tokens=max_length,
        )
        if end_id == -1:
            sampling_config.ignore_eos = True
        else:
            sampling_config.stop_token_ids = [end_id]
        return sampling_config

    async def run(self, prompt_ids, max_length, end_id, request_id=None, turn_id=None):
        sampling_config = self._build_sampling_config(max_length, end_id)
        collector = CumulativeTokenCollector()
        request_key = (
            f"{request_id}.{turn_id}"
            if request_id is not None and turn_id is not None
            else str(id(prompt_ids))
        )
        async for output in self.model.generate(
            request_id=request_key,
            prompt=TokensPrompt(prompt_token_ids=prompt_ids),
            sampling_params=sampling_config,
        ):
            chunk_time = time.perf_counter()
            for completion in output.outputs:
                collector.add_update(completion.index, completion.token_ids, chunk_time)
            if output.finished:
                break
        return build_model_output(
            collector.output_ids(), collector.token_times, collector.chunk_lengths()
        )

    def stop(self):
        import contextlib

        with contextlib.suppress(Exception):
            asyncio.run(self.model.shutdown())
