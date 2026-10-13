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

import itertools
import time

try:
    import tensorrt_llm.bindings.executor as trtllm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import (
        CudaGraphConfig,
        DraftTargetDecodingConfig,
        EagleDecodingConfig,
        KvCacheConfig,
        MoeConfig,
        MTPDecodingConfig,
        NGramDecodingConfig,
    )
except ImportError:
    print("Failed to import tensorrt_llm._torch")
    trtllm = None


from .base import Model


class TRTLLMPYTModel(Model):
    def __init__(
        self,
        model_path,
        max_concurrent_requests,
        sampling_kwargs,
        use_draft_logits=False,
        **kwargs,
    ):
        self.model = create_executor(model_path, max_concurrent_requests, kwargs)
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


def create_executor(model_path: str, max_concurrent_requests, kwargs):
    disable_overlap_schedule = kwargs.get("disable_overlap_schedule", False)
    if kwargs.get("speculative_algorithm", None) == "DRAFT_TARGET":
        specdec = DraftTargetDecodingConfig(
            max_draft_len=kwargs.get("speculative_num_steps", 3),
            speculative_model_dir=kwargs.get("draft_model_dir", None),
        )

    elif kwargs.get("speculative_algorithm", None) == "EAGLE3":
        extra_params = {}
        if "allow_advanced_sampling" in EagleDecodingConfig.model_fields:
            extra_params["allow_advanced_sampling"] = kwargs.get("allow_advanced_sampling", False)
        elif "allow_advanced_sampling" in kwargs:
            print(
                f"WARNING: allow_advanced_sampling unsupported in tensorrt_llm version: {trtllm.__version__}"
            )
        specdec = EagleDecodingConfig(
            max_draft_len=kwargs.get("speculative_num_steps", 3),
            speculative_model_dir=kwargs.get("draft_model_dir", None),
            eagle3_one_model=kwargs.get("use_one_model", True),
            eagle3_layers_to_capture=kwargs.get("eagle3_layers_to_capture", None),
            num_eagle_layers=kwargs.get("num_eagle_layers", 1),
            **extra_params,
        )

    elif kwargs.get("speculative_algorithm", None) == "MTP":
        specdec = MTPDecodingConfig(
            num_nextn_predict_layers=kwargs.get("speculative_num_steps", 3),
            use_relaxed_acceptance_for_thinking=kwargs.get("relaxed_acceptance", False),
            relaxed_topk=kwargs.get("relaxed_topk", 10),
            relaxed_delta=kwargs.get("relaxed_delta", 0.6),
        )
    elif kwargs.get("speculative_algorithm", None) == "NGRAM":
        specdec = NGramDecodingConfig(
            max_draft_len=kwargs.get("speculative_num_steps", 5),
            max_matching_ngram_size=kwargs.get("max_matching_ngram_size", 3),
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=True,
        )
    elif kwargs.get("speculative_algorithm", None) == "NONE":
        specdec = None
    else:
        specdec = None

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=kwargs.get("prefix_cache", False),
        free_gpu_memory_fraction=0.75,
    )

    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[max_concurrent_requests],
        enable_padding=True,
    )

    model = LLM(
        model=model_path,
        tensor_parallel_size=kwargs.get("tensor_parallel_size", 4),
        moe_expert_parallel_size=kwargs.get("moe_expert_parallel_size", 2),
        disable_overlap_scheduler=disable_overlap_schedule,
        cuda_graph_config=cuda_graph_config,
        enable_chunked_prefill=kwargs.get("enable_chunked_prefill", True),
        kv_cache_config=kv_cache_config,
        speculative_config=specdec,
        enable_attention_dp=kwargs.get("enable_attention_dp", False),
        max_batch_size=max_concurrent_requests,
        moe_config=MoeConfig(backend=kwargs.get("moe_backend", "TRTLLM")),
        sampler_type="TorchSampler",
        max_seq_len=kwargs.get("max_seq_len", None),
        max_num_tokens=kwargs.get("max_num_tokens", 8192),
    )
    return model


def check_sampling_config(sampling_config, max_length, end_id):
    return SamplingParams(
        use_beam_search=sampling_config.get("beam_width", 1) > 1,
        n=sampling_config.get("beam_width", 1),  # beam_width=1 for inflight batching
        top_k=sampling_config.get("top_k", None),  # SizeType topK
        top_p=sampling_config.get("top_p", None),
        seed=sampling_config.get("seed", None),
        temperature=sampling_config.get("temperature", 1),
        max_tokens=max_length,
        end_id=end_id,
        detokenize=False,
    )
