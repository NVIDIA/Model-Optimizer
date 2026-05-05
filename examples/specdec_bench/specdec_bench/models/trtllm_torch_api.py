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

import torch

try:
    import tensorrt_llm.bindings.executor as trtllm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import (
        CudaGraphConfig,
        DraftTargetDecodingConfig,
        KvCacheConfig,
        MoeConfig,
        MTPDecodingConfig,
        NGramDecodingConfig,
    )
except ImportError:
    print("Failed to import tensorrt_llm._torch")
    trtllm = None


from .base import CumulativeTokenCollector, Model, build_model_output


class TRTLLMPYTModel(Model):
    def __init__(self, model_path, max_concurrent_requests, sampling_kwargs, **kwargs):
        self.model = create_executor(model_path, max_concurrent_requests, sampling_kwargs, kwargs)
        self.sampling_kwargs = sampling_kwargs

    async def run(self, prompt_ids, max_length, end_id, request_id=None, turn_id=None):
        sampling_config = check_sampling_config(self.sampling_kwargs, max_length, end_id)
        collector = CumulativeTokenCollector()
        async for output in self.model.generate_async(
            prompt_ids,
            streaming=not sampling_config.use_beam_search,
            sampling_params=sampling_config,
        ):
            chunk_time = time.perf_counter()
            for beam in output.outputs:
                collector.add_update(beam.index, beam.token_ids, chunk_time)
        return build_model_output(
            collector.output_ids(), collector.token_times, collector.chunk_lengths()
        )


def create_executor(model_path: str, max_concurrent_requests, sampling_kwargs, kwargs):
    kwargs.pop("parallel_drafting", None)
    trust_remote_code = kwargs.pop("trust_remote_code", False)
    disable_overlap_schedule = kwargs.pop("disable_overlap_schedule", False)
    speculative_algorithm = kwargs.pop("speculative_algorithm", None)
    num_speculative_tokens = kwargs.pop("speculative_num_steps", 3)
    draft_model_dir = kwargs.pop("draft_model_dir", None)
    if speculative_algorithm == "DRAFT_TARGET":
        specdec = DraftTargetDecodingConfig(
            max_draft_len=num_speculative_tokens,
            speculative_model_dir=draft_model_dir,
        )

    elif speculative_algorithm == "EAGLE3":
        from tensorrt_llm.llmapi import Eagle3DecodingConfig

        specdec = Eagle3DecodingConfig(
            max_draft_len=num_speculative_tokens,
            speculative_model_dir=draft_model_dir,
            eagle3_layers_to_capture=kwargs.pop("eagle3_layers_to_capture", None),
            num_eagle_layers=kwargs.pop("num_eagle_layers", 1),
            allow_advanced_sampling=kwargs.pop("allow_advanced_sampling", True),
            use_sa_spec=kwargs.pop("use_sa_spec", False),
            sa_spec_threshold=kwargs.pop("sa_spec_threshold", 4),
        )

    elif speculative_algorithm == "MTP":
        specdec = MTPDecodingConfig(
            num_nextn_predict_layers=num_speculative_tokens,
            use_relaxed_acceptance_for_thinking=kwargs.pop("relaxed_acceptance", False),
            relaxed_topk=kwargs.pop("relaxed_topk", 10),
            relaxed_delta=kwargs.pop("relaxed_delta", 0.6),
            allow_advanced_sampling=kwargs.pop("allow_advanced_sampling", True),
            use_sa_spec=kwargs.pop("use_sa_spec", False),
            sa_spec_threshold=kwargs.pop("sa_spec_threshold", 4),
        )
    elif speculative_algorithm == "NGRAM":
        specdec = NGramDecodingConfig(
            max_draft_len=num_speculative_tokens,
            max_matching_ngram_size=kwargs.pop("max_matching_ngram_size", 3),
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=True,
        )
    elif speculative_algorithm == "PARD":
        from tensorrt_llm.llmapi import PARDDecodingConfig

        specdec = PARDDecodingConfig(
            max_draft_len=num_speculative_tokens,
            speculative_model_dir=draft_model_dir,
        )
    elif speculative_algorithm == "NONE":
        specdec = None
    else:
        print(f"Unknown speculative algorithm: {speculative_algorithm} for TRTLLM PyTorch API")
        specdec = None

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=kwargs.pop("prefix_cache", False),
        free_gpu_memory_fraction=0.75,
        mamba_ssm_cache_dtype=kwargs.pop("mamba_ssm_cache_dtype", "auto"),
        mamba_ssm_stochastic_rounding=kwargs.pop("mamba_ssm_stochastic_rounding", False),
        mamba_ssm_philox_rounds=kwargs.pop("mamba_ssm_philox_rounds", 10),
    )

    cuda_graph_config = CudaGraphConfig(
        max_batch_size=max_concurrent_requests,
        enable_padding=True,
    )

    model = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=kwargs.pop("tensor_parallel_size", 4),
        gpus_per_node=kwargs.pop("gpus_per_node", torch.cuda.device_count()),
        moe_expert_parallel_size=kwargs.pop("moe_expert_parallel_size", 2),
        disable_overlap_scheduler=disable_overlap_schedule,
        cuda_graph_config=cuda_graph_config,
        enable_chunked_prefill=kwargs.pop("enable_chunked_prefill", True),
        kv_cache_config=kv_cache_config,
        speculative_config=specdec,
        enable_attention_dp=kwargs.pop("enable_attention_dp", False),
        max_batch_size=max_concurrent_requests,
        moe_config=MoeConfig(backend=kwargs.pop("moe_backend", "TRTLLM")),
        max_seq_len=kwargs.pop("max_seq_len", None),
        max_num_tokens=kwargs.pop("max_num_tokens", 8192),
        **kwargs,
    )
    return model


def check_sampling_config(sampling_config, max_length, end_id):
    return SamplingParams(
        use_beam_search=sampling_config.get("beam_width", 1) > 1,
        n=sampling_config.get("beam_width", 1),
        top_k=sampling_config.get("top_k", None),
        top_p=sampling_config.get("top_p", None),
        seed=sampling_config.get("seed", None),
        temperature=sampling_config.get("temperature", 1),
        max_tokens=max_length,
        end_id=end_id,
        detokenize=False,
    )
