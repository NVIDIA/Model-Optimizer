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

from .base import Model

try:
    import sglang as sgl
except ImportError:
    print("sglang is not installed.")
    sglang = None


class SGLANGModel(Model):
    def __init__(
        self,
        model_dir,
        max_concurrent_requests,
        sampling_kwargs,
        use_draft_logits=False,
        **kwargs,
    ):
        speculative_algorithm = kwargs.get("speculative_algorithm")
        if speculative_algorithm == "MTP":
            speculative_algorithm = "EAGLE"
        elif speculative_algorithm == "DRAFT_TARGET":
            speculative_algorithm = "STANDALONE"
        elif speculative_algorithm == "NGRAM":
            speculative_algorithm = "LOOKAHEAD"
        elif speculative_algorithm == "DFLASH":
            pass  # SGLang native name, pass through
        elif speculative_algorithm == "NONE":
            speculative_algorithm = None

        engine_kwargs = dict(
            model_path=model_dir,
            skip_tokenizer_init=True,
            trust_remote_code=kwargs.get("trust_remote_code", False),
            mem_fraction_static=kwargs.get("mem_fraction_static", 0.8),
            disable_overlap_schedule=kwargs.get("disable_overlap_schedule", False),
            tp_size=kwargs.get("tensor_parallel_size", 1),
            ep_size=kwargs.get("moe_expert_parallel_size", 1),
            torch_compile_max_bs=max_concurrent_requests,
            max_running_requests=max_concurrent_requests,
            attention_backend=kwargs.get("attention_backend"),
            enable_torch_compile=kwargs.get("enable_torch_compile", False),
            cuda_graph_max_bs=max_concurrent_requests,
            disable_cuda_graph=False,
        )
        if speculative_algorithm is not None:
            # https://github.com/sgl-project/sglang/pull/3582
            engine_kwargs["speculative_algorithm"] = speculative_algorithm
            num_draft_tokens = kwargs.get("speculative_num_draft_tokens", 4)
            engine_kwargs["speculative_num_draft_tokens"] = num_draft_tokens
            engine_kwargs["speculative_draft_model_path"] = kwargs.get("draft_model_dir")
            if speculative_algorithm == "DFLASH":
                if "speculative_dflash_draft_window_size" in kwargs:
                    engine_kwargs["speculative_dflash_draft_window_size"] = kwargs[
                        "speculative_dflash_draft_window_size"
                    ]
                print(
                    f"[specdec_bench] DFLASH ignores --draft_length / speculative_num_steps / "
                    f"speculative_eagle_topk; effective draft block = "
                    f"speculative_num_draft_tokens={num_draft_tokens}"
                )
            else:
                engine_kwargs["speculative_num_steps"] = kwargs.get("speculative_num_steps", 3)
                engine_kwargs["speculative_eagle_topk"] = kwargs.get("speculative_eagle_topk", 1)

        # Forward any other kwargs (e.g. from runtime_params.engine_args) to
        # sgl.Engine, letting yaml override the defaults set above. Skip only
        # specdec_bench-internal routing keys that should never reach SGLang.
        _internal_keys = frozenset({
            "speculative_algorithm",
            "draft_model_dir",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            "speculative_dflash_draft_window_size",
            "tensor_parallel_size",
            "moe_expert_parallel_size",
            "tokenizer_path",
            "use_draft_logits",
        })
        for _k, _v in kwargs.items():
            if _k in _internal_keys:
                continue
            engine_kwargs[_k] = _v

        self.model = sgl.Engine(**engine_kwargs)

        self.sampling_config = sampling_kwargs

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        """Synchronous version of run for use with asyncio.to_thread"""
        timing = []
        output_dict = {}
        self.sampling_config["max_new_tokens"] = max_length
        self.sampling_config["stop_token_ids"] = [end_id]
        timing.append(time.perf_counter())
        assert self.sampling_config.get("beam_width", 1) == 1
        beam_lens = [[] for _ in range(self.sampling_config.get("beam_width", 1))]
        outputs = [None]
        result = await self.model.async_generate(
            sampling_params=self.sampling_config, input_ids=prompt_ids, stream=True
        )
        async for chunk in result:
            timing.append(time.perf_counter())
            outputs = chunk["output_ids"]
            beam_lens[0].append(chunk["meta_info"]["completion_tokens"])

        if end_id == outputs[-1]:
            beam_lens[0].pop(-1)
            outputs.pop(-1)
        reformatted_output_ids = [[] for _ in range(self.sampling_config.get("beam_width", 1))]
        for beam_idx, beam_len in enumerate(beam_lens):
            response = outputs
            if beam_len[0] != 0:
                reformatted_output_ids[beam_idx].append(response[: beam_len[0]])
            for s, e in itertools.pairwise(beam_len):
                reformatted_output_ids[beam_idx].append(response[s:e])
            if len(response) > beam_len[-1]:
                reformatted_output_ids[beam_idx].append(response[beam_len[-1] :])
        output_dict["output_ids"] = reformatted_output_ids
        output_dict["output_logits"] = None
        output_dict["token_times"] = timing
        return output_dict
