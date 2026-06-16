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
    # Cross-engine ``--max_seq_len`` (run.py) lands in kwargs under the
    # vLLM-native name ``max_model_len`` (see run.py's ``_MAX_SEQ_LEN_KEY``)
    # and is read at line ~92 below into AsyncEngineArgs.

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
                "method": "draft_model",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
            if kwargs.get("parallel_draft_block_sizes") is not None:
                specdec["disable_padded_drafter_batch"] = True
                specdec["parallel_draft_block_sizes"] = kwargs.get("parallel_draft_block_sizes")
        elif kwargs.get("speculative_algorithm") == "MTP":
            # vLLM's ``SpeculativeConfig.__post_init__`` (vllm/config/
            # speculative.py:529-602) does method auto-detection ONLY
            # when ``method`` is unset — when ``model`` is provided and
            # ``method`` is None, the default branch sets
            # ``method = "draft_model"`` (the generic same-architecture
            # draft path), NOT MTP. That path enforces equal num_heads
            # between target and draft and raises
            # ``AssertionError: All layers in one attention group must
            # share num_heads`` on heterogeneous-head models like
            # Gemma 4 (target=8 heads, assistant=4).
            #
            # The canonical config for ALL MTP variants is to ALWAYS
            # pass ``method="mtp"`` AND ADD ``model=<assistant>`` only
            # when the family uses a separate assistant model. vLLM's
            # own test at ``tests/v1/e2e/spec_decode/test_spec_decode.py``
            # (lines 818-823) does exactly this for the gemma4-e4b
            # parametrization:
            #
            #     speculative_config = {
            #         "method": "mtp",
            #         "num_speculative_tokens": ...,
            #     }
            #     if draft_model is not None:        # Gemma 4 case
            #         speculative_config["model"] = draft_model
            #
            # Surfaced on OMNIML-5024 pipeline #54356795: dropping the
            # ``method`` key when ``draft_model_dir`` was provided sent
            # the call into the generic draft_model path, hitting the
            # num_heads assertion. Restored both keys.
            specdec = {
                "method": "mtp",
                "num_speculative_tokens": kwargs.get("speculative_num_steps", 3),
            }
            draft_model_dir = kwargs.get("draft_model_dir")
            if draft_model_dir:
                # Gemma 4 family (E2B / E4B / 26B-A4B / 31B) uses a
                # separate assistant checkpoint as the MTP draft.
                # vLLM auto-detects Gemma4 MTP from the assistant
                # ``model_type=gemma4_assistant`` and rewrites it to
                # ``gemma4_mtp`` (speculative.py:511-522). For
                # families where the MTP layer ships inside the
                # target (Qwen 3.5 etc.), omit ``--draft_model_dir``
                # and let vLLM use the target model as its own draft
                # (handled in speculative.py:562-573).
                specdec["model"] = draft_model_dir
        elif kwargs.get("speculative_algorithm") == "DFLASH":
            specdec = {
                "method": "dflash",
                "model": kwargs.get("draft_model_dir"),
                "num_speculative_tokens": kwargs.get("speculative_num_draft_tokens", 8),
            }
        elif kwargs.get("speculative_algorithm") == "NONE":
            specdec = None

        if specdec is None:
            num_speculative_tokens = 1
        else:
            num_speculative_tokens = specdec.get("num_speculative_tokens", 3)

        engine_args = AsyncEngineArgs(
            model=model_dir,
            tokenizer=kwargs.get("tokenizer_path"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            enable_expert_parallel=kwargs.get("moe_expert_parallel_size", 1) > 1,
            enable_prefix_caching=kwargs.get("prefix_cache", False),
            speculative_config=specdec,
            max_num_seqs=max_concurrent_requests * num_speculative_tokens,
            skip_tokenizer_init=False,
            async_scheduling=kwargs.get("async_scheduling", True),
            enforce_eager=False,
            max_model_len=kwargs.get("max_model_len"),
        )
        self.engine_args = engine_args
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

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):  # pragma: no cover
        output_dict = {}
        self.sampling_config.max_tokens = max_length
        self.sampling_config.stop_token_ids = [end_id]
        if end_id == -1:
            self.sampling_config.ignore_eos = True

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

    async def generate(self, prompt_ids, request_id, turn_id):  # pragma: no cover
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

    def get_serving_config(self):  # pragma: no cover
        """Dump the AsyncEngineArgs dataclass plus the runtime vllm_config when available."""
        try:
            import dataclasses

            cfg = dataclasses.asdict(self.engine_args)
        except Exception:
            cfg = {}
        # vllm exposes the resolved engine config on the AsyncLLM instance once
        # initialized — capture max_model_len / kv cache / dtype defaults that
        # don't appear in AsyncEngineArgs.
        try:
            vllm_config = getattr(self.model, "vllm_config", None)
            if vllm_config is not None and hasattr(vllm_config, "to_dict"):
                cfg["vllm_config"] = vllm_config.to_dict()
        except Exception:
            pass
        return cfg

    def stop(self):  # pragma: no cover
        try:
            self.loop.run_until_complete(self.model.shutdown())
            self.loop.close()
        except Exception:
            pass
