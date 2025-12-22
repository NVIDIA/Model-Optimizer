# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A wrapper over the TensorRT-LLM high level API runner."""

import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import tensorrt_llm
import torch
from packaging.version import parse as parse_version
from tensorrt_llm import SamplingParams

try:
    from tensorrt_llm.llmapi import CudaGraphConfig
    from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig
    from tensorrt_llm.llmapi.llm import LLM as TRTLLM
except ImportError:
    print("Please upgrade tensorrt-llm to 1.1.0rc2 or later")
    raise


def _sanitize_temperature_and_top_p(temperature, top_p):
    assert temperature >= 0.0, "Temperature must be greater than 0.0."

    # TRT LLM accepts temperature values only greater than 0.0
    temperature = max(temperature, 0.001)

    kwargs = {"temperature": temperature}
    if top_p is not None:
        # cpp executor only supports topP.value() > 0.f
        top_p = 1e-4 if top_p == 0 else top_p
        kwargs["top_p"] = top_p

    return kwargs


class LLM(TRTLLM):
    """A wrapper over the ``tensorrt_llm.llmapi.llm.LLM`` for LLM profiling and validation."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        tokenizer: "str | Path | None" = None,
        medusa_choices: Any = None,
        tp: int = 0,
        trust_remote_code: bool = False,
        max_seq_len: int = 0,
        max_batch_size: int = 0,
    ):
        """Initializes the LLM runner class.

        Args:
            checkpoint_dir: the directory path of the model checkpoint.
            tokenizer: the tokenizer. For example, a tokenizer from the Huggingface model.
            medusa_choices: The medusa choices for the decoding config.
            tp: the tensor parallel size (for the torch backend). If 0, it will be set to the number of GPUs.
            trust_remote_code: whether to trust the remote code (for the torch backend).
            max_seq_len: Max sequence length for the LLM backend. If 0, it is not specified.
            max_batch_size: Max batch size for the LLM backend. If 0, it is not specified.
        """
        with open(Path(checkpoint_dir) / "config.json") as config_file:
            config = json.load(config_file)

            assert medusa_choices is None, "medusa_choices is not supported with the torch llmapi"

        def _find_max_position_embeddings(cfg: dict) -> int | None:
            if "max_position_embeddings" in cfg:
                return cfg["max_position_embeddings"]
            for v in cfg.values():
                if isinstance(v, dict):
                    res = _find_max_position_embeddings(v)
                    if res is not None:
                        return res
            return None

        # Some VLMs may have a sub-config for max_position_embeddings, so we need to find it.
        if max_seq_len > 0:
            self._max_seq_len = max_seq_len
        else:
            self._max_seq_len = _find_max_position_embeddings(config)
            if self._max_seq_len is None:
                warnings.warn(
                    "max_position_embeddings not found in config.json, using default value 8192"
                )
                self._max_seq_len = 8192
        print(f"max_position_embeddings: {self._max_seq_len}")
        self._max_beam_width = 1

        kwargs = {}
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer

        if tp < 1:
            tp = torch.cuda.device_count()

        # Check if any key in config contains both "num" and "experts"
        ep = 1
        enable_attention_dp = False
        for k in config:
            if "num" in k and "experts" in k:
                ep = torch.cuda.device_count()
                enable_attention_dp = True
                break

        # Sometimes 90% of the GPU memory is not enough for the TRT LLM torch engine.
        trt_kv_cache_config = TRT_KvCacheConfig(free_gpu_memory_fraction=0.7)
        trt_kv_cache_config.max_tokens = self._max_seq_len * (
            max_batch_size if max_batch_size > 0 else 8
        )

        cuda_graph_config = None
        if max_batch_size > 0:
            cuda_graph_config = CudaGraphConfig(
                batch_sizes=[2**i for i in range(int((max_batch_size - 1).bit_length()))]
                + [max_batch_size],
                max_batch_size=max_batch_size,
                enable_padding=True,
            )

        self._support_context_logits_and_stop_words = parse_version(
            tensorrt_llm.__version__
        ) >= parse_version("1.1.0rc2")

        super().__init__(
            backend="pytorch",
            model=checkpoint_dir,
            tensor_parallel_size=tp,
            moe_expert_parallel_size=ep,
            trust_remote_code=trust_remote_code,
            enable_chunked_prefill=True,
            kv_cache_config=trt_kv_cache_config,
            # pytorch backend configs
            cuda_graph_config=cuda_graph_config,
            enable_attention_dp=enable_attention_dp,
            **kwargs,
        )

    @property
    def max_seq_len(self):
        """Get the max sequence length from the LLM instance."""
        return self._max_seq_len

    @property
    def max_beam_width(self):
        """Get the max beam width from the LLM instance."""
        return self._max_beam_width

    @property
    def gather_context_logits(self):
        """Returns whether the context_logits can be returned from the LLM instance."""
        return self._support_context_logits_and_stop_words

    def _generate(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop_words: list[str] | None = None,
    ):
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        if not self._support_context_logits_and_stop_words:
            stop_words = None
        beam_width = self.max_beam_width
        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        sampling_config = SamplingParams(
            max_tokens=max_new_tokens,
            use_beam_search=True,
            best_of=beam_width,
            stop=stop_words,
            **kwargs,
        )

        return self.generate(prompts, sampling_params=sampling_config, use_tqdm=False)

    def generate_tokens(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop_words: list[str] | None = None,
    ) -> list[list[int]] | list[list[list[int]]]:
        """Generates the tokens based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.
            stop_words: A list of words that the generate stops on.

        Returns:
            a list of output token lists if max_beam_width is 1 or a 3D list with shape [batch, beam, sequence_len].
        """
        outputs = self._generate(prompts, max_new_tokens, temperature, top_p, stop_words)
        output_tokens = []

        beam_width = self.max_beam_width
        for request_output in outputs:
            token_ids = [
                completion_output.token_ids for completion_output in request_output.outputs
            ]
            if beam_width == 1:
                # each output_text is a single list of tokens.
                output_tokens += token_ids
            else:
                # each output_text is a list of beam searched token lists.
                output_tokens.append(token_ids)

        return output_tokens

    def generate_text(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        stop_words: list[str] | None = None,
    ) -> list[str] | list[list[str]]:
        """Generates the text based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            max_new_tokens: The max output token length.
            temperature: The sampling temperature
            stop_words: A list of words the generate will stop on.

        Returns:
            a list of output text strings if max_beam_width is 1 or a 2D list with shape [batch, beam].
        """
        outputs = self._generate(prompts, max_new_tokens, temperature, top_p, stop_words)
        output_texts = []

        beam_width = self.max_beam_width
        for request_output in outputs:
            texts = [completion_output.text for completion_output in request_output.outputs]
            if beam_width == 1:
                # each output_text is a single text string.
                output_texts += texts
            else:
                # each output_text is a list of beam searched texts
                output_texts.append(texts)

        return output_texts

    def generate_context_logits(
        self,
        prompts: Iterable[str] | Iterable[list[int]],
        temperature: float = 1.0,
        top_p: float | None = None,
    ) -> list[torch.tensor]:
        """Generates the context logits based on the input prompts.

        Args:
            prompts: The input prompts. Could be a list of strings or token lists.
            temperature: The sampling temperature.
            top_p: The nucleus sampling parameter.

        Returns:
            a tensor list of the context_logits.
        """
        assert self._support_context_logits_and_stop_words, (
            "Context logits are not supported with the current tensorrt_llm version."
        )
        assert temperature >= 0.0, "Temperature must be greater than 0.0."

        kwargs = _sanitize_temperature_and_top_p(temperature, top_p)
        kwargs["return_context_logits"] = True

        sampling_config = SamplingParams(max_tokens=1, use_beam_search=True, best_of=1, **kwargs)

        outputs = self.generate(prompts, sampling_params=sampling_config, use_tqdm=False)

        return [output.context_logits for output in outputs]
