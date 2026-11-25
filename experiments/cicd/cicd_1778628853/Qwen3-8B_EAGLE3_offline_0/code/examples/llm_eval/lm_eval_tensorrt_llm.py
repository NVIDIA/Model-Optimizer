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

import copy
import gc
import logging
import os
import signal
import threading
import time
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from transformers import BatchEncoding

from modelopt.deploy.llm import LLM

logger = logging.getLogger(__name__)

TokenSequence = list[int] | torch.LongTensor | torch.Tensor | BatchEncoding


@register_model("trt-llm")
class TRTLLM(TemplateAPI):
    def __init__(
        self,
        tokenizer: str,
        checkpoint_dir: str,
        batch_size: int = 1,
        **kwargs,
    ):
        assert isinstance(tokenizer, str)
        super().__init__(
            tokenizer=tokenizer,
            batch_size=int(batch_size),
            **kwargs,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        assert isinstance(checkpoint_dir, str)

        max_length = kwargs.get("max_length", self._max_gen_toks + 4096)
        self.llm = LLM(
            checkpoint_dir=checkpoint_dir,
            tokenizer=self.tokenizer,
            max_batch_size=int(batch_size),
            max_seq_len=max_length,
        )
        self.max_length = max_length - 1
        logger.info("Loaded TRT-LLM")

    def model_call(
        self,
        messages: Iterable[list[int]],
        *,
        generate: bool = True,
        gen_kwargs: dict | None = None,
        **kwargs,
    ):
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)

        assert isinstance(messages, Iterable), "Expect the messages to be Iterable[list[int]]"
        first_element = next(iter(messages))
        assert isinstance(first_element, list) and isinstance(first_element[0], int), (
            "Expect the messages to be Iterable[list[int]]"
        )

        if not generate:
            return self.llm.generate_context_logits(prompts=messages)

        llm_kwargs = {}
        max_new_tokens = self._max_gen_toks
        stop_words = []
        if gen_kwargs:
            if "until" in gen_kwargs:
                stop_words = gen_kwargs.pop("until")
                llm_kwargs["stop_words"] = stop_words
            if "temperature" in gen_kwargs:
                llm_kwargs["temperature"] = gen_kwargs.pop("temperature")
            if "top_p" in gen_kwargs:
                llm_kwargs["top_p"] = gen_kwargs.pop("top_p")
            if "max_gen_toks" in gen_kwargs:
                max_new_tokens = gen_kwargs.pop("max_gen_toks")

        output_texts: list[str] = self.llm.generate_text(
            prompts=messages,
            max_new_tokens=max_new_tokens,
            **llm_kwargs,
        )

        # Manually filter out keyword if not supported by llm.
        for i, text in enumerate(output_texts):
            for word in stop_words:
                word_index = text.find(word)
                if word_index >= 0:
                    text = text[:word_index]
            output_texts[i] = text

        return output_texts

    async def amodel_call(
        self,
        session,
        messages: Iterable[list[int]],
        *,
        generate: bool = True,
        cache_keys: list | None = None,
        ctxlens: list[int] | None = None,
        gen_kwargs: dict | None = None,
        **kwargs,
    ):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        *,
        generate: bool = True,
        gen_kwargs: dict | None = None,
        seed: int = 1234,
        **kwargs,
    ) -> dict:
        """This method is responsible for creating the json payload that will be sent to the API."""
        raise NotImplementedError

    @staticmethod
    def parse_generations(outputs: Any | list[Any], **kwargs) -> list[str]:
        """Method used to parse the generations from the (batched) API response."""
        return outputs

    @staticmethod
    def parse_logprobs(
        outputs: Any | list[Any],
        tokens: list[list[int]] | None = None,
        ctxlens: list[int] | None = None,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        """Method used to parse the logprobs from the (batched) API response.

        The provided tokens have two parts: The context tokens (length as ctxlens) and the continuation tokens.
        The logprobs returned is computed from the continuation tokens.
        We return the sum of the logprob of the continuation tokens
            [assuming the continuation tokens are the golden output].
        """
        res = []

        for logits_single_batch, tokens_single_batch, ctxlen_single_batch in zip(
            outputs,
            tokens,  # type: ignore[arg-type]
            ctxlens,  # type: ignore[arg-type]
        ):
            logits_single_batch = logits_single_batch.to("cuda")
            continuation_logprob = F.log_softmax(
                logits_single_batch[(ctxlen_single_batch - 1) : -1], dim=-1
            )
            continuation_tokens = torch.tensor(tokens_single_batch[ctxlen_single_batch:])
            top_tokens = continuation_logprob.argmax(dim=-1).cpu()

            is_greedy = torch.equal(top_tokens, continuation_tokens)

            logprob_sum = (
                continuation_logprob[
                    torch.arange(continuation_logprob.size(0)), continuation_tokens
                ]
                .sum()
                .cpu()
            )

            res.append((logprob_sum, is_greedy))

        return res


if __name__ == "__main__":
    cli_evaluate()
    # Force clean up the LLM instance and void hanging.
    gc.collect()

    # Force terminate in case gc.collect() is not enough.
    def _terminate():
        time.sleep(10)
        os.kill(os.getpid(), signal.SIGTERM)

    termination_thread = threading.Thread(target=_terminate, daemon=True)
    termination_thread.start()
