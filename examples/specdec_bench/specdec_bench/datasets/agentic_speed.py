# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json

from .base import Request
from .speed import SPEEDBench, config_type


def _normalize_message(msg: dict) -> dict:
    """Clean a single message: drop None-valued optional keys, parse tool_call args."""
    out = {"role": msg["role"]}

    out["content"] = msg["content"]

    tool_calls = msg.get("tool_calls")
    if tool_calls is not None and len(tool_calls) > 0:
        normalized_tcs = []
        for tc in tool_calls:
            tc = dict(tc)
            fn = tc.get("function")
            if fn is not None:
                fn = dict(fn)
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                tc["function"] = fn
            normalized_tcs.append(tc)
        out["tool_calls"] = normalized_tcs

    if out["role"] == "tool":
        out["tool_call_id"] = msg["tool_call_id"]

    return out


def _get_cut_points(messages: list, delta: int = 1) -> list:
    """
    Return message slices ending just before each assistant turn.

    For each assistant message at index i, yield messages[:i].
    If the last message is an assistant message, also yield the full list
    (to simulate the final state where the model has all context).

    Args:
        messages: Full conversation as a list of message dicts.
        delta: Take every Nth cut point (1 = all, 2 = every 2nd, etc.).
    """
    filtered = [_normalize_message(m) for m in messages]

    slices = []
    for i, msg in enumerate(filtered):
        if msg["role"] == "assistant":
            if i > 0:
                slices.append(filtered[:i])

    if filtered and filtered[-1]["role"] == "assistant":
        slices.append(filtered)

    if delta > 1:
        slices = slices[::delta]

    return slices


class AgenticSPEEDBench(SPEEDBench):
    def __init__(
        self,
        config_name: config_type = "agentic",
        num_samples: int | None = None,
        category: str | None = None,
        skip_turns_delta: int = 1,
        **kwargs,
    ):
        self._skip_turns_delta = skip_turns_delta
        super().__init__(
            config_name=config_name,
            num_samples=num_samples,
            category=category,
            **kwargs,
        )

    def _preprocess(
        self,
        config_name_or_dataset_path: config_type | str,
        *,
        category: str | None = None,
        _prepare_mode: bool = False,
    ):
        dataset = self._load_dataset(config_name_or_dataset_path, category=category)
        dataset = dataset.with_format("python")

        self._resolved_dataset = dataset
        self.data: list[Request] = []

        for example in dataset:
            question_id = example["question_id"]
            category_name = example["category"]
            turns = example["turns"]
            extras = example.get("extras", {})
            if isinstance(extras, str):
                extras = json.loads(extras)

            tools = json.loads(extras["tools"])

            cut_points = _get_cut_points(turns, delta=self._skip_turns_delta)
            for step, msg_slice in enumerate(cut_points):
                self.data.append(
                    Request(
                        question_id=question_id,
                        category=category_name,
                        messages=msg_slice,
                        tools=tools,
                        step=step * self._skip_turns_delta + 1,
                    )
                )
