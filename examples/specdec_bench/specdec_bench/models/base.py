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

import time
from dataclasses import dataclass, field


class Model:
    async def run(self, prompt_ids, max_length, end_id, request_id=None, turn_id=None):
        """Run a single request and return output_ids plus metadata."""
        raise NotImplementedError

    def stop(self):
        pass


def build_model_output(output_ids, token_times, chunk_lengths=None):
    model_output = {
        "output_ids": output_ids,
        "token_times": token_times,
    }
    if chunk_lengths is not None:
        model_output["chunk_lengths"] = chunk_lengths
    return model_output


@dataclass
class _BeamState:
    token_count: int = 0
    chunk_lengths: list[int] = field(default_factory=list)
    final_tokens: list[int] = field(default_factory=list)


class CumulativeTokenCollector:
    def __init__(self):
        self._beam_states = {}
        self.token_times = [time.perf_counter()]

    def add_update(self, beam_idx, cumulative_tokens, timestamp):
        beam_state = self._beam_states.setdefault(beam_idx, _BeamState())
        new_count = len(cumulative_tokens)
        if new_count <= beam_state.token_count:
            return False

        beam_state.chunk_lengths.append(new_count - beam_state.token_count)
        beam_state.final_tokens = cumulative_tokens
        beam_state.token_count = new_count
        self.token_times.append(timestamp)
        return True

    def output_ids(self):
        return [list(self._beam_states[i].final_tokens) for i in sorted(self._beam_states)]

    def chunk_lengths(self):
        return [list(self._beam_states[i].chunk_lengths) for i in sorted(self._beam_states)]
