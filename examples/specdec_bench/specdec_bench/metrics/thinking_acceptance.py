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

from .base import Metric


class ThinkingAcceptance(Metric):
    """
    Approximate acceptance split between thinking and non-thinking tokens.
    """

    def __init__(self, thinking_end_token=None):
        super().__init__()
        self.name = "thinking_acceptance"
        self.thinking_end_token = thinking_end_token

    def _contains_sequence(self, token_list, token_sequence):
        if isinstance(token_sequence, int):
            return token_sequence in token_list

        seq_len = len(token_sequence)
        if seq_len == 0:
            return False

        return any(
            token_list[i : i + seq_len] == token_sequence
            for i in range(len(token_list) - seq_len + 1)
        )

    def process_final(self, text_outputs, request_records):
        self.out = {}
        prompt_state = {}
        thinking_lengths = []
        non_thinking_lengths = []
        for record in sorted(request_records, key=lambda r: (r["request_id"], r["turn_id"])):
            request_id = record["request_id"]
            turn_id = record["turn_id"]
            if request_id not in prompt_state:
                prompt_state[request_id] = {}
            if turn_id not in prompt_state[request_id]:
                prompt_state[request_id][turn_id] = {"in_thinking": True}

            in_thinking = prompt_state[request_id][turn_id]["in_thinking"]
            for beam_output in self.trimmed_output_chunks(record):
                for output_id_iter in beam_output:
                    acceptance_length = len(output_id_iter)
                    if acceptance_length == 0:
                        continue
                    if in_thinking and self.thinking_end_token is not None:
                        if self._contains_sequence(output_id_iter, self.thinking_end_token):
                            prompt_state[request_id][turn_id]["in_thinking"] = False
                            in_thinking = False
                            continue
                    if in_thinking:
                        thinking_lengths.append(acceptance_length)
                    else:
                        non_thinking_lengths.append(acceptance_length)

        total_thinking = sum(thinking_lengths)
        total_non_thinking = sum(non_thinking_lengths)
        avg_thinking = sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0
        avg_non_thinking = (
            sum(non_thinking_lengths) / len(non_thinking_lengths) if non_thinking_lengths else 0
        )
        ratio = total_thinking / total_non_thinking if total_non_thinking else float("inf")

        self.out["Total_Thinking_Acceptance_Length"] = total_thinking
        self.out["Total_Non_Thinking_Acceptance_Length"] = total_non_thinking
        self.out["Avg_Thinking_Acceptance_Length"] = avg_thinking
        self.out["Avg_Non_Thinking_Acceptance_Length"] = avg_non_thinking
        self.out["Thinking_to_Non_Thinking_Length_Ratio"] = ratio
        self.out["Thinking_Steps"] = len(thinking_lengths)
        self.out["Non_Thinking_Steps"] = len(non_thinking_lengths)

        print(f"Avg Thinking Acceptance Length: {avg_thinking:.2f}")
        print(f"Avg Non-Thinking Acceptance Length: {avg_non_thinking:.2f}")
        print(f"Thinking to Non-Thinking Length Ratio: {ratio:.2f}")

        self.write()

    def clear(self):
        self.out = {}
