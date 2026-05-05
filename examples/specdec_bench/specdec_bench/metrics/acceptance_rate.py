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

import json
import os

from .base import Metric


class AcceptanceRate(Metric):
    def __init__(self):
        super().__init__()
        self.name = "acceptance_rate"
        self.request_acceptance_lengths = {}

    @classmethod
    def build_prompt_ar(cls, request_records):
        prompt_ar = {}
        for record in sorted(request_records, key=lambda r: (r["request_id"], r["turn_id"])):
            request_id = record["request_id"]
            turn_id = record["turn_id"]
            turn_lengths = prompt_ar.setdefault(request_id, {}).setdefault(turn_id, [])
            for beam_lengths in cls.trimmed_chunk_lengths(record):
                turn_lengths.extend(beam_lengths)
        return prompt_ar

    @staticmethod
    def flatten_turn_lengths(turns):
        accepted_lengths = []
        for turn_id in sorted(turns):
            accepted_lengths.extend(turns[turn_id])
        return accepted_lengths

    def _set_request_acceptance_lengths(self, prompt_ar):
        self.request_acceptance_lengths = {}
        for request_id, turns in prompt_ar.items():
            self.request_acceptance_lengths[request_id] = self.flatten_turn_lengths(turns)

    @staticmethod
    def _bin_start(position, bin_size):
        return (position // bin_size) * bin_size

    @staticmethod
    def _format_float(value):
        return f"{value:.4f}"

    @staticmethod
    def _format_bin_label(bin_start, bin_size):
        return f"{bin_start}-{bin_start + bin_size - 1}"

    def _print_mapping(self, title, mapping, bin_size=None):
        print(title)
        for key, value in mapping.items():
            label = self._format_bin_label(key, bin_size) if bin_size is not None else key
            if isinstance(value, float):
                value = self._format_float(value)
            print(f"  {label}: {value}")

    def _process_binned_lengths(self, request_records, bin_size=500):
        generated_bins = {}
        full_bins = {}
        generated_offsets = {}

        for record in sorted(request_records, key=lambda r: (r["request_id"], r["turn_id"])):
            request_id = record["request_id"]
            turn_id = record["turn_id"]
            prompt_length = record.get("prompt_length") or 0
            for beam_idx, beam_lengths in enumerate(self.trimmed_chunk_lengths(record)):
                offset_key = (request_id, turn_id, beam_idx)
                current_generated = generated_offsets.get(offset_key, 0)
                for acceptance_length in beam_lengths:
                    generated_start = current_generated
                    generated_end = current_generated + acceptance_length - 1
                    full_start = prompt_length + current_generated
                    full_end = full_start + acceptance_length - 1
                    for generated_bin in range(
                        self._bin_start(generated_start, bin_size),
                        self._bin_start(generated_end, bin_size) + bin_size,
                        bin_size,
                    ):
                        generated_bins.setdefault(generated_bin, []).append(acceptance_length)
                    for full_bin in range(
                        self._bin_start(full_start, bin_size),
                        self._bin_start(full_end, bin_size) + bin_size,
                        bin_size,
                    ):
                        full_bins.setdefault(full_bin, []).append(acceptance_length)
                    current_generated += acceptance_length
                generated_offsets[offset_key] = current_generated

        self.out["Generated_Token_Binned_Average_AL"] = {
            k: sum(v) / len(v) for k, v in sorted(generated_bins.items())
        }
        self.out["Generated_Token_Binned_Count"] = {
            k: len(v) for k, v in sorted(generated_bins.items())
        }
        self.out["Full_Token_Binned_Average_AL"] = {
            k: sum(v) / len(v) for k, v in sorted(full_bins.items())
        }
        self.out["Full_Token_Binned_Count"] = {k: len(v) for k, v in sorted(full_bins.items())}
        self._print_mapping(
            "Generated token binned average AL",
            self.out["Generated_Token_Binned_Average_AL"],
            bin_size=bin_size,
        )
        self._print_mapping(
            "Full token binned average AL",
            self.out["Full_Token_Binned_Average_AL"],
            bin_size=bin_size,
        )

    def _get_lengths(self, turn, lengths):
        for j in turn:
            if j not in lengths:
                lengths[j] = 0
            lengths[j] += 1

    def _process_lengths(self, lengths):
        lengths = dict(sorted(lengths.items(), key=lambda x: x[0]))
        self.out["Acceptance_Length_Histogram"] = lengths
        self._print_mapping("Acceptance Length Histogram", lengths)
        if not lengths:
            self.out["Conditional_Acceptance_Rate"] = {}
            self.out["Joint_Acceptance_Rate"] = {}
            return
        sum_lengths = sum(lengths.values())
        running_len = sum_lengths
        prev_ratio = 1.0
        self.out["Conditional_Acceptance_Rate"] = {}
        for k, v in lengths.items():
            conditional_ar = running_len / sum_lengths / prev_ratio
            self.out["Conditional_Acceptance_Rate"][k] = conditional_ar
            prev_ratio = running_len / sum_lengths
            running_len -= v
        self._print_mapping("Conditional acceptance rate", self.out["Conditional_Acceptance_Rate"])

        self.out["Joint_Acceptance_Rate"] = {}
        running_joint = 0.0
        running_product = 1.0
        for i, (k, conditional_ar) in enumerate(self.out["Conditional_Acceptance_Rate"].items()):
            running_product *= conditional_ar
            if i == 0:
                running_joint = 0.0
                self.out["Joint_Acceptance_Rate"][k] = 1.0
            else:
                running_joint += running_product
                self.out["Joint_Acceptance_Rate"][k] = running_joint / i
        self._print_mapping("Joint acceptance rate", self.out["Joint_Acceptance_Rate"])

    def process_final(self, text_outputs, request_records):
        all_ar = []
        lengths = {}
        self.out = {}
        self.out["Request_AL"] = {}
        prompt_ar = self.build_prompt_ar(request_records)
        self._set_request_acceptance_lengths(prompt_ar)
        for request_id, turns in prompt_ar.items():
            self.out["Request_AL"][request_id] = {}
            for turn_id, turn in turns.items():
                ar = sum(turn) / len(turn) if turn else 0.0
                self.out["Request_AL"][request_id][turn_id] = ar
                if turn:
                    all_ar.append(ar)
                    self._get_lengths(turn, lengths)
        average_ar = sum(all_ar) / len(all_ar) if all_ar else 0.0
        print(f"Average AL: {self._format_float(average_ar)}")
        self.out["Average_AL"] = average_ar
        self._process_lengths(lengths)
        self._process_binned_lengths(request_records)
        self.write()
        self._format_write_output(text_outputs)

    def clear(self):
        self.out = {}
        self.request_acceptance_lengths = {}

    def _format_write_output(self, outputs):
        with open(os.path.join(self.directory, "responses.jsonl"), "w") as outfile:
            for i, messages in enumerate(outputs):
                q_id = i
                out_line = {}
                out_line["question_id"] = q_id
                if messages[0]["role"] == "system":
                    out_line["system_prompt"] = messages[0]["content"]
                q_turns = [c["content"] for c in messages if c["role"] == "user"]
                generated = [c for c in messages if c.get("generated")]
                if generated:
                    a_turns = [c["content"] for c in generated]
                    raw_a_turns = [c.get("raw_content", c["content"]) for c in generated]
                    output_token_ids = [c.get("output_token_ids") for c in generated]
                else:
                    a_turns = [c["content"] for c in messages if c["role"] == "assistant"]
                    raw_a_turns = [
                        c.get("raw_content", c["content"])
                        for c in messages
                        if c["role"] == "assistant"
                    ]
                    output_token_ids = [
                        c.get("output_token_ids") for c in messages if c["role"] == "assistant"
                    ]
                out_line["turns"] = q_turns
                out_line["acceptance_lengths"] = self.request_acceptance_lengths.get(i, [])
                out_line["choices"] = [{"index": 0, "turns": a_turns}]
                out_line["raw_choices"] = [{"index": 0, "turns": raw_a_turns}]
                out_line["output_token_ids"] = [{"index": 0, "turns": output_token_ids}]
                json.dump(out_line, outfile)
                outfile.write("\n")
