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
from collections import defaultdict

from .acceptance_rate import AcceptanceRate


class MTBench(AcceptanceRate):
    def __init__(self, requests):
        super().__init__()
        self.requests = requests

    def process_final(self, text_outputs, request_records):
        lengths = {}
        self.out = {}
        self.out["Request_AL"] = {}
        prompt_ar = self.build_prompt_ar(request_records)
        self._set_request_acceptance_lengths(prompt_ar)
        for request_id, turns in prompt_ar.items():
            turn_1 = turns[0]
            turn_2 = turns[1]
            request_ar = sum(turn_1 + turn_2) / len(turn_1 + turn_2)
            self.out["Request_AL"][request_id] = request_ar
            self._get_lengths(turn_1, lengths)
            self._get_lengths(turn_2, lengths)
        per_category = defaultdict(list)
        for q_id, ar in self.out["Request_AL"].items():
            per_category[self.requests[q_id].category].append(ar)
        self.out["Category_AL"] = {}
        for category_name, category_ar in per_category.items():
            if len(category_ar) > 0:
                category_ar = sum(category_ar) / len(category_ar)
                self.out["Category_AL"][category_name] = category_ar
                print(f"{category_name} Average AL: {category_ar}")
        average_ar = sum(self.out["Request_AL"].values()) / len(self.out["Request_AL"])
        print("Average AL:", average_ar)
        self.out["Average_AL"] = average_ar
        self._process_lengths(lengths)
        self._process_binned_lengths(request_records)
        self.write()
        self._format_write_output(text_outputs)

    def _format_write_output(self, outputs):
        with open(os.path.join(self.directory, "mtbench_responses.jsonl"), "w") as outfile:
            for i, messages in enumerate(outputs):
                q_id = self.requests[i].question_id
                out_line = {}
                out_line["question_id"] = q_id
                out_line["category"] = self.requests[i].category
                q_turns = [c["content"] for c in messages if c["role"] == "user"]
                a_turns = [c["content"] for c in messages if c["role"] == "assistant"]
                raw_a_turns = [
                    c.get("raw_content", c["content"]) for c in messages if c["role"] == "assistant"
                ]
                out_line["turns"] = q_turns
                out_line["acceptance_lengths"] = self.request_acceptance_lengths.get(i, [])
                out_line["choices"] = [{"index": 0, "turns": a_turns}]
                out_line["raw_choices"] = [{"index": 0, "turns": raw_a_turns}]
                json.dump(out_line, outfile)
                outfile.write("\n")
