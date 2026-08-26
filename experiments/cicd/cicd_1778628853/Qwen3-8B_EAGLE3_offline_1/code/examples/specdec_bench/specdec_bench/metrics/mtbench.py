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

from .acceptance_rate import AcceptanceRate

MTBENCH_TOPICS = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]


class MTBench(AcceptanceRate):
    def process_final(self, text_outputs):
        i = 0
        lengths = {}
        self.out["Request_AR"] = {}
        self.prompt_ar = dict(sorted(self.prompt_ar.items(), key=lambda x: x[0]))
        for request_id, turns in self.prompt_ar.items():
            turn_1 = turns[0]
            turn_2 = turns[1]
            q_id = request_id
            mtbench_topic = MTBENCH_TOPICS[q_id // 10]
            self.out["Request_AR"][request_id] = sum(turn_1 + turn_2) / len(turn_1 + turn_2)
            self._get_lengths(turn_1, lengths)
            self._get_lengths(turn_2, lengths)
            print(mtbench_topic, sum(turn_1 + turn_2) / len(turn_1 + turn_2))
        per_category = [[] for _ in range(len(MTBENCH_TOPICS))]
        for q_id, ar in self.out["Request_AR"].items():
            per_category[q_id // 10].append(ar)
        self.out["Category_AR"] = {}
        for i, category in enumerate(per_category):
            if len(category) > 0:
                category_ar = sum(category) / len(category)
                self.out["Category_AR"][MTBENCH_TOPICS[i]] = category_ar
                print(f"{MTBENCH_TOPICS[i]} Average AR: {category_ar}")
        average_ar = sum(self.out["Request_AR"].values()) / len(self.out["Request_AR"])
        print("Average AR:", average_ar)
        self.out["Average_AR"] = average_ar
        self._process_lengths(lengths)
        self.write()
        self._format_write_output(text_outputs)

    def _format_write_output(self, outputs):
        with open(os.path.join(self.directory, "mtbench_responses.jsonl"), "w") as outfile:
            for i, messages in enumerate(outputs):
                q_id = i + 81
                out_line = {}
                out_line["question_id"] = q_id
                out_line["category"] = MTBENCH_TOPICS[(q_id - 81) // 10]
                q_turns = [c["content"] for c in messages if c["role"] == "user"]
                a_turns = [c["content"] for c in messages if c["role"] == "assistant"]
                out_line["turns"] = q_turns
                out_line["choices"] = [{"index": 0, "turns": a_turns}]
                json.dump(out_line, outfile)
                outfile.write("\n")
