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

from ..speculation_profile import build_profile
from .base import Metric


class AcceptanceRate(Metric):
    # Set once per run by run.py via set_profile_metadata(). Class-level so the
    # MTBench/SpecBench subclasses pick it up without extra wiring, mirroring how
    # Metric.update_directory() distributes the output path.
    profile_metadata = None

    def __init__(self):
        super().__init__()
        self.prompt_ar = {}
        self.name = "acceptance_rate"

    @classmethod
    def set_profile_metadata(cls, metadata):
        """Describe what is being measured, so a speculation_profile.json can be written.

        Without this the acceptance numbers are still computed and written as before;
        only the deployment-facing profile is skipped.
        """
        AcceptanceRate.profile_metadata = metadata

    def process_step(self, step_outputs, request_id, turn_id):
        if request_id not in self.prompt_ar:
            self.prompt_ar[request_id] = {}
        if turn_id not in self.prompt_ar[request_id]:
            self.prompt_ar[request_id][turn_id] = []
        for i, beam_output in enumerate(step_outputs["output_ids"]):
            for output_id_iter in beam_output:
                self.prompt_ar[request_id][turn_id].append(len(output_id_iter))

    def _get_lengths(self, turn, lengths):
        for j in turn:
            if j not in lengths:
                lengths[j] = 0
            lengths[j] += 1

    def _process_lengths(self, lengths):
        lengths = dict(sorted(lengths.items(), key=lambda x: x[0]))
        self.out["Acceptance_Length_Histogram"] = lengths
        print("Acceptance Length Histogram")
        print(lengths)
        sum_lengths = sum(lengths.values())
        running_len = sum_lengths
        prev_ratio = 1
        self.out["Conditional_Acceptance_Rate"] = {}
        print("Conditional acceptance rate")
        for k, v in lengths.items():
            print(k, running_len / sum_lengths / prev_ratio)
            self.out["Conditional_Acceptance_Rate"][k] = running_len / sum_lengths / prev_ratio
            prev_ratio = running_len / sum_lengths
            running_len -= v
        # Joint acceptance rate at step k = product of conditional acceptance
        # rates at steps 1..k = probability that ≥k tokens are accepted in
        # a row. The visualizer renders this as a separate panel.
        self.out["Joint_Acceptance_Rate"] = {}
        running_joint = 1.0
        for k, cond_ar in self.out["Conditional_Acceptance_Rate"].items():
            running_joint *= cond_ar
            self.out["Joint_Acceptance_Rate"][k] = running_joint
        # Emitted here rather than in each process_final(): this is the single point
        # where the acceptance distribution is final, and all three variants
        # (AcceptanceRate / MTBench / SpecBench) route through it, so none can
        # silently stop producing a profile.
        self._write_speculation_profile()

    def _write_speculation_profile(self):
        """Write speculation_profile.json — the deployment-facing view of these numbers.

        Skipped silently when run.py did not supply metadata (e.g. an ad-hoc run with
        no --save_dir): the profile is only meaningful if we can say what it describes.
        """
        metadata = AcceptanceRate.profile_metadata
        if not metadata or not self.directory:
            return
        profile = build_profile(
            self.out,
            per_category=self.out.get("Category_AL"),
            **metadata,
        )
        path = os.path.join(self.directory, "speculation_profile.json")
        os.makedirs(self.directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        validation = profile.get("validation") or {}
        consistency = validation.get("mean_consistency") or {}
        if not consistency.get("passed", True):
            # Loud, because a failure here means the vectors do not describe the
            # measured mean — the profile is wrong in a way downstream cannot detect.
            print(
                "WARNING: speculation profile failed its mean-consistency check "
                f"(implied {consistency.get('implied_mean_accept_length')} vs "
                f"reported {consistency.get('reported_mean_accept_length')}). "
                f"See {path}"
            )
        else:
            print(f"Wrote speculation profile to {path}")

    def process_final(self, text_outputs):
        all_ar = []
        lengths = {}
        self.out["Request_AL"] = {}
        self.prompt_ar = dict(sorted(self.prompt_ar.items(), key=lambda x: x[0]))
        for request_id, turns in self.prompt_ar.items():
            self.out["Request_AL"][request_id] = {}
            for turn_id, turn in turns.items():
                ar = sum(turn) / len(turn)
                self.out["Request_AL"][request_id][turn_id] = ar
                all_ar.append(ar)
                self._get_lengths(turn, lengths)
                print(request_id, turn_id, self.out["Request_AL"][request_id][turn_id])
        average_ar = sum(all_ar) / len(all_ar)
        print("Average AL:", average_ar)
        self.out["Average_AL"] = average_ar
        self._process_lengths(lengths)
        self.write()
        self._format_write_output(text_outputs)

    def clear(self):
        self.prompt_ar = []

    def _format_write_output(self, outputs):
        with open(os.path.join(self.directory, "responses.jsonl"), "w") as outfile:
            for i, messages in enumerate(outputs):
                q_id = i
                out_line = {}
                out_line["question_id"] = q_id
                if messages[0]["role"] == "system":
                    out_line["system_prompt"] = messages[0]["content"]
                q_turns = [c["content"] for c in messages if c["role"] == "user"]
                a_turns = [c["content"] for c in messages if c["role"] == "assistant"]
                out_line["turns"] = q_turns
                out_line["choices"] = [{"index": 0, "turns": a_turns}]
                json.dump(out_line, outfile)
                outfile.write("\n")
