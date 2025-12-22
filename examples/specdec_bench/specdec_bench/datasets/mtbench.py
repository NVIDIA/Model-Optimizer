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

from .base import Dataset, Request

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


class MTBench(Dataset):
    def __init__(self, path, num_samples=80, **kwargs):
        self.data: list[Request] = []  # list of list of questions.
        self.num_samples = num_samples
        self.path = path
        self._preprocess()

    def _preprocess(self):
        with open(self.path) as f:
            for json_line in f:
                line = json.loads(json_line)
                key = "turns" if "turns" in line else "prompt"
                self.data.append(Request(system_prompt=None, turns=line[key]))
        self.data = self.data[: self.num_samples]
