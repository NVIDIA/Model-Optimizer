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

import json

from .base import Dataset, Request


class SpecBench(Dataset):
    def __init__(self, path, num_samples=480, **kwargs):
        self.data: list[Request] = []  # list of list of questions.
        self.num_samples = num_samples
        self._preprocess(path)

    def _preprocess(self, path):
        with open(path) as f:
            for json_line in f:
                line = json.loads(json_line)
                self.data.append(
                    Request(
                        question_id=line["question_id"],
                        category=line["category"],
                        system_prompt=None,
                        turns=line["turns"],
                    )
                )
        self.data = self.data[: self.num_samples]
