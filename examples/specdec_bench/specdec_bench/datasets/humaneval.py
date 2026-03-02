
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

from datasets import load_dataset

from .base import Dataset, Request

def format_prompt(prompt: str) -> str:
    return "Complete the following Python function. Only output the code, no explanations.\n\n" + prompt

class HumanEval(Dataset):
    def __init__(self, path, num_samples=164, **kwargs):
        self.data: list[Request] = []  # list of list of questions.
        self.num_samples = num_samples
        self._preprocess(path)

    def _preprocess(self, path: str):
        dataset = load_dataset(path, split='test')
        for item in dataset:
            self.data.append(Request(system_prompt=None, turns=[format_prompt(item["prompt"])]))
        self.data = self.data[: self.num_samples]
