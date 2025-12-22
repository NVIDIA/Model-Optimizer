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


try:
    from datasets import load_dataset
except ImportError:
    print("datasets is not installed.")
    datasets = None


from .base import Dataset, Request


class BaseHF(Dataset):
    def __init__(self, num_samples=100, **kwargs):
        self.data: list[Request] = []  # list of list of questions.
        self.num_samples = num_samples
        self._preprocess()

    def _preprocess(self):
        dataset = self._load_dataset(self.num_samples)
        for i, line in enumerate(dataset):
            if i == self.num_samples:
                break
            self.data.append(self._single_line_process(line))

    def _single_line_process(self, line):
        raise NotImplementedError

    def _load_dataset(self, num_samples):
        raise NotImplementedError


class OpenOrca(BaseHF):
    def _single_line_process(self, line, **kwargs):
        return Request(system_prompt=line["system_prompt"], turns=[line["question"]])

    def _load_dataset(self, num_samples):
        return load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)


class OpenMathInstructv2(BaseHF):
    def _single_line_process(self, line, **kwargs):
        return Request(system_prompt=None, turns=[line["problem"]])

    def _load_dataset(self, num_samples):
        return load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True)


class UltraChat(BaseHF):
    def _single_line_process(self, line, **kwargs):
        return Request(
            system_prompt=None, turns=[q for i, q in enumerate(line["data"]) if i % 2 == 0]
        )

    def _load_dataset(self, num_samples):
        return load_dataset("stingning/ultrachat", split="train", streaming=True)
