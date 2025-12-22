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


import numpy as np

from .base import Dataset, Request


class RandomToken(Dataset):
    def __init__(self, tokenizer, input_len, num_samples=20, **kwargs):
        self.data: list[Request] = []  # list of list of questions.
        self.num_samples = num_samples
        self.input_len = input_len
        self.tokenizer = tokenizer
        self._preprocess()

    def _preprocess(self):
        np.random.seed(0)
        tokenizer = self.tokenizer
        num_prompts = self.num_samples
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        for i in range(num_prompts):
            prompt = tokenizer.decode(
                [
                    (offsets[i] + i + j) % tokenizer.vocab_size
                    for j in range(int(self.input_len * 1.5))
                ]
            )
            re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[
                : (self.input_len)
            ]
            prompt = tokenizer.decode(re_encoded_sequence)
            self.data.append(Request(system_prompt=None, turns=[prompt]))
        self.data = self.data[: self.num_samples]
