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
    import tiktoken
except ImportError:
    tiktoken = None
from .base import Metric
from .timing import compute_statistics


class AATiming(Metric):
    def __init__(self, base_tokenizer):
        super().__init__()
        self.name = "aa_timing"
        if tiktoken is None:
            raise ImportError(
                "Please install tiktoken to use the AATiming metric, or remove the metric from the run command"
            )
        self.enc = tiktoken.get_encoding("o200k_base")
        self.base_tokenizer = base_tokenizer

    def process_final(self, text_outputs, request_records):
        self.out = {}
        timing = [record["token_times"] for record in request_records]
        total_tokens = []
        for record in request_records:
            target_tokens = self.flat_output_tokens(record)
            target_text = self.base_tokenizer.decode(target_tokens)
            aa_tokens = self.enc.encode(target_text, disallowed_special=())
            total_tokens.append(len(aa_tokens))
        gen_tp_time = []
        start_time = min(t[0] for t in timing)
        end_time = max(t[-1] for t in timing)
        self.out["AA Output TPS"] = sum(total_tokens) / (end_time - start_time)
        for tokens, times in zip(total_tokens, timing):
            if len(times) > 2:
                gen_tp_time.append((tokens - 1) / (times[-1] - times[1]))
        if gen_tp_time:
            self.out["AA Generation Tokens Per Second"] = compute_statistics(gen_tp_time)
        for k, v in self.out.items():
            print(k, v)
        self.write()

    def clear(self):
        self.out = {}
