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
        self.timing = []
        self.name = "aa_timing"
        if tiktoken is None:
            raise ImportError(
                "Please install tiktoken to use the AATiming metric, or remove the metric from the run command"
            )
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.base_tokenizer = base_tokenizer
        self.total_tokens = []

    def process_step(self, step_outputs, request_id, turn_id):
        self.timing.append(step_outputs["token_times"])
        target_tokens = [
            t for tok_list in step_outputs["output_ids"] for tok in tok_list for t in tok
        ]
        target_text = self.base_tokenizer.decode(target_tokens)
        target_tokens = self.enc.encode(target_text, disallowed_special=())
        self.total_tokens.append(len(target_tokens))

    def process_final(self, text_outputs):
        gen_tp_time = []
        start_time = min([t[0] for t in self.timing])
        end_time = max([t[-1] for t in self.timing])
        self.out["AA Output TPS"] = sum(self.total_tokens) / (end_time - start_time)
        for tokens, times in zip(self.total_tokens, self.timing):
            if len(times) > 2:
                gen_tp_time.append((tokens - 1) / (times[-1] - times[1]))
        if gen_tp_time:
            self.out["AA Generation Tokens Per Second"] = compute_statistics(gen_tp_time)
        for k, v in self.out.items():
            print(k, v)
        self.write()

    def clear(self):
        self.timing = []
