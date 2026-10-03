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

from .base import BaseRunner


class SimpleRunner(BaseRunner):
    def __init__(self, model, metrics):
        # initialize the accelerate or the hf model
        self.model = model
        self.metrics = metrics
        self.prompt_ar = []

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        model_output = await self.model.run(prompt_ids, max_length, end_id, request_id, turn_id)
        self.process_metrics_step(model_output, request_id, turn_id)
        output_ids = model_output["output_ids"]
        flattened_output_ids = [[] for _ in range(len(output_ids))]
        for i, beam_output in enumerate(output_ids):
            for output_id_iter in beam_output:
                flattened_output_ids[i].extend(output_id_iter)

        return {
            "output_ids": flattened_output_ids,
            "output_logits": model_output.get("output_logits", None),
        }
