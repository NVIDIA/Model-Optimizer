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


class BaseRunner:
    def __init__(self, model, metrics):
        # initialize the accelerate or the hf model
        self.model = model
        self.metrics = metrics
        self.prompt_ar = []

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        raise NotImplementedError()

    def process_metrics_final(self, text_outputs):
        [metric.process_final(text_outputs) for metric in self.metrics]

    def process_metrics_step(self, step_outputs, request_id, turn_id):
        [metric.process_step(step_outputs, request_id, turn_id) for metric in self.metrics]

    def clear_metrics(self):
        [metric.clear() for metric in self.metrics]

    def stop(self):
        self.model.stop()
