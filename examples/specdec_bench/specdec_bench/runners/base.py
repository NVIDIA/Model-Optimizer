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
        self.model = model
        self.metrics = metrics
        self.metric_records = []

    def _ensure_metric_records(self):
        if not hasattr(self, "metric_records"):
            self.metric_records = []

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        raise NotImplementedError()

    def process_metrics_final(self, text_outputs):
        self._ensure_metric_records()
        for metric in self.metrics:
            metric.process_final(text_outputs, self.metric_records)

    def process_metrics_step(self, step_outputs, request_id, turn_id, prompt_length=None):
        self._ensure_metric_records()
        self.metric_records.append(
            {
                "request_id": request_id,
                "turn_id": turn_id,
                "prompt_length": prompt_length,
                "output_ids": step_outputs["output_ids"],
                "chunk_lengths": step_outputs.get("chunk_lengths"),
                "token_times": step_outputs.get("token_times", []),
            }
        )

    def clear_metrics(self):
        self._ensure_metric_records()
        self.metric_records = []
        for metric in self.metrics:
            metric.clear()

    def stop(self):
        self.model.stop()
