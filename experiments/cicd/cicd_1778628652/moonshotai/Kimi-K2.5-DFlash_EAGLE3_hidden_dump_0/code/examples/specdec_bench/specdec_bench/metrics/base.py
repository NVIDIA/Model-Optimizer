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
import os


class Metric:
    directory = "./"

    def __init__(self):
        self.out = {}
        self.name = "metric"

    def process_step(self, step_outputs, request_id, turn_id):
        raise NotImplementedError

    def process_final(self, text_outputs):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def write(self):
        os.makedirs(self.directory, exist_ok=True)
        if self.out:
            filename = os.path.join(self.directory, f"{self.name}.json")
            if os.path.exists(filename):
                with open(filename) as json_file:
                    existing_data = json.load(json_file)
                existing_data.append(self.out)
            else:
                existing_data = [self.out]

            with open(filename, "w") as json_file:
                json.dump(existing_data, json_file, indent=4)

    @classmethod
    def update_directory(cls, new_dir):
        cls.directory = new_dir
