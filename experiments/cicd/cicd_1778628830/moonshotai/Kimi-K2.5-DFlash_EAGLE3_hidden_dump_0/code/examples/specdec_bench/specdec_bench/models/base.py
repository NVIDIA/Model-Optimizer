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


class Model:
    def __init__(self, model_dir, tokenizer, max_draft_length):
        raise NotImplementedError

    async def run(self, prompt_ids, sampling_params, request_id, turn_id):
        """
        prompt_ids is list of tokens
        output is list of list of tokens
            len(output) = beam width
            len(output[i]) = tokens produced per step?
        """
        raise NotImplementedError

    def stop(self):
        pass
