# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A state-only worker that resolves ReplaySSM before vLLM allocates its cache."""

import os

from vllm.v1.worker.gpu_worker import Worker

from modelopt.recipe import load_recipe
from modelopt.torch.quantization.plugins.vllm_replayssm import configure_vllm_replayssm


class ReplaySSMWorker(Worker):
    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        recipe = load_recipe(os.environ.get("RECIPE_PATH", "general/ptq/gdn_replayssm_fp8_read"))
        configure_vllm_replayssm(self.model_runner, self.vllm_config, recipe.quantize.model_dump())
