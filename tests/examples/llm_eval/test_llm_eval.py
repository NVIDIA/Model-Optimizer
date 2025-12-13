# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess

from _test_utils.examples.models import TINY_LLAMA_PATH
from _test_utils.examples.run_command import run_llm_ptq_command
from _test_utils.torch.misc import minimum_sm


@minimum_sm(89)
def test_llama_eval_fp8():
    try:
        run_llm_ptq_command(
            model=TINY_LLAMA_PATH,
            quant="fp8",
            tasks="mmlu,lm_eval,simple_eval",
            calib=64,
            lm_eval_tasks="hellaswag,gsm8k",
            simple_eval_tasks="humaneval",
            lm_eval_limit=0.1,
            batch=8,
        )
    finally:
        # Force kill llm-serve if it's still running
        subprocess.run(["pkill", "-f", "llm-serve"], check=False)


def test_llama_eval_sparse_attention(tiny_llama_path):
    """Test sparse attention with llm_eval integration."""
    try:
        # Test with default sparse attention config (no quantization)
        run_llm_ptq_command(
            model=tiny_llama_path,
            quant="none",  # No quantization, only sparse attention
            tasks="lm_eval",
            lm_eval_tasks="hellaswag",
            lm_eval_limit=0.05,  # Small limit for fast test
            sparse_cfg="SKIP_SOFTMAX_DEFAULT",
            batch=4,
        )
    finally:
        subprocess.run(["pkill", "-f", "llm-serve"], check=False)
