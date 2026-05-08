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

from _test_utils.examples.run_command import (
    extend_cmd_parts,
    run_example_command,
    run_llm_ptq_command,
)
from _test_utils.torch.misc import minimum_sm
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir


def test_lm_eval_hf(tmp_path):
    model_dir = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)

    cmd_parts = extend_cmd_parts(
        ["python", "lm_eval_hf.py"],
        model="hf",
        model_args=f"pretrained={model_dir}",
        tasks="mmlu",
        num_fewshot=5,
        limit=0.1,
        batch_size=8,
    )
    run_example_command(cmd_parts, "llm_eval")


@minimum_sm(89)
def test_qwen3_eval_fp8(tmp_path):
    model_dir = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    try:
        run_llm_ptq_command(
            model=str(model_dir),
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
