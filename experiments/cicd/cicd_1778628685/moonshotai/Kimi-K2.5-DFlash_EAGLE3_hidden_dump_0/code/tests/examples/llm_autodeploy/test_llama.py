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

import os
import subprocess
import time

from _test_utils.examples.run_command import (
    extend_cmd_parts,
    run_command_in_background,
    run_example_command,
)
from _test_utils.torch.distributed.utils import get_free_port
from _test_utils.torch.misc import minimum_sm


def run_llm_autodeploy_command(
    model: str, quant: str, effective_bits: float, output_dir: str, **kwargs
):
    # Create temporary directory for saving the quantized checkpoint
    port = get_free_port()
    quantized_ckpt_dir = os.path.join(output_dir, "quantized_model")
    kwargs.update(
        {
            "hf_ckpt": model,
            "quant": quant,
            "effective_bits": effective_bits,
            "save_quantized_ckpt": quantized_ckpt_dir,
            "port": port,
        }
    )

    server_handler = None
    try:
        # Quantize and deploy the model to the background
        cmd_parts = extend_cmd_parts(["scripts/run_auto_quant_and_deploy.sh"], **kwargs)
        # Pass None to stdout and stderr to see the output in the console
        server_handler = run_command_in_background(
            cmd_parts, "llm_autodeploy", stdout=None, stderr=None
        )

        # Wait for the server to start. We might need to build
        time.sleep(100)

        # Test the deployment
        run_example_command(
            ["python", "api_client.py", "--prompt", "What is AI?", "--port", str(port)],
            "llm_autodeploy",
        )
    finally:
        if server_handler:
            server_handler.terminate()


@minimum_sm(89)
def test_llama_fp8(tiny_llama_path, tmp_path):
    try:
        run_llm_autodeploy_command(
            model=tiny_llama_path,
            quant="fp8",
            effective_bits=8.5,
            output_dir=tmp_path,
        )
    finally:
        # Force kill llm-serve if it's still running
        subprocess.run(["pkill", "-f", "llm-serve"], check=False)
