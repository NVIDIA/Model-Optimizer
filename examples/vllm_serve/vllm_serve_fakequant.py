# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# MIT License
#
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
import sys
from pathlib import Path

import uvloop
import vllm
from packaging import version
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser

vllm_version = version.parse(vllm.__version__)
if vllm_version <= version.parse("0.11.0"):
    from vllm.executor.ray_distributed_executor import RayDistributedExecutor
    from vllm.utils import FlexibleArgumentParser
else:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.executor.ray_executor import RayDistributedExecutor


# Adding the envs you want to pass to the workers
additional_env_vars = {
    "QUANT_DATASET",
    "QUANT_CALIB_SIZE",
    "QUANT_CFG",
    "AMAX_FILE_PATH",
    "KV_QUANT_CFG",
}

RayDistributedExecutor.ADDITIONAL_ENV_VARS.update(additional_env_vars)


def main():
    # Create parser that handles both quant and serve arguments
    parser = FlexibleArgumentParser(description="vLLM model server with quantization support")
    parser.add_argument("model", type=str, help="The path or name of the model to serve")
    parser = make_arg_parser(parser)
    # Ensure workers can import our custom worker module when using spawn
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + f"{repo_root}"

    # Default to our FakeQuantWorker if user doesn't specify a worker class
    parser.set_defaults(worker_cls="fakequant_worker.FakeQuantWorker")

    # Parse arguments
    args = parser.parse_args()
    # Run the server
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
