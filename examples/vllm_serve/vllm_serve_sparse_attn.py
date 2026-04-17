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

"""Launch vLLM with sparse attention.

Usage:
    SPARSE_ATTN_CFG=SPARSE_SOFTMAX_DEFAULT python vllm_serve_sparse_attn.py \\
        meta-llama/Llama-3.1-8B --max-model-len 8192

Combined with quantization:
    QUANT_CFG=INT8_SMOOTHQUANT_CFG SPARSE_ATTN_CFG=SPARSE_SOFTMAX_DEFAULT \\
        python vllm_serve_sparse_attn.py meta-llama/Llama-3.1-8B
"""

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
    from vllm.utils import FlexibleArgumentParser
else:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

# Pass sparse attention env vars to ray workers (if supported by this vLLM version)
additional_env_vars = {
    "SPARSE_ATTN_CFG",
    "SPARSE_CALIB_CONFIG_PATH",
    "QUANT_DATASET",
    "QUANT_CALIB_SIZE",
    "QUANT_CFG",
    "AMAX_FILE_PATH",
    "KV_QUANT_CFG",
}

try:
    if vllm_version <= version.parse("0.11.0"):
        from vllm.executor.ray_distributed_executor import RayDistributedExecutor
    else:
        from vllm.v1.executor.ray_executor import RayDistributedExecutor
    if hasattr(RayDistributedExecutor, "ADDITIONAL_ENV_VARS"):
        RayDistributedExecutor.ADDITIONAL_ENV_VARS.update(additional_env_vars)
except ImportError:
    pass  # Ray not installed, single-node only


def main():
    """Launch vLLM with sparse attention worker."""
    parser = FlexibleArgumentParser(description="vLLM model server with sparse attention")
    parser.add_argument("model", type=str, help="The path or name of the model to serve")
    parser = make_arg_parser(parser)

    # Ensure workers can import our custom worker module
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + f"{repo_root}"

    # Select worker based on env vars
    has_quant = os.environ.get("QUANT_CFG") or os.environ.get("KV_QUANT_CFG")
    has_sparse = os.environ.get("SPARSE_ATTN_CFG") or os.environ.get("SPARSE_CALIB_CONFIG_PATH")

    if has_quant and has_sparse:
        worker_cls = "sparse_attn_worker.SparseQuantWorker"
    elif has_sparse:
        worker_cls = "sparse_attn_worker.SparseAttnWorker"
    else:
        print("Warning: No SPARSE_ATTN_CFG or QUANT_CFG set. Running standard vLLM.")
        worker_cls = None

    if worker_cls:
        parser.set_defaults(worker_cls=worker_cls)

    args = parser.parse_args()
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
