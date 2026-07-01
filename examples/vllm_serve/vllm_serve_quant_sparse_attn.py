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

"""Launch vLLM with combined attention quantization + sparse attention.

Restores ModelOpt fakequant (env knobs: ``MODELOPT_STATE_PATH`` / ``QUANT_CFG`` /
``KV_QUANT_CFG`` / ...) **and** installs the sparse attention impl driven by the
checkpoint's ``sparse_attention_config`` block. A single served checkpoint then
runs attention quant (Q/K pre-step, P/V in-kernel) together with skip-softmax
sparsity. Layers with neither active quant nor a sparse feature fall back to
vLLM's native attention.

Usage:
    MODELOPT_STATE_PATH=<ckpt>/modelopt_state.pth \\
        python vllm_serve_quant_sparse_attn.py <path/to/modelopt-exported-ckpt>
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


def main():
    """Launch vLLM with the combined quant + sparse attention worker."""
    parser = FlexibleArgumentParser(
        description="vLLM model server with attention quantization + sparse attention"
    )
    parser.add_argument("model", type=str, help="The path or name of the model to serve")
    parser = make_arg_parser(parser)

    # Ensure workers can import our custom worker modules (and its fakequant_worker sibling).
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    current = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([current, repo_root]) if current else repo_root

    parser.set_defaults(worker_cls="quant_sparse_attn_worker.QuantSparseAttnWorker")

    args = parser.parse_args()
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
