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
"""Export a quantized Megatron checkpoint (produced by quantize.py) to a HuggingFace (unified)
checkpoint that can be deployed directly with TensorRT-LLM, vLLM, or SGLang.

The process is as follows:
  1. Build the Megatron-Core model structure + tokenizer from the original HuggingFace model.
  2. Load the quantized Megatron checkpoint (ModelOpt state + weights are restored automatically).
  3. Export the model to a HuggingFace (unified) checkpoint via ModelOpt.

The HuggingFace unified exporter does not gather tensor-parallel-sharded weights, so this script
always loads the checkpoint at tensor_model_parallel_size=1 (re-sharding from whatever TP was used
during quantization). Use --pp_size to shard a large model across GPUs for export.

Example usage to export an FP8 checkpoint produced by quantize.py:

    torchrun --nproc_per_node 1 export.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --megatron_path /tmp/Qwen3-8B-FP8-megatron \
        --export_unified_hf_path /tmp/Qwen3-8B-FP8-hf

See `README.md` in this directory for more details.
"""

import argparse

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.core.utils import unwrap_model

import modelopt.torch.export as mtex
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import print_args, print_rank_0

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--hf_model_name_or_path",
        type=str,
        required=True,
        help="Original HuggingFace model (used for the model structure, tokenizer, and config).",
    )
    parser.add_argument(
        "--megatron_path",
        type=str,
        required=True,
        help="Path to the quantized Megatron checkpoint produced by quantize.py.",
    )
    parser.add_argument(
        "--export_unified_hf_path",
        type=str,
        required=True,
        help="Directory to write the exported HuggingFace (unified) checkpoint to.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")

    # Only Pipeline parallelism is supported for export
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(_DTYPE_MAP),
        help="Data type for the exported weights.",
    )
    parser.add_argument(
        "--export_extra_modules",
        action="store_true",
        help="Export extra modules such as Medusa heads, EAGLE, or MTP.",
    )

    args = parser.parse_args()

    print_args(args)

    return args


def main(args: argparse.Namespace):
    trust_remote_code = is_safe_repo(
        trust_remote_code=args.trust_remote_code, hf_path=args.hf_model_name_or_path
    )
    torch_dtype = _DTYPE_MAP[args.dtype]

    # Build the model structure + tokenizer from HF (weights come from the Megatron checkpoint).
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_name_or_path, trust_remote_code=trust_remote_code
    )
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1  # Tensor parallelism is not supported
    provider.pipeline_model_parallel_size = args.pp_size
    provider.expert_model_parallel_size = 1  # Expert parallelism is not supported
    provider.expert_tensor_parallel_size = 1  # Expert tensor parallelism is not supported
    provider.pipeline_dtype = torch_dtype
    provider.finalize()
    provider.initialize_model_parallel(seed=0)

    # Load the quantized checkpoint. ModelOpt state + weights are restored automatically, and the
    # checkpoint is re-sharded to TP=1 regardless of the parallelism used during quantization.
    print_rank_0(f"Loading quantized Megatron checkpoint from {args.megatron_path}...")
    megatron_model = bridge.load_megatron_model(
        args.megatron_path,
        mp_overrides={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": args.pp_size,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
        },
        wrap_with_ddp=False,
    )
    unwrapped_model = unwrap_model(megatron_model[0])

    # Extra modules (Medusa / EAGLE / MTP) only exist on the last pipeline stage. Use an all-reduce
    # MAX over all ranks (rather than a broadcast from a hard-coded source rank) so the decision is
    # correct regardless of pipeline placement / global rank ordering.
    has_extra_modules = hasattr(unwrapped_model, "eagle_module") or hasattr(
        unwrapped_model, "medusa_heads"
    )
    if torch.distributed.is_initialized():
        flag = torch.tensor(
            [int(has_extra_modules)], dtype=torch.int, device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        has_extra_modules = bool(flag.item())
    export_extra_modules = has_extra_modules and args.export_extra_modules

    print_rank_0(
        f"Exporting to HuggingFace (unified) checkpoint at {args.export_unified_hf_path}..."
    )
    mtex.export_mcore_gpt_to_hf(
        unwrapped_model,
        args.hf_model_name_or_path,
        export_extra_modules=export_extra_modules,
        dtype=torch_dtype,
        export_dir=args.export_unified_hf_path,
        moe_router_dtype=getattr(unwrapped_model.config, "moe_router_dtype", None),
        trust_remote_code=trust_remote_code,
    )
    print_rank_0(f"Exported HuggingFace checkpoint to {args.export_unified_hf_path}")


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        main(args)
    finally:
        dist.cleanup()
