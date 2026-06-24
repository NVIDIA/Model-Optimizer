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

    torchrun --nproc_per_node 2 export.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --megatron_path /tmp/Qwen3-8B-FP8-megatron \
        --pp_size 2 \
        --export_unified_hf_path /tmp/Qwen3-8B-FP8-hf

See `README.md` in this directory for more details.
"""

import argparse

import torch
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.core.utils import unwrap_model

import modelopt.torch.utils.distributed as dist
from modelopt.torch.export import export_mcore_gpt_to_hf
from modelopt.torch.utils import print_args, print_rank_0
from modelopt.torch.utils.plugins.mbridge import (
    load_mbridge_model_from_hf,
    load_modelopt_megatron_checkpoint,
)


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
    parser.add_argument(
        "--export_extra_modules",
        action="store_true",
        help="Export extra modules such as Medusa heads, EAGLE, or MTP.",
    )

    # Only Pipeline parallelism is supported for export
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the first pipeline stage (Uneven Pipeline Parallelism)",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the last pipeline stage (Uneven Pipeline Parallelism)",
    )

    args = parser.parse_args()

    print_args(args)

    return args


def main(args: argparse.Namespace):
    trust_remote_code = is_safe_repo(
        trust_remote_code=args.trust_remote_code, hf_path=args.hf_model_name_or_path
    )

    # Build the model structure from HF
    _bridge, _provider, model, _unwrapped_model, _tokenizer = load_mbridge_model_from_hf(
        hf_model_name_or_path=args.hf_model_name_or_path,
        trust_remote_code=trust_remote_code,
        provider_overrides={
            "tensor_model_parallel_size": 1,  # Tensor parallelism is not supported
            "pipeline_model_parallel_size": args.pp_size,
            "expert_model_parallel_size": 1,  # Expert parallelism is not supported
            "expert_tensor_parallel_size": 1,  # Expert tensor parallelism is not supported
            "num_layers_in_first_pipeline_stage": args.num_layers_in_first_pipeline_stage,
            "num_layers_in_last_pipeline_stage": args.num_layers_in_last_pipeline_stage,
            "pipeline_dtype": torch.bfloat16,
        },
        init_model_parallel=True,
        load_weights=False,  # The weights come from the Megatron checkpoint, so HF weights are not loaded
    )

    # Load the quantized checkpoint (with the correct layer spec) rather than reconstructing it from the checkpoint
    # config, which avoids the non-serializable layer-spec issue for MoE / Mamba models.
    print_rank_0(f"Loading quantized Megatron checkpoint from {args.megatron_path}...")
    load_modelopt_megatron_checkpoint(model, args.megatron_path)
    unwrapped_model = unwrap_model(model[0])

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
    # TODO: Support exporting quantized VLMs. export_mcore_gpt_to_hf uses modelopt's own per-arch
    # mcore<->HF mappings (mcore_qwen.py, ...), which don't cover Qwen3.5-VL / Gemma3-VL. Rather than
    # authoring new per-model mappings, route the megatron->HF quant export through Megatron-Bridge's
    # AutoBridge.export_hf_weights_quant(quantization_checker, quant_fn, quant_block_size): it reuses
    # the bridge's per-model mapping (covering these VLMs plus the vision tower/projector, which the
    # checker leaves in full precision), so modelopt only supplies the checker + pack/scale fn and the
    # hf_quant_config (KV-cache scales need a separate path). quantize.py already saves the
    # quantized-LM VLM Megatron checkpoint.
    export_mcore_gpt_to_hf(
        unwrapped_model,
        args.hf_model_name_or_path,
        export_extra_modules=export_extra_modules,
        dtype=torch.bfloat16,
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
