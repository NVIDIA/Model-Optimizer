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
during quantization). Use --pp_size and/or --ep_size to shard a large model across GPUs for
export.

Example usage to export an FP8 checkpoint produced by quantize.py:

    torchrun --nproc_per_node 2 export_quantized_megatron_to_hf.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --megatron_path /tmp/Qwen3-8B-FP8-megatron \
        --pp_size 2 \
        --export_unified_hf_path /tmp/Qwen3-8B-FP8-hf

With --vllm_fakequant_export, the weight quantizers are instead folded into BF16 weights and the
model is saved through Megatron-Bridge (any TP / EP / PP, LLMs and VLMs). The enabled activation and
KV-cache quantizers go to quantizer_state.pth and quant_recipe.yaml sidecars, which
examples/vllm_serve/vllm_serve_fakequant.py loads to evaluate the checkpoint with vLLM. Formats
without them (e.g. weight-only GGML IQ2) need no sidecars and serve with stock vLLM:

    torchrun --nproc_per_node 2 export_quantized_megatron_to_hf.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --megatron_path /tmp/Qwen3-8B-NVFP4-QAD/checkpoints \
        --tp_size 2 \
        --vllm_fakequant_export \
        --export_unified_hf_path /tmp/Qwen3-8B-NVFP4-QAD-vllm-fq

See `README.md` in this directory for more details.
"""

import argparse
import json
from pathlib import Path

import torch
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.core.utils import unwrap_model
from mlflow_utils import (
    NON_PARAMS,
    add_mlflow_args,
    checkpoint_name,
    checkpoint_root,
    mlflow_run,
    resolve_mlflow_args,
)

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.distributed as dist
from modelopt.torch.export import ensure_local_checkpoint, export_mcore_gpt_to_hf
from modelopt.torch.export.plugins.vllm_fakequant_megatron import (
    check_mcore_vllm_fakequant_quantizers,
)
from modelopt.torch.utils import print_args, print_rank_0
from modelopt.torch.utils.mlflow import Tool, masked_args
from modelopt.torch.utils.plugins.mbridge import (
    is_vlm_config,
    load_mbridge_model_from_hf,
    load_modelopt_megatron_checkpoint,
    use_moe_grouped_gemm,
)

EXPORT = Tool(
    name="megatron_bridge_export",
    tracks=(
        "Track this export on an MLflow server, uploading the command and the run log, and "
        "writing .experiment.json into --export_unified_hf_path."
    ),
    variant_help="the Megatron checkpoint's directory name",
    variant=lambda args: checkpoint_name(args.megatron_path),
    model=lambda args: args.hf_model_name_or_path,
    checkpoint=lambda args: args.export_unified_hf_path,
    source=lambda args: checkpoint_root(args.megatron_path),
    non_params=NON_PARAMS,
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
        "--no_moe_grouped_gemm",
        action="store_true",
        help=(
            "Force SequentialMLP for MoE experts instead of the fused TEGroupedMLP (grouped GEMM). "
            "By default grouped GEMM is used unless the architecture cannot export it to "
            "HuggingFace, in which case SequentialMLP is selected automatically."
        ),
    )
    parser.add_argument(
        "--vllm_fakequant_export",
        action="store_true",
        help=(
            "Fold the weight quantizers into BF16 weights and save through Megatron-Bridge, for "
            "accuracy evaluation in vLLM of formats it has no kernels for (e.g. NVFP4 QAD on "
            "Hopper, GGML IQ2). Enabled activation and KV-cache quantizers go to "
            "quantizer_state.pth and quant_recipe.yaml sidecars for vllm_serve_fakequant.py."
        ),
    )
    parser.add_argument(
        "--export_extra_modules",
        action="store_true",
        help="Export extra modules such as Medusa heads, EAGLE, or MTP.",
    )

    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size (--vllm_fakequant_export only; the unified exporter needs 1)",
    )
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--ep_size",
        type=int,
        default=1,
        help="Expert parallel size; shards the experts across ranks",
    )
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

    add_mlflow_args(parser, EXPORT)

    args = parser.parse_args()
    resolve_mlflow_args(args, parser, EXPORT)

    print_args(masked_args(args))

    # Flipped by main() once the HuggingFace checkpoint is on disk, which is what the MLflow
    # provenance pointer claims authorship of.
    args.checkpoint_exported = False

    return args


def export_vllm_fakequant(bridge, model: list, export_dir: str) -> None:
    """Fold the weight quantizers into BF16 weights and save them with Megatron-Bridge.

    Megatron-Bridge writes the enabled quantizers' ranges and settings under HF names to
    ``quantizer_state.pth`` and ``quant_recipe.yaml``, the sidecars ``vllm_serve_fakequant.py``
    loads; it writes none when no quantizer is left enabled (e.g. weight-only formats).
    """
    if not hasattr(MegatronModelBridge, "build_modelopt_quantizer_recipe"):
        raise RuntimeError(
            "--vllm_fakequant_export needs a Megatron-Bridge that writes the quantizer sidecars "
            "(quantizer_state.pth and quant_recipe.yaml); see the Pre-Requisites in "
            "examples/megatron_bridge/README.md."
        )
    for chunk in model:
        check_mcore_vllm_fakequant_quantizers(unwrap_model(chunk))
        mtq.fold_weight(unwrap_model(chunk))
    bridge.save_hf_pretrained(model, export_dir)
    if dist.is_master():
        # The weights are BF16, so a pre-quantized source's quantization config would misdescribe them.
        config_path = Path(export_dir) / "config.json"
        config = json.loads(config_path.read_text())
        if config.pop("quantization_config", None) is not None:
            config_path.write_text(json.dumps(config, indent=2))
        (Path(export_dir) / "hf_quant_config.json").unlink(missing_ok=True)
    dist.barrier()


def export_unified_hf(unwrapped_model, args: argparse.Namespace, trust_remote_code: bool) -> None:
    """Export the quantized model to a unified HF checkpoint with ModelOpt's Megatron exporter."""
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
    # TODO: Gemma3-VL is not in export_mcore_gpt_to_hf's per-arch mappings yet.
    export_mcore_gpt_to_hf(
        unwrapped_model,
        args.local_checkpoint_path,
        export_extra_modules=export_extra_modules,
        dtype=torch.bfloat16,
        export_dir=args.export_unified_hf_path,
        trust_remote_code=trust_remote_code,
    )


def main(args: argparse.Namespace):
    # --hf_model_name_or_path stays as given. Later steps read the local copy of the whole
    # checkpoint in local_checkpoint_path, since exporters read the source's files from local disk
    # only; hub_model_id is the Hub ID, or None for a local path.
    args.hub_model_id, args.local_checkpoint_path = ensure_local_checkpoint(
        args.hf_model_name_or_path
    )
    trust_remote_code = is_safe_repo(
        trust_remote_code=args.trust_remote_code, hf_path=args.local_checkpoint_path
    )
    if not args.vllm_fakequant_export and args.tp_size != 1:
        raise ValueError("--tp_size > 1 needs --vllm_fakequant_export.")
    if args.vllm_fakequant_export and args.export_extra_modules:
        raise ValueError("--export_extra_modules is not supported with --vllm_fakequant_export.")

    # Build the model structure from HF
    bridge, _provider, model, _unwrapped_model, _tokenizer = load_mbridge_model_from_hf(
        hf_model_name_or_path=args.local_checkpoint_path,
        trust_remote_code=trust_remote_code,
        moe_grouped_gemm=use_moe_grouped_gemm(
            args.local_checkpoint_path,
            trust_remote_code=trust_remote_code,
            force_sequential=args.no_moe_grouped_gemm,
        ),
        provider_overrides={
            "tensor_model_parallel_size": args.tp_size,
            "pipeline_model_parallel_size": args.pp_size,
            "expert_model_parallel_size": args.ep_size,
            "expert_tensor_parallel_size": 1,  # Expert tensor parallelism is not supported
            "num_layers_in_first_pipeline_stage": args.num_layers_in_first_pipeline_stage,
            "num_layers_in_last_pipeline_stage": args.num_layers_in_last_pipeline_stage,
            "pipeline_dtype": torch.bfloat16,
        },
        init_model_parallel=True,
        # The weights come from the Megatron checkpoint. Megatron-Bridge saves the whole in-memory
        # model, and a VLM's QAD checkpoint holds only the language model, so load the HF vision
        # tower first.
        load_weights=args.vllm_fakequant_export
        and is_vlm_config(args.local_checkpoint_path, trust_remote_code=trust_remote_code),
    )

    # Load the quantized checkpoint (with the correct layer spec) rather than reconstructing it from the checkpoint
    # config, which avoids the non-serializable layer-spec issue for MoE / Mamba models.
    print_rank_0(f"Loading quantized Megatron checkpoint from {args.megatron_path}...")
    load_modelopt_megatron_checkpoint(model, args.megatron_path)
    unwrapped_model = unwrap_model(model[0])
    mtq.merge_quant_lora(unwrapped_model)

    if args.vllm_fakequant_export:
        print_rank_0(f"Exporting vLLM fake-quant checkpoint to {args.export_unified_hf_path}...")
        export_vllm_fakequant(bridge, model, args.export_unified_hf_path)
    else:
        export_unified_hf(unwrapped_model, args, trust_remote_code)
    args.checkpoint_exported = True
    print_rank_0(f"Exported HuggingFace checkpoint to {args.export_unified_hf_path}")


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        # Entered inside the try: opening the run is fatal by design, and the peers of a rank
        # that exits without dist.abort() stay blocked on the first collective.
        with mlflow_run(args, EXPORT):
            main(args)
    except BaseException:
        dist.abort()  # peers may be stuck in a collective this rank will never reach
    finally:
        dist.cleanup()
