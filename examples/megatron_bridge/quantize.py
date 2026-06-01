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
"""Example script for post-training quantization (PTQ) of a GPT / Mamba model using ModelOpt on a
Megatron-Bridge model (loaded from HF).

The process is as follows:
  1. Load a pretrained HuggingFace model into a Megatron-Core model via Megatron-Bridge.
  2. Apply ModelOpt quantization (fake-quant) with calibration on a few samples from a dataset.
     The quantization format is specified either by a short --quant_cfg alias or a --recipe YAML.
  3. (Optional) Compress weights to a real low-bit representation.
  4. Save the quantized model as a Megatron checkpoint (with ModelOpt state). The checkpoint can be
     reloaded for further training (QAT / distillation) or converted to a HuggingFace (unified)
     checkpoint for deployment with `export.py` (see that script for TensorRT-LLM / vLLM / SGLang).

Tensor / pipeline / expert parallelism are all supported here — the Megatron checkpoint is saved
sharded and can be re-sharded on load (e.g. `export.py` reloads it at TP=1 for the HF export).

Example usage to quantize Qwen3-8B to FP8 on 2 GPUs (Tensor Parallelism = 2):
    1024 samples from nemotron-post-training-dataset-v2 are used for calibration.

    torchrun --nproc_per_node 2 quantize.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --quant_cfg fp8 \
        --tp_size 2 \
        --export_megatron_path /tmp/Qwen3-8B-FP8-megatron

Equivalent run using a YAML recipe (authoritative for quant_cfg + algorithm + KV-cache config):

    torchrun --nproc_per_node 2 quantize.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --recipe general/ptq/fp8_default-kv_fp8 \
        --tp_size 2 \
        --export_megatron_path /tmp/Qwen3-8B-FP8-megatron

To convert the saved Megatron checkpoint to a deployable HuggingFace checkpoint, run `export.py`.

To see the full usage for advanced configurations, run:
    torchrun --nproc_per_node 1 quantize.py --help

See `README.md` in this directory for more details.
"""

import argparse
import copy

import torch

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.distributed as dist
from modelopt.recipe import ModelOptPTQRecipe, load_recipe
from modelopt.torch.utils import print_rank_0, warn_rank_0
from modelopt.torch.utils.plugins.mbridge import load_mbridge_model_from_hf
from modelopt.torch.utils.plugins.megatron_calibration import get_megatron_calibration_forward_loop
from modelopt.torch.utils.plugins.megatron_generate import megatron_generate

# Curated short-name aliases for the most common quantization configs. Any other config exposed by
# ``mtq.config.choices`` (e.g. ``FP8_DEFAULT_CFG``) can also be passed by its full name.
QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "fp8_blockwise": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
}

# KV-cache quantization configs (applied on top of the weight/activation quant config).
KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_model_name_or_path", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--export_megatron_path",
        type=str,
        required=True,
        help="Path to save the quantized model in Megatron checkpoint format (with ModelOpt state).",
    )

    # Parallelism arguments
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp_size", type=int, default=1, help="Expert tensor parallel size")

    # Quantization arguments
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help=(
            "PTQ recipe YAML file or builtin name (e.g. 'general/ptq/fp8_default-kv_fp8'). "
            "When set, --quant_cfg, --kv_cache_quant, --weight_only, and --moe_calib_experts_ratio "
            "are ignored; the recipe is authoritative for quant_cfg, algorithm, and KV-cache config."
        ),
    )
    parser.add_argument(
        "--quant_cfg",
        type=str,
        default="fp8",
        help=(
            f"Quantization config. Short aliases: {', '.join(QUANT_CFG_CHOICES)}. "
            "You can also pass any full config name exposed by modelopt (e.g. FP8_DEFAULT_CFG). "
            "Ignored when --recipe is set."
        ),
    )
    parser.add_argument(
        "--kv_cache_quant",
        type=str,
        default="none",
        choices=list(KV_QUANT_CFG_CHOICES),
        help="KV-cache quantization config to apply on top of --quant_cfg. Ignored when --recipe is set.",
    )
    parser.add_argument(
        "--weight_only",
        action="store_true",
        help="Disable input (activation) quantization, i.e. weight-only quantization.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress weights to a real low-bit representation (instead of fake quantization).",
    )
    parser.add_argument(
        "--moe_calib_experts_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of experts (in (0.0, 1.0]) to calibrate per forward pass for MoE models. "
            "Lower values speed up calibration of models with many experts; ignored for dense models."
        ),
    )

    # Calibration dataset arguments
    parser.add_argument(
        "--calib_dataset_name",
        type=str,
        default="nemotron-post-training-dataset-v2",
        help="HF Dataset name or local path used for calibration.",
    )
    parser.add_argument(
        "--calib_num_samples", type=int, default=1024, help="Number of samples for calibration"
    )
    parser.add_argument("--calib_batch_size", type=int, default=1, help="Calibration batch size")
    parser.add_argument("--seq_length", type=int, default=4096, help="Calibration sequence length")

    # Post-quantization generation (sanity check) arguments
    parser.add_argument(
        "--prompts",
        type=str,
        default="Hello!|Born in California, Soyer trained as a",
        help="Prompts to sanity-check the quantized model. Use | to separate batches.",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=32,
        help="Output sequence length for the generation sanity check.",
    )
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="Skip the post-quantization generation sanity check.",
    )

    args = parser.parse_args()

    if args.moe_calib_experts_ratio is not None and not (0.0 < args.moe_calib_experts_ratio <= 1.0):
        parser.error("--moe_calib_experts_ratio must be in the range (0.0, 1.0].")

    print_rank_0("\n==================== Arguments ====================")
    for k, v in args.__dict__.items():
        print_rank_0(f"{k:<35} {v}")
    print_rank_0("===================================================\n")

    return args


def get_quant_config(args: argparse.Namespace) -> dict:
    """Build the ModelOpt quantization config dict from the parsed arguments."""
    if args.recipe is not None:
        # A YAML recipe is authoritative: it encodes quant_cfg + algorithm + KV-cache config
        # directly, so the --quant_cfg / --kv_cache_quant / --weight_only / --moe_calib_experts_ratio
        # customizations below are skipped.
        print_rank_0(f"Using recipe {args.recipe} for quantization")
        if (
            args.kv_cache_quant != "none"
            or args.weight_only
            or args.moe_calib_experts_ratio is not None
        ):
            warn_rank_0(
                "--kv_cache_quant / --weight_only / --moe_calib_experts_ratio are ignored when "
                "--recipe is set; the recipe is authoritative."
            )
        recipe = load_recipe(args.recipe)
        if not isinstance(recipe, ModelOptPTQRecipe):
            raise TypeError(
                f"Expected a PTQ recipe but got {type(recipe).__name__} from {args.recipe}"
            )
        return recipe.quantize.model_dump()

    if args.quant_cfg in QUANT_CFG_CHOICES:
        mtq_config = QUANT_CFG_CHOICES[args.quant_cfg]
    elif args.quant_cfg in mtq.config.choices:
        mtq_config = getattr(mtq, args.quant_cfg)
    else:
        raise ValueError(
            f"Unsupported --quant_cfg '{args.quant_cfg}'. Choose one of the short aliases "
            f"({', '.join(QUANT_CFG_CHOICES)}) or a full config name from {mtq.config.choices}."
        )

    # Deepcopy so we don't mutate the shared module-level config, and normalize the inner quant_cfg
    # to the list format so we can safely append customizations below.
    mtq_config = copy.deepcopy(mtq_config)
    mtq_config["quant_cfg"] = mtq.normalize_quant_cfg_list(mtq_config["quant_cfg"])

    if args.weight_only:
        mtq_config["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})

    # KV-cache quantization is incompatible with weight compression.
    if args.kv_cache_quant != "none":
        if args.compress:
            raise ValueError("--kv_cache_quant cannot be combined with --compress.")
        kv_cache_quant_cfg = getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_quant])["quant_cfg"]
        mtq_config = mtq.utils.update_quant_cfg_with_kv_cache_quant(mtq_config, kv_cache_quant_cfg)

    # For MoE models, optionally calibrate only a fraction of experts per forward pass for speed.
    if args.moe_calib_experts_ratio is not None:
        algorithm = mtq_config.get("algorithm")
        if isinstance(algorithm, str):
            mtq_config["algorithm"] = {
                "method": algorithm,
                "moe_calib_experts_ratio": args.moe_calib_experts_ratio,
            }
        elif isinstance(algorithm, dict):
            algorithm["moe_calib_experts_ratio"] = args.moe_calib_experts_ratio
        else:
            warn_rank_0(
                f"Quantization algorithm {algorithm!r} does not support moe_calib_experts_ratio; ignoring."
            )

    return mtq_config


def main(args: argparse.Namespace):
    bridge, _provider, model, unwrapped_model, tokenizer = load_mbridge_model_from_hf(
        hf_model_name_or_path=args.hf_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        provider_overrides={
            "tensor_model_parallel_size": args.tp_size,
            "pipeline_model_parallel_size": args.pp_size,
            "expert_model_parallel_size": args.ep_size,
            "expert_tensor_parallel_size": args.etp_size,
            "pipeline_dtype": torch.bfloat16,
            "seq_length": args.seq_length,
        },
        init_model_parallel=True,
        # Grouped GEMM is not supported for PTQ + export; use the per-expert (sequential) MLP.
        moe_grouped_gemm=False,
    )

    mtq_config = get_quant_config(args)
    print_rank_0(f"Quantizing the model with: {args.recipe or args.quant_cfg}")
    if "awq" in str(mtq_config.get("algorithm")):
        print_rank_0(
            "AWQ calibration can take longer than other methods; "
            "reduce --calib_num_samples to speed it up."
        )

    # Dynamic and weight-only configs need no activation statistics, so skip both the
    # (potentially expensive) calibration dataset download and the calibration forward pass.
    if mtq.need_calibration(mtq_config):
        forward_loop = get_megatron_calibration_forward_loop(
            tokenizer,
            dataset_name=args.calib_dataset_name,
            num_samples=args.calib_num_samples,
            seq_length=args.seq_length,
            batch_size=args.calib_batch_size,
            # pack=True uses Megatron-LM pretraining-style global-stream document packing
            # Leave to False for backward compatibility
            pack=False,
        )
    else:
        warn_rank_0("Dynamic or weight-only quantization detected; skipping calibration.")
        forward_loop = None

    if hasattr(unwrapped_model, "calibration_mode"):
        # Some model wrappers (e.g. distillation/speculative) gate calibration behind a flag.
        unwrapped_model.calibration_mode = True
        mtq.quantize(unwrapped_model, mtq_config, forward_loop)
        unwrapped_model.calibration_mode = False
    else:
        mtq.quantize(unwrapped_model, mtq_config, forward_loop)

    if args.compress:
        mtq.compress(unwrapped_model)
        print_rank_0("Weights are now compressed to low-bit!")

    # Save the quantizer summary alongside the checkpoint for later inspection. Only the master
    # rank writes the file to avoid a multi-rank race on the same path.
    if dist.is_master():
        mtq.print_quant_summary(unwrapped_model, args.export_megatron_path)

    print_rank_0(f"Saving quantized model to {args.export_megatron_path} in Megatron format...")
    bridge.save_megatron_model(
        model,
        args.export_megatron_path,
        hf_tokenizer_path=args.hf_model_name_or_path,
        hf_tokenizer_kwargs={"trust_remote_code": args.trust_remote_code},
    )
    print_rank_0(f"Saved quantized model to {args.export_megatron_path} in Megatron format")
    print_rank_0(
        "To deploy this model (TensorRT-LLM / vLLM / SGLang), convert it to a HuggingFace "
        f"checkpoint with export.py:\n"
        f"    torchrun --nproc_per_node <N> export.py "
        f"--hf_model_name_or_path {args.hf_model_name_or_path} "
        f"--megatron_path {args.export_megatron_path} "
        f"--export_unified_hf_path {args.export_megatron_path}_hf"
    )

    # Sanity-check generation with the fake-quantized model.
    if not args.skip_generate:
        print_rank_0("Testing quantized model with custom prompts...")
        unwrapped_model.eval()
        for idx, prompt in enumerate(args.prompts.split("|")):
            tokens = tokenizer(prompt, return_tensors="pt")
            # enable_kv_cache=False avoids pre-allocating the static KV cache: this is a short
            # sanity-check generation and the KV-cache allocation can OOM tight quantization runs
            # on large MoE models.
            generated_ids = megatron_generate(
                unwrapped_model, tokens.input_ids.cuda(), osl=args.osl, enable_kv_cache=False
            )
            generated_texts = tokenizer.batch_decode(generated_ids)
            print_rank_0(f"Prompt {idx + 1}: {prompt}")
            print_rank_0(f"Generated: {generated_texts}")

    print_rank_0("Done!")


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        main(args)
    finally:
        dist.cleanup()
