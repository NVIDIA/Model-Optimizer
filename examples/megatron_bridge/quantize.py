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

Example usage to quantize Qwen3-8B to NVFP4 on 2 GPUs (Tensor Parallelism = 2):
    1024 samples from default dataset are used for calibration (sequence length = 4096).

    torchrun --nproc_per_node 2 quantize.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --quant_cfg nvfp4 \
        --tp_size 2 \
        --calib_batch_size 1 \
        --seq_length 4096 \
        --export_megatron_path /tmp/Qwen3-8B-NVFP4-megatron

Equivalent run using a YAML recipe (authoritative for quant_cfg + algorithm + KV-cache config):

    torchrun --nproc_per_node 2 quantize.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --recipe general/ptq/nvfp4_default-kv_fp8 \
        --tp_size 2 \
        --calib_batch_size 1 \
        --seq_length 4096 \
        --export_megatron_path /tmp/Qwen3-8B-NVFP4-megatron

To convert the saved Megatron checkpoint to a deployable HuggingFace checkpoint, run `export.py`.

To see the full usage for advanced configurations, run:
    torchrun --nproc_per_node 1 quantize.py --help

See `README.md` in this directory for more details.
"""

import argparse
import copy
import gc

import torch
from transformers import AutoProcessor

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.distributed as dist
from modelopt.recipe import ModelOptPTQRecipe, load_recipe
from modelopt.recipe.presets import KV_CACHE_NONE, KV_QUANT_CFG_CHOICES, QUANT_CFG_CHOICES
from modelopt.torch.utils import print_args, print_rank_0, warn_rank_0
from modelopt.torch.utils.dataset_utils import get_supported_datasets
from modelopt.torch.utils.plugins.mbridge import load_mbridge_model_from_hf
from modelopt.torch.utils.plugins.megatron_calibration import (
    get_megatron_calibration_forward_loop,
    get_megatron_vlm_calibration_forward_loop,
)
from modelopt.torch.utils.plugins.megatron_generate import megatron_generate
from modelopt.torch.utils.vlm_dataset_utils import get_supported_vlm_datasets

# Default calibration datasets when --calib_dataset_name is not set
DEFAULT_TEXT_CALIB_DATASET = "cnn_nemotron_v2_mix"  # cnn_dailymail + nemotron-post-training-v2
DEFAULT_VLM_CALIB_DATASET = "nemotron_vlm_dataset_v2"

# The --quant_cfg / --kv_cache_quant CLI vocabularies are discovered from the preset
# YAMLs (shared with the hf_ptq examples via modelopt.recipe.presets). --quant_cfg
# additionally accepts any full config name from ``mtq.config.choices`` (e.g.
# ``FP8_DEFAULT_CFG``); see get_quant_config below.

# TODO: Add AutoQuantize (mtq.auto_quantize) support to automatically search a per-layer mix of
# quantization formats that meets a target compression / accuracy constraint, instead of applying a
# single fixed --quant_cfg / --recipe to the whole model.


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

    # Parallelism arguments. Data parallelism is implicit: DP = world_size / (tp * pp * cp).
    # e.g. `torchrun --nproc_per_node 8 quantize.py --tp_size 2` runs with DP=4.
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size")

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
            f"Quantization config. Preset names / short aliases: {', '.join(QUANT_CFG_CHOICES)}. "
            "You can also pass any full config name exposed by modelopt (e.g. FP8_DEFAULT_CFG). "
            "Ignored when --recipe is set."
        ),
    )
    parser.add_argument(
        "--kv_cache_quant",
        type=str,
        default=KV_CACHE_NONE,
        choices=[KV_CACHE_NONE, *KV_QUANT_CFG_CHOICES],
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

    # Calibration dataset arguments (matched to hf_ptq.py)
    parser.add_argument(
        "--calib_dataset_name",
        type=str,
        default=None,
        help=(
            "Calibration dataset. If unset, it is auto-selected by model type: a text dataset "
            f"({DEFAULT_TEXT_CALIB_DATASET}) for language models, and an image-text dataset "
            f"({DEFAULT_VLM_CALIB_DATASET}) for VLMs. Passing a text dataset for a VLM estimates importance from text "
            f"only. Text dataset options: {get_supported_datasets()}; VLM (image) dataset options: "
            f"{get_supported_vlm_datasets()}."
        ),
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

    print_args(args)

    return args


def get_quant_config(args: argparse.Namespace) -> dict:
    """Build the ModelOpt quantization config dict from the parsed arguments."""
    if args.recipe is not None:
        # A YAML recipe is authoritative: it encodes quant_cfg + algorithm + KV-cache config
        # directly, so the --quant_cfg / --kv_cache_quant / --weight_only / --moe_calib_experts_ratio
        # customizations below are skipped.
        print_rank_0(f"Using recipe {args.recipe} for quantization")
        if (
            args.kv_cache_quant != KV_CACHE_NONE
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
            f"Unsupported --quant_cfg '{args.quant_cfg}'. Choose a preset name / short alias "
            f"({', '.join(QUANT_CFG_CHOICES)}) or a full config name from {mtq.config.choices}."
        )

    # Deepcopy so we don't mutate a shared module-level config (the ``mtq.config.choices``
    # full-name branch returns one; QUANT_CFG_CHOICES already hands back a fresh copy), and
    # normalize the inner quant_cfg to the list format so we can safely append customizations below.
    mtq_config = copy.deepcopy(mtq_config)
    mtq_config["quant_cfg"] = mtq.normalize_quant_cfg_list(mtq_config["quant_cfg"])

    if args.weight_only:
        mtq_config["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})

    if args.kv_cache_quant != KV_CACHE_NONE:
        kv_cache_quant_cfg = KV_QUANT_CFG_CHOICES[args.kv_cache_quant]["quant_cfg"]
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
            "context_parallel_size": args.cp_size,
            "expert_tensor_parallel_size": 1,  # Expert tensor parallelism is not supported
            "pipeline_dtype": torch.bfloat16,
            "seq_length": args.seq_length,
            "gradient_accumulation_fusion": False,  # not supported
        },
        init_model_parallel=True,
    )

    # Only the language model is quantized (vision tower + projector stay full precision)
    language_model = getattr(unwrapped_model, "language_model", unwrapped_model)
    is_vlm = language_model is not unwrapped_model
    if is_vlm:
        warn_rank_0(
            "VLM detected: quantizing `model.language_model` only (vision tower left in full precision)."
        )

    # Auto-select the calibration dataset by model type when not explicitly provided.
    if args.calib_dataset_name is None:
        args.calib_dataset_name = (
            DEFAULT_VLM_CALIB_DATASET if is_vlm else DEFAULT_TEXT_CALIB_DATASET
        )

    # Infer the calibration modality from the dataset: the known image-text datasets require a VLM, everything
    # else is text. Passing a text dataset for a VLM estimates importance from text only (vision tower idle).
    use_image_calib = args.calib_dataset_name in get_supported_vlm_datasets()
    if use_image_calib and not is_vlm:
        raise ValueError(
            f"Calibration dataset '{args.calib_dataset_name}' is image-text and requires a VLM; "
            "pass a text dataset for a language model."
        )
    if is_vlm and not use_image_calib:
        warn_rank_0(
            f"Text-only calibration on a VLM (dataset '{args.calib_dataset_name}'): the language "
            "model's calibration statistics will not see vision tokens."
        )
    print_rank_0(f"Using calibration dataset: {args.calib_dataset_name}")

    mtq_config = get_quant_config(args)

    # Quantize only the language model: disable quantizers on every top-level submodule that is not
    # the language model (vision tower + projector). Skip aliases of language-model submodules (e.g.
    # Qwen's ``self.decoder = language_model.decoder``) so the LM's own layers stay enabled.
    if is_vlm:
        lm_module_ids = {id(m) for m in language_model.modules()}
        non_lm_children = sorted(
            name
            for name, child in unwrapped_model.named_children()
            if name != "language_model" and id(child) not in lm_module_ids
        )
        for name in non_lm_children:
            # Anchor to the child subtree (top-level child of the quantized root) so a short non-LM
            # name cannot accidentally match a language-model quantizer path by substring.
            mtq_config["quant_cfg"].append({"quantizer_name": f"{name}.*", "enable": False})
        print_rank_0(f"Disabling quantizers on non-language-model submodules: {non_lm_children}")

    # KV-cache quantization is incompatible with weight compression. Validate on the *resolved*
    # config (KV-cache quantizers are named ``*[kv]_bmm_quantizer``) so this also covers
    # recipe-driven KV-cache configs, not just the --kv_cache_quant flag.
    if args.compress and any(
        isinstance(entry, dict) and "bmm_quantizer" in str(entry.get("quantizer_name", ""))
        for entry in mtq.normalize_quant_cfg_list(mtq_config["quant_cfg"])
    ):
        raise ValueError("--compress cannot be combined with KV-cache quantization.")

    print_rank_0(f"Quantizing the model with: {args.recipe or args.quant_cfg}")
    if "awq" in str(mtq_config.get("algorithm")):
        print_rank_0(
            "AWQ calibration can take longer than other methods; reduce --calib_num_samples to speed it up."
        )

    # Dynamic and weight-only configs need no activation statistics, so skip both the
    # (potentially expensive) calibration dataset download and the calibration forward pass.
    if not mtq.need_calibration(mtq_config):
        warn_rank_0("Dynamic or weight-only quantization detected; skipping calibration.")
        forward_loop = None
    elif not use_image_calib:
        text_forward_loop = get_megatron_calibration_forward_loop(
            tokenizer,
            dataset_name=args.calib_dataset_name,
            num_samples=args.calib_num_samples,
            seq_length=args.seq_length,
            batch_size=args.calib_batch_size,
            pack=True,  # Megatron pretraining-style global-stream document packing
        )

        # Run text prefill on the language model: we quantize the root (a VLM root forward expects
        # vision inputs), but text calibration must drive the inner LM. For plain LMs these are the same.
        def forward_loop(_model=None):
            text_forward_loop(language_model)
    else:
        # VLMs: drive the full VLM forward on image-text pairs so the language model's quantizers
        # see vision-conditioned activations (we still quantize the LM only).
        processor = AutoProcessor.from_pretrained(
            args.hf_model_name_or_path, trust_remote_code=args.trust_remote_code
        )
        forward_loop = get_megatron_vlm_calibration_forward_loop(
            unwrapped_model,  # full VLM (vision encoder + projector + language model)
            processor,
            dataset_name=args.calib_dataset_name,
            num_samples=args.calib_num_samples,
            seq_length=args.seq_length,
            batch_size=args.calib_batch_size,
        )

    if hasattr(unwrapped_model, "calibration_mode"):
        # Some model wrappers (e.g. distillation/speculative) gate calibration behind a flag.
        unwrapped_model.calibration_mode = True
        mtq.quantize(unwrapped_model, mtq_config, forward_loop)
        unwrapped_model.calibration_mode = False
    else:
        mtq.quantize(unwrapped_model, mtq_config, forward_loop)

    # Free calibration/quantization memory before generate
    gc.collect()
    torch.cuda.empty_cache()

    if args.compress:
        mtq.compress(unwrapped_model)
        print_rank_0("Weights are now compressed to low-bit!")

    # Save the quantizer summary alongside the checkpoint for later inspection. Only the master
    # rank writes the file to avoid a multi-rank race on the same path.
    if dist.is_master():
        mtq.print_quant_summary(unwrapped_model, args.export_megatron_path)

    bridge.save_megatron_model(
        model,
        args.export_megatron_path,
        hf_tokenizer_path=args.hf_model_name_or_path,
        hf_tokenizer_kwargs={"trust_remote_code": args.trust_remote_code},
    )
    print_rank_0(
        f"\nSaved quantized model to {args.export_megatron_path} in Megatron format. "
        "To deploy this model (TensorRT-LLM / vLLM / SGLang), convert it to a Unified HF ckpt with export.py"
    )

    # Sanity-check generation with the fake-quantized model. Skipped when --compress is set: the
    # weights are now real low-bit and megatron_generate may not support compressed forward for
    # every quant format.
    if args.compress and not args.skip_generate:
        warn_rank_0(
            "Skipping the post-quantization generation sanity check because --compress is set."
        )
    if not args.skip_generate and not args.compress:
        print_rank_0("\nTesting quantized model with custom prompts...")
        # Sanity-check text generation on the quantized language model.
        language_model.eval()
        for idx, prompt in enumerate(args.prompts.split("|")):
            tokens = tokenizer(prompt, return_tensors="pt")
            # enable_kv_cache=False avoids pre-allocating the static KV cache: this is a short sanity-check
            # generation and the KV-cache allocation can OOM tight quantization runs on large MoE models.
            generated_ids = megatron_generate(
                language_model, tokens.input_ids.cuda(), osl=args.osl, enable_kv_cache=False
            )
            generated_texts = tokenizer.batch_decode(generated_ids)
            print_rank_0(f"\nPrompt {idx + 1}: {prompt}\nGenerated: {generated_texts}")

    print_rank_0("\nDone!")


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        main(args)
    finally:
        dist.cleanup()
