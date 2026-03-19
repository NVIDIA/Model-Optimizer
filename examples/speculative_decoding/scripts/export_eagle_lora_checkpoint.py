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

"""Export Eagle3 + LoRA co-trained checkpoint for deployment.

Produces:
  1. A merged base model (original weights + LoRA merged in) for serving.
  2. An eagle head checkpoint in HF-compatible format for vLLM speculative decoding.

The checkpoint from co-training stores LoRA-wrapped key names (e.g.
``q_proj.base_layer.weight``, ``q_proj.lora_A.default.weight``).  To load it
we must reconstruct the model the same way training did: load the vanilla base
model, inject LoRA adapters, convert to Eagle3, then load the checkpoint state
dict into that structure.

Usage:
  python scripts/export_eagle_lora_checkpoint.py \\
      --model_path ckpts/.../checkpoint-XXXXX \\
      --base_model nvidia/Cosmos-Reason2-8B \\
      --export_eagle_path export/eagle-head \\
      --export_merged_path export/merged-model
"""

import argparse
import json
import os

import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.export import export_speculative_decoding
from modelopt.torch.speculative.utils import load_vlm_or_llm_with_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Export Eagle3 + LoRA co-trained checkpoint.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the co-trained checkpoint directory.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Original base model name/path (e.g. nvidia/Cosmos-Reason2-8B).",
    )
    parser.add_argument(
        "--export_eagle_path",
        type=str,
        required=True,
        help="Output directory for the eagle head checkpoint.",
    )
    parser.add_argument(
        "--export_merged_path",
        type=str,
        default=None,
        help="Output directory for the LoRA-merged base model. "
        "If not provided, only the eagle head is exported.",
    )
    parser.add_argument(
        "--eagle_decoder_type",
        type=str,
        default="llama",
        help="Eagle decoder type used during training.",
    )
    parser.add_argument(
        "--eagle_config",
        type=str,
        default=None,
        help="Path to eagle_config.json (if custom config was used during training).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (must match training config).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (must match training config).",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target modules (must match training config).",
    )
    return parser.parse_args()


def _load_checkpoint_state_dict(model_path):
    """Load the full state dict from a checkpoint directory (safetensors or bin)."""
    state_dict = {}
    for fn in sorted(os.listdir(model_path)):
        full = os.path.join(model_path, fn)
        if fn.endswith(".safetensors"):
            state_dict.update(load_file(full))
        elif fn.startswith("model") and fn.endswith(".bin"):
            state_dict.update(torch.load(full, map_location="cpu", weights_only=True))
    return state_dict


def _merge_lora_in_state_dict(state_dict, lora_r, lora_alpha):
    """Merge LoRA A/B deltas into base_layer weights and return a clean state dict.

    Handles keys produced by ``peft.inject_adapter_in_model`` where the base
    weight is stored under ``<prefix>.base_layer.weight`` and LoRA matrices are
    ``<prefix>.lora_A.default.weight`` / ``<prefix>.lora_B.default.weight``.

    Returns a new dict with plain ``<prefix>.weight`` keys (LoRA merged in) and
    all non-LoRA keys passed through unchanged.
    """
    scaling = lora_alpha / lora_r
    merged = {}
    base_layer_keys = {k for k in state_dict if ".base_layer.weight" in k}

    processed_prefixes = set()
    for bl_key in base_layer_keys:
        prefix = bl_key.rsplit(".base_layer.weight", 1)[0]
        processed_prefixes.add(prefix)

        a_key = f"{prefix}.lora_A.default.weight"
        b_key = f"{prefix}.lora_B.default.weight"
        out_key = f"{prefix}.weight"

        w = state_dict[bl_key].clone()
        if a_key in state_dict and b_key in state_dict:
            lora_a = state_dict[a_key]
            lora_b = state_dict[b_key]
            w += (lora_b @ lora_a).to(w.dtype) * scaling
        merged[out_key] = w

    for k, v in state_dict.items():
        skip = False
        for pfx in processed_prefixes:
            if k.startswith((pfx + ".base_layer.", pfx + ".lora_")):
                skip = True
                break
        if not skip:
            merged[k] = v

    return merged


mto.enable_huggingface_checkpointing()


def main():
    args = parse_args()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        bias="none",
    )

    custom_config = json.load(open(args.eagle_config)) if args.eagle_config else {}
    eagle_convert_config = {
        "eagle_decoder_type": args.eagle_decoder_type,
        "eagle_freeze_base_model": False,
        "eagle_architecture_config": custom_config,
    }

    # --- Rebuild the model architecture to match the checkpoint ---
    print(f"Loading base model architecture from {args.base_model}...")
    _, model = load_vlm_or_llm_with_kwargs(
        args.base_model, torch_dtype="auto", device_map="cpu", trust_remote_code=True
    )

    print("Injecting LoRA adapters...")
    inject_adapter_in_model(lora_config, model, adapter_name="default")

    print("Converting to Eagle3...")
    mtsp.convert(model, [("eagle", eagle_convert_config)])

    # --- Load checkpoint weights into the reconstructed model ---
    print(f"Loading checkpoint weights from {args.model_path}...")
    ckpt_state = _load_checkpoint_state_dict(args.model_path)
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys (may be expected for optimizer states)")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys")

    model.eval()

    # --- Export Eagle head ---
    print("Exporting eagle head...")
    with torch.inference_mode():
        export_speculative_decoding(model, export_dir=args.export_eagle_path)
    print(f"Eagle head exported to {args.export_eagle_path}")

    # --- Optionally export LoRA-merged base model ---
    if args.export_merged_path:
        print("Merging LoRA into base model and exporting...")
        merged_state = _merge_lora_in_state_dict(
            ckpt_state, lora_r=args.lora_r, lora_alpha=args.lora_alpha
        )

        # Filter to only base model keys (exclude eagle_module)
        base_keys = {k: v for k, v in merged_state.items() if "eagle_module" not in k}

        config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
        if "vl" in config.model_type.lower():
            base_model = AutoModelForVision2Seq.from_pretrained(
                args.base_model, torch_dtype="auto", trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype="auto", trust_remote_code=True
            )

        missing, unexpected = base_model.load_state_dict(base_keys, strict=False)
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys when loading merged weights")

        os.makedirs(args.export_merged_path, exist_ok=True)
        base_model.save_pretrained(args.export_merged_path)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(args.export_merged_path)
        print(f"Merged model saved to {args.export_merged_path}")


if __name__ == "__main__":
    main()
