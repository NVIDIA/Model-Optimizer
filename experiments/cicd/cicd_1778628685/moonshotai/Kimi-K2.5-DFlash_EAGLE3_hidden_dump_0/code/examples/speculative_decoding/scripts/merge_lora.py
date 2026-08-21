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

"""Merge LoRA weights from an exported EAGLE checkpoint into the base model and save.

Usage:
    python merge_lora.py \
        --base_model_path /path/to/original/base/model \
        --exported_lora_dir /path/to/exported/eagle/checkpoint \
        --output_path /path/to/merged/output

The exported checkpoint (from export_hf_checkpoint.py) contains
adapter_model.safetensors and adapter_config.json in standard peft format.
This script loads the original base model, applies the trained LoRA adapters,
merges them into the base weights, and saves the fused model + tokenizer.
"""

import argparse
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights from an exported EAGLE checkpoint into the base model."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the original base model (HF model name or local path).",
    )
    parser.add_argument(
        "--exported_lora_dir",
        type=str,
        required=True,
        help="Path to the exported EAGLE checkpoint containing adapter_model.safetensors.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save the merged (fused) base model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lora_dir = Path(args.exported_lora_dir)

    # Verify exported files exist (standard peft naming)
    config_path = lora_dir / "adapter_config.json"
    weights_path = lora_dir / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected adapter_config.json and adapter_model.safetensors "
            f"in {lora_dir}. Run export_hf_checkpoint.py first."
        )

    lora_sd = load_file(weights_path)
    print(f"Loaded {len(lora_sd)} LoRA tensors from {lora_dir}")
    print(f"  Sample keys: {list(lora_sd.keys())[:4]}")

    # Load the original base model
    print(f"Loading base model from {args.base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype="auto", device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # Load LoRA adapter into the base model (export dir uses standard peft naming)
    print("Loading LoRA adapter via PeftModel.from_pretrained...")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(lora_dir))
    print("  PeftModel loaded successfully")

    # Debug: check adapter file keys vs model keys and values
    adapter_keys = set(lora_sd.keys())
    model_lora_keys = {k for k in model.state_dict() if ".lora_A." in k or ".lora_B." in k}
    print(f"  Adapter file keys (first 4): {sorted(adapter_keys)[:4]}")
    print(f"  Model LoRA keys (first 4): {sorted(model_lora_keys)[:4]}")
    # Check if exported lora_B values are actually non-zero
    for k, v in lora_sd.items():
        if ".lora_B." in k:
            print(f"  Exported {k}: shape={v.shape}, norm={v.norm().item():.6f}")
            break

    # Verify lora_B weights are non-zero (B is init'd to zero, so non-zero means loaded)
    lora_b_norms = [v.norm().item() for k, v in model.state_dict().items() if ".lora_B." in k]
    if not lora_b_norms or all(n == 0 for n in lora_b_norms):
        raise RuntimeError("LoRA-B weights are all zero — adapter loading failed.")
    print(
        f"  Verified: {len(lora_b_norms)} LoRA-B matrices "
        f"(mean norm={sum(lora_b_norms) / len(lora_b_norms):.4f})"
    )

    # Merge LoRA into base weights and remove adapter wrappers
    model = model.merge_and_unload()
    print("LoRA merged successfully.")

    # Save
    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Restore the original base model's config.json.  save_pretrained() with newer
    # transformers (>=5.x) rewrites config fields (e.g. rope_theta → rope_parameters,
    # torch_dtype → dtype) which can confuse downstream engines like TRT-LLM or vLLM.
    # Since LoRA only changes weights — not architecture — the original config is correct.
    import shutil

    base_config = Path(args.base_model_path) / "config.json"
    output_config = Path(args.output_path) / "config.json"
    if base_config.exists():
        shutil.copy2(str(base_config), str(output_config))
        print(f"  Restored original config.json from {base_config}")

    print(f"Done! Merged model saved to {args.output_path}")


if __name__ == "__main__":
    main()
