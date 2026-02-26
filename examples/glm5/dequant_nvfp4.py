#!/usr/bin/env python3
"""Dequantize NVFP4/FP8 checkpoint back to bf16 for HuggingFace inference testing.

Usage:
    python dequant_nvfp4.py dequant --nvfp4_dir /path/to/nvfp4 --output_dir /path/to/output
    python dequant_nvfp4.py generate --model_dir /path/to/dequanted
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# e2m1 lookup table for NVFP4: indices 0-7 positive, 8-15 negative
E2M1_LUT = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)

NVFP4_BLOCK_SIZE = 16  # 16 FP4 values per scale
FP8_BLOCK_SIZE = 128


def dequant_nvfp4(weight_uint8, weight_scale, weight_scale_2):
    """Dequantize NVFP4 packed uint8 weight to bf16.

    Args:
        weight_uint8: [out, packed_in] uint8 tensor (2 FP4 values per byte)
        weight_scale: [out, packed_in // 8] float8_e4m3fn per-block scale
        weight_scale_2: scalar float32 double-scale
    Returns:
        [out, packed_in * 2] bf16 tensor
    """
    out_dim, packed_in = weight_uint8.shape
    unpacked_in = packed_in * 2

    # Unpack uint8 to two 4-bit indices
    unpacked = torch.empty(out_dim, unpacked_in, dtype=torch.long)
    unpacked[:, 0::2] = (weight_uint8 & 0x0F).long()  # low nibble
    unpacked[:, 1::2] = (weight_uint8 >> 4).long()  # high nibble

    # Lookup e2m1 values
    fp_values = E2M1_LUT[unpacked]  # [out, unpacked_in] f32

    # Compute per-block scales: scale * scale_2
    per_block_scale = weight_scale.float() * weight_scale_2.float()  # [out, num_blocks]

    # Reshape to [out, num_blocks, block_size] and apply scale
    num_blocks = per_block_scale.shape[1]
    fp_values = fp_values.view(out_dim, num_blocks, NVFP4_BLOCK_SIZE)
    result = fp_values * per_block_scale.unsqueeze(-1)

    return result.reshape(out_dim, unpacked_in).to(torch.bfloat16)


def dequant_fp8(weight_fp8, scale_inv):
    """Dequantize FP8 weight with block-128 scale_inv to bf16.

    Args:
        weight_fp8: [M, N] float8_e4m3fn tensor
        scale_inv: [ceil(M/128), ceil(N/128)] float32 tensor
    Returns:
        [M, N] bf16 tensor
    """
    M, N = weight_fp8.shape
    scale_M, scale_N = scale_inv.shape
    bs = FP8_BLOCK_SIZE

    # Convert to float for computation
    w = weight_fp8.float()

    # Pad if dimensions not divisible by block size
    padded_M = scale_M * bs
    padded_N = scale_N * bs
    if padded_M != M or padded_N != N:
        w_padded = torch.zeros(padded_M, padded_N, dtype=torch.float32)
        w_padded[:M, :N] = w
        w = w_padded

    # Reshape to blocks and apply scale
    w = w.view(scale_M, bs, scale_N, bs)
    w = w * scale_inv[:, None, :, None]
    w = w.reshape(padded_M, padded_N)

    # Trim padding
    return w[:M, :N].to(torch.bfloat16)


def is_scale_key(name):
    """Check if tensor name is a scale/companion tensor that should be dropped."""
    return (
        name.endswith(".weight_scale")
        or name.endswith(".weight_scale_2")
        or name.endswith(".weight_scale_inv")
        or name.endswith(".input_scale")
    )


def process_shard(src_path, dst_path):
    """Process a single safetensors shard: dequantize quantized weights, pass through others."""
    tensors = load_file(src_path, device="cpu")

    output = {}
    processed_nvfp4 = 0
    processed_fp8 = 0
    passthrough = 0

    # First pass: identify weight keys and their companions
    for name, tensor in tensors.items():
        if is_scale_key(name):
            continue  # skip scale tensors

        if tensor.dtype == torch.uint8:
            # NVFP4 quantized weight
            scale_key = name + "_scale"
            scale2_key = name + "_scale_2"
            weight_scale = tensors[scale_key]
            weight_scale_2 = tensors[scale2_key]
            output[name] = dequant_nvfp4(tensor, weight_scale, weight_scale_2)
            processed_nvfp4 += 1

        elif tensor.dtype == torch.float8_e4m3fn:
            # FP8 quantized weight
            scale_inv_key = name + "_scale_inv"
            scale_inv = tensors[scale_inv_key]
            output[name] = dequant_fp8(tensor, scale_inv)
            processed_fp8 += 1

        else:
            # bf16, f32, int — pass through as-is
            output[name] = tensor
            passthrough += 1

    save_file(output, dst_path)
    return src_path.name, processed_nvfp4, processed_fp8, passthrough


def cmd_dequant(args):
    nvfp4_dir = Path(args.nvfp4_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all safetensors files
    st_files = sorted(nvfp4_dir.glob("*.safetensors"))
    print(f"Found {len(st_files)} safetensors files", flush=True)

    # Process shards in parallel
    t0 = time.time()
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sf in st_files:
            dst = output_dir / sf.name
            fut = pool.submit(process_shard, sf, dst)
            futures[fut] = sf.name

        for i, fut in enumerate(as_completed(futures), 1):
            name, nvfp4, fp8, pt = fut.result()
            elapsed = time.time() - t0
            print(
                f"[{i}/{len(st_files)}] {name}: "
                f"nvfp4={nvfp4}, fp8={fp8}, passthrough={pt}  "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nAll shards processed in {elapsed:.0f}s")

    # Build new index.json with scale keys removed
    index_path = nvfp4_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    new_weight_map = {}
    for key, filename in index["weight_map"].items():
        if not is_scale_key(key):
            new_weight_map[key] = filename

    # Recompute total_size from output files
    total_size = 0
    for sf in output_dir.glob("*.safetensors"):
        total_size += sf.stat().st_size

    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"Wrote model.safetensors.index.json ({len(new_weight_map)} keys)")

    # Copy config files
    for pattern in ["config.json", "generation_config.json", "tokenizer*", "special_tokens*"]:
        for src in nvfp4_dir.glob(pattern):
            dst = output_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"Copied {src.name}")

    # Patch tokenizer_config.json to remove fields that break HF loading
    tok_cfg_path = output_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with open(tok_cfg_path) as f:
            tok_cfg = json.load(f)
        changed = False
        for key in ["tokenizer_class", "is_local", "extra_special_tokens"]:
            if key in tok_cfg:
                del tok_cfg[key]
                changed = True
        if changed:
            with open(tok_cfg_path, "w") as f:
                json.dump(tok_cfg, f, indent=2)
            print("Patched tokenizer_config.json")

    print("\nDone! Output directory:", output_dir)


def cmd_generate(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = args.model_dir
    print(f"Loading model from {model_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    prompts = [
        "What is the capital of France?",
        "Write a short poem about the ocean.",
        "Explain quantum computing in one sentence.",
    ]

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

    print(f"\n{'='*60}")
    print("Generation test complete.")


def main():
    parser = argparse.ArgumentParser(description="Dequantize NVFP4/FP8 checkpoint to bf16")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dequant = sub.add_parser("dequant", help="Dequantize checkpoint")
    p_dequant.add_argument("--nvfp4_dir", required=True, help="Path to NVFP4 checkpoint")
    p_dequant.add_argument("--output_dir", required=True, help="Path to output bf16 checkpoint")
    p_dequant.add_argument("--workers", type=int, default=32, help="Number of parallel workers")

    p_gen = sub.add_parser("generate", help="Run generation test")
    p_gen.add_argument("--model_dir", required=True, help="Path to dequanted model")

    args = parser.parse_args()
    if args.command == "dequant":
        cmd_dequant(args)
    elif args.command == "generate":
        cmd_generate(args)


if __name__ == "__main__":
    main()
