#!/usr/bin/env python3
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

"""Example: TriAttention KV cache sparsity calibration on HuggingFace models.

Demonstrates the TriAttention calibration pipeline:
1. Load a pretrained HF model
2. Apply KV cache sparsity mode (sparsify)
3. Run calibration with a forward pass to compute per-head frequency statistics
4. Verify calibration data was produced
5. Optionally save calibration data

Usage:
    python hf_triattention.py --model Qwen/Qwen3-0.6B

    # With custom budget and calibration length
    python hf_triattention.py --model Qwen/Qwen3-0.6B --budget 1024 --calib-seq-len 4096

    # Save calibration data
    python hf_triattention.py --model Qwen/Qwen3-0.6B --output calibration.pt
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.sparsity.kv_cache as mtskv
from modelopt.torch.sparsity.kv_cache.config import TriAttentionConfig


def make_calibration_forward_loop(tokenizer, seq_len: int = 2048, input_file: str | None = None):
    """Create a forward loop that runs calibration data through the model.

    Args:
        tokenizer: Tokenizer for the model.
        seq_len: Sequence length for calibration input.
        input_file: Path to a plain text file for calibration. If None, uses
            built-in placeholder text.

    Returns:
        Callable that takes a model and runs a forward pass.
    """
    if input_file is not None:
        from pathlib import Path

        calib_text = Path(input_file).read_text(encoding="utf-8")
    else:
        calib_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed. "
            "Deep learning, a branch of machine learning, uses neural networks with many "
            "layers to model complex patterns in data. Transformers have revolutionized "
            "natural language processing by introducing self-attention mechanisms that "
            "allow models to weigh the importance of different parts of the input. "
        ) * 100

    input_ids = tokenizer.encode(
        calib_text, return_tensors="pt", truncation=True, max_length=seq_len
    )
    print(f"  Calibration tokens: {input_ids.shape[1]}")

    def forward_loop(model):
        device = next(model.parameters()).device
        inputs = input_ids.to(device)
        with torch.no_grad():
            model(input_ids=inputs)

    return forward_loop


def main(args):
    """Run TriAttention calibration pipeline."""
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    # Print model info
    config = model.config
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Heads: {config.num_attention_heads}")
    print(f"  KV heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
    print(f"  Hidden size: {config.hidden_size}")

    # Step 1: Calibrate (before sparsify — apply_mode adds wrappers that alter forward pass)
    print(f"\nRunning calibration (seq_len={args.calib_seq_len})...")
    forward_loop = make_calibration_forward_loop(
        tokenizer, seq_len=args.calib_seq_len, input_file=args.input
    )

    t0 = time.time()
    model = mtskv.calibrate(model, forward_loop=forward_loop)
    elapsed = time.time() - t0
    print(f"  Calibration complete in {elapsed:.1f}s")

    # Step 2: Apply KV cache sparsity mode
    print(f"\nApplying TriAttention mode (budget={args.budget})...")
    triattention_config = TriAttentionConfig(
        budget=args.budget,
        prune_interval=args.prune_interval,
    )
    model = mtskv.sparsify(model, triattention_config)
    print("  Mode applied (no-op on weights).")

    # Step 3: Verify calibration data
    calib_data = getattr(model, "_triattention_calibration", None)
    if calib_data is None:
        print("\n  ERROR: No calibration data found on model!")
        return

    print("\n  Calibration results:")
    print(f"    Head dim: {calib_data.head_dim}")
    print(f"    RoPE style: {calib_data.rope_style}")
    print(f"    Num layers: {calib_data.num_layers}")
    print(f"    Num KV heads: {calib_data.num_kv_heads}")
    print(f"    Heads calibrated: {len(calib_data.head_stats)}")

    # Check concentration (Mean Resultant Length)
    total_heads = 0
    concentrated = 0
    for stats in calib_data.head_stats.values():
        abs_mean = torch.abs(stats.q_mean_complex)  # |E[q]|
        mean_abs = stats.q_abs_mean  # E[|q|]
        # R = |E[q]| / E[|q|] — concentration metric
        r_values = abs_mean / (mean_abs + 1e-8)
        r_mean = r_values.mean().item()
        total_heads += 1
        if r_mean > 0.9:
            concentrated += 1

    print(f"    Concentrated heads (R > 0.9): {concentrated}/{total_heads}")
    print(f"    Concentration ratio: {concentrated / total_heads:.1%}")

    # Step 4: Optionally save calibration data
    if args.output:
        state = calib_data.state_dict()
        torch.save(state, args.output)
        print(f"\n  Calibration data saved to: {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TriAttention KV cache sparsity calibration example."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2048,
        help="KV token budget (tokens to retain per head).",
    )
    parser.add_argument(
        "--prune-interval",
        type=int,
        default=128,
        help="Re-score and evict every N tokens.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Plain text file for calibration input. "
        "If not provided, uses built-in placeholder text. "
        "Use the same file as triattention/scripts/calibrate.py --input for comparison.",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=2048,
        help="Sequence length for calibration input.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save calibration data (.pt file).",
    )

    args = parser.parse_args()
    main(args)
