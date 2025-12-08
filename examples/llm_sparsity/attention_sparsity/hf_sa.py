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

"""Example script for applying sparse attention to HuggingFace models."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.sparsity.attention_sparsity import SparseAttentionConfig
from modelopt.torch.sparsity.attention_sparsity.config import SKIP_SOFTMAX_DEFAULT
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule
from modelopt.torch.utils.memory_monitor import launch_memory_monitor

RAND_SEED = 1234

# Enable HuggingFace checkpointing support
mto.enable_huggingface_checkpointing()

# You can define custom configurations or use the default
SPARSE_ATTN_CFG_CHOICES = {
    "skip_softmax": SKIP_SOFTMAX_DEFAULT,
}


def get_narrativeqa_samples(num_samples=3):
    """Load samples from NarrativeQA dataset for testing.

    Args:
        num_samples: Number of samples to generate

    Raises:
        RuntimeError: If dataset loading fails
        ValueError: If no valid samples could be loaded
    """
    # Load NarrativeQA dataset with retry logic
    try:
        dataset = load_dataset("narrativeqa", split="test", streaming=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load NarrativeQA dataset: {e}")

    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break

        # Combine document context and question
        context = item.get("document", {}).get("text", "")
        question = item.get("question", {}).get("text", "")

        if context and question:
            # Use the full context as-is
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            samples.append(prompt)

    if not samples:
        raise ValueError("Could not load NarrativeQA samples")

    print(f"Loaded {len(samples)} NarrativeQA samples")
    return samples


def truncate_text(text: str, tokenizer, max_length: int):
    """Truncate text from the middle to preserve beginning and end.

    Args:
        text: Input text to truncate
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum number of tokens

    Returns:
        Truncated text that fits within max_length tokens
    """
    # First tokenize to see if truncation is needed
    tokens = tokenizer.encode(text, add_special_tokens=True)

    if len(tokens) <= max_length:
        return text

    # Need to truncate - preserve beginning and end
    # Calculate actual special tokens used
    dummy_tokens = tokenizer.encode("", add_special_tokens=True)
    special_token_count = len(dummy_tokens)
    available_tokens = max_length - special_token_count

    # Split tokens roughly in half for beginning and end
    begin_tokens = available_tokens // 2
    end_tokens = available_tokens - begin_tokens

    # Decode beginning and end parts
    begin_text = tokenizer.decode(tokens[:begin_tokens], skip_special_tokens=True)
    end_text = tokenizer.decode(tokens[-end_tokens:], skip_special_tokens=True)

    # Combine with ellipsis marker
    return begin_text + " [...] " + end_text


def verify_outputs(model, tokenizer, args):
    """Compare outputs between baseline and sparse attention models."""
    # Update seq_len to match calibration max_seqlen if calibration was used
    base_config = SPARSE_ATTN_CFG_CHOICES.get(args.sparse_attn, {})
    if "calibration" in base_config and "max_seqlen" in base_config["calibration"]:
        calib_max_seqlen = base_config["calibration"]["max_seqlen"]
        if args.seq_len != calib_max_seqlen:
            print(
                f"\nNote: Updating test seq_len from {args.seq_len} to {calib_max_seqlen} "
                f"to match calibration config"
            )
            args.seq_len = calib_max_seqlen

    # Load and prepare a single test prompt
    print(f"\nLoading test sample (will be tokenized up to {args.seq_len} tokens)")
    prompts = get_narrativeqa_samples(num_samples=1)
    prompt = prompts[0]

    # Prepare inputs
    truncated_prompt = truncate_text(prompt, tokenizer, args.seq_len)
    display_prompt = (
        truncated_prompt[:150] + "..." if len(truncated_prompt) > 150 else truncated_prompt
    )

    inputs = tokenizer(
        truncated_prompt,
        return_tensors="pt",
        max_length=args.seq_len,
        truncation=True,
        padding=False,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    print("\n" + "=" * 60)
    print("BASELINE vs SPARSE ATTENTION COMPARISON")
    print("=" * 60)
    print(f"\nTest prompt: {display_prompt}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    # Helper function to generate text
    def generate_text(model, inputs, args, tokenizer):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Find all sparse attention modules
    sparse_modules = [m for m in model.modules() if isinstance(m, SparseAttentionModule)]

    # Generate baseline by temporarily disabling sparse attention
    print("\n" + "-" * 60)
    print("Generating baseline (sparse attention disabled)...")
    for module in sparse_modules:
        module.disable()
    baseline_text = generate_text(model, inputs, args, tokenizer)

    # Generate with sparse attention enabled
    print("\nGenerating with sparse attention (calibrated thresholds)...")
    for module in sparse_modules:
        module.enable()
    sparse_text = generate_text(model, inputs, args, tokenizer)

    # Display comparison
    print("\n" + "-" * 60)
    print("RESULTS:")
    baseline_display = baseline_text[:300] + "..." if len(baseline_text) > 300 else baseline_text
    sparse_display = sparse_text[:300] + "..." if len(sparse_text) > 300 else sparse_text

    print(f"\nBaseline:    {baseline_display}")
    print(f"With Sparse: {sparse_display}")

    if baseline_text == sparse_text:
        print("\nOutputs are identical")
    else:
        print("\nOutputs differ")


def sparsify_model(model, args):
    """Apply sparse attention to the model with optional calibration."""
    print(f"\nApplying sparse attention: {args.sparse_attn} with backend: {args.backend}")
    base_config = SPARSE_ATTN_CFG_CHOICES[args.sparse_attn]

    # Create modified config with selected backend
    modified_sparse_cfg = {}
    for pattern, cfg in base_config["sparse_cfg"].items():
        modified_cfg = cfg.copy()
        modified_cfg["backend"] = args.backend
        modified_sparse_cfg[pattern] = modified_cfg

    # Create new config with modified settings
    sparse_config = SparseAttentionConfig(sparse_cfg=modified_sparse_cfg)

    # Sparsify the model
    model = mtsa.sparsify(model, config=sparse_config)

    print("Sparse attention applied successfully!")

    return model


def main(args):
    """Main function to run the selected mode."""
    if not torch.cuda.is_available():
        raise OSError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    launch_memory_monitor()

    print(f"Loading model: {args.pyt_ckpt_path}")

    # Load model and tokenizer
    # Note: attn_implementation="eager" is required for calibration to work properly
    # (flash_attention_2 or sdpa would bypass the softmax patching needed for stats collection)
    model = AutoModelForCausalLM.from_pretrained(
        args.pyt_ckpt_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pyt_ckpt_path)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")

    # Apply sparse attention to the model (with calibration if configured)
    model = sparsify_model(model, args)

    # Verify outputs if requested (compares baseline vs calibrated sparse model)
    if args.verify_output:
        verify_outputs(model, tokenizer, args)

    # Export if requested
    if args.export_dir:
        print(f"\nExporting model to: {args.export_dir}")
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=export_dir)

        tokenizer.save_pretrained(export_dir)
        print(f"Model exported successfully to: {export_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # Model arguments
    parser.add_argument(
        "--pyt_ckpt_path",
        type=str,
        required=True,
        help="Specify where the PyTorch checkpoint path is",
    )
    parser.add_argument(
        "--sparse_attn",
        type=str,
        default="skip_softmax",
        choices=list(SPARSE_ATTN_CFG_CHOICES.keys()),
        help="Sparse attention configuration to apply.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch"],
        help="Backend for sparse attention (default: pytorch). More backends coming soon.",
    )

    # Sequence length arguments
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length for input prompts (will be truncated if longer)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to use from NarrativeQA dataset",
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Maximum new tokens to generate"
    )
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")

    # Operation arguments
    parser.add_argument(
        "--verify_output",
        action="store_true",
        help="Verify that sparse attention outputs match baseline",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Directory to export the model with sparse attention applied",
    )

    args = parser.parse_args()
    main(args)
