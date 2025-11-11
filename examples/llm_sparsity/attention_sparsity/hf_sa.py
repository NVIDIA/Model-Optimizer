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
from modelopt.torch.sparsity.attention_sparsity.config import (
    SKIP_SOFTMAX_CALIB,
    SKIP_SOFTMAX_DEFAULT,
)
from modelopt.torch.utils.memory_monitor import launch_memory_monitor

RAND_SEED = 1234

# Enable HuggingFace checkpointing support
mto.enable_huggingface_checkpointing()

# Sparse attention configuration choices
SPARSE_ATTN_CFG_CHOICES = {
    "skip_softmax": SKIP_SOFTMAX_DEFAULT,
    "skip_softmax_calib": SKIP_SOFTMAX_CALIB,
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


def generate_sample_output(model, tokenizer, args):
    """Generate sample output for comparison.

    Args:
        model: The model to generate with
        tokenizer: Tokenizer for encoding/decoding
        args: Command line arguments

    Returns:
        Tuple of (generated_text, input_prompt, input_ids)
    """
    # Load test sample
    prompts = get_narrativeqa_samples(num_samples=1)
    prompt = prompts[0]

    # Prepare inputs
    truncated_prompt = truncate_text(prompt, tokenizer, args.seq_len)
    inputs = tokenizer(
        truncated_prompt,
        return_tensors="pt",
        max_length=args.seq_len,
        truncation=True,
        padding=False,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
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
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, truncated_prompt, inputs["input_ids"]


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

    # Generate sample output BEFORE sparse attention
    print("\nGenerating sample output before sparse attention...")
    output_before, test_prompt, input_ids = generate_sample_output(model, tokenizer, args)

    # Apply sparse attention with optional calibration
    print(f"\nApplying sparse attention: {args.sparse_attn}")
    sparse_config = SPARSE_ATTN_CFG_CHOICES[args.sparse_attn]
    model = mtsa.sparsify(model, config=sparse_config)
    print("Sparse attention applied successfully!")

    # Generate sample output AFTER sparse attention
    print("\nGenerating sample output after sparse attention...")
    output_after, _, _ = generate_sample_output(model, tokenizer, args)

    # Display comparison
    print("\n" + "=" * 60)
    print("OUTPUT COMPARISON (Before vs After Sparse Attention)")
    print("=" * 60)
    display_prompt = test_prompt[:150] + "..." if len(test_prompt) > 150 else test_prompt
    print(f"\nTest prompt: {display_prompt}")
    print(f"Input tokens: {input_ids.shape[1]}")

    output_before_display = (
        output_before[:300] + "..." if len(output_before) > 300 else output_before
    )
    output_after_display = output_after[:300] + "..." if len(output_after) > 300 else output_after

    print(f"\nBefore sparse attention: {output_before_display}")
    print(f"After sparse attention:  {output_after_display}")

    if output_before == output_after:
        print("\nOutputs are identical")
    else:
        print("\nOutputs differ")

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

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Maximum new tokens to generate"
    )
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")

    # Operation arguments
    parser.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Directory to export the model with sparse attention applied",
    )

    args = parser.parse_args()
    main(args)
