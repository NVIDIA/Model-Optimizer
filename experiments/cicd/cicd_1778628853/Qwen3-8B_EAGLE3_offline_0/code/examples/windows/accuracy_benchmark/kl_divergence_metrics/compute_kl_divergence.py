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

"""
KL divergence computation between two models.

This script supports two comparison modes:
1. HF model vs GenAI model
2. GenAI model vs GenAI model (same execution provider)


Usage:
    # HF vs GenAI comparison
    python compute_kl_divergence_hf_vs_genai_sequential.py \
        --model1 "path/to/hf/model" --model1_type hf \
        --model2 "path/to/genai/model" --model2_type genai \
        --device cuda

    # GenAI vs GenAI comparison
    python compute_kl_divergence_hf_vs_genai_sequential.py \
        --model1 "path/to/genai/model1" --model1_type genai \
        --model2 "path/to/genai/model2" --model2_type genai

Note:
    The GenAI models automatically use the appropriate execution provider
    based on the installed onnxruntime-genai package.

Requirements:
    - Only requires VRAM for one model at a time (~8GB for 8B models)
    - Sufficient system RAM to store logits (~2-4GB)
"""

import argparse
import gc
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Lazy import for onnxruntime_genai
og: Any = None

DEBUG = False


def debug_print(message):
    """Print debug message only if DEBUG flag is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")


def cleanup_vram():
    """Aggressively clean up VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        debug_print("VRAM cleaned")
        try:
            debug_print(
                f"[INFO] GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            debug_print(f"[INFO] GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        except Exception:
            pass


def get_wikitext2():
    """
    Load and concatenate the WikiText-2 test dataset.

    Returns:
        str: Concatenated text from all samples, separated by double newlines.
    """
    print("\n[INFO] Loading Wikitext-2 'test' split...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    debug_print(f"Number of raw samples: {len(test)}")

    result = "\n\n".join(text for text in test["text"])
    debug_print(f"[INFO] Dataset loaded ({len(result):,} characters)")
    return result


def compute_kl_divergence(log_probs_ref, log_probs_tar):
    """
    Compute Kullback-Leibler divergence between two log probability distributions.

    Args:
        log_probs_ref (np.ndarray): Reference log probabilities with shape (seq_len, vocab_size).
        log_probs_tar (np.ndarray): Target log probabilities with shape (seq_len, vocab_size).

    Returns:
        float: Average KL divergence across all positions.
    """
    kl_divergence = 0.0
    for i in range(log_probs_ref.shape[0]):
        log_probs_ref_i = np.array(log_probs_ref[i])
        log_probs_tar_i = np.array(log_probs_tar[i])
        prob_ref_i = np.exp(log_probs_ref_i)
        kl_divergence += np.sum(prob_ref_i * np.abs(log_probs_ref_i - log_probs_tar_i))
    kl_divergence = kl_divergence / log_probs_ref.shape[0]
    return kl_divergence


def extract_genai_logits(model_path, dataset, max_context_length=4096):
    """
    Extract logits from GenAI model and store in CPU memory.

    Args:
        model_path (str): Path to ONNX Runtime GenAI model directory.
        dataset (str): Text dataset to process.
        max_context_length (int): Maximum context length for chunks.

    Returns:
        dict: Dictionary containing logits list, chunk info, and metadata.
    """
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING MODEL 1 LOGITS (GenAI)")
    print("=" * 80)
    print(f"[INFO] Loading ONNX Runtime GenAI model from: {model_path}")

    # Import onnxruntime_genai
    global og
    try:
        import onnxruntime_genai as og_module

        og = og_module
        debug_print("[INFO] Successfully imported onnxruntime_genai")
    except ImportError as e:
        raise ImportError(
            f"Failed to import onnxruntime_genai: {e}. "
            f"Make sure the correct onnxruntime-genai package is installed"
        )

    assert og is not None, "onnxruntime_genai module not loaded"

    # Load GenAI model
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    print("[INFO] GenAI model loaded successfully")

    if torch.cuda.is_available():
        try:
            debug_print(
                f"[INFO] GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            debug_print(f"[INFO] GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        except Exception:
            pass

    # Tokenize
    print("[INFO] Tokenizing dataset...")
    input_ids = tokenizer.encode_batch([dataset])

    if isinstance(input_ids, dict) and "input_ids" in input_ids:
        input_ids = input_ids["input_ids"]
    if hasattr(input_ids, "as_numpy"):
        input_ids = input_ids.as_numpy()

    input_ids = np.array(input_ids)

    if input_ids.ndim == 1:
        input_ids = np.expand_dims(input_ids, 0)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    seq_len = int(input_ids.shape[1])
    debug_print(f"[INFO] Input sequence length: {seq_len}")

    # Store all logits in CPU memory
    all_logits = []
    chunk_info = []

    print("[INFO] Extracting logits ...")
    for chunk_count, begin_loc in enumerate(range(0, seq_len, max_context_length), start=1):
        end_loc = min(begin_loc + max_context_length, seq_len)

        # Extract chunk
        input_ids_chunk = input_ids[:, begin_loc:end_loc].clone()
        debug_print(f"Chunk shape: {input_ids_chunk.shape}")

        # Run GenAI model
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=int(input_ids_chunk.shape[1]), do_sample=False, early_stopping=False
        )

        generator = og.Generator(model, params)
        generator.append_tokens(input_ids_chunk.numpy())

        with torch.no_grad():
            try:
                generator.generate_next_token()
                logits = generator.get_output("logits")

                if hasattr(logits, "as_numpy"):
                    logits = logits.as_numpy()

                logits = torch.tensor(logits, dtype=torch.float32)
                debug_print(f"Logits shape: {logits.shape}")

                # Store logits in CPU memory (important!)
                logits_cpu = logits.cpu().numpy()
                all_logits.append(logits_cpu)

                chunk_info.append(
                    {
                        "chunk_id": chunk_count,
                        "begin_loc": begin_loc,
                        "end_loc": end_loc,
                        "shape": logits_cpu.shape,
                    }
                )

                # Clean up chunk tensors immediately
                del generator, params, logits, input_ids_chunk

            except Exception as e:
                print(f"[ERROR] GenAI model forward pass failed: {e}")
                break

    debug_print(f"[INFO] Extracted {len(all_logits)} chunks from GenAI model")

    genai_data = {
        "logits": all_logits,
        "chunk_info": chunk_info,
        "model_path": model_path,
        "model_type": "genai",
        "seq_len": seq_len,
        "max_context_length": max_context_length,
        "total_chunks": len(all_logits),
        "input_ids": input_ids.cpu(),  # Move to CPU to free GPU memory
    }

    # Cleanup: Delete model and free VRAM
    debug_print("\n[INFO] Cleaning up GenAI model from VRAM...")
    del model
    del tokenizer
    del input_ids

    cleanup_vram()

    debug_print("[INFO] GenAI model cleaned from VRAM")
    debug_print("=" * 80)

    return genai_data


def extract_hf_logits(hf_model_path, dataset, device="cuda", max_context_length=4096):
    """
    Extract logits from Hugging Face model and store in CPU memory.

    Args:
        hf_model_path (str): Path to Hugging Face model directory or model name.
        dataset (str): Text dataset to process.
        device (str): Device for inference ('cuda' or 'cpu').
        max_context_length (int): Maximum context length for chunks.

    Returns:
        dict: Dictionary containing logits list, chunk info, and metadata.
    """
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING MODEL 1 LOGITS (HF)")
    print("=" * 80)
    print(f"[INFO] Loading Hugging Face model from: {hf_model_path}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None,  # Don't use device_map="auto" for proper cleanup
    )

    if device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    model.eval()

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(getattr(model, "config", None), "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("[INFO] Hugging Face model loaded successfully")

    if device == "cuda":
        try:
            debug_print(
                f"[INFO] GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            debug_print(f"[INFO] GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        except Exception:
            pass

    # Tokenize
    print("[INFO] Tokenizing dataset...")
    inputs = tokenizer(dataset, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])
    debug_print(f"[INFO] Input sequence length: {seq_len}")

    # Store all logits in CPU memory
    all_logits = []
    chunk_info = []

    print("[INFO] Extracting logits ...")
    for chunk_count, begin_loc in enumerate(range(0, seq_len, max_context_length), 1):
        end_loc = min(begin_loc + max_context_length, seq_len)

        # Extract chunk
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        debug_print(f"Chunk shape: {input_ids_chunk.shape}")

        # Move to device
        input_ids_chunk = input_ids_chunk.to(model.device)

        # Run model
        with torch.no_grad():
            try:
                outputs = model(input_ids_chunk)
                logits = outputs.logits
                debug_print(f"Logits shape: {logits.shape}")

                # Store logits in CPU memory (important!)
                logits_cpu = logits.cpu().numpy()
                all_logits.append(logits_cpu)

                chunk_info.append(
                    {
                        "chunk_id": chunk_count,
                        "begin_loc": begin_loc,
                        "end_loc": end_loc,
                        "shape": logits_cpu.shape,
                    }
                )

                # Clean up chunk tensors immediately
                del outputs, logits, input_ids_chunk

            except Exception as e:
                print(f"[ERROR] HF model forward pass failed: {e}")
                break

    debug_print(f"[INFO] Extracted {len(all_logits)} chunks from HF model")

    hf_data = {
        "logits": all_logits,
        "chunk_info": chunk_info,
        "model_path": hf_model_path,
        "model_type": "hf",
        "seq_len": seq_len,
        "max_context_length": max_context_length,
        "total_chunks": len(all_logits),
        "input_ids": input_ids.cpu(),  # Move to CPU to free GPU memory
    }

    # Cleanup: Delete model and free VRAM
    debug_print("\n[INFO] Cleaning up HF model from VRAM...")

    # Move model to CPU first to free GPU memory
    if device == "cuda":
        model = model.to("cpu")

    # Delete all references
    del model
    del tokenizer
    del input_ids
    del inputs

    # Aggressive cleanup
    cleanup_vram()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    debug_print("[INFO] HF model cleaned from VRAM")
    if torch.cuda.is_available():
        try:
            debug_print(
                f"[INFO] GPU Memory allocated after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            debug_print(
                f"[INFO] GPU Memory reserved after cleanup: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )
        except Exception:
            pass
    debug_print("=" * 80)

    return hf_data


def compute_kl_with_model2(model_path, model1_data, dataset, model2_type, device="cuda"):
    """
    Load second model and compute KL divergence against stored model1 logits.

    Args:
        model_path (str): Path to second model directory.
        model1_data (dict): Dictionary containing model1 logits and metadata.
        dataset (str): Text dataset to process.
        model2_type (str): Type of model2 ('hf' or 'genai').
        device (str): Device for HF model inference if model2 is HF.

    Returns:
        dict: Results containing KL divergence metrics.
    """
    print("\n" + "=" * 80)
    print(f"STEP 2: LOADING MODEL 2 ({model2_type.upper()}) AND COMPUTING KL DIVERGENCE")
    print("=" * 80)
    print(f"[INFO] Loading model from: {model_path}")

    # Load model based on type
    if model2_type == "genai":
        # Import onnxruntime_genai
        global og
        try:
            import onnxruntime_genai as og_module

            og = og_module
            debug_print("[INFO] Successfully imported onnxruntime_genai")
        except ImportError as e:
            raise ImportError(
                f"Failed to import onnxruntime_genai: {e}. "
                f"Make sure the correct onnxruntime-genai package is installed"
            )

        assert og is not None, "onnxruntime_genai module not loaded"

        # Load GenAI model
        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
        print("[INFO] GenAI model loaded successfully")

        if torch.cuda.is_available():
            try:
                debug_print(
                    f"[INFO] GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                )
                debug_print(
                    f"[INFO] GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
                )
            except Exception:
                pass

        # Tokenize with GenAI tokenizer
        print("\n[INFO] Tokenizing dataset...")
        input_ids = tokenizer.encode_batch([dataset])

        if isinstance(input_ids, dict) and "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]
        if hasattr(input_ids, "as_numpy"):
            input_ids = input_ids.as_numpy()

        input_ids = np.array(input_ids)

        if input_ids.ndim == 1:
            input_ids = np.expand_dims(input_ids, 0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)

    else:  # model2_type == "hf"
        # Load HF model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=None,  # Don't use device_map="auto" for proper cleanup
        )

        if device == "cuda":
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        model.eval()

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(getattr(model, "config", None), "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        print("[INFO] HF model loaded successfully")

        if device == "cuda" and torch.cuda.is_available():
            try:
                debug_print(
                    f"[INFO] GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                )
                debug_print(
                    f"[INFO] GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
                )
            except Exception:
                pass

        # Tokenize with HF tokenizer
        print("\n[INFO] Tokenizing dataset...")
        inputs = tokenizer(dataset, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"]

    seq_len = int(input_ids.shape[1])
    debug_print(f"[INFO] Input sequence length: {seq_len}")

    # Get config from model1 data
    max_context_length = model1_data["max_context_length"]
    total_chunks = model1_data["total_chunks"]

    debug_print(f"\n[INFO] Computing KL divergence for {total_chunks} chunks...")
    debug_print("=" * 80)

    # Process chunks and compute KL divergence
    total_kl_divergence = 0.0
    chunk_results = []
    chunk_count = 0

    for begin_loc in range(0, seq_len, max_context_length):
        chunk_count += 1

        if chunk_count > total_chunks:
            break

        end_loc = min(begin_loc + max_context_length, seq_len)

        # Extract chunk
        input_ids_chunk = input_ids[:, begin_loc:end_loc].clone()
        debug_print(f"Chunk shape: {input_ids_chunk.shape}")

        # Run model2 based on type
        with torch.no_grad():
            try:
                if model2_type == "genai":
                    # Run GenAI model
                    params = og.GeneratorParams(model)
                    params.set_search_options(
                        max_length=int(input_ids_chunk.shape[1]),
                        do_sample=False,
                        early_stopping=False,
                    )

                    generator = og.Generator(model, params)
                    generator.append_tokens(input_ids_chunk.numpy())
                    generator.generate_next_token()

                    model2_logits = generator.get_output("logits")
                    if hasattr(model2_logits, "as_numpy"):
                        model2_logits = model2_logits.as_numpy()
                    model2_logits = torch.tensor(model2_logits, dtype=torch.float32)

                else:  # model2_type == "hf"
                    # Run HF model
                    input_ids_chunk_device = input_ids_chunk.to(model.device)
                    outputs = model(input_ids_chunk_device)
                    model2_logits = outputs.logits

                debug_print(f"Model2 logits shape: {model2_logits.shape}")

            except Exception as e:
                print(f"[ERROR] Model2 forward pass failed: {e}")
                break

        # Get corresponding model1 logits from stored data
        model1_logits = torch.tensor(model1_data["logits"][chunk_count - 1], dtype=torch.float32)
        debug_print(f"Model1 logits shape: {model1_logits.shape}")

        # Ensure logits are compatible (trim to same dimensions)
        min_seq = min(model1_logits.shape[1], model2_logits.shape[1])
        min_vocab = min(model1_logits.shape[2], model2_logits.shape[2])

        model1_logits_trimmed = model1_logits[:, :min_seq, :min_vocab]
        model2_logits_trimmed = model2_logits[:, :min_seq, :min_vocab]

        debug_print(f"Trimmed Model1 logits: {model1_logits_trimmed.shape}")
        debug_print(f"Trimmed Model2 logits: {model2_logits_trimmed.shape}")

        # Compute log probabilities
        model1_log_probs = (
            torch.nn.functional.log_softmax(model1_logits_trimmed, dim=2).cpu().numpy()
        )
        model2_log_probs = (
            torch.nn.functional.log_softmax(model2_logits_trimmed, dim=2).cpu().numpy()
        )

        # Squeeze batch dimension
        model1_log_probs = np.squeeze(model1_log_probs, axis=0)
        model2_log_probs = np.squeeze(model2_log_probs, axis=0)

        debug_print(f"Model1 log probs shape: {model1_log_probs.shape}")
        debug_print(f"Model2 log probs shape: {model2_log_probs.shape}")

        # Compute KL divergence for this chunk
        chunk_kl = compute_kl_divergence(model1_log_probs, model2_log_probs)
        total_kl_divergence += chunk_kl

        debug_print(f"[RESULT] Chunk {chunk_count} KL divergence: {chunk_kl:.6f}")

        chunk_results.append(
            {
                "chunk_id": chunk_count,
                "begin_loc": int(model1_data["chunk_info"][chunk_count - 1]["begin_loc"]),
                "end_loc": int(model1_data["chunk_info"][chunk_count - 1]["end_loc"]),
                "kl_divergence": float(chunk_kl),
            }
        )

        # Cleanup chunk data immediately
        del model2_logits, model1_logits, model1_logits_trimmed, model2_logits_trimmed
        del model1_log_probs, model2_log_probs, input_ids_chunk
        if model2_type == "genai":
            del generator, params
        else:
            del outputs, input_ids_chunk_device

    avg_kl_divergence = total_kl_divergence / chunk_count

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    debug_print(f"Total chunks processed: {chunk_count}")
    debug_print(f"Total KL divergence: {total_kl_divergence:.6f}")
    print(f"Average KL divergence: {avg_kl_divergence:.6f}")
    print("=" * 80)

    # Cleanup model2
    debug_print(f"\n[INFO] Cleaning up model2 ({model2_type}) from VRAM...")

    # For HF models, move to CPU first
    if model2_type == "hf" and device == "cuda":
        model = model.to("cpu")

    del model
    del tokenizer
    del input_ids

    # Aggressive cleanup
    cleanup_vram()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    debug_print(f"[INFO] Model2 ({model2_type}) cleaned from VRAM")
    if torch.cuda.is_available():
        try:
            debug_print(
                f"[INFO] GPU Memory allocated after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            debug_print(
                f"[INFO] GPU Memory reserved after cleanup: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )
        except Exception:
            pass

    results = {
        "total_kl_divergence": float(total_kl_divergence),
        "average_kl_divergence": float(avg_kl_divergence),
        "total_chunks": int(chunk_count),
        "chunk_results": chunk_results,
    }

    return results


def main():
    """
    Command-line entry point for sequential KL divergence comparison.
    """
    parser = argparse.ArgumentParser(
        description="Memory-efficient sequential KL divergence comparison between HF and GenAI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare HF vs GenAI model
  python compute_kl_divergence_hf_vs_genai_sequential.py \\
      --model1 "meta-llama/Llama-3.1-8B-Instruct" --model1_type hf \\
      --model2 "G:\\models\\genai_model" --model2_type genai \\
      --device cuda \\
      --output results.json

  # Compare two GenAI models (same EP)
  python compute_kl_divergence_hf_vs_genai_sequential.py \\
      --model1 "G:\\models\\genai_fp16" --model1_type genai \\
      --model2 "G:\\models\\genai_int4" --model2_type genai \\
      --output results.json

  # Compare GenAI vs HF model
  python compute_kl_divergence_hf_vs_genai_sequential.py \\
      --model1 "G:\\models\\genai_model" --model1_type genai \\
      --model2 "F:\\shared\\Llama-3.1-8B-Instruct" --model2_type hf \\
      --device cuda \\
      --output results.json

Note:
  - GenAI models automatically use the appropriate execution provider
    based on the installed onnxruntime-genai package (cuda, directml, cpu, tensorrt)
  - The --device flag only controls HF model inference (applies to any HF model)
  - For GenAI vs GenAI comparison, both models must use the same execution provider

Advantages:
  - Only loads one model at a time (minimal VRAM usage)
  - No disk I/O overhead (stores logits in RAM)
  - Single process execution
  - Real-time KL divergence computation

VRAM Requirements:
  - Only requires VRAM for one model at a time (~8GB for 8B models)
  - Requires system RAM to store logits (~2-4GB)
        """,
    )

    parser.add_argument("--model1", required=True, help="Path to first model directory")
    parser.add_argument(
        "--model1_type",
        required=True,
        choices=["hf", "genai"],
        help="Type of first model (hf or genai)",
    )
    parser.add_argument("--model2", required=True, help="Path to second model directory")
    parser.add_argument(
        "--model2_type",
        required=True,
        choices=["hf", "genai"],
        help="Type of second model (hf or genai)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for HF model inference (default: cuda)",
    )
    parser.add_argument("--output", required=False, help="Output JSON file for results (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")

    args = parser.parse_args()

    # Set global debug flag
    global DEBUG
    DEBUG = args.debug

    # Validate model paths (only check local paths, HF model identifiers will be downloaded)
    if args.model1_type == "genai" and not os.path.exists(args.model1):
        print(f"[ERROR] Model1 path does not exist: {args.model1}")
        return 1
    elif (
        args.model1_type == "hf" and os.path.sep in args.model1 and not os.path.exists(args.model1)
    ):
        # Only validate if it looks like a local path (contains path separators)
        print(f"[ERROR] Model1 path does not exist: {args.model1}")
        return 1

    if args.model2_type == "genai" and not os.path.exists(args.model2):
        print(f"[ERROR] Model2 path does not exist: {args.model2}")
        return 1
    elif (
        args.model2_type == "hf" and os.path.sep in args.model2 and not os.path.exists(args.model2)
    ):
        # Only validate if it looks like a local path (contains path separators)
        print(f"[ERROR] Model2 path does not exist: {args.model2}")
        return 1

    # Validate GenAI vs GenAI comparison uses same EP
    if args.model1_type == "genai" and args.model2_type == "genai":
        print("[INFO] Comparing two GenAI models (same execution provider)")

    print("=" * 80)
    print("KL DIVERGENCE COMPUTATION BETWEEN TWO MODELS")
    print("=" * 80)
    print(f"Model 1: {args.model1} ({args.model1_type.upper()})")
    print(f"Model 2: {args.model2} ({args.model2_type.upper()})")
    if args.model1_type == "hf" or args.model2_type == "hf":
        print(f"HF Device: {args.device}")
    if args.output:
        print(f"Output: {args.output}")
    print("=" * 80)

    overall_start_time = time.time()

    try:
        # Load dataset once
        dataset = get_wikitext2()

        # Step 1: Extract Model1 logits and store in memory
        model1_start_time = time.time()
        if args.model1_type == "hf":
            model1_data = extract_hf_logits(args.model1, dataset, args.device)
        else:  # genai
            model1_data = extract_genai_logits(args.model1, dataset)
        model1_end_time = time.time()
        print(
            f"\n[TIMING] Model1 extraction time: {model1_end_time - model1_start_time:.2f} seconds"
        )

        # Step 2: Load Model2 and compute KL divergence
        model2_start_time = time.time()
        kl_results = compute_kl_with_model2(
            args.model2, model1_data, dataset, args.model2_type, args.device
        )
        model2_end_time = time.time()
        debug_print(
            f"\n[TIMING] Model2 computation time: {model2_end_time - model2_start_time:.2f} seconds"
        )

        overall_end_time = time.time()

        # Prepare final results
        final_results = {
            "models": {
                "model1": {"path": str(args.model1), "type": args.model1_type},
                "model2": {"path": str(args.model2), "type": args.model2_type},
            },
            "device": args.device
            if (args.model1_type == "hf" or args.model2_type == "hf")
            else "N/A",
            "total_chunks": kl_results["total_chunks"],
            "max_context_length": model1_data["max_context_length"],
            "kl_divergence": {
                "total": kl_results["total_kl_divergence"],
                "average": kl_results["average_kl_divergence"],
            },
            "chunk_results": kl_results["chunk_results"],
            "timing": {
                "model1_extraction_seconds": float(model1_end_time - model1_start_time),
                "model2_computation_seconds": float(model2_end_time - model2_start_time),
                "total_seconds": float(overall_end_time - overall_start_time),
            },
            "computation_timestamp": datetime.now().isoformat(),
        }

        # Save results if output file specified
        if args.output:
            print(f"\n[INFO] Saving results to: {args.output}")
        with open(args.output, "w") as f:
            json.dump(final_results, f, indent=2)
            print("[INFO] Results saved successfully")

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
        print(
            f"  - Model1 ({args.model1_type}) extraction: {model1_end_time - model1_start_time:.2f} seconds"
        )
        print(
            f"  - Model2 ({args.model2_type}) + KL computation: {model2_end_time - model2_start_time:.2f} seconds"
        )
        print(f"\nAverage KL divergence: {kl_results['average_kl_divergence']:.6f}")
        print("=" * 80)

        print("\n[SUCCESS] KL divergence computation completed!")
        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Computation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Computation failed: {e}")
        if DEBUG:
            import traceback

            traceback.print_exc()
        return 1
    finally:
        # Final cleanup
        cleanup_vram()


if __name__ == "__main__":
    import sys

    sys.exit(main())
