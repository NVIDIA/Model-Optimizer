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

# SPDX-License-Identifier: MIT
#
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# This file is based on perplexity_metrics.py from the ONNX Runtime GenAI project:
# https://github.com/microsoft/onnxruntime-genai/blob/main/tools/python/model_validation/perplexity_metrics.py
#
# Modifications Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modifications made:
# - Added support for multiple context lengths
# - Added configurable chunk sizes
# - Enhanced prefill chunking handling

import json
import time

import numpy as np
import onnxruntime_genai as og
import torch
from datasets import load_dataset

# Global debug flag - set to True for verbose output
DEBUG = False


def calculate_perplexity_hf(
    model_name_or_path, max_length=1024, stride=512, device="cuda", torch_dtype=None
):
    """
    Evaluate perplexity of a HuggingFace model on the WikiText-2 dataset.

    This function computes perplexity using a sliding window approach similar to the
    ONNX Runtime GenAI version, but using native HuggingFace transformers.

    Args:
        model_name_or_path (str): HuggingFace model name (e.g., 'meta-llama/Llama-2-7b-hf')
                                   or path to a local model directory.
        max_length (int, optional): Maximum input sequence length for evaluation.
                                    Defaults to 1024.
        stride (int, optional): Stride for sliding window evaluation.
                                Defaults to 512.
        device (str, optional): Device to run the model on ('cuda', 'cpu', etc.).
                                Defaults to 'cuda'.
        torch_dtype: PyTorch dtype for the model. If None, uses default (float32).
                     Common options: torch.float16, torch.bfloat16, torch.float32.

    Returns:
        float: Computed perplexity score. Lower values indicate better model performance.

    Raises:
        ImportError: If transformers package is not installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "The 'transformers' package is required for HuggingFace model evaluation. "
            "Install it with: pip install transformers"
        ) from e

    time_start = time.time()
    print(f"\n[RUN] === BEGIN calculate_perplexity_hf('{model_name_or_path}') ===")
    print(f"[RUN] Loading HuggingFace model from: {model_name_or_path}")

    # Load tokenizer
    print("[TOKENIZER] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"[MODEL] Loading model on device: {device}")
    model_kwargs = {"device_map": device}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
        print(f"[MODEL] Using dtype: {torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()

    # Load and prepare the evaluation dataset
    dataset = get_wikitext2()
    print("[TOKENIZER] Tokenizing ...")

    # Tokenize the entire dataset
    encodings = tokenizer(dataset, return_tensors="pt", add_special_tokens=True)
    input_ids = encodings.input_ids

    if DEBUG:
        print(f"[TOKENIZER] Input shape: {input_ids.shape}, dtype: {input_ids.dtype}")

    seq_len = input_ids.size(1)
    print(f"[INFO] Full input length: {seq_len}")
    print(f"[INFO] max_length: {max_length}, stride: {stride}")

    max_eval_length = seq_len

    # Initialize accumulators for log probabilities
    total_log_probs = 0.0
    total_token_count = 0
    prev_end_loc = 0

    # Slide a window over the input to compute perplexity in chunks
    for chunk_idx, begin_loc in enumerate(range(0, max_eval_length, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        if DEBUG:
            print(
                f"\n[LOOP] chunk_idx={chunk_idx} [begin={begin_loc} end={end_loc}] trg_len={trg_len}"
            )

        # Extract the current chunk of input tokens (keep on CPU until needed)
        input_ids_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids_chunk.clone()

        # Mask context tokens: only predict for last trg_len tokens in chunk
        mask = np.ones(target_ids.shape, dtype=bool)
        mask[:, :-trg_len] = False
        target_ids_masked = target_ids.clone()
        target_ids_masked[~torch.from_numpy(mask)] = -100  # -100 is the ignore index

        if DEBUG:
            print(f"[MASK] Mask shape: {mask.shape}")
            print(f"[TARGET_IDS_MASKED] Target ids masked: {target_ids_masked}")

        # Run the model forward pass without gradient calculation
        with torch.no_grad():
            if DEBUG:
                print("[INFER] Running model forward pass ...")

            outputs = model(input_ids_chunk)
            logits = outputs.logits

            if DEBUG:
                print(f"[LOGITS] Shape: {logits.shape}, dtype: {logits.dtype}")

        # Compute log probabilities over vocabulary for each position
        log_probs = torch.nn.functional.log_softmax(logits, dim=2).cpu().numpy()
        chunk_seq_len = log_probs.shape[1]

        # Language models predict next token: logits[i] predicts token[i+1]
        # So we need logits[:-1] to match with target_ids[1:]
        if chunk_seq_len > 1:
            # Get log probabilities for all positions except the last
            pred_log_probs = log_probs[0, :-1, :]  # predictions for positions 0 to max_length-2
            # Get the target token ids for positions 1 to max_length-1
            target_ids_shifted = (
                target_ids_masked[0, 1:].cpu().numpy()
            )  # targets at positions 1 to max_length-1

            if DEBUG:
                print(f"[TARGET_IDS_SHIFTED] Target ids shifted shape: {target_ids_shifted.shape}")
                print(f"[PRED_LOG_PROBS] Pred log probs shape: {pred_log_probs.shape}")
                print(f"chunk_seq_len: {chunk_seq_len}")

            # Only include tokens with label != -100 (matching masking)
            mask_flat = target_ids_shifted != -100
            valid_indices = np.arange(len(target_ids_shifted))[mask_flat]
            valid_targets = target_ids_shifted[mask_flat]

            if DEBUG:
                print(f"[VALID_INDICES] Valid indices shape: {valid_indices.shape}")
                print(f"[VALID_TARGETS] Valid targets shape: {valid_targets.shape}")

            # Gather the log probabilities for the correct target tokens
            valid_log_probs = pred_log_probs[valid_indices, valid_targets]

            if DEBUG:
                print(f"[VALID_LOG_PROBS] Valid log probs shape: {valid_log_probs.shape}")
        else:
            valid_log_probs = np.array([])
            mask_flat = np.array([], dtype=bool)

        # Accumulate log probabilities and token count (same as ONNX)
        total_log_probs += float(np.sum(valid_log_probs))
        total_token_count += int(valid_log_probs.size)

        if DEBUG:
            print(
                f"[LOOP] This chunk: valid tokens={valid_log_probs.size}, sum={np.sum(valid_log_probs)}"
            )
            print(f"[TALLY] total_log_probs: {total_log_probs}")
            print(f"[TALLY] total_token_count: {total_token_count}")

        # Clear GPU cache to prevent OOM
        del (
            outputs,
            logits,
            log_probs,
            pred_log_probs,
            input_ids_chunk,
            target_ids,
            target_ids_masked,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

        # Update for next chunk
        prev_end_loc = end_loc
        if end_loc >= max_eval_length:
            if DEBUG:
                print("[LOOP] Reached evaluation limit.")
            break

    # Compute average log probability and perplexity (same as ONNX)
    avg_log_prob = total_log_probs / total_token_count
    perplexity = np.exp(-avg_log_prob)  # Note the negative sign!

    if DEBUG:
        print(f"[FINAL] avg_log_prob: {avg_log_prob}")

    print(f"\n[RESULT] Perplexity of {model_name_or_path}: {perplexity}")
    print("[RUN] === END calculate_perplexity_hf ===\n")
    time_end = time.time()
    print(f"[RUN] Time taken: {time_end - time_start:.2f} seconds")

    # Cleanup: Unload model from GPU memory
    print("[CLEANUP] Unloading model from GPU...")
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    print("[CLEANUP] Model unloaded")

    return perplexity


def get_wikitext2():
    """
    Load and concatenate the WikiText-2 test dataset.

    Returns:
        str: Concatenated text from all samples in the WikiText-2 test split,
             with samples separated by double newlines.

    Note:
        Requires HuggingFace CLI authentication to access the dataset.
    """
    # Load the Wikitext-2 test split using HuggingFace datasets
    print("\n[INFO] Loading Wikitext-2 'test' split ...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if DEBUG:
        print(f"[DATASET] Number of raw samples: {len(test)}")
        for i in range(3):
            print(f"[DATASET] Sample[{i}]: {repr(test[i]['text'])[:200]} ...")
    # Concatenate all text samples into a single string, separated by double newlines
    result = "\n\n".join(text for text in test["text"])
    if DEBUG:
        print(
            f"[DATASET] Concatenated text preview: {result[:512]!r} ... [total chars: {len(result)}]"
        )
    return result


def perplexity_eval(model_dir, input_len=1024, chunk_size=None):
    """
    Evaluate perplexity of an ONNX Runtime GenAI model on the WikiText-2 dataset.

    This function computes perplexity using a sliding window approach. It supports
    both standard evaluation and prefill chunking for longer context lengths.

    Args:
        model_dir (str): Path to the ONNX Runtime GenAI model directory.
                        Must contain genai_config.json and tokenizer files.
        input_len (int, optional): Maximum input sequence length for evaluation.
                                   Used as context length when KV chunking is enabled.
                                   Defaults to 1024.
        chunk_size (int, optional): Prefill chunk size for prefill chunking.
                                   If provided, overrides the chunk_size in genai_config.json.
                                   When set, enables evaluation with longer context lengths.
                                   Defaults to None.

    Returns:
        float: Computed perplexity score. Lower values indicate better model performance.
               Typical ranges: 2-20 (excellent), 20-40 (good), 40-80 (ok), 100+ (poor).

    """
    time_start = time.time()
    print(f"\n[RUN] === BEGIN perplexity_eval('{model_dir}') ===")
    print(f"[RUN] Loading ONNX model from: {model_dir}")
    chunking_failed = False
    # Load the ONNX model
    # Apply chunk_size overlay if provided
    config = og.Config(model_dir)
    if chunk_size is not None:
        search_config = {"chunk_size": int(chunk_size)}
        try:
            print(f"[CONFIG] Applying chunk_size overlay: {chunk_size}")
            config.overlay(json.dumps({"search": search_config}))
            print(f"[CONFIG] Successfully applied chunk_size: {chunk_size}")
        except Exception as e:
            print(f"[WARNING] Failed to apply chunk_size overlay: {e}")
            chunking_failed = True
    model = og.Model(config)

    if DEBUG:
        print("[RUN] Creating tokenizer ...")
    # Create the tokenizer for the model
    tokenizer = og.Tokenizer(model)
    # Load model configuration from JSON file (optional)
    model_cfg_json = None
    try:
        with open(f"{model_dir}/genai_config.json") as file:
            model_cfg_json = json.load(file)
        if DEBUG:
            print(
                f"[CONFIG] Model config loaded: {json.dumps(model_cfg_json.get('model', {}), indent=2)}"
            )
    except Exception as e:
        print(f"[WARNING] Could not read genai_config.json: {e}. Falling back to defaults.")

    max_context_length = 1024
    stride = 512
    kv_chunking_enabled = False

    # Check for chunk_size - prioritize parameter over config file
    effective_chunk_size = None
    if chunk_size is not None and not chunking_failed:
        # Use the provided chunk_size parameter (overlaid)
        effective_chunk_size = int(chunk_size)
        kv_chunking_enabled = True
        if DEBUG:
            print(f"[CONFIG] Using provided chunk_size: {effective_chunk_size}")
    elif model_cfg_json and "search" in model_cfg_json and "chunk_size" in model_cfg_json["search"]:
        # Use chunk_size from existing config file
        effective_chunk_size = model_cfg_json["search"]["chunk_size"]
        kv_chunking_enabled = True
        if DEBUG:
            print(f"[CONFIG] Using config file chunk_size: {effective_chunk_size}")

    if DEBUG:
        print(
            f"[CONFIG] Effective chunk_size: {effective_chunk_size if kv_chunking_enabled else 'disabled'}"
        )

    if kv_chunking_enabled and effective_chunk_size:
        if DEBUG:
            print(f"[INFO] chunk size: {effective_chunk_size}")
            print(f"[INFO] input length: {input_len}")
        max_context_length = int(input_len)  # Use input_len when chunking is enabled
        stride = effective_chunk_size
        if DEBUG:
            print(
                f"[CONFIG] KV chunking enabled with chunk_size: {effective_chunk_size}, input_len: {input_len}"
            )
    elif DEBUG:
        print(f"[CONFIG] KV chunking disabled, using default stride: {stride}")

    # Set chunk and stride lengths for evaluation
    model_context_len = (
        int(model_cfg_json["model"]["context_length"])
        if model_cfg_json
        and "model" in model_cfg_json
        and "context_length" in model_cfg_json["model"]
        else max_context_length
    )
    max_length = min(max_context_length, model_context_len)
    if DEBUG:
        print(f"[INFO] max_length for chunk: {max_length}, stride for sliding window: {stride}")

    # Load and prepare the evaluation dataset
    dataset = get_wikitext2()
    print("[TOKENIZER] Tokenizing ...")
    # Tokenize the entire dataset
    input_ids = tokenizer.encode_batch([dataset])
    # Handle possible dict output from tokenizer
    if isinstance(input_ids, dict) and "input_ids" in input_ids:
        input_ids = input_ids["input_ids"]
    # Convert to numpy if needed
    if hasattr(input_ids, "as_numpy"):
        input_ids = input_ids.as_numpy()
        if DEBUG:
            print("[TOKENIZER] Used as_numpy()")
    input_ids = np.array(input_ids)
    if DEBUG:
        print(f"[TOKENIZER] Numpy array shape: {input_ids.shape}, dtype: {input_ids.dtype}")
    # Ensure input_ids is 2D (batch, seq_len)
    if input_ids.ndim == 1:
        input_ids = np.expand_dims(input_ids, 0)
        if DEBUG:
            print(f"[SHAPE] Expanded dims, now: {input_ids.shape}")

    # Convert input_ids to torch tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    if DEBUG:
        print(f"[TENSOR] Torch tensor shape: {input_ids.shape}, dtype: {input_ids.dtype}")

    # Determine the sequence length to use
    seq_len = int(input_ids.shape[1])
    if DEBUG:
        print(f"[INFO] Full input length: {seq_len}")

    # Initialize accumulators for log probabilities and token count
    total_log_probs = 0.0
    total_token_count = 0
    prev_end_loc = 0
    if kv_chunking_enabled:
        assert stride == effective_chunk_size, (
            f"For chunking case, stride must equal chunk_size. "
            f"Got stride={stride}, chunk_size={effective_chunk_size}"
        )
    # Slide a window over the input to compute perplexity in chunks
    for chunk_idx, begin_loc in enumerate(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if DEBUG:
            print(
                f"\n[LOOP] chunk_idx={chunk_idx} [begin={begin_loc} end={end_loc}] trg_len={trg_len}"
            )

        # Extract the current chunk of input tokens
        input_ids_chunk = input_ids[:, begin_loc:end_loc].clone()
        target_ids = input_ids_chunk.clone()
        if DEBUG:
            print(f"input_ids_chunk.shape: {input_ids_chunk.shape}")
        # Mask context tokens: only predict for last trg_len tokens in chunk
        mask = np.ones(target_ids.shape, dtype=bool)
        mask[:, :-trg_len] = False
        target_ids_masked = target_ids.clone()
        target_ids_masked[~torch.from_numpy(mask)] = -100  # -100 is the ignore index
        if DEBUG:
            print(f"[MASK] Mask : {mask}")
            print(f"[TARGET_IDS_MASKED] Target ids masked : {target_ids_masked}")
        # Set up generator parameters for deterministic generation (no sampling)
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=int(input_ids_chunk.shape[1]), do_sample=False, early_stopping=False
        )
        # Create generator and append input tokens
        generator = og.Generator(model, params)
        generator.append_tokens(input_ids_chunk.numpy())

        # Run the model forward pass without gradient calculation
        with torch.no_grad():
            if DEBUG:
                print("[INFER] Running model forward pass ...")
            try:
                generator.generate_next_token()
            except Exception as e:
                print(f"[INFER] .generate_next_token() failed: {e}")
                break  # Fatal error
            # Get logits output from the model
            logits = generator.get_output("logits")
            if hasattr(logits, "as_numpy"):
                logits = logits.as_numpy()
                if DEBUG:
                    print("[LOGITS] Used as_numpy()")
            logits = torch.tensor(logits, dtype=torch.float32)
            if DEBUG:
                print(f"[LOGITS] Torch tensor shape: {logits.shape}, dtype: {logits.dtype}")

        # Compute log probabilities over vocabulary for each position
        log_probs = torch.nn.functional.log_softmax(logits, dim=2).cpu().numpy()
        chunk_seq_len = log_probs.shape[1]
        # Language models predict next token: logits[i] predicts token[i+1]
        # So we need logits[:-1] to match with target_ids[1:]
        if chunk_seq_len > 1:
            # Get log probabilities for all positions except the last
            pred_log_probs = log_probs[0, :-1, :]  # predictions for positions 0 to max_length-2
            # Get the target token ids for positions 1 to max_length-1
            target_ids_shifted = (
                target_ids_masked[0, 1:].cpu().numpy()
            )  # targets at positions 1 to max_length-1
            if DEBUG:
                print(f"[TARGET_IDS_SHIFTED] Target ids shifted shape: {target_ids_shifted.shape}")
                print(f"[PRED_LOG_PROBS] Pred log probs shape: {pred_log_probs.shape}")
                print(f"chunk_seq_len: {chunk_seq_len}")

            # Only include tokens with label != -100 (matching HF masking)
            mask_flat = target_ids_shifted != -100
            if kv_chunking_enabled:
                trg_len = min(trg_len, stride)
                mask_flat = np.ones(trg_len, dtype=bool)
                valid_indices = np.arange(0, trg_len - 1)
                valid_targets = target_ids_shifted[-trg_len + 1 :]
            else:
                valid_indices = np.arange(len(target_ids_shifted))[mask_flat]
                valid_targets = target_ids_shifted[mask_flat]
            if DEBUG:
                print(f"[VALID_INDICES] Valid indices shape: {valid_indices.shape}")
                print(f"[VALID_TARGETS] Valid targets shape: {valid_targets.shape}")
            # Gather the log probabilities for the correct target tokens
            valid_log_probs = pred_log_probs[valid_indices, valid_targets]
            if DEBUG:
                print(f"[VALID_LOG_PROBS] Valid log probs shape: {valid_log_probs.shape}")
        else:
            valid_log_probs = np.array([])
            mask_flat = np.array([], dtype=bool)

        # Accumulate log probabilities and token count
        total_log_probs += float(np.sum(valid_log_probs))
        total_token_count += int(valid_log_probs.size)

        if DEBUG:
            print(
                f"[LOOP] This chunk: valid tokens={valid_log_probs.size}, sum={np.sum(valid_log_probs)}"
            )
            print(f"[TALLY] total_log_probs: {total_log_probs}")
            print(f"[TALLY] total_token_count: {total_token_count}")

        # Update for next chunk
        prev_end_loc = end_loc
        if end_loc == seq_len:
            if DEBUG:
                print("[LOOP] Reached end of sequence.")
            break

    # Compute average log probability and perplexity
    avg_log_prob = total_log_probs / total_token_count
    perplexity = np.exp(-avg_log_prob)
    if DEBUG:
        print(f"[FINAL] avg_log_prob: {avg_log_prob}")
    print(f"\n[RESULT] Perplexity of {model_dir}: {perplexity}")
    print("[RUN] === END perplexity_eval ===\n")
    time_end = time.time()
    print(f"[RUN] Time taken: {time_end - time_start:.2f} seconds")
    return perplexity


# Example usage:
# perplexity_eval("/path/to/model_dir")
#
# To enable debug output, set DEBUG = True at the top of this file
