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

"""Calibration functions for sparse attention."""

import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from ..config import CalibrationConfig
from ..conversion import print_sparse_attention_summary
from ..sparse_attention import SparseAttentionModule
from .calibrator import DynamicThresholdCalibrator
from .dataset import RulerDatasetBuilder


def _extract_tokenizer_from_model(model: nn.Module) -> str:
    """Extract tokenizer name/path from model config.

    Args:
        model: Model to extract tokenizer from

    Returns:
        Tokenizer name or path

    Raises:
        ValueError: If tokenizer path cannot be determined from model
    """
    # Extract tokenizer path from model config
    tokenizer_path = getattr(getattr(model, "config", None), "_name_or_path", None)

    if not tokenizer_path:
        raise ValueError("Could not load tokenizer from model.")

    return tokenizer_path


def _extract_calibration_config(config: dict[str, Any]) -> CalibrationConfig | None:
    """Extract and validate calibration config from sparse_cfg.

    Args:
        config: Sparse attention configuration dict

    Returns:
        Validated CalibrationConfig instance, or None if calibration is not configured

    Raises:
        ValueError: If calibration config has invalid type or contains invalid values
    """
    sparse_cfg = config.get("sparse_cfg", {})

    # Calibration is optional
    if "calibration" not in sparse_cfg:
        return None

    calib_dict = sparse_cfg["calibration"]

    # Validate calibration is a dict
    if not isinstance(calib_dict, dict):
        raise ValueError(f"Calibration config must be a dict, got {type(calib_dict).__name__}. ")

    # Create and validate CalibrationConfig
    return CalibrationConfig(**calib_dict)


def _parse_target_sparse_ratio(
    target_sparse_ratio: dict[str, float],
) -> dict[str, float]:
    """Parse target_sparse_ratio dict.

    Args:
        target_sparse_ratio: Target sparsity ratio dict with 'prefill' and 'decode' keys

    Returns:
        Dict with 'prefill' and 'decode' keys
    """
    return target_sparse_ratio


def create_calibration_forward_loop(
    calibration_data: list[dict[str, Any]],
    tokenizer_name_or_path: str,
    batch_size: int = 1,
    chunk_size: int = 2048,
) -> Callable:
    """Create forward loop for calibration.

    Args:
        calibration_data: List of samples with 'input' and 'length' fields
        tokenizer_name_or_path: HuggingFace tokenizer path
        batch_size: Batch size (currently unused, always 1)
        chunk_size: Chunk size for chunked prefill to avoid OOM. Set to -1 to disable.

    Returns:
        Forward loop function that takes model as argument
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def forward_loop(model: nn.Module) -> None:
        device = next(model.parameters()).device

        for sample in calibration_data:
            inputs = tokenizer(
                sample["input"], return_tensors="pt", truncation=True, max_length=sample["length"]
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"].to(device)
            seq_len = input_ids.shape[1]

            with torch.no_grad():
                if chunk_size > 0 and seq_len > chunk_size:
                    # Chunked prefill to avoid OOM with long sequences
                    past_key_values = None
                    for start_idx in range(0, seq_len, chunk_size):
                        end_idx = min(start_idx + chunk_size, seq_len)
                        chunk_input_ids = input_ids[:, start_idx:end_idx]

                        outputs = model(
                            chunk_input_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = outputs.past_key_values

                    # Clean up KV cache
                    del past_key_values
                    torch.cuda.empty_cache()
                else:
                    # Full prefill without chunking
                    model(input_ids, use_cache=False)

    return forward_loop


def create_decode_calibration_forward_loop(
    calibration_data: list[dict[str, Any]],
    tokenizer_name_or_path: str,
    num_decode_tokens: int = 10,
) -> Callable:
    """Create forward loop for decode phase calibration.

    Uses flash attention for fast prefill, then switches to eager attention
    for decode token generation with softmax hook measurement.

    Args:
        calibration_data: List of samples with 'input' and 'length' fields
        tokenizer_name_or_path: HuggingFace tokenizer path
        num_decode_tokens: Number of decode tokens to generate per sample

    Returns:
        Forward loop function that takes model as argument
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def forward_loop(model: nn.Module) -> None:
        device = next(model.parameters()).device

        for sample in calibration_data:
            inputs = tokenizer(
                sample["input"], return_tensors="pt", truncation=True, max_length=sample["length"]
            )
            input_ids = inputs["input_ids"].to(device)

            # Save original attention implementation
            original_attn_impl = getattr(model.config, "_attn_implementation", "eager")

            with torch.no_grad():
                # Step 1: Fast prefill with flash attention (no measurement)
                model.config._attn_implementation = "flash_attention_2"
                outputs = model(input_ids, use_cache=True)
                past_key_values = outputs.past_key_values

                # Step 2: Switch to eager for decode (enables softmax hook)
                model.config._attn_implementation = "eager"

                # Step 3: Manual decode loop for explicit control over token generation
                # model.generate() method is not used here because it doesn't allow explicit control over KV cache
                # Get the last token's logits and sample next token
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

                for _ in range(num_decode_tokens):
                    outputs = model(
                        next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            # Restore original attention implementation
            model.config._attn_implementation = original_attn_impl

            # Clean up
            del past_key_values
            torch.cuda.empty_cache()

    return forward_loop


def calibrate_sparse_attention(
    model: nn.Module,
    config: dict[str, Any],
    forward_loop: Callable | None = None,
) -> dict[str, Any]:
    """Calibrate sparse attention parameters for optimal sparsity.

    Supports both prefill and decode phase calibration with per-phase target sparsity.

    Args:
        model: Model with sparse attention modules
        config: Sparse attention configuration dict
        forward_loop: Callable that forwards calibration data through model.
                     If None, auto-generates RULER dataset. Only used for prefill.

    Returns:
        Dictionary with calibration results for each phase
    """
    # Extract and validate calibration config
    calib_config = _extract_calibration_config(config)

    # Skip calibration if not configured
    if calib_config is None:
        return {}

    # Parse target_sparse_ratio into per-phase targets
    target_dict = _parse_target_sparse_ratio(calib_config.target_sparse_ratio)
    calibrate_prefill = target_dict.get("prefill", 0.0) > 0.0
    calibrate_decode = target_dict.get("decode", 0.0) > 0.0

    # Skip if both phases are disabled
    if not calibrate_prefill and not calibrate_decode:
        print("Both prefill and decode target sparsity are 0.0, skipping calibration")
        return {}

    # Get sparse attention modules
    sparse_modules = [
        (name, m) for name, m in model.named_modules() if isinstance(m, SparseAttentionModule)
    ]

    if not sparse_modules:
        print("No sparse attention modules found for calibration")
        return {}

    print(f"Calibrating {len(sparse_modules)} sparse attention modules together...")

    # Extract tokenizer and build calibration data if needed
    tokenizer = _extract_tokenizer_from_model(model)
    calibration_data = None

    if calibrate_prefill or calibrate_decode:
        builder = RulerDatasetBuilder(
            samples=calib_config.samples,
            max_seqlen=calib_config.max_seqlen,
            tokenizer_name_or_path=tokenizer,
            num_length_bins=calib_config.num_length_bins,
            max_length_filter=int(calib_config.max_seqlen * 1.5),
        )
        calibration_data = builder.build_calibration_dataset()
        print(f"Generated {len(calibration_data)} calibration samples")

    # Initialize results
    threshold_scale_factor: dict[str, float] = {}
    calibration_results: dict[str, Any] = {}

    # Run prefill calibration if enabled
    if calibrate_prefill:
        print("\n" + "=" * 60)
        print("PREFILL PHASE CALIBRATION")
        print("=" * 60)

        assert calibration_data is not None, "calibration_data must be built before prefill"
        prefill_forward_loop = forward_loop or create_calibration_forward_loop(
            calibration_data, tokenizer, chunk_size=calib_config.chunk_size
        )

        prefill_calibrator = DynamicThresholdCalibrator(
            target_sparse_ratio=target_dict,
            threshold_trials=calib_config.threshold_trials,
        )
        prefill_result = prefill_calibrator.calibrate(model, prefill_forward_loop, phase="prefill")

        if "scale_factor" in prefill_result:
            threshold_scale_factor["prefill"] = prefill_result["scale_factor"]
            calibration_results["prefill"] = prefill_result
        else:
            warnings.warn("Prefill calibration did not produce valid results")

    # Run decode calibration if enabled
    if calibrate_decode:
        print("\n" + "=" * 60)
        print("DECODE PHASE CALIBRATION")
        print("=" * 60)

        assert calibration_data is not None, "calibration_data must be built before decode"
        decode_forward_loop = create_decode_calibration_forward_loop(
            calibration_data, tokenizer, num_decode_tokens=calib_config.num_decode_tokens
        )

        decode_calibrator = DynamicThresholdCalibrator(
            target_sparse_ratio=target_dict,
            threshold_trials=calib_config.threshold_trials,
        )
        decode_result = decode_calibrator.calibrate(model, decode_forward_loop, phase="decode")

        if "scale_factor" in decode_result:
            threshold_scale_factor["decode"] = decode_result["scale_factor"]
            calibration_results["decode"] = decode_result
        else:
            warnings.warn("Decode calibration did not produce valid results")

    # Check if any calibration succeeded
    if not threshold_scale_factor:
        warnings.warn("No calibration produced valid results")
        return {}

    # Apply combined threshold_scale_factor dict to all modules
    print("\n" + "=" * 60)
    print("APPLYING CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Applying threshold_scale_factor to {len(sparse_modules)} modules:")
    for phase, scale_factor in threshold_scale_factor.items():
        print(f"  {phase}: {scale_factor:.6f}")

    for module_name, module in sparse_modules:
        module._sparse_method_instance.threshold_scale_factor = threshold_scale_factor

    # Print final summary
    print("\nCalibration complete!")
    print_sparse_attention_summary(model)

    return {
        "threshold_scale_factor": threshold_scale_factor,
        "calibration_results": calibration_results,
    }
