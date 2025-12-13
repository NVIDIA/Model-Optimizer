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


def calibrate_sparse_attention(
    model: nn.Module,
    config: dict[str, Any],
    forward_loop: Callable | None = None,
) -> dict[str, Any]:
    """Calibrate sparse attention parameters for optimal sparsity.

    Args:
        model: Model with sparse attention modules
        config: Sparse attention configuration dict
        forward_loop: Callable that forwards calibration data through model.
                     If None, auto-generates RULER dataset.

    Returns:
        Dictionary with calibration results
    """
    # Extract and validate calibration config
    calib_config = _extract_calibration_config(config)

    # Skip calibration if not configured
    if calib_config is None:
        return {}

    # Generate forward_loop if not provided
    if not forward_loop:
        tokenizer = _extract_tokenizer_from_model(model)
        builder = RulerDatasetBuilder(
            samples=calib_config.samples,
            max_seqlen=calib_config.max_seqlen,
            tokenizer_name_or_path=tokenizer,
            num_length_bins=calib_config.num_length_bins,
            max_length_filter=int(calib_config.max_seqlen * 1.5),
        )
        calibration_data = builder.build_calibration_dataset()
        print(f"Generated {len(calibration_data)} calibration samples")
        forward_loop = create_calibration_forward_loop(
            calibration_data, tokenizer, chunk_size=calib_config.chunk_size
        )

    # Get sparse attention modules
    sparse_modules = [
        (name, m) for name, m in model.named_modules() if isinstance(m, SparseAttentionModule)
    ]

    if not sparse_modules:
        print("No sparse attention modules found for calibration")
        return {}

    print(f"Calibrating {len(sparse_modules)} sparse attention modules together...")

    # Run calibration
    calibrator = DynamicThresholdCalibrator(
        target_sparse_ratio=calib_config.target_sparse_ratio,
        threshold_trials=calib_config.threshold_trials,
    )
    calibration_result = calibrator.calibrate(model, forward_loop)

    # Print calibration statistics (regardless of success/failure for debugging)
    print("\nCalibration complete!")
    print_sparse_attention_summary(model)

    if "scale_factor" not in calibration_result:
        warnings.warn("Calibration did not produce valid results")
        return {}

    # Apply calibrated scale factor to all modules
    scale_factor = calibration_result["scale_factor"]
    print(f"\nApplying calibrated scale factor={scale_factor:.6f} to {len(sparse_modules)} modules")

    for module_name, module in sparse_modules:
        module._sparse_method_instance.threshold_scale_factor = scale_factor

    return {"calibration_results": {name: calibration_result for name, _ in sparse_modules}}
