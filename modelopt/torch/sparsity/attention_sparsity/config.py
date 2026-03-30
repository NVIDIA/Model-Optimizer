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

"""Configuration classes for sparse attention optimization."""

from collections.abc import Callable
from typing import Any

from pydantic import Field, field_validator, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

# Type definitions for sparse configuration
SparseAttributeConfig = dict[str, Any]  # Configuration for a specific pattern

SparseAttentionCfgType = dict[
    str | Callable,  # Pattern or callable for matching modules
    SparseAttributeConfig,  # Configuration dict with threshold, enable, etc.
]


class SparseAttentionAttributeConfig(ModeloptBaseConfig):
    """Sparse attention attribute configuration for pattern-based module config."""

    method: str = ModeloptField(
        default="flash_skip_softmax",
        title="Sparse attention method.",
        description="The sparse attention method to use (e.g., 'flash_skip_softmax').",
    )

    enable: bool = ModeloptField(
        default=True,
        title="Enable sparse attention.",
        description="If True, enables sparse attention. If False, bypasses sparsity.",
    )

    thresholds: dict[str, list[float]] = ModeloptField(
        default={"prefill": [1e-3], "decode": [1e-4]},
        title="Sparsity thresholds.",
        description=(
            "Thresholds for determining which attention values to skip. "
            "Must be a dict with 'prefill' and/or 'decode' keys, each mapping to a list of floats. "
            "Prefill and decode lists must have the same length. "
            "Sparsity is computed per threshold; the first threshold's mask is applied."
        ),
    )

    br: int = ModeloptField(
        default=128,
        title="Block row size.",
        description="Block row size for block-wise sparsity in Flash Attention.",
    )

    bc: int = ModeloptField(
        default=128,
        title="Block column size.",
        description="Block column size for block-wise sparsity in Flash Attention.",
    )

    backend: str = ModeloptField(
        default="pytorch",
        title="Backend implementation.",
        description=(
            "Backend to use for sparse attention computation. "
            "'pytorch' uses softmax patching with F.softmax (requires attn_implementation='eager'). "
            "'triton' uses the fused Triton kernel (requires attn_implementation='modelopt_triton')."
        ),
    )

    collect_stats: bool = ModeloptField(
        default=False,
        title="Collect statistics.",
        description="Whether to collect sparsity statistics during forward pass for monitoring.",
    )

    is_causal: bool = ModeloptField(
        default=True,
        title="Causal attention flag.",
        description=(
            "Whether the model uses causal (autoregressive) attention. "
            "If True, sparsity statistics are calculated over the lower triangle only. "
            "Set to False for cross-attention models. "
            "Defaults to True for decoder-only models like GPT, LLaMA, etc."
        ),
    )

    sparsity_n: int = ModeloptField(
        default=2,
        title="N in N:M sparsity.",
        description=(
            "Keep top-N of every M attention scores. Only used by triton_sparse_softmax. "
            "Set to 0 to disable sparsity."
        ),
    )

    sparsity_m: int = ModeloptField(
        default=4,
        title="M in N:M sparsity.",
        description="Group size for N:M sparsity (4 or 8). Only used by triton_sparse_softmax.",
    )

    num_sink_tokens: int = ModeloptField(
        default=0,
        title="Number of sink tokens.",
        description=(
            "KV positions before this index are kept dense (attention sinks). "
            "Absolute token count. Only used by triton_sparse_softmax."
        ),
    )

    dense_window_size: int = ModeloptField(
        default=64,
        title="Dense window size in tokens.",
        description=(
            "Tokens near the query diagonal kept dense (local attention window). "
            "Absolute token count. Default 64. Only used by triton_sparse_softmax."
        ),
    )

    skip_softmax_threshold: float = ModeloptField(
        default=0.1,
        title="Skip-softmax threshold.",
        description=(
            "Tiles contributing less than this fraction are skipped entirely. "
            "Only used by triton_skip_softmax. Typical values: 1e-3 to 1e-1. "
            "Set to 0 to disable."
        ),
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        """Validate method is a string."""
        if not isinstance(v, str):
            raise ValueError("method must be a string")
        return v

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        """Validate backend is pytorch or triton."""
        if v not in ("pytorch", "triton"):
            raise ValueError(
                f"Invalid backend: {v}. Supported backends: 'pytorch' (requires "
                f"attn_implementation='eager'), 'triton' (requires "
                f"attn_implementation='modelopt_triton')."
            )
        return v

    @field_validator("sparsity_m")
    @classmethod
    def validate_sparsity_m(cls, v):
        """Validate sparsity_m is 4 or 8."""
        if v not in (4, 8):
            raise ValueError(f"sparsity_m must be 4 or 8, got {v}")
        return v

    @field_validator("sparsity_n")
    @classmethod
    def validate_sparsity_n(cls, v):
        """Validate sparsity_n is non-negative."""
        if v < 0:
            raise ValueError(f"sparsity_n must be >= 0, got {v}")
        return v

    @field_validator("num_sink_tokens")
    @classmethod
    def validate_num_sink_tokens(cls, v):
        """Validate num_sink_tokens is non-negative."""
        if v < 0:
            raise ValueError(f"num_sink_tokens must be >= 0, got {v}")
        return v

    @field_validator("dense_window_size")
    @classmethod
    def validate_dense_window_size(cls, v):
        """Validate dense_window_size is non-negative."""
        if v < 0:
            raise ValueError(f"dense_window_size must be >= 0, got {v}")
        return v

    @field_validator("br", "bc")
    @classmethod
    def validate_block_size(cls, v):
        """Validate block sizes are positive integers."""
        if v <= 0:
            raise ValueError(f"Block size must be positive, got {v}")
        return v

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, v):
        """Validate thresholds is a dict of lists with valid phases and values in range (0, 1)."""
        if not isinstance(v, dict):
            raise ValueError(
                f"Threshold must be a dict with 'prefill' and/or 'decode' keys, got {type(v).__name__}"
            )
        # Validate phase keys
        valid_phases = {"prefill", "decode"}
        invalid_keys = set(v.keys()) - valid_phases
        if invalid_keys:
            raise ValueError(
                f"Invalid threshold phases: {invalid_keys}. Valid phases: {valid_phases}"
            )
        # Validate all values are lists of floats in range (0, 1)
        lengths = {}
        for phase, threshold_list in v.items():
            if not isinstance(threshold_list, list) or len(threshold_list) == 0:
                raise ValueError(
                    f"Thresholds for phase '{phase}' must be a non-empty list, got {threshold_list}"
                )
            for threshold in threshold_list:
                if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 1:
                    raise ValueError(
                        f"Each threshold for phase '{phase}' must be in range (0, 1), got {threshold}"
                    )
            lengths[phase] = len(threshold_list)
        # Validate prefill and decode lists have the same length
        if len(lengths) == 2 and len(set(lengths.values())) != 1:
            raise ValueError(
                f"Prefill and decode threshold lists must have the same length, "
                f"got prefill={lengths['prefill']}, decode={lengths['decode']}"
            )
        return v

    @model_validator(mode="after")
    def validate_sparsity_n_vs_m(self):
        """Validate sparsity_n is within the supported range for the given sparsity_m."""
        if self.sparsity_n > 0:
            max_n = 3 if self.sparsity_m == 4 else self.sparsity_m - 1
            if self.sparsity_n > max_n:
                raise ValueError(
                    f"sparsity_n={self.sparsity_n} exceeds max for sparsity_m={self.sparsity_m}. "
                    f"Valid range: 1..{max_n}"
                )
        return self


class CalibrationConfig(ModeloptBaseConfig):
    """Configuration for automatic threshold calibration using RULER dataset.

    Calibration fits an Exponential model to determine dynamic thresholds that
    achieve target sparsity. The model learns parameters a and b per phase:

        scale_factor = a * exp(b * target_sparsity)

    At inference time, the threshold is computed as:

        threshold = scale_factor / sequence_length

    Key benefits:
    - Target sparsity can be changed at runtime without recalibration
    - Threshold automatically adapts to sequence length
    - Supports independent prefill and decode phase calibration
    - Exponential model provides better fit (lower RMSE)
    """

    target_sparse_ratio: dict[str, float] = ModeloptField(
        default={"prefill": 0.5, "decode": 0.5},
        title="Target sparsity ratio",
        description=(
            "Target ratio of sparse attention blocks (0.0 to 1.0). "
            "Dict with 'prefill' and 'decode' keys for per-phase targets. "
            "Set a phase value to 0.0 to skip calibration for that phase."
        ),
    )

    samples: int = ModeloptField(
        default=24,
        title="Calibration samples",
        description=(
            "Total number of RULER samples for calibration (distributed across length bins). "
            "Default (24) provides 1 sample per task per length bin (4 bins * 6 RULER tasks). "
            "Increase for more robust calibration."
        ),
    )

    max_seqlen: int = ModeloptField(
        default=32768,
        title="Maximum sequence length",
        description="Maximum sequence length for calibration (length bins auto-generated as powers of 2).",
    )

    num_length_bins: int = ModeloptField(
        default=4,
        title="Number of length bins",
        description="Number of length bins to generate (hidden parameter, default: 4).",
    )

    chunk_size: int = ModeloptField(
        default=2048,
        title="Chunk size for prefill",
        description=(
            "Chunk size for chunked prefill to avoid OOM with long sequences. "
            "When sequence length exceeds chunk_size, prefill is done in chunks using KV cache. "
            "Set to -1 to disable chunking (full prefill)."
        ),
    )

    num_decode_tokens: int = ModeloptField(
        default=10,
        title="Number of decode tokens",
        description="Number of decode tokens to generate for decode phase calibration.",
    )

    threshold_trials: list[float] | None = ModeloptField(
        default=None,
        title="Threshold trials",
        description=(
            "List of threshold values to test during calibration. "
            "If None, uses default: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, "
            "1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1]. "
            "Increasing the number of trials improves calibration accuracy but slows down calibration."
        ),
    )

    cache_dir: str | None = ModeloptField(
        default=None,
        title="Cache directory",
        description=(
            "Directory to cache generated calibration samples. "
            "Caching avoids regenerating samples on repeated calibration runs."
        ),
    )

    data_dir: str | None = ModeloptField(
        default=None,
        title="RULER data directory",
        description=(
            "Path to RULER data directory (contains 'essays' subdir with Paul Graham .txt files). "
            "Required for NIAH essay tasks when not using repo layout. Set from example script or CLI."
        ),
    )

    @field_validator("threshold_trials")
    @classmethod
    def validate_threshold_trials(cls, v):
        """Validate threshold_trials are in valid range."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError(f"threshold_trials must be a list, got {type(v)}")
            if len(v) == 0:
                raise ValueError("threshold_trials must not be empty")
            for threshold in v:
                if not isinstance(threshold, (int, float)):
                    raise ValueError(f"All threshold_trials must be numbers, got {type(threshold)}")
                if threshold <= 0 or threshold >= 1:
                    raise ValueError(
                        f"All threshold_trials must be in range (0, 1), got {threshold}"
                    )
        return v

    @field_validator("target_sparse_ratio")
    @classmethod
    def validate_target_sparse_ratio(cls, v):
        """Validate target sparsity ratio dict."""
        if not isinstance(v, dict):
            raise ValueError(
                f"target_sparse_ratio must be a dict with 'prefill' and 'decode' keys, got {type(v)}"
            )
        # Validate phase keys
        valid_phases = {"prefill", "decode"}
        invalid_keys = set(v.keys()) - valid_phases
        if invalid_keys:
            raise ValueError(
                f"Invalid target_sparse_ratio phases: {invalid_keys}. Valid phases: {valid_phases}"
            )
        # Validate all values are in range [0, 1]
        for phase, ratio in v.items():
            if not isinstance(ratio, (int, float)) or not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    f"target_sparse_ratio for phase '{phase}' must be between 0.0 and 1.0, got {ratio}"
                )
        return v

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v):
        """Validate samples is positive."""
        if v <= 0:
            raise ValueError(f"samples must be positive, got {v}")
        return v

    @field_validator("max_seqlen")
    @classmethod
    def validate_max_seqlen(cls, v):
        """Validate max_seqlen is at least 1024."""
        if v < 1024:
            raise ValueError(f"max_seqlen must be >= 1024, got {v}")
        return v

    @field_validator("num_length_bins")
    @classmethod
    def validate_num_length_bins(cls, v):
        """Validate num_length_bins is positive."""
        if v <= 0:
            raise ValueError(f"num_length_bins must be positive, got {v}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v):
        """Validate chunk_size is positive or -1 (disabled)."""
        if v != -1 and v <= 0:
            raise ValueError(f"chunk_size must be positive or -1 (disabled), got {v}")
        return v

    @field_validator("num_decode_tokens")
    @classmethod
    def validate_num_decode_tokens(cls, v):
        """Validate num_decode_tokens is positive."""
        if v <= 0:
            raise ValueError(f"num_decode_tokens must be positive, got {v}")
        return v


class SparseAttentionConfig(ModeloptBaseConfig):
    """Base configuration for sparse attention optimization.

    This base configuration provides the common structure for all sparse
    attention methods and supports pattern-based layer configuration.
    """

    # Pattern-based sparse configuration (similar to quant_cfg in quantization)
    sparse_cfg: SparseAttentionCfgType = ModeloptField(
        default={
            "*attention*": {"method": "flash_skip_softmax", "enable": True},
            "default": {"enable": False},
        },
        title="Sparse attention configuration",
        description="Pattern-based configuration for sparse attention. Keys are patterns to match module names "
        "(or 'calibration' for global calibration settings), values are configuration dicts with parameters like "
        "'threshold', 'enable', etc.",
        validate_default=True,
    )

    # Export configuration
    export_format: str | None = Field(
        None, description="Export format for sparse attention (e.g., 'onnx', 'tensorrt')"
    )


class FlashSkipSoftmaxConfig(SparseAttentionConfig):
    """Configuration for Flash Attention-aware softmax skip sparse attention."""

    # Override sparse_cfg with flash_skip_softmax specific defaults
    sparse_cfg: SparseAttentionCfgType = ModeloptField(
        default={
            "*attention*": {
                "method": "flash_skip_softmax",
                "thresholds": {"prefill": [1e-3], "decode": [1e-5]},
                "br": 128,  # Flash Attention block rows
                "bc": 128,  # Flash Attention block columns
                "backend": "pytorch",  # Only pytorch backend supported
                "collect_stats": True,  # Enable statistics collection
                "enable": True,
            },
            "default": {"enable": False},
        },
        title="Flash softmax skip sparse configuration",
        description="Pattern-based configuration with flash_skip_softmax specific defaults. "
        "Includes FA block sizes (br, bc) and correction factor settings.",
        validate_default=True,
    )


# Pre-defined Sparse Attention Configuration
# Default configuration with block-wise sparsity optimized for Flash Attention
SKIP_SOFTMAX_DEFAULT = {
    "sparse_cfg": {
        "*attn*": {
            "method": "flash_skip_softmax",
            "thresholds": {
                "prefill": [1e-3],  # More aggressive during prefill
                "decode": [1e-4],  # Conservative during decode
            },
            "br": 128,  # Flash Attention block rows
            "bc": 128,  # Flash Attention block columns
            "backend": "pytorch",  # Only pytorch backend supported
            "collect_stats": True,
            "enable": True,
        },
        "default": {"enable": False},
    },
}


# Configuration with RULER calibration
# Note: threshold field is omitted - calibration determines dynamic threshold λ = a / length
# The calibrated threshold adapts to sequence length for optimal sparsity
SKIP_SOFTMAX_CALIB = {
    "sparse_cfg": {
        "calibration": {
            "target_sparse_ratio": {"prefill": 0.5, "decode": 0.5},
            "samples": 64,
            "max_seqlen": 16384,
            "chunk_size": 4096,
        },
        "*attn*": {
            "method": "flash_skip_softmax",
            "br": 128,
            "bc": 128,
            "backend": "pytorch",  # Only pytorch backend supported
            "collect_stats": True,
            "enable": True,
        },
        "default": {"enable": False},
    },
}


# Default N:M sparse softmax configuration
SPARSE_SOFTMAX_DEFAULT = {
    "sparse_cfg": {
        "*attn*": {
            "method": "triton_sparse_softmax",
            "sparsity_n": 2,
            "sparsity_m": 4,
            "num_sink_tokens": 0,
            "dense_window_size": 64,
            "backend": "triton",
            "enable": True,
        },
        "default": {"enable": False},
    },
}


# Default skip-softmax configuration for Triton kernel
SKIP_SOFTMAX_TRITON_DEFAULT = {
    "sparse_cfg": {
        "*attn*": {
            "method": "triton_skip_softmax",
            "skip_softmax_threshold": 0.1,
            "backend": "triton",
            "enable": True,
        },
        "default": {"enable": False},
    },
}


__all__ = [
    "SKIP_SOFTMAX_CALIB",
    "SKIP_SOFTMAX_DEFAULT",
    "SKIP_SOFTMAX_TRITON_DEFAULT",
    "SPARSE_SOFTMAX_DEFAULT",
    "CalibrationConfig",
    "FlashSkipSoftmaxConfig",
    "SparseAttentionAttributeConfig",
    "SparseAttentionCfgType",
    "SparseAttentionConfig",
    "SparseAttributeConfig",
]
