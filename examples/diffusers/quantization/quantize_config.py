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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from models_utils import MODEL_REGISTRY, ModelType


class DataType(str, Enum):
    """Supported data types for model loading."""

    HALF = "Half"
    BFLOAT16 = "BFloat16"
    FLOAT = "Float"

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._dtype_map[self.value]


DataType._dtype_map = {
    DataType.HALF: torch.float16,
    DataType.BFLOAT16: torch.bfloat16,
    DataType.FLOAT: torch.float32,
}


class QuantFormat(str, Enum):
    """Supported quantization formats."""

    INT8 = "int8"
    FP8 = "fp8"
    FP4 = "fp4"


class QuantAlgo(str, Enum):
    """Supported quantization algorithms."""

    MAX = "max"
    SVDQUANT = "svdquant"
    SMOOTHQUANT = "smoothquant"


class CollectMethod(str, Enum):
    """Calibration collection methods."""

    GLOBAL_MIN = "global_min"
    MIN_MAX = "min-max"
    MIN_MEAN = "min-mean"
    MEAN_MAX = "mean-max"
    DEFAULT = "default"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    format: QuantFormat = QuantFormat.INT8
    algo: QuantAlgo = QuantAlgo.MAX
    percentile: float = 1.0
    collect_method: CollectMethod = CollectMethod.DEFAULT
    alpha: float = 1.0  # SmoothQuant alpha
    lowrank: int = 32  # SVDQuant lowrank
    quantize_mha: bool = False
    compress: bool = False

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.format == QuantFormat.FP8 and self.collect_method != CollectMethod.DEFAULT:
            raise NotImplementedError("Only 'default' collect method is implemented for FP8.")
        if self.quantize_mha and self.format == QuantFormat.INT8:
            raise ValueError("MHA quantization is only supported for FP8, not INT8.")
        if self.compress and self.format == QuantFormat.INT8:
            raise ValueError("Compression is only supported for FP8 and FP4, not INT8.")


@dataclass
class CalibrationConfig:
    """Configuration for calibration process."""

    prompts_dataset: dict | Path
    batch_size: int = 2
    calib_size: int = 128
    n_steps: int = 30

    def validate(self) -> None:
        """Validate calibration configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.calib_size <= 0:
            raise ValueError("Calibration size must be positive.")
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive.")

    @property
    def num_batches(self) -> int:
        """Calculate number of calibration batches."""
        return self.calib_size // self.batch_size


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""

    model_type: ModelType = ModelType.FLUX_DEV
    model_dtype: dict[str, torch.dtype] = field(default_factory=lambda: {"default": torch.float16})
    backbone: str = ""
    trt_high_precision_dtype: DataType = DataType.HALF
    override_model_path: Path | None = None
    cpu_offloading: bool = False
    ltx_skip_upsampler: bool = False  # Skip upsampler for LTX-Video (faster calibration)
    extra_params: dict[str, Any] = field(default_factory=dict)

    @property
    def model_path(self) -> str:
        """Get the model path (override or default)."""
        if self.override_model_path:
            return str(self.override_model_path)
        return MODEL_REGISTRY[self.model_type]


@dataclass
class ExportConfig:
    """Configuration for model export."""

    quantized_torch_ckpt_path: Path | None = None
    onnx_dir: Path | None = None
    hf_ckpt_dir: Path | None = None
    restore_from: Path | None = None

    def validate(self) -> None:
        """Validate export configuration."""
        if self.restore_from and not self.restore_from.exists():
            raise FileNotFoundError(f"Restore checkpoint not found: {self.restore_from}")

        if self.quantized_torch_ckpt_path:
            parent_dir = self.quantized_torch_ckpt_path.parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)

        if self.onnx_dir and not self.onnx_dir.exists():
            self.onnx_dir.mkdir(parents=True, exist_ok=True)

        if self.hf_ckpt_dir and not self.hf_ckpt_dir.exists():
            self.hf_ckpt_dir.mkdir(parents=True, exist_ok=True)
