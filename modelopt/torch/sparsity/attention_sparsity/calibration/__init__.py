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

"""Calibration framework for sparse attention methods."""

from .calibrate import calibrate_sparse_attention
from .calibrator import DynamicThresholdCalibrator
from .checkpoint_manifest import (
    CHECKPOINT_MANIFEST_NAME,
    CheckpointManifestError,
    StableFileSnapshot,
    VerifiedCheckpointManifest,
    create_checkpoint_manifest,
    read_stable_file_snapshot,
    verify_checkpoint_manifest,
)
from .mask_reuse import (
    AnchorLayerStats,
    MaskReuseCalibrationError,
    MaskReuseObservation,
    calibrate_mask_reuse_policy,
    canonical_prefill_threshold_scale_factor,
    load_mask_reuse_observations,
    parse_mask_reuse_observations,
)
from .mask_reuse_compact import (
    CompactMaskReuseCapture,
    CompactMaskReuseCaptureSource,
    calibrate_compact_mask_reuse_policy,
    load_compact_mask_reuse_captures,
)
from .ruler_dataset import RulerDatasetBuilder

__all__ = [
    "CHECKPOINT_MANIFEST_NAME",
    "AnchorLayerStats",
    "CheckpointManifestError",
    "CompactMaskReuseCapture",
    "CompactMaskReuseCaptureSource",
    "DynamicThresholdCalibrator",
    "MaskReuseCalibrationError",
    "MaskReuseObservation",
    "RulerDatasetBuilder",
    "StableFileSnapshot",
    "VerifiedCheckpointManifest",
    "calibrate_compact_mask_reuse_policy",
    "calibrate_mask_reuse_policy",
    "calibrate_sparse_attention",
    "canonical_prefill_threshold_scale_factor",
    "create_checkpoint_manifest",
    "load_compact_mask_reuse_captures",
    "load_mask_reuse_observations",
    "parse_mask_reuse_observations",
    "read_stable_file_snapshot",
    "verify_checkpoint_manifest",
]
