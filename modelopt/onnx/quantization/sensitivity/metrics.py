# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Proxy metrics for ONNX quantization sensitivity scoring.

Each metric maps a pair ``(fp16_act, quant_act)`` of aligned activation tensors to a non-negative
scalar. Higher values mean the quantization target under test caused more distortion of the model's
output, so the caller ranks targets by increasing sensitivity to decide what to keep at higher
precision. Callers pass the raw activations exactly as ORT returned them; each metric normalizes
internally where relevant (e.g. softmax for KL) and averages across the leading batch dimension.
"""

import numpy as np

__all__ = ["cos_dist", "kl_div", "mse"]

_EPS = 1e-12


def _flatten_per_sample(tensor: np.ndarray) -> np.ndarray:
    """Flatten every non-batch dimension into a single feature dim.

    Args:
        tensor: Any-shape numpy array whose first axis is the sample/batch axis. Scalar tensors
            (0-D) are treated as a single sample with one feature.

    Returns:
        A ``(num_samples, num_features)`` array.
    """
    arr = np.asarray(tensor)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    return arr.reshape(arr.shape[0], -1)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + _EPS)


def kl_div(fp16_act: np.ndarray, quant_act: np.ndarray) -> float:
    """KL divergence between softmax-normalized FP16 and quantized activations.

    Both tensors are flattened per-sample and passed through softmax to obtain probability
    distributions, then the KL divergence ``sum(p * log(p / q))`` is computed per sample and
    averaged. This is the recommended default metric because it matches the intuition "output
    distribution should be similar" and is robust to activation magnitude scale.

    Args:
        fp16_act: FP16 reference activations, shape ``(num_samples, ...)``.
        quant_act: Activations from the quantized model, shape ``(num_samples, ...)``.

    Returns:
        Mean KL divergence across the sample axis, as a Python float.
    """
    p = _softmax(_flatten_per_sample(fp16_act).astype(np.float64))
    q = _softmax(_flatten_per_sample(quant_act).astype(np.float64))
    per_sample = np.sum(p * (np.log(p + _EPS) - np.log(q + _EPS)), axis=-1)
    return float(np.mean(per_sample))


def mse(fp16_act: np.ndarray, quant_act: np.ndarray) -> float:
    """Mean squared error on raw activation values.

    Cheap to compute but sensitive to activation magnitude scale; a target whose output happens to
    be large in absolute value will look more sensitive under MSE than under KL / cosine.

    Args:
        fp16_act: FP16 reference activations.
        quant_act: Activations from the quantized model with the same shape as ``fp16_act``.

    Returns:
        Mean squared error across all elements, as a Python float.
    """
    diff = _flatten_per_sample(fp16_act).astype(np.float64) - _flatten_per_sample(quant_act).astype(
        np.float64
    )
    return float(np.mean(diff * diff))


def cos_dist(fp16_act: np.ndarray, quant_act: np.ndarray) -> float:
    """Cosine distance ``1 - cos(fp16, quant)`` averaged across samples.

    Scale-invariant: robust to models with wide activation-magnitude variance where MSE would be
    dominated by the largest tensors.

    Args:
        fp16_act: FP16 reference activations.
        quant_act: Activations from the quantized model with the same shape as ``fp16_act``.

    Returns:
        Mean cosine distance across the sample axis, as a Python float in ``[0, 2]``.
    """
    p = _flatten_per_sample(fp16_act).astype(np.float64)
    q = _flatten_per_sample(quant_act).astype(np.float64)
    dot = np.sum(p * q, axis=-1)
    norm = np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1)
    cos = dot / (norm + _EPS)
    return float(np.mean(1.0 - cos))
