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

"""Utilities for sparse attention integration with llm_eval."""

import modelopt.torch.sparsity.attention_sparsity as mtsa


def _extract_model(model_obj):
    """Extract actual model from wrapper (HFLM or EvalModel)."""
    if hasattr(model_obj, "gpt2"):
        return model_obj.gpt2
    elif hasattr(model_obj, "model"):
        return model_obj.model
    else:
        return model_obj


def sparsify_model(
    model,
    sparse_cfg: str,
    backend=None,
):
    """Apply sparse attention to model with optional RULER calibration.

    Args:
        model: Model wrapper (HFLM or EvalModel) or raw model
        sparse_cfg: Sparse attention config name or dict
        backend: Backend to use (optional, overrides config backend)

    Returns:
        The model with sparse attention applied

    Note:
        Calibration is automatically triggered if the config contains a 'calibration' field.
        The calibration will auto-generate RULER dataset from the model's tokenizer.
    """
    # Extract actual model
    net = _extract_model(model)

    # Resolve config
    if isinstance(sparse_cfg, str):
        # Get config from mtsa module (e.g., SKIP_SOFTMAX_CALIB, SKIP_SOFTMAX_DEFAULT)
        mtsa_cfg = getattr(mtsa, sparse_cfg, None)
        if mtsa_cfg is None:
            raise ValueError(f"Unknown sparse_cfg: {sparse_cfg}.")
    else:
        mtsa_cfg = sparse_cfg

    # Override backend if specified
    if backend:
        if isinstance(mtsa_cfg, dict) and "sparse_cfg" in mtsa_cfg:
            modified_sparse_cfg = {}
            for pattern, cfg in mtsa_cfg["sparse_cfg"].items():
                modified_cfg = cfg.copy() if isinstance(cfg, dict) else cfg
                if isinstance(modified_cfg, dict):
                    modified_cfg["backend"] = backend
                modified_sparse_cfg[pattern] = modified_cfg
            mtsa_cfg = {"sparse_cfg": modified_sparse_cfg}

    # Apply sparsification
    print(f"\nApplying sparse attention with config: {sparse_cfg}")
    mtsa.sparsify(net, mtsa_cfg)
    print("Sparse attention applied successfully!")

    return model
