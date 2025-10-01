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

"""Main API functions for sparse attention optimization."""

from typing import Any

import torch

from modelopt.torch.opt.conversion import ModeloptStateManager, apply_mode
from modelopt.torch.opt.searcher import ForwardLoop

from .calibration import calibrate_sparse_attention
from .config import SparseAttentionConfig
from .mode import SparseAttentionModeRegistry

__all__ = [
    "calibrate",
    "sparsify",
]


def sparsify(
    model: torch.nn.Module,
    config: dict[str, Any] | SparseAttentionConfig,
    forward_loop: ForwardLoop | None = None,
) -> torch.nn.Module:
    """Applies sparse attention optimization to the model in-place.

    This method performs replacement of attention modules with their sparse counterparts and
    optionally performs calibration as specified by ``config``.
    ``forward_loop`` is used to forward data through the model and gather statistics for calibration.

    Args:
        model: A pytorch model
        config: A dictionary or an instance of
            :class:`SparseAttentionConfig <modelopt.torch.sparsity.attention_sparsity.config.SparseAttentionConfig>`
            specifying the values for keys ``"sparse_cfg"``, ``"method"``, and optionally ``"calibration"``.

            The ``"sparse_cfg"`` key specifies the sparse attention configurations.
            The ``"method"`` key specifies the sparse attention method (e.g., "softmax_skip").
            The ``"calibration"`` key specifies calibration settings if automatic threshold tuning is desired.

            Sparse attention configurations is a dictionary mapping wildcards or filter functions
            to its sparse attention attributes. The wildcards or filter functions are matched
            against the module names. The sparse attention attributes include ``"threshold"``,
            ``"enable"``, and method-specific parameters.

            An example ``config`` dictionary is given below:

            .. code-block::python

                config = {
                    "method": "softmax_skip",
                    "sparse_cfg": {
                        # Phase-aware thresholds with backend selection and calibration
                        "*attention*": {
                            "threshold": {"prefill": 1e-3, "decode": 1e-5},
                            "backend": "pytorch",  # Only pytorch backend supported
                            "enable": True,
                            "calibration": {  # Optional: enables automatic threshold calibration
                                "target_sparse_ratio": 0.5,
                                "samples": 48,
                                "max_seqlen": 8192,
                            },
                        },
                        # Disable for specific layers
                        "*layer_0*": {"enable": False},
                        # Default settings
                        "default": {"enable": False},
                    },
                }

            The ``"backend"`` parameter must be set to ``"pytorch"``:

            - ``"pytorch"``: Softmax patching approach (only supported backend)

            This requires the model to be loaded with ``attn_implementation="eager"``.

        forward_loop: A callable that forwards all calibration data through the model. This is used
            to gather statistics for calibration. It should take model as the argument. It does not need
            to return anything.

            This argument is only required when calibration is enabled in the config.

            Here are a few examples for correct ``forward_loop`` definitions:

            Example 1:

            .. code-block::

                def forward_loop(model) -> None:
                    # iterate over the data loader and forward data through the model
                    for batch in data_loader:
                        model(batch)

            Example 2:

            .. code-block::

                def forward_loop(model) -> float:
                    # evaluate the model on the task
                    return evaluate(model, task, ....)

            .. note::

                Calibration does not require forwarding the entire dataset through the model.
                Please subsample the dataset or reduce the number of batches if needed.

            .. important::

                The model must always be loaded with ``attn_implementation="eager"``
                for sparse attention to work correctly:

                .. code-block:: python

                    from transformers import AutoModelForCausalLM

                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        attn_implementation="eager",  # Required for sparse attention
                        torch_dtype=torch.bfloat16,
                    )

                This is because sparse attention works by patching torch.nn.functional.softmax,
                which is only called in the eager attention implementation.

    Returns:
        A pytorch model which has sparse attention applied and optionally calibrated.
    """
    model = apply_mode(
        model, mode=[("sparse_attention", config)], registry=SparseAttentionModeRegistry
    )

    # Calibrate the sparsity ratio of the attention modules
    return calibrate(model, forward_loop=forward_loop)


def calibrate(
    model: torch.nn.Module,
    forward_loop: ForwardLoop | None = None,
) -> torch.nn.Module:
    """Calibrates sparse attention thresholds based on target sparsity.

    This function performs calibration to find optimal thresholds that achieve
    the target sparsity ratio specified in the sparse attention configuration.

    Args:
        model: A pytorch model with sparse attention already applied
        forward_loop: Optional callable that forwards calibration data through the model.
            It should take model as the argument and can optionally return metrics.
            If None, will auto-generate RULER dataset for calibration.

    Returns:
        The calibrated model with optimized sparse attention thresholds.
        If no calibration is configured, returns the model unchanged.
    """
    # Get the sparse attention config from the model's state
    if not ModeloptStateManager.is_converted(model):
        return model

    manager = ModeloptStateManager(model)

    sparse_attn_config = next(
        (state["config"] for name, state in manager._state if name == "sparse_attention"), None
    )

    if sparse_attn_config is None:
        return model

    # Check if calibration is configured in any sparse_cfg pattern
    # Note: sparse_attn_config is always a dict (stored via config.model_dump())
    sparse_cfg = sparse_attn_config.get("sparse_cfg", {})

    has_calibration = any(
        isinstance(cfg, dict) and "calibration" in cfg for cfg in sparse_cfg.values()
    )

    if not has_calibration:
        return model

    # Run calibration (handles stats collection internally)
    calibrate_sparse_attention(model, sparse_attn_config, forward_loop=forward_loop)

    return model
