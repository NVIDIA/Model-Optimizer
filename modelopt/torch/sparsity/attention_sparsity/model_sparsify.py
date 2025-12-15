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

from modelopt.torch.opt.conversion import apply_mode
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

    This method performs replacement of attention modules with their sparse counterparts.

    Args:
        model: A pytorch model
        config: A dictionary or an instance of
            :class:`SparseAttentionConfig <modelopt.torch.sparsity.attention_sparsity.config.SparseAttentionConfig>`
            specifying the values for keys ``"sparse_cfg"`` and ``"method"``.

            The ``"sparse_cfg"`` key specifies the sparse attention configurations.
            The ``"method"`` key specifies the sparse attention method (e.g., "flash_skip_softmax").

            Sparse attention configurations is a dictionary mapping wildcards or filter functions
            to its sparse attention attributes. The wildcards or filter functions are matched
            against the module names. The sparse attention attributes include ``"threshold"``,
            ``"enable"``, and method-specific parameters.

            An example ``config`` dictionary is given below:

            .. code-block::python

                config = {
                    "sparse_cfg": {
                        # Phase-aware thresholds with backend selection
                        "*attention*": {
                            "method": "flash_skip_softmax",
                            "threshold": {"prefill": 1e-3, "decode": 1e-5},
                            "backend": "pytorch",  # Only pytorch backend supported
                            "enable": True,
                        },
                        # Disable for specific layers
                        "*layer.0*": {"enable": False},
                        # Default settings
                        "default": {"enable": False},
                    },
                }

            For automatic threshold calibration using RULER dataset:

            .. code-block::python

                config = {
                    "sparse_cfg": {
                        "*attention*": {
                            "method": "flash_skip_softmax",
                            "backend": "pytorch",
                            "enable": True,
                            "calibration": {  # Enables automatic threshold calibration
                                "target_sparse_ratio": 0.5,
                                "samples": 48,
                                "max_seqlen": 8192,
                            },
                        },
                        "default": {"enable": False},
                    },
                }

            The ``"backend"`` parameter must be set to ``"pytorch"``:

            - ``"pytorch"``: Softmax patching approach (only supported backend)

            This requires the model to be loaded with ``attn_implementation="eager"``.

        forward_loop: Reserved for future use.

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
    return calibrate(model, config, forward_loop=forward_loop)


def calibrate(
    model: torch.nn.Module,
    config: dict[str, Any] | SparseAttentionConfig,
    forward_loop: ForwardLoop | None = None,
) -> torch.nn.Module:
    """Calibrates sparse attention thresholds based on target sparsity.

    Args:
        model: Model with sparse attention modules
        config: Sparse attention configuration with calibration settings
        forward_loop: Optional callable that forwards calibration data through the model.
            If provided, uses this for calibration data.
            If None, will auto-generate RULER dataset for calibration.

    Returns:
        The calibrated model with optimized sparse attention thresholds.
    """
    calibrate_sparse_attention(model, config, forward_loop=forward_loop)
    return model
