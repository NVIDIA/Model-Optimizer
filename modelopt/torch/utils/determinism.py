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

"""Best-effort determinism for reproducible runs."""

import os
import random

import numpy as np
import torch

from .logging import print_rank_0, warn_rank_0

__all__ = [
    "DETERMINISTIC_ATTN_IMPLEMENTATIONS",
    "deterministic_attn_implementation",
    "enable_determinism",
]

_DETERMINISTIC_CUBLAS_CONFIGS = (":4096:8", ":16:8")
_SDPA_EFFICIENT = "modelopt_sdpa_efficient"

# The attention implementations that give a sample the same activations whatever else is in its
# batch. transformers' default SDPA does not: it runs another kernel for a sample when a row of its
# batch is padded, and yet another at batch size 1.
DETERMINISTIC_ATTN_IMPLEMENTATIONS = ("eager", _SDPA_EFFICIENT)


def enable_determinism(seed: int) -> None:
    """Seed every random number generator and ask PyTorch for deterministic kernels.

    An operation without a deterministic kernel only warns, so repeated runs on the same hardware,
    software and parallel layout agree as closely as the remaining kernels allow. Call this at
    startup, before the first CUDA matmul: cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when it creates
    its handle. A value of that variable that is already set is kept, with a warning if cuBLAS does
    not treat it as deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cublas_config = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", _DETERMINISTIC_CUBLAS_CONFIGS[0]
    )
    if cublas_config not in _DETERMINISTIC_CUBLAS_CONFIGS:
        warn_rank_0(
            f"CUBLAS_WORKSPACE_CONFIG={cublas_config} is not one of {_DETERMINISTIC_CUBLAS_CONFIGS}: "
            "cuBLAS reductions are not made deterministic."
        )
    torch.use_deterministic_algorithms(True, warn_only=True)
    print_rank_0(f"Deterministic mode: seeded every RNG with {seed}, deterministic torch kernels.")


def deterministic_attn_implementation(requested: str | None = None) -> str:
    """Return the attention implementation to load a model with for a reproducible run.

    That is ``requested`` if it is one of ``DETERMINISTIC_ATTN_IMPLEMENTATIONS``, else ``"eager"``,
    with a warning when something else was asked for. ``"modelopt_sdpa_efficient"`` is SDPA pinned
    to PyTorch's memory-efficient kernel, with a mask for every batch: like eager it runs one kernel
    whatever the batch holds, and unlike eager it never builds the attention matrix. It is
    registered with transformers here, so call this before loading the model.
    """
    if requested is None or requested == "eager":
        return "eager"
    if requested not in DETERMINISTIC_ATTN_IMPLEMENTATIONS:
        warn_rank_0(
            f"Attention implementation {requested!r} runs another kernel for a sample depending on "
            f"the other samples in its batch; using eager. Reproducible runs accept "
            f"{', '.join(DETERMINISTIC_ATTN_IMPLEMENTATIONS)}."
        )
        return "eager"
    _register_sdpa_efficient()
    return requested


def _register_sdpa_efficient() -> None:
    """Make ``modelopt_sdpa_efficient`` an attention implementation of transformers' models."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    def attention(module, query, key, value, attention_mask, **kwargs):
        # What eager implements and SDPA does not: dropping it would calibrate another model.
        unsupported = [name for name in ("softcap", "s_aux") if kwargs.get(name) is not None]
        if unsupported:
            raise NotImplementedError(
                f"{_SDPA_EFFICIENT} does not support {' and '.join(unsupported)}, which "
                f"{type(module).__name__} uses: use eager attention."
            )
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    def mask(*args, **kwargs):
        # transformers drops the mask of a batch without a padded row, so that SDPA can run its
        # flash kernel on it. Every batch gets a mask here, so every batch runs the same kernel.
        kwargs["allow_is_causal_skip"] = False
        return sdpa_mask(*args, **kwargs)

    AttentionInterface.register(_SDPA_EFFICIENT, attention)
    AttentionMaskInterface.register(_SDPA_EFFICIENT, mask)
