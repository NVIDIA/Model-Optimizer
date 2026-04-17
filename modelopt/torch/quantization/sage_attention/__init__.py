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

"""SageAttention-style attention quantization for diffusers models.

``apply_sage_attention`` patches a diffusers transformer to quantize the
post-softmax P tile to NVFP4 E2M1 inside ModelOpt's Triton flash-attention
kernel (``quantize_p=True``).  This is purely a **quantization** feature —
it is independent of, and can be freely combined with, the sparse attention
methods in ``modelopt.torch.sparsity.attention_sparsity``.

Design
------
SageAttention wraps the transformer's ``forward`` once:

1. Before the forward, it sets ``quantize_p=True`` in a thread-local store
   that the Triton kernel reads.
2. It activates the ``modelopt_triton`` diffusers attention backend for the
   duration of the forward pass so that attention calls are routed to the
   ModelOpt Triton kernel.
3. After the forward (``finally`` block), it resets ``quantize_p=False``.

Sparse attention methods (skip-softmax / N:M sparse softmax) manage their
own thread-local params (threshold, sparsity_n/m, …) and deliberately **do
not touch** ``quantize_p``, enabling transparent combination:

.. code-block:: python

    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from modelopt.torch.quantization import apply_sage_attention

    # SageAttention standalone — NVFP4 P-matrix quantization only
    apply_sage_attention(transformer)

    # Combined with N:M sparse softmax
    mtsa.sparsify(transformer, mtsa.SPARSE_SOFTMAX_DEFAULT)
    apply_sage_attention(transformer)

    # Combined with skip-softmax tile pruning
    mtsa.sparsify(transformer, mtsa.SKIP_SOFTMAX_TRITON_DEFAULT)
    apply_sage_attention(transformer)

Supported models
----------------
Currently targets **diffusers** transformer models (WAN, LTX, …) that use
the diffusers attention-dispatch mechanism.  The ``modelopt_triton`` backend
is registered in ``diffusers._AttentionBackendRegistry`` on first call.

Requirements
------------
- CUDA GPU + Triton installed
- ``modelopt.torch.sparsity.attention_sparsity`` (provides the Triton kernel
  and diffusers backend registration)
"""

import torch

__all__ = ["apply_sage_attention", "apply_sage_attention_v3"]


def apply_sage_attention(
    transformer: torch.nn.Module,
    quantize_p: bool = True,
) -> None:
    """Patch a diffusers transformer to use NVFP4 P-matrix quantization.

    Wraps ``transformer.forward`` so that every call activates the
    ``modelopt_triton`` diffusers attention backend with ``quantize_p=True``
    inside the Triton flash-attention kernel.

    This is a standalone quantization feature and does not depend on or
    conflict with ``mtsa.sparsify()``.  Both can be applied to the same
    transformer — sparsity parameters and quantization parameters are stored
    in independent thread-local slots.

    Args:
        transformer: A diffusers transformer module (e.g. ``pipe.transformer``
            for WAN2.2 / LTX Video).
        quantize_p: If True (default), quantize the post-softmax P tile to
            NVFP4 E2M1 with per-tile max scaling inside the Triton kernel.

    Raises:
        ImportError: If ``modelopt.torch.sparsity.attention_sparsity`` is not
            installed (required for the Triton kernel and diffusers backend).
    """
    try:
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            clear_sage_attention_config,
            get_triton_attention_backend,
            register_diffusers_triton_attention,
            set_sage_attention_config,
        )
    except ImportError as exc:
        raise ImportError(
            "apply_sage_attention requires modelopt.torch.sparsity.attention_sparsity "
            "(Triton kernel + diffusers backend). Install modelopt with the [all] extra "
            "or ensure triton is available."
        ) from exc

    register_diffusers_triton_attention()

    original_forward = transformer.forward

    def _sage_forward(*args, **kwargs):
        set_sage_attention_config(quantize_p=quantize_p)
        with get_triton_attention_backend():
            try:
                return original_forward(*args, **kwargs)
            finally:
                clear_sage_attention_config()

    transformer.forward = _sage_forward
    transformer._modelopt_sage_attention = True  # mark for inspection

    q_str = "NVFP4 E2M1" if quantize_p else "disabled"
    print(
        f"[ModelOpt] SageAttention applied: quantize_p={quantize_p} ({q_str} P-tile quantization)"
    )


def apply_sage_attention_v3(transformer: torch.nn.Module) -> None:
    """Patch a diffusers transformer with SageAttention v3 microscaling NVFP4.

    Wraps ``transformer.forward`` to quantize **Q, K, V, and P** to NVFP4 E2M1
    using per-group microscaling (groups of 16 elements along the head dimension),
    following the SageAttention v3 paper (arxiv 2505.11594).

    Compared to :func:`apply_sage_attention` (which only quantizes P with per-tile
    scaling), this also quantizes Q, K, and V with finer per-group scales, targeting
    Blackwell / Ada GPUs where FP4 tensor cores provide maximum throughput.

    Args:
        transformer: A diffusers transformer module (e.g. ``pipe.transformer``).

    Raises:
        ImportError: If ``modelopt.torch.sparsity.attention_sparsity`` is not
            installed (required for the Triton kernel and diffusers backend).
    """
    try:
        from modelopt.torch.sparsity.attention_sparsity.kernels.diffusers_triton_attention import (
            clear_sage_attention_config,
            get_triton_attention_backend,
            register_diffusers_triton_attention,
            set_sage_attention_config,
        )
    except ImportError as exc:
        raise ImportError(
            "apply_sage_attention_v3 requires modelopt.torch.sparsity.attention_sparsity "
            "(Triton kernel + diffusers backend). Install modelopt with the [all] extra "
            "or ensure triton is available."
        ) from exc

    register_diffusers_triton_attention()

    original_forward = transformer.forward

    def _sage_v3_forward(*args, **kwargs):
        set_sage_attention_config(quantize_p=False, quantize_qkv=True)
        with get_triton_attention_backend():
            try:
                return original_forward(*args, **kwargs)
            finally:
                clear_sage_attention_config()

    transformer.forward = _sage_v3_forward
    transformer._modelopt_sage_attention_v3 = True  # mark for inspection
    print("[ModelOpt] SageAttention v3 applied: per-group MX NVFP4 on Q, K, V, and P")
