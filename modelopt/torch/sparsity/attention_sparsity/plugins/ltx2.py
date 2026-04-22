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

"""Plugin for LTX-2 video diffusion models with VSA support.

LTX-2 uses a native ``LTXSelfAttention`` module whose forward signature is
``(x, context, pe, k_pe)`` and which does not call
``F.scaled_dot_product_attention``.  VSA's default SDPA patching in
``SparseAttentionModule`` therefore has no effect on it, so this plugin
installs a model-specific wrapper that:

1. Projects Q/K/V from ``x`` (and ``context`` for self-attention: ``context = x``)
2. Applies LTX-2's ``q_norm`` / ``k_norm`` RMSNorms and RoPE via ``ltx_core``
3. Computes an optional ``gate_compress`` from a trainable zero-initialised
   projection (used by VSA's compression branch, trained later)
4. Calls ``VSA.forward_attention()`` directly, bypassing SDPA
5. Applies the original module's ``to_out`` projection

A forward pre-hook on the root ``LTXModel`` extracts the ``(T, H, W)``
shape from ``Modality.positions`` (same source FastVideo uses) and stores it
on the model, so the wrapper can read it per-step without module-level global
state.
"""

import logging
import weakref

import torch
import torch.nn as nn

from modelopt.torch.utils.logging import warn_rank_0

from ..sparse_attention import SparseAttentionModule, SparseAttentionRegistry
from . import CUSTOM_MODEL_PLUGINS

logger = logging.getLogger(__name__)

_LTX2_LICENSE_WARNING = (
    "LTX-2 packages (ltx-core, ltx-pipelines, ltx-trainer) are provided by "
    "Lightricks and are NOT covered by the Apache 2.0 license governing NVIDIA "
    "Model Optimizer. You MUST comply with the LTX Community License Agreement "
    "when installing and using LTX-2 with NVIDIA Model Optimizer. Any derivative "
    "models or fine-tuned weights from LTX-2 (including quantized or distilled "
    "checkpoints) remain subject to the LTX Community License Agreement, not "
    "Apache 2.0. See: https://github.com/Lightricks/LTX-2/blob/main/LICENSE"
)


def _extract_video_shape_hook(module: nn.Module, args: tuple) -> None:
    """Forward pre-hook on LTXModel to extract ``dit_seq_shape`` from Modality.positions.

    Mirrors FastVideo's ``VideoSparseAttentionMetadataBuilder.build()`` which
    computes ``dit_seq_shape = raw_latent_shape // patch_size``.  Here we
    derive the same shape by counting unique position values per dimension in
    ``Modality.positions``, which is available at the LTXModel entry point
    (before ``TransformerArgsPreprocessor`` converts it to RoPE embeddings).

    The result is stored on the model instance as ``module._vsa_video_shape``
    so ``_LTX2SparseAttention._resolve_video_shape()`` can read it via its
    weak reference to the root model.  Using an instance attribute (not a
    global) makes this safe for concurrent models.
    """
    # LTXModel.forward(self, video: Modality | None, audio, perturbations)
    video = args[0] if len(args) > 0 else None
    if video is None or not hasattr(video, "positions") or video.positions is None:
        return

    positions = video.positions  # (B, 3, T) or (B, 3, T, 2)

    try:
        if positions.ndim == 4:
            # (B, 3, T, 2) -- take start coordinates
            pos_per_dim = positions[0, :, :, 0]  # (3, T)
        elif positions.ndim == 3:
            # (B, 3, T)
            pos_per_dim = positions[0]  # (3, T)
        else:
            return

        t_dim = pos_per_dim[0].unique().numel()
        h_dim = pos_per_dim[1].unique().numel()
        w_dim = pos_per_dim[2].unique().numel()
        seq_len = positions.shape[2]

        if t_dim * h_dim * w_dim == seq_len:
            module._vsa_video_shape = (t_dim, h_dim, w_dim)
            logger.debug(
                f"Extracted dit_seq_shape={module._vsa_video_shape} from "
                f"Modality.positions (seq_len={seq_len})"
            )
        else:
            logger.debug(
                f"Position-derived shape {(t_dim, h_dim, w_dim)} product "
                f"({t_dim * h_dim * w_dim}) != seq_len ({seq_len}), skipping"
            )
    except Exception:
        logger.debug("Failed to extract video_shape from Modality.positions", exc_info=True)


def _is_ltx2_model(model: nn.Module) -> bool:
    """Check if model is an LTX-2 model.

    Uses ``LTXModel`` / ``LTXSelfAttention`` class names to avoid false
    positives from other DiTs (e.g., LongCat) that share similar attribute
    patterns.
    """
    if type(model).__name__ == "LTXModel":
        return True
    return any(type(m).__name__ == "LTXSelfAttention" for m in model.modules())


def _is_ltx2_attention_module(module: nn.Module, name: str = "") -> bool:
    """Check if a module is an LTX-2 Attention module by class name or structure.

    Primary: class name is ``LTXSelfAttention``. Fallback: has ``to_q/k/v``,
    ``q_norm``, ``k_norm``, and ``rope_type`` (unique to LTX-2 among DiTs we
    support).
    """
    class_name = type(module).__name__
    if class_name == "LTXSelfAttention":
        return True
    return (
        hasattr(module, "to_q")
        and hasattr(module, "to_k")
        and hasattr(module, "to_v")
        and hasattr(module, "q_norm")
        and hasattr(module, "k_norm")
        and hasattr(module, "rope_type")
    )


class _LTX2SparseAttention(SparseAttentionModule):
    """Sparse-attention wrapper for LTX-2 ``LTXSelfAttention`` modules.

    Handles LTX-2 specifics (native forward args, RMSNorm, RoPE, trainable
    ``gate_compress``) and delegates the actual attention computation to
    ``VSA.forward_attention``.  Falls back to the original module forward
    for cross-attention / incompatible sequence lengths / missing video
    shape, matching how the core SDPA patch falls through to original SDPA.
    """

    def _setup(self):
        super()._setup()

        # Add trainable gate_compress projection if not already present.
        # Zero-init so its initial contribution is 0 — matches VSA's behaviour
        # when gate_compress is None but leaves room for fine-tuning.
        if not hasattr(self, "to_gate_compress"):
            to_q = self.to_q
            in_features = to_q.in_features
            out_features = to_q.out_features

            self.to_gate_compress = nn.Linear(in_features, out_features, bias=True)
            nn.init.zeros_(self.to_gate_compress.weight)
            nn.init.zeros_(self.to_gate_compress.bias)

            self.to_gate_compress = self.to_gate_compress.to(
                device=to_q.weight.device,
                dtype=to_q.weight.dtype,
            )

    def _compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q/K/V with LTX-2 norms and RoPE.

        Inputs are ``[batch, seq, hidden_dim]``; output tensors share the same
        layout and are reshaped to heads later in ``forward``.
        """
        context = context if context is not None else x

        query = self.to_q(x)
        key = self.to_k(context)
        value = self.to_v(context)

        if hasattr(self, "q_norm"):
            query = self.q_norm(query)
        if hasattr(self, "k_norm"):
            key = self.k_norm(key)

        if pe is not None and hasattr(self, "rope_type"):
            try:
                from ltx_core.model.transformer.rope import apply_rotary_emb
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "LTX-2 VSA plugin requires the 'ltx_core' package for RoPE "
                    "support. The plugin registered successfully, but 'ltx_core' "
                    "is needed at runtime. Install with: pip install ltx-core"
                ) from None

            query = apply_rotary_emb(query, pe, self.rope_type)
            key = apply_rotary_emb(key, pe if k_pe is None else k_pe, self.rope_type)

        return query, key, value

    @staticmethod
    def _reshape_for_vsa(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """``[batch, seq, hidden]`` → ``[batch, heads, seq, head_dim]``."""
        batch, seq_len, hidden_dim = tensor.shape
        head_dim = hidden_dim // num_heads
        return tensor.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    @staticmethod
    def _reshape_from_vsa(tensor: torch.Tensor) -> torch.Tensor:
        """``[batch, heads, seq, head_dim]`` → ``[batch, seq, hidden]``."""
        batch, heads, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)

    def _resolve_video_shape(self, seq_len: int) -> tuple[int, int, int] | None:
        """Resolve video_shape for the current forward pass.

        Resolution order (mirrors FastVideo's metadata flow):
        1. ``root_model._vsa_video_shape`` -- set by the forward pre-hook from
           ``Modality.positions``
        2. ``method.video_shape`` -- explicitly set via the sparsify config
        """
        root_ref = getattr(self, "_vsa_root_model_ref", None)
        root = root_ref() if root_ref is not None else None
        if root is not None:
            shape = getattr(root, "_vsa_video_shape", None)
            if shape is not None:
                t, h, w = shape
                if t * h * w == seq_len:
                    return shape

        method = getattr(self, "_sparse_method_instance", None)
        if method is not None and method.video_shape is not None:
            t, h, w = method.video_shape
            if t * h * w == seq_len:
                return method.video_shape

        return None

    def forward(self, *args, **kwargs):
        """Run the LTX-2 attention forward through VSA.

        Consumes LTX-2's native call signature (``x``, ``context``, ``pe``,
        ``k_pe``) and dispatches to ``VSA.forward_attention``; falls through
        to the original module for cross-attention or incompatible inputs.
        """
        if not self.is_enabled:
            return self._call_original_forward(*args, **kwargs)

        x = kwargs.get("x")
        if x is None and len(args) > 0:
            x = args[0]

        if x is None:
            return self._call_original_forward(*args, **kwargs)

        context = kwargs.get("context")
        pe = kwargs.get("pe")
        k_pe = kwargs.get("k_pe")

        # Cross-attention: fall through to the original module
        if context is not None and x.shape[1] != context.shape[1]:
            return self._call_original_forward(*args, **kwargs)

        method = getattr(self, "_sparse_method_instance", None)
        if method is None:
            return self._call_original_forward(*args, **kwargs)

        query, key, value = self._compute_qkv(x, context, pe, k_pe)

        # Incompatible seq_len (e.g., audio attention with seq=32)
        seq_len = query.shape[1]
        block_size_3d = method.block_size_3d
        block_elements = block_size_3d[0] * block_size_3d[1] * block_size_3d[2]
        if seq_len < block_elements:
            logger.debug(f"VSA skipped: seq_len={seq_len} < block_elements={block_elements}")
            return self._call_original_forward(*args, **kwargs)

        video_shape = self._resolve_video_shape(seq_len)
        if video_shape is None:
            logger.debug(f"VSA skipped: no matching video_shape for seq_len={seq_len}")
            return self._call_original_forward(*args, **kwargs)

        gate_compress = None
        if hasattr(self, "to_gate_compress"):
            gate_compress = self.to_gate_compress(x)

        # Reshape to [batch, heads, seq, head_dim]
        query = self._reshape_for_vsa(query, self.heads)
        key = self._reshape_for_vsa(key, self.heads)
        value = self._reshape_for_vsa(value, self.heads)
        if gate_compress is not None:
            gate_compress = self._reshape_for_vsa(gate_compress, self.heads)

        output, stats = method.forward_attention(
            query=query,
            key=key,
            value=value,
            gate_compress=gate_compress,
            video_shape=video_shape,
        )

        # Bubble stats up through SparseAttentionModule's stats path
        self._last_stats = stats
        if self._stats_manager is not None:
            self._stats_manager.collect(stats)
            self._last_stats = None

        output = self._reshape_from_vsa(output)

        if hasattr(self, "to_out"):
            output = self.to_out(output)

        return output

    def _call_original_forward(self, *args, **kwargs):
        """Invoke the original module's forward, bypassing VSA.

        ``SparseAttentionModule.forward`` passes through to the original
        module when ``is_enabled`` is False — exploit that to avoid
        reimplementing the fallback path.
        """
        was_enabled = getattr(self, "_enabled", True)
        self._enabled = False
        try:
            result = SparseAttentionModule.forward(self, *args, **kwargs)
        finally:
            self._enabled = was_enabled
        return result

    def get_gate_compress_parameters(self):
        """Return trainable ``gate_compress`` parameters for later fine-tuning."""
        if hasattr(self, "to_gate_compress"):
            return self.to_gate_compress.parameters()
        return iter([])


def register_ltx2_attention(model: nn.Module) -> int:
    """Register LTX-2 Attention modules for VSA wrapping.

    Replaces any existing generic wrapper in ``SparseAttentionRegistry``
    with ``_LTX2SparseAttention`` for each LTX-2 attention type found, wires
    a weakref back to the root model on every attention instance, and
    installs the ``Modality.positions`` extraction pre-hook.
    """
    if not _is_ltx2_model(model):
        return 0

    # Third-party-license notice: emit once per LTX-2 model detection,
    # matching the pattern used by modelopt's quantization and kernel LTX-2
    # plugins.  The wrapper touches ``ltx_core`` (RoPE) at forward time, so
    # users must comply with the LTX Community License Agreement.
    warn_rank_0(_LTX2_LICENSE_WARNING, UserWarning, stacklevel=2)

    registered_types = set()
    num_modules = 0

    for name, module in model.named_modules():
        if not _is_ltx2_attention_module(module, name):
            continue

        num_modules += 1
        module_type = type(module)

        if module_type in registered_types:
            continue

        if module_type in SparseAttentionRegistry:
            logger.debug(f"Unregistering generic wrapper for {module_type.__name__}")
            SparseAttentionRegistry.unregister(module_type)

        SparseAttentionRegistry.register({module_type: module_type.__name__})(_LTX2SparseAttention)
        registered_types.add(module_type)
        logger.info(f"Registered LTX-2 attention: {module_type.__name__}")

    if num_modules > 0:
        logger.info(f"Found {num_modules} LTX-2 Attention modules in model")

        # Weakref avoids the circular-submodule problem (nn.Module.__setattr__
        # would otherwise register the root model as a submodule of every
        # attention, causing infinite recursion in named_children()).
        root_ref = weakref.ref(model)
        for _, module in model.named_modules():
            if _is_ltx2_attention_module(module):
                object.__setattr__(module, "_vsa_root_model_ref", root_ref)

        model.register_forward_pre_hook(_extract_video_shape_hook)
        logger.debug("Registered VSA video_shape extraction hook on model")

    return len(registered_types)


def register_ltx2_on_the_fly(model: nn.Module) -> bool:
    """Plugin entry point: wire up LTX-2 VSA if this is an LTX-2 model."""
    num_registered = register_ltx2_attention(model)
    if num_registered > 0:
        logger.info(f"Registered {num_registered} LTX-2 attention types for VSA")
        return True
    return False


# Idempotent: plugins/__init__.py stores plugins in a set so re-imports are safe.
CUSTOM_MODEL_PLUGINS.add(register_ltx2_on_the_fly)
