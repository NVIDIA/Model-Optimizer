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

LTX-2 uses a specific Attention module structure that differs from standard
HuggingFace/Diffusers attention. This plugin provides:

1. Detection of LTX-2's native Attention modules
2. Q/K/V projection, RMSNorm, and RoPE handling
3. Support for trainable gate_compress for VSA quality optimization
"""

import logging
import weakref

import torch
import torch.nn as nn

from ..sparse_attention import SparseAttentionModule, SparseAttentionRegistry
from . import CUSTOM_MODEL_PLUGINS

logger = logging.getLogger(__name__)


def _extract_video_shape_hook(module: nn.Module, args: tuple) -> None:
    """Forward pre-hook on LTXModel to extract dit_seq_shape from Modality.positions.

    Mirrors FastVideo's ``VideoSparseAttentionMetadataBuilder.build()`` which
    computes ``dit_seq_shape = raw_latent_shape // patch_size``.  Here we derive
    the same shape by counting unique position values per dimension in the
    ``Modality.positions`` tensor, which is available at the LTXModel entry
    point (before ``TransformerArgsPreprocessor`` converts it to RoPE embeddings).

    The result is stored on the model instance as ``module._vsa_video_shape``
    so that ``_LTX2SparseAttention._resolve_video_shape()`` can read it via
    its ``_vsa_root_model`` reference.  This avoids module-level global state
    and is safe for concurrent models.
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

    Uses LTXModel / LTXSelfAttention class names to avoid false positives
    from other DiTs (e.g., LongCat) that share similar attribute patterns.

    Args:
        model: PyTorch model to check.

    Returns:
        True if model is LTX-2 (root class LTXModel or contains LTXSelfAttention).
    """
    if type(model).__name__ == "LTXModel":
        return True
    return any(type(m).__name__ == "LTXSelfAttention" for m in model.modules())


def _is_ltx2_attention_module(module: nn.Module, name: str = "") -> bool:
    """Check if a module is an LTX-2 Attention module by class name or structure.

    Primary: class name is LTXSelfAttention. Fallback: has to_q/k/v, q_norm,
    k_norm, and rope_type (unique to LTX-2 among DiTs).

    Args:
        module: Module to check.
        name: Module name in model hierarchy.

    Returns:
        True if module is an LTX-2 attention module.
    """
    class_name = type(module).__name__
    if class_name == "LTXSelfAttention":
        return True
    # Fallback for subclasses or renamed variants: must have rope_type (LTX-2 only)
    return (
        hasattr(module, "to_q")
        and hasattr(module, "to_k")
        and hasattr(module, "to_v")
        and hasattr(module, "q_norm")
        and hasattr(module, "k_norm")
        and hasattr(module, "rope_type")
    )


class _LTX2SparseAttention(SparseAttentionModule):
    """Sparse attention wrapper for LTX-2 Attention modules.

    This plugin handles all LTX-2 specific logic:
    - Q/K/V projection and normalization (using native LTX-2 args: x, context, pe, k_pe)
    - RoPE application via ltx_core
    - Trainable gate_compress for VSA quality optimization

    The plugin computes Q, K, V directly and calls VSA.forward_attention(),
    keeping VSA as a pure algorithm without module-specific knowledge.
    """

    def _setup(self):
        """Setup the VSA wrapper with trainable gate_compress."""
        super()._setup()

        # Check if we need to add gate_compress projection
        if not hasattr(self, "to_gate_compress"):
            to_q = self.to_q
            in_features = to_q.in_features
            out_features = to_q.out_features

            # Create gate_compress projection (zero-initialized)
            self.to_gate_compress = nn.Linear(in_features, out_features, bias=True)
            nn.init.zeros_(self.to_gate_compress.weight)
            nn.init.zeros_(self.to_gate_compress.bias)

            # Move to same device/dtype as to_q
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
        """Compute Q, K, V projections with LTX-2 specific processing.

        Args:
            x: Input tensor [batch, seq, hidden_dim].
            context: Context for cross-attention, or None for self-attention.
            pe: Positional embeddings for RoPE.
            k_pe: Optional separate positional embeddings for keys.

        Returns:
            Tuple of (query, key, value) tensors in [batch, seq, hidden_dim] format.
        """
        # For self-attention, use x for K, V
        context = context if context is not None else x

        # Project to Q, K, V
        query = self.to_q(x)
        key = self.to_k(context)
        value = self.to_v(context)

        # Apply Q/K norms (LTX-2 specific)
        if hasattr(self, "q_norm"):
            query = self.q_norm(query)
        if hasattr(self, "k_norm"):
            key = self.k_norm(key)

        # Apply RoPE if provided (LTX-2 specific)
        if pe is not None and hasattr(self, "rope_type"):
            try:
                from ltx_core.model.transformer.rope import apply_rotary_emb
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "LTX-2 VSA plugin requires the 'ltx_core' package for RoPE support. "
                    "The plugin registered successfully, but 'ltx_core' is needed at runtime. "
                    "Install it with:  pip install ltx-core"
                ) from None

            query = apply_rotary_emb(query, pe, self.rope_type)
            key = apply_rotary_emb(key, pe if k_pe is None else k_pe, self.rope_type)

        return query, key, value

    def _reshape_for_vsa(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Reshape tensor from [batch, seq, hidden_dim] to [batch, heads, seq, head_dim].

        Args:
            tensor: Input tensor [batch, seq, hidden_dim].
            num_heads: Number of attention heads.

        Returns:
            Reshaped tensor [batch, heads, seq, head_dim].
        """
        batch, seq_len, hidden_dim = tensor.shape
        head_dim = hidden_dim // num_heads
        return tensor.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    def _reshape_from_vsa(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from [batch, heads, seq, head_dim] to [batch, seq, hidden_dim].

        Args:
            tensor: Input tensor [batch, heads, seq, head_dim].

        Returns:
            Reshaped tensor [batch, seq, hidden_dim].
        """
        batch, heads, seq_len, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)

    def _resolve_video_shape(self, seq_len: int) -> tuple[int, int, int] | None:
        """Resolve video_shape for the current forward pass.

        Resolution order (mirrors FastVideo's metadata flow):
        1. ``root_model._vsa_video_shape`` -- set by the forward pre-hook from
           ``Modality.positions`` (analogous to ``get_forward_context().attn_metadata``)
        2. ``method.video_shape`` -- explicitly set via the sparsify config

        Args:
            seq_len: Current sequence length (for validation).

        Returns:
            Tuple (T, H, W) or None if not determinable.
        """
        # 1. Primary: video_shape extracted by forward pre-hook on root model
        root_ref = getattr(self, "_vsa_root_model_ref", None)
        root = root_ref() if root_ref is not None else None
        if root is not None:
            shape = getattr(root, "_vsa_video_shape", None)
            if shape is not None:
                t, h, w = shape
                if t * h * w == seq_len:
                    return shape

        # 2. Fallback: explicit video_shape from sparsify config
        method = getattr(self, "_sparse_method_instance", None)
        if method is not None and method.video_shape is not None:
            t, h, w = method.video_shape
            if t * h * w == seq_len:
                return method.video_shape

        return None

    def forward(self, *args, **kwargs):
        """Forward pass computing Q/K/V directly and calling VSA.forward_attention().

        This method handles all LTX-2 specific logic:
        1. Extract arguments (uses LTX-2 native names: x, context, pe, k_pe)
        2. Compute Q, K, V projections with norms and RoPE
        3. Compute gate_compress
        4. Resolve video_shape from hook or config
        5. Check compatibility and call VSA or fallback
        6. Apply output projection
        """
        x = kwargs.get("x")
        if x is None and len(args) > 0:
            x = args[0]

        if x is None:
            return self._call_original_forward(*args, **kwargs)

        context = kwargs.get("context")
        pe = kwargs.get("pe")
        k_pe = kwargs.get("k_pe")

        # === Check cross-attention ===
        if context is not None:
            if x.shape[1] != context.shape[1]:
                # NOTE: skip VSA for Cross-attention, use original attention
                return self._call_original_forward(*args, **kwargs)

        # === Check VSA method availability ===
        if not hasattr(self, "_sparse_method_instance") or self._sparse_method_instance is None:
            return self._call_original_forward(*args, **kwargs)

        method = self._sparse_method_instance  # VSA instance

        # === Compute Q, K, V ===
        query, key, value = self._compute_qkv(x, context, pe, k_pe)

        # === Check sequence length compatibility ===
        seq_len = query.shape[1]
        block_size_3d = method.block_size_3d  # type: ignore[attr-defined]
        block_elements = block_size_3d[0] * block_size_3d[1] * block_size_3d[2]

        if seq_len < block_elements:
            # Incompatible sequence length (e.g., audio attention with seq_len=32)
            logger.debug(f"VSA skipped: seq_len={seq_len} < block_elements={block_elements}")
            return self._call_original_forward(*args, **kwargs)

        # === Resolve video_shape ===
        video_shape = self._resolve_video_shape(seq_len)
        if video_shape is None:
            logger.debug(f"VSA skipped: no matching video_shape for seq_len={seq_len}")
            return self._call_original_forward(*args, **kwargs)

        # === Compute gate_compress ===
        gate_compress = None
        if hasattr(self, "to_gate_compress"):
            gate_compress = self.to_gate_compress(x)

        # === Reshape for VSA: [batch, seq, hidden] -> [batch, heads, seq, head_dim] ===
        query = self._reshape_for_vsa(query, self.heads)
        key = self._reshape_for_vsa(key, self.heads)
        value = self._reshape_for_vsa(value, self.heads)
        if gate_compress is not None:
            gate_compress = self._reshape_for_vsa(gate_compress, self.heads)

        # === Call VSA forward_attention directly ===
        output, stats = method.forward_attention(  # type: ignore[attr-defined]
            query=query,
            key=key,
            value=value,
            gate_compress=gate_compress,
            video_shape=video_shape,
        )

        # Store stats for collection
        method._last_stats = stats

        # === Reshape output: [batch, heads, seq, head_dim] -> [batch, seq, hidden] ===
        output = self._reshape_from_vsa(output)

        # === Apply output projection ===
        if hasattr(self, "to_out"):
            output = self.to_out(output)

        return output

    def _call_original_forward(self, *args, **kwargs):
        """Call the original module's forward method, bypassing VSA.

        Temporarily disables sparse attention so SparseAttentionModule.forward()
        passes through to the original module.
        """
        # Temporarily disable sparse attention to bypass sparse logic
        # SparseAttentionModule.forward() checks is_enabled and passes through if False
        was_enabled = getattr(self, "_enabled", True)
        self._enabled = False
        try:
            # This goes through SparseAttentionModule.forward() which checks is_enabled,
            # sees it's disabled, and calls DynamicModule.forward() -> original module
            result = SparseAttentionModule.forward(self, *args, **kwargs)
        finally:
            self._enabled = was_enabled
        return result

    def get_gate_compress_parameters(self):
        """Get trainable gate_compress parameters.

        Returns:
            Iterator of gate_compress parameters for optimization.
        """
        if hasattr(self, "to_gate_compress"):
            return self.to_gate_compress.parameters()
        return iter([])  # Empty iterator


def register_ltx2_attention(model: nn.Module) -> int:
    """Register LTX-2 Attention modules for VSA wrapping.

    This function detects LTX-2 Attention modules and registers them with
    the SparseAttentionRegistry. It also handles unregistering any generic
    wrappers that may have been registered first.

    Args:
        model: LTX-2 model to process.

    Returns:
        Number of module types registered.
    """
    if not _is_ltx2_model(model):
        return 0

    registered_types = set()
    num_modules = 0

    for name, module in model.named_modules():
        if not _is_ltx2_attention_module(module, name):
            continue

        num_modules += 1
        module_type = type(module)

        if module_type in registered_types:
            continue

        # Unregister any existing generic wrapper
        if module_type in SparseAttentionRegistry:
            logger.debug(f"Unregistering generic wrapper for {module_type.__name__}")
            SparseAttentionRegistry.unregister(module_type)

        # Register LTX-2 specific wrapper
        SparseAttentionRegistry.register({module_type: module_type.__name__})(_LTX2SparseAttention)
        registered_types.add(module_type)
        logger.info(f"Registered LTX-2 attention: {module_type.__name__}")

    if num_modules > 0:
        logger.info(f"Found {num_modules} LTX-2 Attention modules in model")

        # Store a weak reference to the root model on each attention module so
        # _resolve_video_shape() can read model._vsa_video_shape without globals.
        # Using weakref avoids circular module registration (nn.Module.__setattr__
        # would register a plain Module reference as a submodule, causing infinite
        # recursion in named_children()).
        root_ref = weakref.ref(model)
        for _, module in model.named_modules():
            if _is_ltx2_attention_module(module):
                object.__setattr__(module, "_vsa_root_model_ref", root_ref)

        # Register forward pre-hook to extract video_shape from Modality.positions
        # before each forward pass -- analogous to FastVideo's
        # set_forward_context(attn_metadata=builder.build(...))
        model.register_forward_pre_hook(_extract_video_shape_hook)
        logger.debug("Registered VSA video_shape extraction hook on model")

    return len(registered_types)


def register_ltx2_on_the_fly(model: nn.Module) -> bool:
    """Plugin entry point for LTX-2 VSA registration.

    Args:
        model: PyTorch model to process.

    Returns:
        True if any LTX-2 modules were registered.
    """
    num_registered = register_ltx2_attention(model)

    if num_registered > 0:
        logger.info(f"Registered {num_registered} LTX-2 attention types for VSA")
        return True

    return False


# Add to plugin set (order-independent: guards against re-registration internally)
CUSTOM_MODEL_PLUGINS.add(register_ltx2_on_the_fly)
