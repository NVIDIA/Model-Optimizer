# Adapted from https://github.com/sgl-project/SpecForge/blob/8ea5ca6/specforge/core/dflash.py
# Copyright (c) 2025 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""DFlash speculative decoding plugin for HuggingFace models.

Architecture:
- Feature Fusion: multi-layer target hidden states → FC + RMSNorm
- KV Injection: fused features as K/V in every draft layer with QK-norm
- Parallel Drafting: mask_token_id for unknown positions, bidirectional within blocks
- Random anchor sampling with exponential loss decay
- Logit distillation from target model

Reference: "DFlash: Block Diffusion for Flash Speculative Decoding" (arXiv:2602.06036)

Draft model components:
    The draft model currently uses Qwen3 components (MLP, RMSNorm, RotaryEmbedding)
    from ``transformers.models.qwen3``, matching z-lab's reference checkpoint format.
    Qwen3 sliding window attention is supported via ``config.layer_types``.
    The draft architecture is independent of the target model — any target model can
    be used as long as it provides hidden states.

    To add support for other draft architectures:

    Qwen3MoE (MoE MLP):
        1. Import ``Qwen3MoeMLP`` from ``transformers.models.qwen3_moe``
        2. Add a config flag (e.g., ``use_moe``) in ``dflash_architecture_config``
        3. In ``DFlashDecoderLayer.__init__``, select MLP based on the flag
        RMSNorm, RotaryEmbedding, and attention are shared across Qwen3 variants.

    MLA (Multi-head Latent Attention, e.g., DeepseekV3/Kimi-K2):
        MLA compresses K/V into a low-rank latent space. To support MLA in DFlash:
        1. Replace ``DFlashAttention`` with an MLA-aware variant that handles
           compressed KV injection (project target_hidden through MLA's down/up
           projections before concatenating with noise K/V)
        2. Handle lazy rope initialization (see ``_setup_kimi_k2_decoder`` in
           ``modelopt.torch.speculative.utils`` for the EAGLE3 approach)
        3. The ``_apply`` meta buffer fix in ``DFlashModule`` already handles the
           lazy rope pattern needed for MLA models.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as _Qwen3Config
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..dflash.conversion import DFlashDMRegistry
from ..dflash.dflash_model import DFlashModel
from .modeling_dflash import (  # noqa: F401
    DFlashAttention,
    DFlashBaseModelOutput,
    DFlashModule,
    build_target_layer_ids,
)
from .modeling_fakebase import (
    _BASE_MODEL_PATHS,
    _EMBED_TOKENS_PATHS,
    _FINAL_NORM_PATHS,
    _LM_HEAD_PATHS,
)

logger = logging.getLogger(__name__)

__all__ = ["HFDFlashModel"]


_QWEN3_VL_MROPE_WORKAROUND_VERSION = "5.3.0"
_MULTIMODAL_FORWARD_KWARGS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "mm_token_type_ids",
        "image_sizes",
        "images",
        "videos",
    }
)


def _multimodal_forward_kwargs(model_kwargs: dict) -> dict:
    """Return collator fields accepted by Hugging Face multimodal forwards."""
    return {
        name: value
        for name, value in model_kwargs.items()
        if name in _MULTIMODAL_FORWARD_KWARGS and value is not None
    }


def _expand_qwen3_video_grid_thw(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """Return the per-frame video grid representation used by Qwen3-VL RoPE.

    Qwen3-VL's video processor emits one ``[T, H, W]`` row per source video, but
    its rendered prompt contains a separate visual-token group for every temporal
    frame.  Transformers 5.3's ``get_rope_index`` consumes one grid row per
    rendered group, while the vision encoder still requires the original one-row-
    per-video representation.  This helper is therefore used *only* for mRoPE
    position construction; callers must keep the original tensor for the model
    forward.
    """
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
        raise ValueError(
            "Qwen3-VL video_grid_thw must have shape [num_videos, 3], got "
            f"{tuple(video_grid_thw.shape)}."
        )
    if torch.any(video_grid_thw[:, 0] <= 0):
        raise ValueError("Qwen3-VL video_grid_thw temporal lengths must be positive.")

    expanded_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    expanded_grid_thw[:, 0] = 1
    return expanded_grid_thw


def _dpace_position_weights(
    confidences: torch.Tensor, alpha: float, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute detached D-PACE per-position weights from draft confidences.

    Derived from D-PACE (arXiv:2605.18810). The paper factorizes the per-position
    weight (Fig. 2 / Eq. 8) into a *cumulative confidence* times a *continuation
    value*, which is equivalently the suffix sum of the cumulative confidences::

        C_j = prod_{i<=j} q~_i                 # cumulative confidence (Eq. 8)
        w_j = sum_{m>=j} C_m                    # = C_j * continuation value f~_j

    Each confidence is asymmetrically smoothed toward 1 (Eq. 7)::

        q~_i = (1 - alpha) * q_i + alpha,   alpha in (0, 1],

    so the floor ``q~_i >= alpha`` keeps every cumulative product (hence every
    weight) strictly positive. We evaluate the suffix sum from its definition as
    ``total - exclusive_prefix_sum`` of ``C`` rather than reversing the tensor.
    Positions with ``valid_mask == 0`` are multiplicative no-ops in ``C`` and
    contribute nothing to the sum, matching the per-token loss mask. Weights are
    detached (Eq. 9): they reweight the cross-entropy without adding gradient.

    Args:
        confidences: ``[..., L]`` draft confidence ``q_i = exp(-CE_i)`` per position.
        alpha: smoothing factor in (0, 1]; raises if outside that range.
        valid_mask: optional ``[..., L]`` 0/1 mask; ``None`` treats all positions valid.

    Returns:
        Detached weights with the same shape and dtype as ``confidences``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in (0, 1], got {alpha}")

    with torch.no_grad():
        smoothed = alpha + (1.0 - alpha) * confidences.float()  # Eq. 7
        if valid_mask is not None:
            keep = valid_mask.to(torch.bool)
            smoothed = torch.where(keep, smoothed, torch.ones_like(smoothed))
        cum_conf = torch.cumprod(smoothed, dim=-1)  # Eq. 8 cumulative confidence C_j
        if valid_mask is not None:
            cum_conf = cum_conf * keep.to(cum_conf.dtype)
        # Suffix sum w_j = sum_{m>=j} C_m, written as total minus the exclusive
        # prefix sum so no axis reversal is needed (Eq. 8).
        inclusive = torch.cumsum(cum_conf, dim=-1)
        weights = inclusive[..., -1:] - inclusive + cum_conf
        return weights.to(dtype=confidences.dtype)


@DFlashDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})

def _top1_id_and_prob(logits):
    """Argmax token id and its softmax probability, without keeping the softmax around.

    ``max - logsumexp`` is the log-probability of the argmax, so the full probability
    vector is never materialised -- which matters because callers hand this multi-hundred
    -megabyte chunks of a 202k-wide vocabulary.
    """
    logits = logits.float()
    top_val, top_id = logits.max(dim=-1)
    return top_id, (top_val - torch.logsumexp(logits, dim=-1)).exp()


class HFDFlashModel(DFlashModel):
    """DFlash Model for HuggingFace transformers."""

    @property
    def _base_model(self):
        return self.get_submodule(self.base_model_path)

    @property
    def _base_model_embeddings(self):
        return self.get_submodule(self.base_model_embeddings_path)

    @property
    def _base_model_lm_head(self):
        return self.get_submodule(self.base_model_lm_head_path)

    @property
    def _base_model_norm(self):
        """Base model's final pre-lm_head RMSNorm, or None if none was located.

        Applied before lm_head in the offline/streaming distillation path only when the
        producer captured a pre-norm hidden (base_hidden_prenorm), to reconstruct true logits.
        """
        path = getattr(self, "base_model_norm_path", None)
        return self.get_submodule(path) if path else None

    @property
    def _base_llm_config(self):
        return (
            getattr(self.config, "text_config", None)
            or getattr(self.config, "llm_config", None)
            or self.config
        )

    def _qwen3_vl_position_ids(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        model_kwargs,
    ):
        """Precompute Qwen3-VL mRoPE positions for Transformers 5.3.0 batches.

        The video encoder consumes one grid row per source video, whereas mRoPE
        consumes one row per rendered temporal-frame group.  Calling the
        top-level model with the original video grid makes the two contracts
        conflict.  Construct the mRoPE positions with a frame-expanded copy,
        then pass the original grid to the vision encoder in ``forward``.

        Transformers 5.4.0 performs this frame expansion in ``get_rope_index``
        itself; only 5.3.0 needs the external workaround. See
        https://github.com/huggingface/transformers/blob/v5.4.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py

        Prefer ``get_rope_index`` over ``compute_3d_position_ids``. The latter
        writes ``rope_deltas`` into the base model even though DFlash training
        never supplies a cache.  Keeping this calculation side-effect free is
        important when the frozen target is reused for consecutive training
        batches or validation.
        """
        model_type = str(getattr(self.config, "model_type", ""))
        if (
            position_ids is not None
            or not model_type.startswith("qwen3_vl")
            # Cached decoding uses the base model's rope_deltas path.  DFlash
            # training has no cache and is the only path that needs the
            # frame-expanded construction below.
            or past_key_values is not None
        ):
            return position_ids

        image_grid_thw = model_kwargs.get("image_grid_thw")
        video_grid_thw = model_kwargs.get("video_grid_thw")
        if not isinstance(image_grid_thw, torch.Tensor) and not isinstance(
            video_grid_thw, torch.Tensor
        ):
            return position_ids

        if transformers.__version__ != _QWEN3_VL_MROPE_WORKAROUND_VERSION:
            if transformers.__version__.startswith("5.3."):
                raise RuntimeError(
                    "Qwen3-VL DFlash mRoPE supports Transformers 5.3.0 or >=5.4.0; "
                    f"got {transformers.__version__}. A 5.3.x patch release may already "
                    "expand video_grid_thw internally."
                )
            return position_ids

        mm_token_type_ids = model_kwargs.get("mm_token_type_ids")
        backbone = getattr(self, "model", None)
        # Probed dynamically: which one exists depends on the Transformers version.
        get_rope_index: Any = getattr(backbone, "get_rope_index", None)
        compute_position_ids: Any = getattr(backbone, "compute_3d_position_ids", None)
        if (
            not isinstance(mm_token_type_ids, torch.Tensor)
            or input_ids is None
            or (not callable(get_rope_index) and not callable(compute_position_ids))
        ):
            raise ValueError(
                "Qwen3-VL DFlash training requires input_ids, mm_token_type_ids, and "
                "a Qwen3-VL model with get_rope_index or compute_3d_position_ids. "
                "Use the Qwen3-VL AutoProcessor without dropping mm_token_type_ids."
            )

        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "Qwen3-VL mm_token_type_ids must have the same shape as input_ids, got "
                f"{tuple(mm_token_type_ids.shape)} and {tuple(input_ids.shape)}."
            )

        rope_video_grid_thw = video_grid_thw
        if isinstance(video_grid_thw, torch.Tensor) and video_grid_thw.numel() > 0:
            video_token_mask = mm_token_type_ids == 2
            if isinstance(attention_mask, torch.Tensor):
                video_token_mask = video_token_mask & attention_mask.bool()
            video_group_starts = video_token_mask.clone()
            video_group_starts[:, 1:] &= ~video_token_mask[:, :-1]
            expected_video_groups = int(video_grid_thw[:, 0].sum())
            actual_video_groups = int(video_group_starts.sum())
            if actual_video_groups != expected_video_groups:
                raise ValueError(
                    "Qwen3-VL video frame groups do not match video_grid_thw: "
                    f"expected {expected_video_groups}, found {actual_video_groups}."
                )
            rope_video_grid_thw = _expand_qwen3_video_grid_thw(video_grid_thw)

        rope_kwargs = {
            "input_ids": input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": rope_video_grid_thw,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        if callable(get_rope_index):
            position_ids, _ = get_rope_index(**rope_kwargs)
        else:
            position_ids = compute_position_ids(
                **rope_kwargs,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
            )

        expected_shape = (3, *input_ids.shape)
        valid_position_ids = isinstance(position_ids, torch.Tensor) and (
            tuple(position_ids.shape) == expected_shape
        )
        if not valid_position_ids:
            raise RuntimeError(
                "Qwen3-VL produced invalid mRoPE position ids: expected shape "
                f"{expected_shape}, got {getattr(position_ids, 'shape', None)}."
            )
        return position_ids

    def _find_base_model_parts(self):
        """Locate base model submodules (backbone, embeddings, lm_head) by probing known paths.

        Reuses the shared path constants from modeling_fakebase (same as EAGLE).
        """
        for name, paths in {
            "base_model_path": _BASE_MODEL_PATHS,
            "base_model_embeddings_path": _EMBED_TOKENS_PATHS,
            "base_model_lm_head_path": _LM_HEAD_PATHS,
        }.items():
            for path in paths:
                try:
                    submodule = self.get_submodule(path)
                    assert isinstance(submodule, torch.nn.Module)
                    setattr(self, name, path)
                    break
                except Exception:
                    continue
            else:
                raise ValueError(f"Part {name} not found in model")
        # Final pre-lm_head norm is OPTIONAL (set None if absent): used to re-normalize the
        # un-normed final hidden collect by vllm.
        self.base_model_norm_path = None
        for path in _FINAL_NORM_PATHS:
            try:
                assert isinstance(self.get_submodule(path), torch.nn.Module)
                self.base_model_norm_path = path
                break
            except Exception:
                continue

    def modify(self, config):
        """Initialize DFlash draft module."""
        super().modify(config)

        self.dflash_fp32_master_weights = getattr(config, "dflash_fp32_master_weights", False)
        self.dflash_report_teacher_metrics = getattr(
            config, "dflash_report_teacher_metrics", True
        )
        # Extra per-step scalars handed to the trainer's logging callback.
        self._dflash_metrics = {}

        base_config = self._base_llm_config
        # Use Qwen3Config (not generic PretrainedConfig) so rope_parameters is
        # auto-populated from rope_theta. DFlash draft uses Qwen3 components.
        self.dflash_config = _Qwen3Config(**config.dflash_architecture_config)

        # hidden_size and vocab_size MUST match the base model.
        self.dflash_config.hidden_size = base_config.hidden_size
        self.dflash_config.vocab_size = base_config.vocab_size

        # Inherit architecture settings from base model when not specified by user
        # (setdefault). Static defaults (hidden_act, attention_bias, etc.) are in
        # dflash/default_config.py.
        _setdefault_attrs = [
            "max_position_embeddings",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
        ]
        for attr in _setdefault_attrs:
            if not hasattr(self.dflash_config, attr) or getattr(self.dflash_config, attr) is None:
                if hasattr(base_config, attr):
                    setattr(self.dflash_config, attr, getattr(base_config, attr))

        # RoPE base settings are ENFORCED to match the base model (not setdefault): the
        # DFlash draft injects the target's KV into every layer, so its RoPE base must
        # match the target's for the injected positions to align — and the exporter writes
        # the base model's rope_theta. Letting dflash_architecture_config override these
        # would make training (draft rope) and inference (base rope) disagree, so we
        # overwrite any user value and warn. (rope_scaling is intentionally NOT inherited:
        # DFlash uses standard Qwen3 RotaryEmbedding; the long-context YaRN scaling is
        # added only at export via dflash_export_rope_scaling.)
        # A config can carry BOTH a top-level rope_theta and a rope_parameters dict
        # with different values: Transformers 5 keeps the real base in
        # rope_parameters while the class default (10000.0 for Qwen3) stays visible
        # as rope_theta. rope_parameters wins, otherwise the draft trains against a
        # RoPE base 100x off the target's.
        base_rope_params = getattr(base_config, "rope_parameters", None)
        if not isinstance(base_rope_params, dict):
            base_rope_params = {}
        # Only rope_theta is taken from the dict. rope_parameters also carries the target's
        # scaling family and that family's own fields, and copying rope_type without them
        # builds a draft whose rotary init function looks up keys the draft config has not
        # got -- a llama3 or yarn target would fail in convert() with a bare KeyError. The
        # scaling family is deliberately not inherited (see above); it is added at export
        # via dflash_export_rope_scaling.
        for attr in ("rope_theta", "rope_type", "rope_interleaved"):
            if attr == "rope_theta" and attr in base_rope_params:
                base_val = base_rope_params[attr]
            elif hasattr(base_config, attr):
                base_val = getattr(base_config, attr)
            else:
                continue
            user_val = getattr(self.dflash_config, attr, None)
            if user_val is not None and user_val != base_val:
                logger.warning(
                    "DFlash: ignoring dflash_architecture_config.%s=%r and enforcing the "
                    "base model's value %r — the draft injects the target's KV, so its RoPE "
                    "base must match the target's.",
                    attr,
                    user_val,
                    base_val,
                )
            setattr(self.dflash_config, attr, base_val)
            # Qwen3Config populates rope_parameters at construction, so a later
            # setattr on the flat field alone would leave the dict — which is what
            # the rotary module reads — holding the stale value.
            draft_rope_params = getattr(self.dflash_config, "rope_parameters", None)
            if isinstance(draft_rope_params, dict) and attr in draft_rope_params:
                draft_rope_params[attr] = base_val

        self.dflash_config.head_dim = getattr(
            self.dflash_config,
            "head_dim",
            self.dflash_config.hidden_size // self.dflash_config.num_attention_heads,
        )
        self.dflash_config.block_size = self.dflash_block_size
        # On the draft config so _build_draft_module stays a pure function of it.
        self.dflash_config.attention_sink_bias = self.dflash_attention_sink

        # Which base layers feed the draft's `fc`: explicit ids win, else the uniform default.
        num_target_layers = (
            base_config.num_orig_hidden_layers
            if self.dflash_offline
            else base_config.num_hidden_layers
        )
        num_draft_layers = self.dflash_config.num_hidden_layers
        user_target_layer_ids = config.dflash_architecture_config.get("target_layer_ids")
        if user_target_layer_ids:
            if len(user_target_layer_ids) != num_draft_layers:
                raise ValueError(
                    f"dflash_architecture_config.target_layer_ids has "
                    f"{len(user_target_layer_ids)} entries but the draft has "
                    f"{num_draft_layers} layers; one target layer per draft layer is required."
                )
            if max(user_target_layer_ids) >= num_target_layers:
                raise ValueError(
                    f"dflash_architecture_config.target_layer_ids {user_target_layer_ids} "
                    f"references a layer beyond the base model's {num_target_layers} layers."
                )
            if len(set(user_target_layer_ids)) != len(user_target_layer_ids):
                raise ValueError(
                    f"dflash_architecture_config.target_layer_ids {user_target_layer_ids} "
                    "contains duplicates; each draft layer needs a distinct capture layer "
                    "(the streaming producer captures each base layer at most once)."
                )
            # forward() indexes hidden_states[lid + 1], where a negative id silently wraps.
            if min(user_target_layer_ids) < 0:
                raise ValueError(
                    f"dflash_architecture_config.target_layer_ids {user_target_layer_ids} "
                    "must be non-negative base-layer indices."
                )
            self.target_layer_ids = list(user_target_layer_ids)
            logger.info("DFlash: using explicit target_layer_ids %s", self.target_layer_ids)
        else:
            self.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)
        self.dflash_config.target_layer_ids = self.target_layer_ids

        # mask_token_id: validated by DFlashConfig, auto-detected from tokenizer context
        self.mask_token_id = config.dflash_mask_token_id
        logger.info("DFlash mask_token_id: %s", self.mask_token_id)

        # Freeze base model
        if self.dflash_freeze_base_model:
            for param in self.parameters():
                param.requires_grad = False

        self._find_base_model_parts()

        # Factory hook: subclasses (e.g. Domino) override to build an augmented
        # draft module while reusing all of DFlash's modify() setup.
        self.dflash_module = self._build_draft_module(self.dflash_config)
        # Warm start from an exported draft checkpoint, before the dtype/device move below
        # so the loaded tensors get cast alongside the rest of the module.
        if self.dflash_init_checkpoint:
            self._load_init_checkpoint(self.dflash_init_checkpoint)
        # Match base model dtype/device. Skipped when the base is on meta, which is the
        # from_pretrained restore path: the weights are not loaded yet and the draft cannot be
        # placed. `restore_draft_precision` is what picks it up again there.
        base_device = self._base_device()
        if base_device.type != "meta":
            self._place_draft(base_device)

        # Delete base model layers for offline training (save memory)
        if self.dflash_offline:
            self._base_model._modules.pop("layers")

        self.is_quantized = False
        self._num_anchors = self.dflash_num_anchors
        # Opt-in Inductor fusion of the draft stack; see DFlashModule._body for why this is
        # only affordable now that the block count is static.
        self.dflash_module._dflash_compile_stack = bool(
            getattr(self, "dflash_use_torch_compile", False)
        )

    def _build_draft_module(self, dflash_config):
        """Build the draft module. Subclasses override to use an augmented module."""
        return DFlashModule(dflash_config)

    def _base_device(self):
        """Device the frozen base model sits on. ``meta`` during a from_pretrained restore."""
        if self.dflash_offline:
            return self._base_model_lm_head.weight.device
        return next(self._base_model.layers[-1].parameters()).device

    def _place_draft(self, base_device):
        """Put the draft on its training device and dtype and build its rotary buffer.

        The dtype has to be settled before the Trainer builds the optimizer, because AdamW
        allocates its moments with ``zeros_like(p)``: a cast afterwards would fix the
        parameters and leave the moments in the base model's dtype, which is the half that
        cannot represent its own updates. Only the draft takes fp32 under
        ``dflash_fp32_master_weights`` -- the frozen base has no trainable parameters and no
        optimizer state, so promoting it would double its memory for nothing and change the
        hidden states the draft is trained against.

        The rotary buffer is built here rather than on the draft's first forward.
        ``_maybe_init_rotary_emb`` creates the non-persistent ``inv_freq`` lazily, and the
        DFlash forward returns early -- without running the draft -- for a rank whose batch
        has no valid anchor. That rank ends the step one buffer short, and DDP's
        ``broadcast_buffers`` then coalesces buffer lists of differing flattened size across
        ranks, which hangs rather than raising. Building it up front makes every rank's
        buffer list identical for the whole run. Numerically inert: ``inv_freq`` is a pure
        function of the config and, being non-persistent, is absent from the state dict
        either way.

        Idempotent, so the resume path can call it again through
        ``restore_draft_precision``.
        """
        draft_dtype = torch.float32 if self.dflash_fp32_master_weights else self._base_model.dtype
        self.dflash_module.to(device=base_device, dtype=draft_dtype)
        # Logged, not assumed: the optimizer dtype this decides is not visible until step 1.
        logger.info(
            "DFlash draft on %s in %s (dflash_fp32_master_weights=%s); Adam moments will "
            "follow. Frozen base left at %s.",
            base_device,
            draft_dtype,
            self.dflash_fp32_master_weights,
            self._base_model.dtype,
        )
        self.dflash_module._maybe_init_rotary_emb(device=base_device)

    def _require_autocast_for_promoted_draft(self, device):
        """Fail early, and by name, when a promoted draft is about to run unautocast.

        ``dflash_fp32_master_weights`` needs a bf16 autocast around the forward, and HF
        ``Trainer`` only wraps ``forward``. AR validation reaches the draft through
        ``pseudo_speculative_generate``, which is called directly, so it runs outside that
        wrapper: with ``estimate_ar: true`` a run trains normally and then dies at the first
        ``ar_validate_steps`` boundary on a bare matmul dtype mismatch, possibly hours in.
        Name the two knobs instead of letting ``F.linear`` report it.
        """
        if not self.dflash_fp32_master_weights or self._base_model.dtype == torch.float32:
            return
        if torch.is_autocast_enabled(device.type):
            return
        raise RuntimeError(
            f"DFlash: dflash_fp32_master_weights holds the draft in fp32 while the frozen "
            f"base is {self._base_model.dtype}, and this path runs outside the autocast that "
            f"reconciles them, so the draft's first matmul would fail. Wrap the call in "
            f"torch.autocast(device_type={device.type!r}, dtype={self._base_model.dtype}), "
            f"or set estimate_ar=false, or set dflash_fp32_master_weights=false."
        )

    def restore_draft_precision(self, checkpoint_dir=None):
        """Re-apply the draft's device, dtype and rotary buffer after a checkpoint restore.

        ``modify()`` runs during ``from_pretrained`` with the base model still on meta, so it
        skips all three. Left alone, a resumed run keeps whatever dtype the checkpoint loaded
        at, AdamW allocates its moments to match, and ``dflash_fp32_master_weights`` is
        silently inert for the rest of the run -- the resumed half of a long training job
        quietly loses the feature. Call this once the weights are loaded and before the
        Trainer builds the optimizer. A no-op on a freshly converted model, which did all
        three in ``modify()``.

        ``checkpoint_dir`` additionally restores the precision the draft was *saved* at.
        ``from_pretrained(dtype="auto")`` gives every tensor a single dtype -- the base
        model's -- which discards the extra mantissa bits an fp32 draft wrote to disk.
        Reloading those tensors at their stored dtype is the only way to get them back, and
        without it a resume silently costs the run one rounding of its master weights. Pass
        it only for an HF-format checkpoint; it raises if there are no safetensors there,
        which is better than silently keeping the wrong precision.
        """
        base_device = self._base_device()
        if base_device.type == "meta":
            return
        self._place_draft(base_device)
        if checkpoint_dir is not None:
            self._reload_draft_weights_at_stored_precision(checkpoint_dir)

    def _reload_draft_weights_at_stored_precision(self, checkpoint_dir):
        """Copy the draft's tensors back out of the checkpoint at the dtype they were saved in."""
        # Imported here rather than at module scope: that module pulls in accelerate,
        # huggingface_hub and torch.distributed.tensor.
        from modelopt.torch.utils.plugins.model_load_utils import (
            read_safetensors_subset,
            weight_map_for,
        )

        prefix = "dflash_module."
        stored = read_safetensors_subset(
            checkpoint_dir, weight_map_for(checkpoint_dir), lambda k: k.startswith(prefix)
        )
        own = dict(self.dflash_module.named_parameters())
        with torch.no_grad():
            for key, saved in stored.items():
                param = own.get(key[len(prefix) :])
                if param is not None and param.shape == saved.shape:
                    param.copy_(saved.to(param.dtype))
        logger.info(
            "DFlash draft precision restore: %d/%d tensors reloaded from %s.",
            len(stored),
            len(own),
            checkpoint_dir,
        )

    # Draft-module entries that legitimately come from the base model rather than the
    # exported draft checkpoint, so their absence (or presence) is not an error.
    _INIT_CKPT_IGNORED_KEYS = ("embed_tokens.weight", "lm_head.weight")

    def _load_init_checkpoint(self, path: str):
        """Warm-start ``self.dflash_module`` from an exported draft checkpoint.

        Accepts either the export directory (containing ``model.safetensors``) or the
        safetensors file itself. The architecture is fixed by ``dflash_architecture_config``
        at this point, so the checkpoint has to match it: any missing, unexpected, or
        wrong-shaped tensor raises. Loading part of a draft and leaving the rest randomly
        initialized looks like a warm start but trains from a corrupted starting point, so
        it is rejected instead of warned about.
        """
        from safetensors.torch import load_file

        ckpt = Path(path)
        if ckpt.is_dir():
            ckpt = ckpt / "model.safetensors"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"dflash_init_checkpoint: no draft weights at {ckpt}. Expected an exported "
                "draft directory containing model.safetensors, or the file itself."
            )

        state_dict = load_file(str(ckpt))
        # Tolerate a `dflash_module.` prefix so a raw training checkpoint also works.
        state_dict = {
            (k.split("dflash_module.", 1)[1] if "dflash_module." in k else k): v
            for k, v in state_dict.items()
        }
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k not in self._INIT_CKPT_IGNORED_KEYS and "rotary_emb" not in k
        }

        # Shape-check against the module's own view of each key. Subclasses may remap keys
        # on load (DSpark accepts a nested ``markov_head.`` layout), so resolve through the
        # same hooks first — otherwise a wrong-shaped remapped tensor would skip this check
        # and fail later with a much less obvious error.
        module_sd = self.dflash_module.state_dict()
        resolved = dict(state_dict)
        for hook in self.dflash_module._load_state_dict_pre_hooks.values():
            hook(resolved, "", None, True, [], [], [])
        mismatched = [
            f"{k}: checkpoint {tuple(v.shape)} vs module {tuple(module_sd[k].shape)}"
            for k, v in resolved.items()
            if k in module_sd and v.shape != module_sd[k].shape
        ]
        if mismatched:
            raise ValueError(
                "dflash_init_checkpoint: shape mismatch between "
                f"{ckpt} and the configured draft architecture:\n  " + "\n  ".join(mismatched)
            )

        # strict=False, then check by hand: the module's own load hooks (e.g. DSpark's
        # markov_head remap) run first, and buffers such as rotary_emb are excluded above.
        incompatible = self.dflash_module.load_state_dict(state_dict, strict=False)
        missing = [
            k
            for k in incompatible.missing_keys
            if "rotary_emb" not in k and k not in self._INIT_CKPT_IGNORED_KEYS
        ]
        if missing or incompatible.unexpected_keys:
            raise ValueError(
                f"dflash_init_checkpoint: {ckpt} does not match the configured draft "
                "architecture.\n"
                f"  missing from checkpoint: {sorted(missing)}\n"
                f"  unexpected in checkpoint: {sorted(incompatible.unexpected_keys)}"
            )
        logger.info(
            "DFlash: warm-started draft module from %s (%d tensors).", ckpt, len(state_dict)
        )

    def get_exporter(self):
        """Get the exporter for the DFlash draft model."""
        from modelopt.torch.export.plugins.hf_spec_export import DFlashExporter

        return DFlashExporter(self)

    def _sample_anchor_positions(self, seq_len, loss_mask, device):
        """Randomly sample anchor positions per sample.

        Returns (anchor_positions [B, N], block_keep_mask [B, N]).

        ``N`` is fixed by the config and the sequence length, never by the batch
        contents. It used to be ``min(num_anchors, valid_counts.max() - 1)``, i.e. a
        function of the longest answer in the batch. ``n_blocks`` sets the draft's
        ``q_len`` and ``kv_len``, and both the block-mask builder and the attention are
        ``torch.compile(..., dynamic=False)``, so every distinct value cost a fresh
        170-300 s recompile -- and new values kept appearing indefinitely, since any
        batch whose answers are all shorter than the cap mints one. A 3-node production
        run spent 1360 of its first 1510 training seconds frozen in recompilation
        against a 0.6 s/step steady state.

        Which anchors get sampled is unchanged: ``cap`` below still applies the old
        data-dependent bound, just on-device and before the sort rather than as a slice
        width. The surplus columns carry ``keep=False``, which every consumer already
        handles because rows shorter than the batch maximum have always produced them:
        the losses normalize by the weight sum derived from ``block_keep_mask``, and
        ``mask_mod`` ands the same flag in, so FlexAttention skips those tiles instead
        of computing them.

        TODO: Fix the random seed per epoch (change between epochs) so that anchor
        positions are deterministic within an epoch. This would allow caching the derived
        masks and position IDs across steps while preserving the same data augmentation
        effect. Currently, anchors are re-sampled every forward pass.
        """
        bs = self.dflash_block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)
        num_anchors = getattr(self, "_num_anchors", 512)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)

        # Static. Bounded by max_anchor + 1 as well as num_anchors so that short sequences
        # (unit tests, short-context recipes) cannot ask for more columns than exist.
        max_n = min(num_anchors, max_anchor + 1)

        # The bound the old code computed, kept on-device so the shapes above stay static
        # and the .item() sync is gone. valid_counts <= max_anchor + 1 by construction, so
        # clamping to max_n leaves this equal to min(num_anchors, valid_counts.max() - 1).
        cap = (valid_counts.max() - 1).clamp(min=0, max=max_n)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        fill = torch.tensor(seq_len + 1, device=device)
        masked_indices = torch.where(valid, indices, fill)

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Blank past `cap` BEFORE sorting, not after. Sorting a wider slice would pull
        # anchors from beyond the old bound into the front of the row and silently change
        # which positions are trained on; blanking first reproduces the old
        # ``gathered[:, :cap].sort()`` exactly and leaves the surplus as fill.
        cols = torch.arange(max_n, device=device).unsqueeze(0)
        anchors = torch.where(cols < cap, gathered[:, :max_n], fill).sort(dim=1).values

        keep = cols < torch.minimum(valid_counts, cap).unsqueeze(1)
        anchors = torch.where(keep, anchors, torch.tensor(0, dtype=torch.long, device=device))
        return anchors, keep

    def _build_noise_embedding(self, input_ids, anchor_positions, block_keep_mask, n_blocks):
        """Build noise embeddings: anchor token at block start, mask_token elsewhere."""
        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n_blocks * block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_starts = torch.arange(n_blocks, device=device) * block_size
        block_starts_exp = block_starts.unsqueeze(0).expand(bsz, -1)
        valid_anchors = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchors)
        batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n_blocks)
        noise_ids[batch_idx, block_starts_exp] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )
        return self._base_model_embeddings(noise_ids)

    def _build_position_ids(self, seq_len, anchor_positions, device):
        """Build position IDs: context [0..S-1], draft blocks [anchor+0..anchor+B-1]."""
        bsz = anchor_positions.shape[0]
        block_size = self.dflash_block_size

        ctx_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(block_size, device=device).view(1, 1, -1)
        draft_pos = (anchor_positions.unsqueeze(-1) + offsets).view(bsz, -1)
        return torch.cat([ctx_pos, draft_pos], dim=1)

    def _build_draft_attention_mask(
        self, seq_len, anchor_positions, block_keep_mask, n_blocks, dtype, device, window=None
    ):
        """Build SDPA attention mask: context (causal) + draft (per ``dflash_draft_attention``).

        When ``window`` is not None, all layers use sliding-window attention: each draft
        query only sees context positions within ``window`` tokens before its own position.
        Block-internal attention is left un-windowed (the config enforces
        ``window >= block_size``, so a full block always fits inside the window and windowing
        it would be a no-op).

        Block-internal visibility follows ``self.dflash_draft_attention``:
        ``"bidirectional"`` (default, MiMo-style) lets every query see the whole block, while
        ``"causal"`` restricts a query at block position ``i`` to draft positions ``<= i`` so
        the block is modelled autoregressively.
        """
        bsz = anchor_positions.shape[0]
        block_size = self.dflash_block_size
        q_len = n_blocks * block_size
        kv_len = seq_len + q_len

        # FlexAttention consumes the same predicate as a BlockMask and skips the ~67% of
        # tiles that are entirely masked; DFlashAttention routes a BlockMask to
        # flex_attention_forward. See dflash_flex_attention.py for why this matters so
        # much here (dense mask -> sm80 memory-efficient kernel -> 59% of the step).
        if getattr(self, "dflash_use_flex_attention", False):
            from .dflash_flex_attention import build_draft_block_mask

            return build_draft_block_mask(
                seq_len,
                anchor_positions,
                block_keep_mask,
                n_blocks,
                block_size,
                window,
                device,
                head_dim=self.dflash_module.layers[0].self_attn.head_dim,
            )

        q_indices = torch.arange(q_len, device=device).view(1, 1, -1, 1)
        kv_indices = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
        q_block_ids = q_indices // block_size

        anchor_exp = anchor_positions.view(bsz, 1, n_blocks, 1).repeat_interleave(block_size, dim=2)

        # Context: kv < S and kv < anchor
        mask_ctx = (kv_indices < seq_len) & (kv_indices < anchor_exp)

        # Sliding window on the context: keep only context kv whose real position is within
        # `window` tokens before the query's real position (anchor + position-in-block).
        if window is not None:
            q_real_pos = anchor_exp + (q_indices % block_size)  # [B, 1, q_len, 1]
            mask_ctx = mask_ctx & (kv_indices > q_real_pos - window)
        # Draft: kv >= S and same block
        is_draft = kv_indices >= seq_len
        kv_block_ids = (kv_indices - seq_len) // block_size
        mask_draft = is_draft & (q_block_ids == kv_block_ids)
        if self.dflash_draft_attention == "causal":
            # Autoregressive within the block: query at block position i sees draft
            # positions <= i only. Compare positions *within* the block so the term is
            # independent of which block the query belongs to.
            kv_pos_in_block = (kv_indices - seq_len) % block_size
            mask_draft = mask_draft & (kv_pos_in_block <= (q_indices % block_size))
        # Valid block
        valid_block = block_keep_mask.view(bsz, 1, n_blocks, 1).repeat_interleave(block_size, dim=2)

        final_mask = (mask_ctx | mask_draft) & valid_block  # [B, 1, Q, KV]

        # Convert bool mask to float additive mask for SDPA
        attn_mask = torch.zeros(bsz, 1, q_len, kv_len, device=device, dtype=dtype)
        attn_mask.masked_fill_(~final_mask, torch.finfo(dtype).min)
        return attn_mask

    def _build_generate_swa_mask(self, ctx_len, bsz, dtype, device):
        """Generation-time mask [B, 1, block_size, ctx_len + block_size], or None.

        Returns None only when there is nothing to mask: full attention over the context
        *and* bidirectional blocks (KV cache with no mask). With sliding-window attention,
        each block query only sees context within ``dflash_swa_window_size`` tokens before
        its real position (ctx_len + position-in-block), matching training and vLLM
        inference; block kv is left un-windowed. With ``dflash_draft_attention="causal"``
        the block is additionally lower-triangular, mirroring
        :meth:`_build_draft_attention_mask` so generation matches training.
        """
        window = self.dflash_swa_window_size
        causal = self.dflash_draft_attention == "causal"
        if window is None and not causal:
            return None
        block_size = self.dflash_block_size
        kv_len = ctx_len + block_size
        kv_idx = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
        q_pos_in_block = torch.arange(block_size, device=device).view(1, 1, -1, 1)
        q_real_pos = ctx_len + q_pos_in_block
        is_ctx = kv_idx < ctx_len
        # Context kv kept iff within the window (when windowing); block kv always visible
        # unless the block is causal, in which case only positions <= the query's.
        keep_ctx = is_ctx if window is None else (is_ctx & (kv_idx > q_real_pos - window))
        keep_block = ~is_ctx
        if causal:
            keep_block = keep_block & ((kv_idx - ctx_len) <= q_pos_in_block)
        keep = keep_ctx | keep_block
        attn_mask = torch.zeros(bsz, 1, block_size, kv_len, device=device, dtype=dtype)
        attn_mask.masked_fill_(~keep, torch.finfo(dtype).min)
        return attn_mask

    def _teacher_top1(self, base_logits, base_outputs, teacher_indices, chunk=256):
        """Teacher top-1 token id and its probability at the supervised block positions.

        Returns flat ``[B * N * block_size]`` tensors, or ``(None, None)`` when the base
        distribution is not available here.

        Two sources, in order: the full base logits when something already materialised
        them (self-logit-distillation), otherwise the post-final-norm base hidden, which
        streaming/offline training carries precisely so consumers can project only the
        rows they need. That second path is why this is not simply
        ``base_outputs.logits.softmax(-1)``: at PineDrift's 202048-token vocabulary a
        full-sequence logits tensor is ~10 GB and a gathered one ~6.6 GB, on top of the
        draft logits already resident, so the projection is done on gathered rows in
        chunks and reduced to two scalars per row immediately. The chunk is small on
        purpose: this runs beside draft logits that are already ~6.6 GB, and a near-OOM
        step on PyTorch's expandable segments silently retries instead of crashing, so
        it shows up as a slowdown rather than an error.
        """
        if not getattr(self, "dflash_report_teacher_metrics", False):
            return None, None

        _, n_blocks, _ = teacher_indices.shape
        full_logits = base_logits if base_logits is not None else getattr(base_outputs, "logits", None)
        if full_logits is not None:
            gathered = torch.gather(
                full_logits.unsqueeze(1).expand(-1, n_blocks, -1, -1),
                2,
                teacher_indices.unsqueeze(-1).expand(-1, -1, -1, full_logits.size(-1)),
            ).reshape(-1, full_logits.size(-1))
            return _top1_id_and_prob(gathered)

        base_hidden = getattr(base_outputs, "base_hidden", None)
        if base_hidden is None:
            return None, None
        rows = torch.gather(
            base_hidden.unsqueeze(1).expand(-1, n_blocks, -1, -1),
            2,
            teacher_indices.unsqueeze(-1).expand(-1, -1, -1, base_hidden.size(-1)),
        ).reshape(-1, base_hidden.size(-1))
        ids, probs = [], []
        for start in range(0, rows.shape[0], chunk):
            piece = self._base_model_lm_head(rows[start : start + chunk])
            piece_ids, piece_probs = _top1_id_and_prob(piece)
            ids.append(piece_ids)
            probs.append(piece_probs)
        return torch.cat(ids), torch.cat(probs)

    def _compute_loss(
        self,
        logits,
        input_ids,
        anchor_positions,
        block_keep_mask,
        loss_mask,
        base_logits=None,
        draft_hidden=None,
        base_outputs=None,
    ):
        """Compute weighted cross-entropy (or KD) loss and accuracy.

        Args:
            logits: Draft model output [B, N*block_size, vocab].
            input_ids: Original input token IDs [B, seq_len].
            anchor_positions: Anchor positions per block [B, N].
            block_keep_mask: Valid block mask [B, N].
            loss_mask: Token-level loss mask [B, seq_len].
            base_logits: Base model logits for KD loss [B, seq_len, vocab], or None for CE.
            draft_hidden: Draft hidden states [B, N*block_size, H] behind ``logits``.
                Unused here; passed for variants whose head consumes them.

        Returns:
            (loss, accuracy) tuple.
        """
        self._dflash_metrics = {}
        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        n_blocks = anchor_positions.shape[1]
        device = input_ids.device

        label_offsets = torch.arange(0, block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )

        # Weight mask: valid block * in bounds * exclude anchor (pos 0) * loss_mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, block_size).float()
        weight_mask = weight_mask * valid_label.float()
        pos_in_block = torch.arange(block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        orig_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        weight_mask = weight_mask * orig_loss_mask

        binary_eval_mask = weight_mask.view(-1)

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)

        # Non-KD loss is per-token cross-entropy; compute it once (grad enabled) so the
        # D-PACE confidences below can reuse it instead of a second CE pass. The KD path
        # (base_logits is not None) optimizes KL, so its confidences need a dedicated
        # no_grad CE pass.
        loss_per_token = None
        if base_logits is None:
            loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Block-position loss weighting: dynamic D-PACE weights or static exponential decay.
        if self.dflash_loss_objective == "dpace" and block_size > 1:
            # Draft confidence q_i = exp(-CE) on the target-selected token, over the
            # predicted positions (slot 0 is the given anchor, already masked above).
            # Weights are detached (paper Eq.9), so this adds the documented ~2.3%
            # training overhead without altering the cross-entropy gradient.
            with torch.no_grad():
                conf_ce = (
                    loss_per_token.detach()
                    if loss_per_token is not None
                    else F.cross_entropy(flat_logits, flat_targets, reduction="none")
                ).view(bsz, n_blocks, block_size)
                confidences = torch.exp(-conf_ce[..., 1:].float())
                dpace = torch.ones_like(weight_mask)
                dpace[..., 1:] = _dpace_position_weights(
                    confidences, self.dflash_dpace_alpha, valid_mask=weight_mask[..., 1:]
                )
            weight_mask = weight_mask * dpace
        elif self.dflash_loss_decay_factor > 0:
            k = torch.arange(block_size, device=device).view(1, 1, -1)
            decay = torch.exp(-(k - 1).clamp(min=0).float() / self.dflash_loss_decay_factor)
            weight_mask = weight_mask * decay

        flat_weights = weight_mask.view(-1)
        valid_count = flat_weights.sum() + 1e-6

        if valid_count > 1.0:
            if base_logits is not None:
                # KD loss: teacher logits for token anchor+k are at position anchor+k-1
                teacher_indices = (safe_label_indices - 1).clamp(min=0)
                teacher_logits = torch.gather(
                    base_logits.unsqueeze(1).expand(-1, n_blocks, -1, -1),
                    2,
                    teacher_indices.unsqueeze(-1).expand(-1, -1, -1, base_logits.size(-1)),
                )
                flat_teacher = teacher_logits.reshape(-1, base_logits.size(-1)).detach()
                target_soft = torch.softmax(flat_teacher, dim=-1)
                draft_logsoft = torch.log_softmax(flat_logits, dim=-1)
                kd_loss = -(target_soft * draft_logsoft).sum(dim=-1)
                loss = (kd_loss * flat_weights).sum() / valid_count
            else:
                loss = (loss_per_token * flat_weights).sum() / valid_count

            with torch.no_grad():
                preds = flat_logits.argmax(dim=-1)
                correct = (preds == flat_targets) & (binary_eval_mask > 0.5)
                # Per block POSITION, not just the mean over them. The mean is what
                # the drafter's own loss sees, but what decides accepted length is the
                # accuracy of positions 1, 2, 3 ... -- a mean of 0.10 is a healthy
                # 0.45-at-position-1 decaying away, or a flat 0.10 everywhere, and
                # those two mean completely different things. Slot 0 is the anchor
                # (already excluded by weight_mask), so it reports 0.
                c = correct.view(bsz, n_blocks, block_size).float().sum(dim=(0, 1))
                d = (binary_eval_mask > 0.5).view(bsz, n_blocks, block_size).float().sum(dim=(0, 1))
                # Drop slot 0: it is the anchor, always given, and the AR estimate in
                # eagle_utils.on_log chains these as a cumulative product -- a leading
                # 0.0 would zero the whole chain, and a leading 1.0 would look exactly
                # like the empty-loss-mask failure signature. So index 0 of the reported
                # vector is block position 1, the first token the draft has to guess.
                accuracy = (c / (d + 1e-6))[1:].tolist()

                # Teacher-side diagnostics over the same supervised positions. These
                # describe the TARGET model, not the draft: how often the teacher's own
                # top-1 IS the corpus's next token, and how peaked that top-1 is. They
                # bound what any draft can learn here -- a low match rate means the
                # corpus is off-distribution for the target, and a flat top-1 means the
                # target itself is uncertain there, so acceptance cannot be high no
                # matter how good the draft gets. Teacher logits for the token at label
                # index j sit at position j-1, the same alignment the KD term uses.
                t_ids, t_probs = self._teacher_top1(
                    base_logits, base_outputs, (safe_label_indices - 1).clamp(min=0)
                )
                if t_ids is not None:
                    keep = binary_eval_mask > 0.5
                    eval_count = keep.sum().float() + 1e-6
                    self._dflash_metrics["teacher_top1_match"] = (
                        ((t_ids == flat_targets) & keep).sum().float() / eval_count
                    ).item()
                    self._dflash_metrics["teacher_top1_confidence"] = (
                        (t_probs * keep.float()).sum() / eval_count
                    ).item()
        else:
            loss = flat_logits.sum() * 0.0
            accuracy = [0.0] * max(block_size - 1, 1)

        return loss, accuracy

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **kwargs,
    ):
        """Training forward with random anchor sampling.

        - Random anchor sampling instead of uniform block division
        - Bidirectional intra-block attention (no causal constraint)
        - Context sees strictly before anchor position
        - Label alignment: position k predicts token at anchor+k
        - Optional loss decay weighting
        """
        if self.training:
            position_ids = self._qwen3_vl_position_ids(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                kwargs,
            )

        if not self.training:
            if self.dflash_offline:
                raise RuntimeError(
                    "DFlash offline model cannot run eval/inference forward — base model "
                    "layers were deleted during offline conversion to save memory. "
                    "Reload the full base model before running evaluation."
                )
            # Don't pass labels to base model — DFlash uses unshifted labels
            # which are incompatible with the base model's shifted loss.
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, seq_len = input_ids.shape
        block_size = self.dflash_block_size
        device = input_ids.device

        if seq_len % block_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by block_size ({block_size}). "
                f"Adjust training_seq_len or use padding."
            )

        # 1. Run base model → extract target hidden states
        if self.dflash_offline:
            assert "base_model_outputs" in kwargs
            # For self-logit-distillation, from_offline_dict reconstructs base logits from the
            # captured hidden (final norm re-applied as needed) when the producer didn't supply
            # them, and raises if anything needed for that is missing.
            base_outputs = DFlashBaseModelOutput.from_offline_dict(
                kwargs["base_model_outputs"],
                self._base_model_norm,
                self._base_model_lm_head,
                # The teacher-side metrics need the base distribution too, but only at a
                # few thousand rows -- so ask for the post-norm base hidden and defer the
                # lm_head, instead of materialising [B, seq, vocab] logits nobody wants.
                need_logits=self.dflash_self_logit_distillation
                or getattr(self, "dflash_report_teacher_metrics", False),
                defer_lm_head=not self.dflash_self_logit_distillation,
            )
            target_hidden = base_outputs.target_hidden
        else:
            # Multimodal models need the top-level conditional-generation forward so their
            # image/video features are inserted before the language model runs.  Keep the
            # long-standing narrow call for text-only models.
            base_forward_kwargs = _multimodal_forward_kwargs(kwargs)
            use_top_level_forward = bool(base_forward_kwargs)
            with torch.no_grad():
                if use_top_level_forward:
                    raw_outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=False,
                        output_attentions=output_attentions,
                        output_hidden_states=True,
                        cache_position=cache_position,
                        return_dict=True,
                        **base_forward_kwargs,
                    )
                else:
                    raw_outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

            if not getattr(raw_outputs, "hidden_states", None):
                raise RuntimeError(
                    "The base model did not return hidden states required for DFlash training. "
                    "Ensure its top-level multimodal forward supports output_hidden_states=True."
                )
            offset = 1
            selected = [raw_outputs.hidden_states[lid + offset] for lid in self.target_layer_ids]
            target_hidden = torch.cat(selected, dim=-1)  # [B, seq, num_layers * H]
            base_outputs = DFlashBaseModelOutput(
                target_hidden=target_hidden, logits=raw_outputs.logits
            )

        # 2. Build loss mask. Labels carry optional answer-only masking, but do
        # not in general mark padded tokens with -100 (the VLM collator creates
        # them from padded input_ids). Always intersect with attention_mask so
        # anchor sampling and loss never include the padded tail.
        loss_mask = torch.ones(bsz, seq_len, device=device)
        if labels is not None:
            loss_mask = loss_mask * (labels != LabelSmoother.ignore_index).float()
        if attention_mask is not None:
            loss_mask = loss_mask * attention_mask.float()

        # In offline training, assistant mask is dumped and passed as kwarg.
        if kwargs.get("loss_mask") is not None:
            loss_mask = loss_mask * kwargs["loss_mask"]

        # 3. Random anchor sampling
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        if n_blocks == 0 or not block_keep_mask.any():
            # Keep all trainable draft parameters in the graph so DDP can reduce a rank
            # that receives an all-masked answer-only batch.
            dummy = sum(
                (
                    parameter.reshape(-1)[0] * 0.0
                    for parameter in self.dflash_module.parameters()
                    if parameter.requires_grad
                ),
                torch.zeros((), device=device),
            )
            return ModelOutput(
                loss=dummy, logits=base_outputs.logits,
                train_acc=[[0.0] * max(self.dflash_block_size - 1, 1)],
                dflash_metrics={},
            )

        # 4. Build draft inputs
        noise_embedding = self._build_noise_embedding(
            input_ids, anchor_positions, block_keep_mask, n_blocks
        )
        full_pos = self._build_position_ids(seq_len, anchor_positions, device)
        attn_mask = self._build_draft_attention_mask(
            seq_len,
            anchor_positions,
            block_keep_mask,
            n_blocks,
            target_hidden.dtype,
            device,
            window=self.dflash_swa_window_size,
        )

        # 5. Draft forward
        hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=full_pos,
            attention_mask=attn_mask,
        )

        # 6. Compute loss and accuracy
        logits = self._base_model_lm_head(hidden)
        loss, accuracy = self._compute_loss(
            logits,
            input_ids,
            anchor_positions,
            block_keep_mask,
            loss_mask,
            base_outputs.logits if self.dflash_self_logit_distillation else None,
            draft_hidden=hidden,
            base_outputs=base_outputs,
        )

        return ModelOutput(
            loss=loss,
            logits=base_outputs.logits,
            train_acc=[accuracy],
            dflash_metrics=dict(self._dflash_metrics),
        )

    @torch.no_grad()
    def pseudo_speculative_generate(self, input_ids, steps=1):
        """Generate draft tokens using one DFlash block for AR validation.

        This method implements a single speculative decoding step:

        1. **Base model forward**: Run the full target model on ``input_ids`` to get:
           - ``base_token``: greedy next token (argmax of last position logits)
           - ``hidden_states``: intermediate hidden states from target layers

        2. **Extract target hidden states**: Concatenate hidden states from
           ``target_layer_ids`` (e.g., layers [1, 9, 17, 25, 33] for 5-layer draft).
           Shape: ``[B, seq_len, num_layers * hidden_size]``.

        3. **Build block input**: Create a block of ``block_size`` tokens where:
           - Position 0 = ``base_token`` (the anchor/known token)
           - Positions 1..block_size-1 = ``mask_token_id`` (unknown, to be predicted)
           Embed this block via the base model's embedding layer.

        4. **Position IDs**: Context positions ``[0..seq_len-1]`` followed by block
           positions ``[seq_len..seq_len+block_size-1]``. The draft model's attention
           uses RoPE on these positions so Q (block only) attends to K (context + block)
           with correct relative position encoding.

        5. **Draft forward**: Run ``DFlashModule`` with:
           - ``noise_embedding``: embedded block tokens
           - ``target_hidden``: extracted hidden states from step 2
           - ``position_ids``: context + block positions
           - ``attention_mask=None``: no mask at inference (all positions attend freely)
           The draft model's KV injection concatenates projected target_hidden as K/V
           with the block's own K/V, enabling the draft to "see" the target's context.

        6. **Decode**: Apply ``lm_head`` to draft hidden states at positions 1..block_size-1
           (skip position 0 which is the known anchor). Argmax gives draft tokens.

        7. **Return**: ``(base_token, draft_tokens[:steps])`` — base token is always
           returned; draft tokens are truncated to ``steps`` (default: block_size-1).

        Note:
            This method re-runs the full target model from scratch on each call
            (no KV cache). For AR validation, it is called repeatedly with growing
            ``input_ids`` by ``AcceptanceRateValidation.validate()``. The ``steps``
            parameter should be set to ``block_size - 1`` for full block evaluation.

        Args:
            input_ids: Input token IDs [B, seq_len].
            steps: Number of draft tokens to return (capped at block_size-1).

        Returns:
            base_token: Next token from base model [B, 1].
            draft_tokens: Draft tokens [B, min(steps, block_size-1)] or None if steps < 1.
        """
        if self.dflash_offline:
            raise RuntimeError(
                "DFlash offline model cannot run AR validation / pseudo_speculative_generate — "
                "base model layers were deleted during offline conversion to save memory. "
                "Reload the full base model before running AR validation."
            )
        # Call the base model's inner model directly (avoids DynamicModule dispatch)
        model_output = self._base_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        # Compute logits via lm_head
        base_logits = self._base_model_lm_head(model_output.last_hidden_state)
        # Build output with hidden_states
        base_outputs = ModelOutput(
            logits=base_logits,
            hidden_states=model_output.hidden_states,
        )
        base_logits = base_outputs.logits
        base_token = base_logits[:, -1:, :].argmax(dim=-1).to(input_ids.device)

        if steps < 1:
            return base_token, None

        # Extract target hidden states (raw, before FC projection)
        hid_offset = 1
        selected = [base_outputs.hidden_states[lid + hid_offset] for lid in self.target_layer_ids]
        target_hidden = torch.cat(selected, dim=-1)

        block_size = self.dflash_block_size
        bsz = input_ids.shape[0]
        device = input_ids.device

        # Block: first token is base_token (anchor), rest are mask
        block_ids = torch.full(
            (bsz, block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_ids[:, 0] = base_token.squeeze(-1)
        noise_embedding = self._base_model_embeddings(block_ids)

        # Position IDs: training uses [0..L-1, 0..L-1] where noise positions
        # mirror context positions. At inference, block predicts tokens at
        # seq_len..seq_len+B-1, so noise positions continue from ctx_len.
        ctx_len = target_hidden.shape[1]
        ctx_positions = torch.arange(ctx_len, device=device)
        block_positions = torch.arange(ctx_len, ctx_len + block_size, device=device)
        pos_ids = torch.cat([ctx_positions, block_positions]).unsqueeze(0).expand(bsz, -1)

        attn_mask = self._build_generate_swa_mask(ctx_len, bsz, target_hidden.dtype, device)

        # Draft forward
        self._require_autocast_for_promoted_draft(device)
        draft_hidden = self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=attn_mask,
        )

        # Logits on positions 1..block_size-1 (skip anchor at position 0)
        draft_logits = self._base_model_lm_head(draft_hidden[:, 1:, :])
        draft_tokens = draft_logits.argmax(dim=-1)  # [B, block_size-1]

        # Return up to `steps` tokens
        num_tokens = min(steps, block_size - 1)
        return base_token, draft_tokens[:, :num_tokens]
