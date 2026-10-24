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
        3. ``modify()`` builds the draft's rotary buffer eagerly once the base model is
           off meta; ``_maybe_init_rotary_emb`` stays idempotent for the lazy pattern.
"""

import functools
import logging
from pathlib import Path
from typing import NamedTuple

import torch
from transformers import PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as _Qwen3Config
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..dflash.conversion import DFlashDMRegistry
from ..dflash.dflash_model import DFlashModel
from .dflash_flex_attention import build_draft_block_mask
from .dflash_losses import DraftLogits, LossContext, LossOutput, _block_targets, _masked_top1
from .modeling_dflash import (  # noqa: F401
    DFlashAttention,
    DFlashBaseModelOutput,
    DFlashModule,
    build_target_layer_ids,
)
from .modeling_fakebase import (
    _BASE_MODEL_PATHS,
    _EMBED_TOKENS_PATHS,
    _LM_HEAD_PATHS,
    find_final_norm_path,
)
from .modeling_final_norm import _apply_base_final_norm

logger = logging.getLogger(__name__)

__all__ = ["HFDFlashModel"]


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


def _gather_rows(x, positions):
    """``x[b, positions[b, ...]]`` for ``x`` [B, seq, D] and ``positions`` [B, ...]."""
    idx = positions.reshape(positions.shape[0], -1, 1).expand(-1, -1, x.size(-1))
    return torch.gather(x, 1, idx).reshape(*positions.shape, x.size(-1))


def _draft_mask_mod(seq_len, anchor_positions, block_keep_mask, block_size, window, causal):
    """The draft's attention visibility rule, as a FlexAttention ``mask_mod``.

    Elementwise ops and indexing only, so the one rule feeds ``create_block_mask`` and,
    evaluated on broadcast index grids, the dense SDPA mask.
    """
    # Indexed inside mask_mod, which create_block_mask runs under vmap: keep them integral.
    anchors = anchor_positions.to(torch.int32)
    keep = block_keep_mask.to(torch.bool)
    last_block = anchors.shape[1] - 1

    def mask_mod(b, h, q_idx, kv_idx):
        # torch 2.8's flex backward also passes padded rows (q_idx >= Q_LEN): keep reads in bounds.
        q_block = (q_idx // block_size).clamp(max=last_block)
        anchor = anchors[b, q_block]
        is_ctx = kv_idx < seq_len
        ctx_ok = is_ctx & (kv_idx < anchor)
        if window is not None:
            # Window from the query's real position (anchor + position in block).
            ctx_ok = ctx_ok & (kv_idx > anchor + (q_idx % block_size) - window)
        draft_ok = (~is_ctx) & (q_block == (kv_idx - seq_len) // block_size)
        if causal:
            # Block-causal: block position i sees draft positions <= i.
            draft_ok = draft_ok & (((kv_idx - seq_len) % block_size) <= (q_idx % block_size))
        return (ctx_ok | draft_ok) & keep[b, q_block]

    return mask_mod


def _dense_additive_mask(mask_mod, bsz, q_len, kv_len, dtype, device):
    """``mask_mod`` evaluated on broadcast index grids, as SDPA's additive [B, 1, Q, KV] mask."""
    visible = mask_mod(
        torch.arange(bsz, device=device).view(-1, 1, 1, 1),
        None,
        torch.arange(q_len, device=device).view(1, 1, -1, 1),
        torch.arange(kv_len, device=device).view(1, 1, 1, -1),
    )
    attn_mask = torch.zeros(bsz, 1, q_len, kv_len, device=device, dtype=dtype)
    attn_mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return attn_mask


class DraftBatch(NamedTuple):
    """What a training step hands its loss: the batch, its blocks and the draft's output."""

    input_ids: torch.Tensor  # [B, seq]
    loss_mask: torch.Tensor  # [B, seq] 1.0 where a token may be a label
    anchors: torch.Tensor  # [B, N] each block's anchor position
    keep: torch.Tensor  # [B, N] whether each block is trained
    base_outputs: DFlashBaseModelOutput
    hidden: torch.Tensor  # [B, N * bs, H] the draft's hidden states


@DFlashDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFDFlashModel(DFlashModel):
    """DFlash Model for HuggingFace transformers."""

    # Block position k is trained on the token at anchor + k + _label_shift: DFlash predicts
    # the anchor's own position onward, so slot 0 (the given anchor) is never supervised.
    _label_shift = 0

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
    def _needs_base_logits(self) -> bool:
        """Whether the loss reads the target's logits; offline training then reconstructs them."""
        return bool(self.dflash_self_logit_distillation)

    @property
    def _base_llm_config(self):
        return (
            getattr(self.config, "text_config", None)
            or getattr(self.config, "llm_config", None)
            or self.config
        )

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
        self.base_model_norm_path = find_final_norm_path(self)

    def modify(self, config):
        """Initialize DFlash draft module."""
        super().modify(config)

        # shift_label names a next-token variant's alignment in its export; the alignment is
        # fixed by _label_shift, so a config turning it off is refused rather than ignored.
        arch_config = config.dflash_architecture_config
        if self._label_shift and not arch_config.get("shift_label", True):
            raise ValueError(
                f"dflash_architecture_config.shift_label={arch_config['shift_label']!r} is not "
                f"supported: projector_type {arch_config.get('projector_type')!r} trains only "
                "next-token alignment (block position k predicts the token at anchor + k + 1)."
            )

        self.dflash_fp32_master_weights = config.dflash_fp32_master_weights

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
            # A NoPE target's config class can declare rope_theta without the checkpoint
            # setting it. There is no target rotary to align to, so the draft keeps its own
            # rather than inheriting None as the RoPE base.
            if base_val is None:
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
        if self.dflash_use_flex_attention and self.dflash_attention_sink:
            raise ValueError(
                "dflash_use_flex_attention does not support dflash_attention_sink: the flex "
                "kernel has no sink column, so the draft would train without the sink it is "
                "exported with. Unset one of them."
            )
        if self.dflash_use_flex_attention and self.dflash_config.attention_dropout:
            raise ValueError(
                "dflash_use_flex_attention does not support attention_dropout > 0 (got "
                f"{self.dflash_config.attention_dropout}): the flex path applies no dropout, "
                "so it would be silently skipped. Unset one of them."
            )

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
            # _target_hidden indexes hidden_states[lid + 1], where a negative id silently wraps.
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
        #
        # The draft matches the frozen base exactly, in dtype as well as device, so nothing
        # downstream has to reconcile the two. The extra precision `dflash_fp32_master_weights`
        # asks for lives in the optimizer instead; see `plugins/master_weight_adamw.py`.
        #
        # The rotary buffer is built here rather than on the draft's first forward: that
        # forward returns early for a rank whose batch has no valid anchor, leaving its buffer
        # list one short, and DDP's `broadcast_buffers` hangs rather than raises on mismatched
        # lists. After the cast, because `Module.to` casts float buffers and building first
        # would round the RoPE frequencies.
        base_device = self._base_device()
        if base_device.type != "meta":
            self.dflash_module.to(device=base_device, dtype=self._base_model.dtype)
            self.dflash_module._maybe_init_rotary_emb(device=base_device)

        # Delete base model layers for offline training (save memory)
        if self.dflash_offline:
            self._base_model._modules.pop("layers")

        self.is_quantized = False
        self._num_anchors = self.dflash_num_anchors
        if self.dflash_use_torch_compile:
            self.dflash_module.compile_body()

    def _build_draft_module(self, dflash_config):
        """Build the draft module. Subclasses override to use an augmented module."""
        return DFlashModule(dflash_config)

    def _base_device(self):
        """Device the frozen base model sits on. ``meta`` during a from_pretrained restore."""
        if self.dflash_offline:
            return self._base_model_lm_head.weight.device
        return next(self._base_model.layers[-1].parameters()).device

    def restore_draft_precision(self):
        """Re-apply what ``modify()`` could not, after a ``from_pretrained`` restore.

        ``modify()`` runs with the base model still on meta there, so it skips the draft's
        placement entirely. A no-op on a freshly converted model.
        """
        base_device = self._base_device()
        if base_device.type == "meta":
            return
        self.dflash_module.to(device=base_device, dtype=self._base_model.dtype)
        self.dflash_module._maybe_init_rotary_emb(device=base_device)

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

    @staticmethod
    def _build_loss_mask(input_ids, attention_mask, labels, loss_mask):
        """Return the float ``[B, seq_len]`` mask of positions anchor sampling and the loss may use.

        Every DFlash-family training forward calls this so the variants mask the same positions.
        """
        mask = torch.ones(input_ids.shape, device=input_ids.device)
        if labels is not None:
            mask = mask * (labels != LabelSmoother.ignore_index).float()
        # Labels carry optional answer-only masking but do not in general mark padding with -100
        # (collators copy them from padded input_ids), so always intersect with attention_mask.
        if attention_mask is not None:
            mask = mask * attention_mask.float()
        # In offline training, assistant mask is dumped and passed as kwarg.
        if loss_mask is not None:
            mask = mask * loss_mask
        return mask

    def _sample_anchor_positions(self, seq_len, loss_mask, device):
        """Randomly sample anchor positions per sample.

        Returns (anchor_positions [B, N], block_keep_mask [B, N]).

        With ``torch.compile`` (``dynamic=False``) or FlexAttention on, ``N`` is fixed by the
        config and ``seq_len``, not by the batch, so those shape-specialized graphs see one
        shape and never recompile. Eager attention sizes ``N`` to the batch instead.

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

        if self.dflash_use_torch_compile or self.dflash_use_flex_attention:
            # Also bounded by max_anchor + 1 so short sequences cannot ask for more columns.
            max_n = min(num_anchors, max_anchor + 1)
        else:
            # Padding blocks would cost eager attention their full compute; one host sync
            # is cheaper, e.g. with answer-only loss on short answers.
            max_n = max(min(num_anchors, int(valid_counts.max().item()) - 1), 1)

        # The data-dependent bound on sampled anchors, kept on-device so no shape depends on it.
        cap = (valid_counts.max() - 1).clamp(min=0, max=max_n)

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        fill = torch.tensor(seq_len + 1, device=device)
        masked_indices = torch.where(valid, indices, fill)

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)

        # Blank past `cap` before sorting: sorting first would pull anchors from beyond the
        # bound into the row and silently change which positions are trained on.
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
        """Build the draft attention mask: context (causal) + draft (per ``dflash_draft_attention``).

        When ``window`` is not None, all layers use sliding-window attention: each draft
        query only sees context positions within ``window`` tokens before its own position.
        Block-internal attention is left un-windowed (the config enforces
        ``window >= block_size``, so a full block always fits inside the window and windowing
        it would be a no-op).

        Block-internal visibility follows ``self.dflash_draft_attention``:
        ``"bidirectional"`` (default, MiMo-style) lets every query see the whole block, while
        ``"causal"`` restricts a query at block position ``i`` to draft positions ``<= i`` so
        the block is modelled autoregressively.

        Returns a dense additive SDPA mask, or a ``BlockMask`` under
        ``dflash_use_flex_attention``; both evaluate the one rule in ``_draft_mask_mod``.
        """
        bsz = anchor_positions.shape[0]
        q_len = n_blocks * self.dflash_block_size
        kv_len = seq_len + q_len
        mask_mod = _draft_mask_mod(
            seq_len,
            anchor_positions,
            block_keep_mask,
            self.dflash_block_size,
            window,
            causal=self.dflash_draft_attention == "causal",
        )

        if self.dflash_use_flex_attention:
            return build_draft_block_mask(
                mask_mod,
                bsz,
                q_len,
                kv_len,
                device,
                head_dims={layer.self_attn.head_dim for layer in self.dflash_module.layers},
            )

        return _dense_additive_mask(mask_mod, bsz, q_len, kv_len, dtype, device)

    def _build_generate_swa_mask(self, ctx_len, bsz, dtype, device):
        """Generation-time mask [B, 1, block_size, ctx_len + block_size], or None.

        Returns None only when there is nothing to mask: full attention over the context
        *and* bidirectional blocks (KV cache with no mask). Otherwise it is the training rule,
        :func:`_draft_mask_mod`, for the one block being drafted, anchored at ``ctx_len``:
        the window counts from each query's real position, and under
        ``dflash_draft_attention="causal"`` the block is lower-triangular.
        """
        window = self.dflash_swa_window_size
        causal = self.dflash_draft_attention == "causal"
        if window is None and not causal:
            return None
        block_size = self.dflash_block_size
        anchors = torch.full((bsz, 1), ctx_len, dtype=torch.long, device=device)
        keep = torch.ones(bsz, 1, dtype=torch.bool, device=device)
        mask_mod = _draft_mask_mod(ctx_len, anchors, keep, block_size, window, causal)
        return _dense_additive_mask(mask_mod, bsz, block_size, ctx_len + block_size, dtype, device)

    def _target_hidden(self, hidden_states, device=None):
        """The base hidden states the draft's ``fc`` reads, concatenated: [B, seq, n * H].

        ``hidden_states[0]`` is the embedding output, so layer ``lid``'s output is at ``lid + 1``.
        ``device`` gathers the layers first, for a base model sharded across GPUs.
        """
        selected = [hidden_states[lid + 1] for lid in self.target_layer_ids]
        if device is not None:
            selected = [h.to(device) for h in selected]
        return torch.cat(selected, dim=-1)

    def _generation_base(self, input_ids):
        """The base model's greedy next token after ``input_ids`` ([B, 1]), and its hidden states."""
        if self.dflash_offline:
            raise RuntimeError(
                "An offline DFlash-family model cannot run AR validation / "
                "pseudo_speculative_generate — base model layers were deleted during offline "
                "conversion to save memory. Reload the full base model before running AR validation."
            )
        # Call the base model's inner model directly (avoids DynamicModule dispatch)
        model_output = self._base_model(input_ids=input_ids, output_hidden_states=True)
        base_logits = self._base_model_lm_head(model_output.last_hidden_state)
        base_token = base_logits[:, -1:, :].argmax(dim=-1).to(input_ids.device)
        return base_token, model_output.hidden_states

    def _draft_block(self, base_token, hidden_states):
        """Draft hidden states [B, block_size, H] for the block that ``base_token`` anchors.

        Re-runs no base layer: the context is ``hidden_states`` from :meth:`_generation_base`.
        """
        # target_layer_ids spans early and late layers, so under device_map="auto" these come
        # off different GPUs and the cat fails. Everything the draft consumes is gathered
        # onto the draft's own device -- the last base layer's, per modify() -- which is
        # also not input_ids.device once the base model is sharded. All no-ops single-device.
        device = self._base_device()
        target_hidden = self._target_hidden(hidden_states, device=device)

        block_size = self.dflash_block_size
        bsz = base_token.shape[0]

        # Block: first token is base_token (anchor), rest are mask
        block_ids = torch.full(
            (bsz, block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_ids[:, 0] = base_token.squeeze(-1).to(device)
        # The embedding table is on the first shard; its output follows it, not block_ids.
        noise_embedding = self._base_model_embeddings(block_ids).to(device)

        # The training layout for one block anchored right after the context.
        ctx_len = target_hidden.shape[1]
        anchors = torch.full((bsz, 1), ctx_len, dtype=torch.long, device=device)
        pos_ids = self._build_position_ids(ctx_len, anchors, device)

        attn_mask = self._build_generate_swa_mask(ctx_len, bsz, target_hidden.dtype, device)

        return self.dflash_module(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=attn_mask,
        )

    def _draft_zero_loss(self, device):
        """A zero loss that every trainable draft parameter feeds, for a step with nothing to train.

        DDP expects a gradient for each parameter on every rank; a rank returning a constant
        loss would leave the others waiting on its reduction.
        """
        return sum(
            (p.reshape(-1)[0] * 0.0 for p in self.dflash_module.parameters() if p.requires_grad),
            torch.zeros((), device=device),
        )

    @torch.no_grad()
    def _teacher_logits(self, base_outputs, label_positions, token_ids=None):
        """The base model's logits for the tokens at ``label_positions`` ([B, ...]) -> [B, ..., vocab].

        Next-token logits: the ones that predict the token at position ``i`` sit at ``i - 1``.
        Only those rows of the base hidden go through the final norm and lm_head, so no
        full-sequence logits are built from it; ``token_ids`` ([B, ..., k]) narrows the
        projection to just those vocab entries.
        """
        positions = (label_positions - 1).clamp(min=0)
        if base_outputs.logits is not None:
            if token_ids is None:
                return _gather_rows(base_outputs.logits, positions)
            batch = torch.arange(positions.shape[0], device=positions.device)
            batch = batch.view(-1, *[1] * positions.dim())
            return base_outputs.logits[batch, positions.unsqueeze(-1), token_ids]
        if base_outputs.base_hidden is None:
            raise ValueError(
                "This objective needs the base model's distribution, but the batch carries "
                "neither its logits nor its final hidden states (base_model_hidden_states)."
            )
        lm_head = self._base_model_lm_head
        # A producer can store the hidden states in a wider dtype than the target's weights.
        # Cast before the final norm too, which online training runs in the target's dtype.
        rows = _gather_rows(base_outputs.base_hidden, positions).to(lm_head.weight.dtype)
        rows = _apply_base_final_norm(rows, base_outputs.base_hidden_prenorm, self._base_model_norm)
        if token_ids is None:
            return lm_head(rows)
        logits = torch.einsum("...h,...kh->...k", rows, lm_head.weight[token_ids])
        bias = getattr(lm_head, "bias", None)
        return logits if bias is None else logits + bias[token_ids]

    def _online_base_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        output_attentions=None,
        cache_position=None,
        **kwargs,
    ):
        """Run the frozen base model: the target hidden states, and its logits for KD."""
        # Multimodal models need the top-level conditional-generation forward so their
        # image/video features are inserted before the language model runs.  Keep the
        # long-standing narrow call for text-only models.
        base_forward_kwargs = _multimodal_forward_kwargs(kwargs)
        with torch.no_grad():
            if base_forward_kwargs:
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
        return DFlashBaseModelOutput(
            target_hidden=self._target_hidden(raw_outputs.hidden_states), logits=raw_outputs.logits
        )

    def _draft_logits(self, batch):
        """The draft's logits per block position; a variant with a head applies it here."""
        logits = self._base_model_lm_head(batch.hidden)
        logits = logits.view(*batch.anchors.shape, self.dflash_block_size, -1)
        return DraftLogits(final=logits, backbone=logits)

    @property
    def _position_weighting(self):
        """How block positions are weighted in the loss: ``"dpace"`` or ``"decay"``."""
        return self.dflash_loss_objective

    def _loss_context(self, batch):
        """The step's labels, the draft's logits and the position weights."""
        targets = _block_targets(
            batch.input_ids,
            batch.anchors,
            batch.keep,
            batch.loss_mask,
            self.dflash_block_size,
            self._label_shift,
        )
        return LossContext(
            batch,
            self._draft_logits(batch),
            targets,
            weighting={
                "scheme": self._position_weighting,
                "first": 1 - self._label_shift,
                "decay_factor": self.dflash_loss_decay_factor,
                "dpace_alpha": self.dflash_dpace_alpha,
            },
            teacher=functools.partial(self._teacher_logits, batch.base_outputs, targets.positions),
            # The KD path optimizes the soft target; its D-PACE confidences need no gradient.
            ce_grad=not self.dflash_self_logit_distillation,
        )

    def _compute_loss(self, ctx):
        """The weighted cross-entropy (or KD) loss on the served logits, and its accuracy.

        A step whose position weights leave nothing to train returns a zero loss and
        accuracy 0.0.
        """
        final = ctx.logits.final
        if not ctx.trainable:
            # Through the draft's output, so an FSDP unit around the draft still runs its backward.
            return LossOutput(final.sum() * 0.0, 0.0, {})

        if self.dflash_self_logit_distillation:
            flat_logits = ctx.flat_logits("final")
            target_soft = torch.softmax(ctx.teacher().reshape(flat_logits.shape), dim=-1)
            draft_logsoft = torch.log_softmax(flat_logits, dim=-1)
            per_token = -(target_soft * draft_logsoft).sum(dim=-1)
        else:
            per_token = ctx.ce("final")
        loss = (per_token * ctx.weights.view(-1)).sum() / ctx.weight_sum

        with torch.no_grad():
            accuracy = _masked_top1(final, ctx.targets.token_ids, ctx.targets.mask)
        return LossOutput(loss, accuracy.item(), {})

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
        """Training forward with random anchor sampling, shared by every DFlash-family variant.

        - Random anchor sampling instead of uniform block division
        - Bidirectional intra-block attention (no causal constraint)
        - Context sees strictly before anchor position
        - Label alignment: position k predicts the token at anchor + k + ``_label_shift``
        - Position weighting by D-PACE or exponential decay

        A variant changes the step only through its hooks: ``_online_base_outputs``,
        ``_draft_logits`` (its head), ``_position_weighting`` and ``_compute_loss``.
        """
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

        seq_len = input_ids.shape[1]
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
            base_outputs = DFlashBaseModelOutput.from_offline_dict(kwargs["base_model_outputs"])
            # Fail at the entry point rather than in the loss, after the draft has run.
            if (
                self._needs_base_logits
                and base_outputs.logits is None
                and base_outputs.base_hidden is None
            ):
                raise KeyError("base_model_hidden_states")
        else:
            base_outputs = self._online_base_outputs(
                input_ids,
                attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                cache_position=cache_position,
                **kwargs,
            )
        target_hidden = base_outputs.target_hidden

        # 2. Build loss mask.
        loss_mask = self._build_loss_mask(
            input_ids, attention_mask, labels, kwargs.get("loss_mask")
        )

        # 3. Random anchor sampling
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        if n_blocks == 0 or not block_keep_mask.any():
            # An all-masked answer-only batch: nothing to train, but DDP still reduces.
            return ModelOutput(
                loss=self._draft_zero_loss(device), logits=base_outputs.logits, train_acc=[[0.0]]
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
        batch = DraftBatch(
            input_ids, loss_mask, anchor_positions, block_keep_mask, base_outputs, hidden
        )
        out = self._compute_loss(self._loss_context(batch))
        return ModelOutput(
            loss=out.loss,
            logits=base_outputs.logits,
            train_acc=[[out.accuracy]],
            draft_metrics=out.metrics,
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
           - ``attention_mask``: the training rule for this one block (None when it masks nothing)
           The draft model's KV injection concatenates projected target_hidden as K/V
           with the block's own K/V, enabling the draft to "see" the target's context.

        6. **Decode**: Apply ``lm_head`` to draft hidden states from position
           ``1 - _label_shift`` on, the first one that predicts a token after the anchor
           (position 1 for DFlash, whose position 0 is the known anchor). Argmax gives draft
           tokens.

        7. **Return**: ``(base_token, draft_tokens[:steps])`` — base token is always
           returned; draft tokens are truncated to ``steps``.

        Note:
            This method re-runs the full target model from scratch on each call
            (no KV cache). For AR validation, it is called repeatedly with growing
            ``input_ids`` by ``AcceptanceRateValidation.validate()``. The ``steps``
            parameter should be set to ``block_size - 1 + _label_shift`` for full block
            evaluation.

        Args:
            input_ids: Input token IDs [B, seq_len].
            steps: Number of draft tokens to return (capped at block_size - 1 + _label_shift).

        Returns:
            base_token: Next token from base model [B, 1].
            draft_tokens: Draft tokens [B, min(steps, block_size - 1 + _label_shift)] or None
                if steps < 1.
        """
        base_token, hidden_states = self._generation_base(input_ids)
        if steps < 1:
            return base_token, None
        draft_hidden = self._draft_block(base_token, hidden_states)

        # Position k predicts the token at anchor + k + _label_shift, so this one drafts the
        # first token after the anchor.
        first = 1 - self._label_shift
        draft_logits = self._base_model_lm_head(draft_hidden[:, first:, :])
        draft_tokens = draft_logits.argmax(dim=-1)  # [B, block_size - first]

        # Return up to `steps` tokens. base_token already follows input_ids; draft_tokens is
        # produced on the draft's device, and callers cat both onto the running sequence.
        num_tokens = min(steps, self.dflash_block_size - first)
        return base_token, draft_tokens[:, :num_tokens].to(input_ids.device)
