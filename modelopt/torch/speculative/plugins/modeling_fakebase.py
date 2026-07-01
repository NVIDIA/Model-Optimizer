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

"""Lightweight fake base model for offline speculative decoding training."""

import json
import os

import torch
import torch.nn as nn
import transformers
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from .modeling_final_norm import _FINAL_NORM_CLASSES, _select_final_norm_type

# Candidate module paths searched in order — shared with HFEagleModel._find_base_model_parts
_EMBED_TOKENS_PATHS = [
    "embed_tokens",
    "language_model.model.embed_tokens",
    "model.embed_tokens",
    "backbone.embeddings",
    "language_model.backbone.embeddings",
    "model.language_model.embed_tokens",
    "tok_embeddings",  # Mistral native checkpoints (consolidated.safetensors)
]
_LM_HEAD_PATHS = [
    "lm_head",
    "language_model.lm_head",
    "output",  # Mistral native checkpoints (consolidated.safetensors)
]
_FINAL_NORM_PATHS = [
    "model.norm",
    "language_model.model.norm",
    "norm",
    "backbone.norm_f",
    "backbone.norm",
    "language_model.backbone.norm",
    "model.language_model.norm",
]
_BASE_MODEL_PATHS = [
    "language_model.model",
    "model.language_model",
    "model",
    "backbone",
    "language_model.backbone",
]
_VLM_CONFIG_ATTRS = ["text_config", "llm_config"]
_SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"
# Single-file safetensors names to try, in order.  Mistral native checkpoints
# use ``consolidated.safetensors`` instead of the HF-standard ``model.safetensors``.
_SAFETENSORS_SINGLE_FILENAMES = ["model.safetensors", "consolidated.safetensors"]


class FakeBaseConfig(PretrainedConfig):
    """Minimal config for FakeBaseModel that supports offline speculative decoding training."""

    model_type = "fake_base_model"

    def __init__(
        self,
        num_hidden_layers=None,
        hidden_size=None,
        vocab_size=None,
        max_position_embeddings=None,
        dtype=torch.bfloat16,
        tie_word_embeddings=False,
        num_orig_hidden_layers=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        intermediate_size=None,
        rms_norm_eps=1e-6,
        rope_theta=None,
        final_norm_type=None,
        **kwargs,
    ):
        """Initialize FakeBaseConfig with minimal model configuration parameters."""
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.rms_norm_eps = rms_norm_eps
        # Which self-implemented final-norm class FakeBaseModel builds, or None to build no norm
        # (model whose final-norm type we don't know). See _FINAL_NORM_CLASSES /
        # _FINAL_NORM_TYPE_BY_MODEL_TYPE. Persisted so a reloaded config rebuilds the same norm.
        self.final_norm_type = final_norm_type
        self.num_hidden_layers = num_hidden_layers
        # Mirror the original base layer count. The non-fake offline path loads with
        # num_hidden_layers=0 and stashes the real count here (see utils.load_vlm_or_llm);
        # the fake base keeps num_hidden_layers as the real count, so default to it. DFlash's
        # offline modify() reads num_orig_hidden_layers directly (hf_dflash.py), so it must
        # always be present on the base config.
        # TODO: Deprecate the old offline path.
        self.num_orig_hidden_layers = (
            num_orig_hidden_layers if num_orig_hidden_layers is not None else num_hidden_layers
        )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # Attention/MLP dims needed when exporting a draft head built on a fake base: the
        # DFlash exporter (hf_spec_export._export_config) references base_config.{num_attention_heads,
        # num_key_value_heads, intermediate_size} as getattr fallbacks, which Python evaluates
        # eagerly, so they must exist even though the fake base has no real layers.
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.intermediate_size = intermediate_size
        # For some drafter algo (e.g. DFlash) rope theta must match target model. Extract here.
        self.rope_theta = rope_theta
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype


class FakeBaseModel(PreTrainedModel):
    """Minimal base model for offline speculative decoding.

    Contains only ``lm_head``, ``embed_tokens``, and the minimal config needed by the EAGLE
    training loop. The full model weights are never loaded, keeping memory usage low.

    Weights are loaded from a local HuggingFace checkpoint directory. Weight key names and
    VLM config nesting are auto-detected from the shared path constants.
    """

    config_class = FakeBaseConfig

    def __init__(self, config: FakeBaseConfig, **kwargs):
        """Initialize FakeBaseModel structure from a FakeBaseConfig.

        To construct a FakeBaseModel from an original HuggingFace checkpoint (e.g. a Llama
        repo), use the :meth:`from_source` classmethod instead.
        """
        super().__init__(config, **kwargs)
        # Initialize dummy module and attributes for compatibility with HFEagleModel
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        self.model.dtype = config.dtype
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype
        )
        # Final pre-lm_head norm, applied before lm_head when reconstructing base logits for
        # self-logit-distillation (vLLM-captured final hidden states are un-normed). Built ONLY
        # when the base model's final-norm type is known (config.final_norm_type set); otherwise
        # no ``norm`` attribute exists and downstream skips re-norming. The concrete class comes
        # from config.final_norm_type (see _FINAL_NORM_CLASSES); weight loaded in from_source.
        if config.final_norm_type is not None:
            norm_cls = _FINAL_NORM_CLASSES[config.final_norm_type]
            self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype)
        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_source(cls, source: str, trust_remote_code: bool = False) -> "FakeBaseModel":
        """Load lm_head and embed_tokens from a local directory or HuggingFace Hub repo.

        Args:
            source: Path to a local HuggingFace checkpoint directory, or a HuggingFace Hub
                repo ID (e.g. ``"meta-llama/Llama-3.1-8B"``). The source type is detected
                automatically: if ``source`` is an existing local directory it is treated as a
                local checkpoint; otherwise it is treated as a Hub repo ID and the required
                files are downloaded via ``huggingface_hub``.
        """
        orig_config = transformers.AutoConfig.from_pretrained(
            source, trust_remote_code=trust_remote_code
        )
        # For vlms, detect language model config based on _VLM_CONFIG_ATTRS
        base_cfg = next(
            (
                getattr(orig_config, attr)
                for attr in _VLM_CONFIG_ATTRS
                if getattr(orig_config, attr, None) is not None
            ),
            orig_config,
        )
        # Extract necessary info for spec training from base config
        config = FakeBaseConfig(
            num_hidden_layers=getattr(base_cfg, "num_hidden_layers", None),
            hidden_size=getattr(base_cfg, "hidden_size", None),
            vocab_size=getattr(base_cfg, "vocab_size", None),
            max_position_embeddings=getattr(base_cfg, "max_position_embeddings", None),
            dtype=getattr(base_cfg, "dtype", torch.bfloat16),
            tie_word_embeddings=getattr(base_cfg, "tie_word_embeddings", False),
            num_attention_heads=getattr(base_cfg, "num_attention_heads", None),
            num_key_value_heads=getattr(base_cfg, "num_key_value_heads", None),
            intermediate_size=getattr(base_cfg, "intermediate_size", None),
            rms_norm_eps=getattr(base_cfg, "rms_norm_eps", 1e-6),
            rope_theta=getattr(base_cfg, "rope_theta", None),
            final_norm_type=_select_final_norm_type(getattr(base_cfg, "model_type", None)),
        )
        model = cls(config)
        # Load lm_head, embed_tokens, and (for known models) the final norm into the model.
        model._load_weights(source)
        return model

    @staticmethod
    def _find_weight_key(weight_map: dict, paths: list[str], label: str) -> str:
        """Return the first ``path + '.weight'`` found in ``weight_map``."""
        for path in paths:
            key = path + ".weight"
            if key in weight_map:
                return key
        tried = [p + ".weight" for p in paths]
        raise RuntimeError(f"Cannot find {label} in checkpoint; tried: {tried}")

    @staticmethod
    def _load_index(source: str) -> dict:
        """Load weight_map from a sharded index, or synthesize one from a single safetensors file.

        Sharded checkpoints ship ``model.safetensors.index.json`` mapping every key to its shard;
        small checkpoints ship a single ``model.safetensors`` with no index — we read its keys
        and synthesize the equivalent weight_map so downstream code stays the same.
        """

        def _try_fetch(name: str) -> str | None:
            if os.path.isdir(source):
                path = os.path.join(source, name)
                return path if os.path.isfile(path) else None
            try:
                return hf_hub_download(repo_id=source, filename=name)
            except EntryNotFoundError:
                return None

        if (index_path := _try_fetch(_SAFETENSORS_INDEX_FILENAME)) is not None:
            with open(index_path) as f:
                return json.load(f).get("weight_map", {})
        for single_name in _SAFETENSORS_SINGLE_FILENAMES:
            if (single_path := _try_fetch(single_name)) is not None:
                with safe_open(single_path, framework="pt") as h:
                    return dict.fromkeys(h.keys(), single_name)
        raise FileNotFoundError(
            f"No {_SAFETENSORS_INDEX_FILENAME} or {_SAFETENSORS_SINGLE_FILENAMES} found at "
            f"{source!r}. FakeBaseModel only supports safetensors checkpoints; "
            "pytorch_model.bin is not supported."
        )

    @staticmethod
    def _resolve_shard_paths(source: str, shard_filenames: list[str]) -> list[str]:
        """Return local filesystem paths for each shard filename.

        For a local directory the paths are joined directly; for a HuggingFace Hub repo ID the
        shards are downloaded via ``hf_hub_download`` (cached on subsequent calls).
        """
        if os.path.isdir(source):
            return [os.path.join(source, name) for name in shard_filenames]
        return [hf_hub_download(repo_id=source, filename=name) for name in shard_filenames]

    def _load_weights(self, source: str) -> None:
        """Load lm_head, embed_tokens, and (for known models) the final norm into this model.

        Reads only the tensors needed (never materializes a whole shard) and copies them into
        the already-constructed submodules. For unknown models (``final_norm_type`` unset) the
        norm is neither loaded nor present (``self.norm`` does not exist; see :meth:`__init__`).
        """
        weight_map = self._load_index(source)

        embed_tokens_key = self._find_weight_key(weight_map, _EMBED_TOKENS_PATHS, "embed_tokens")
        try:
            lm_head_key = self._find_weight_key(weight_map, _LM_HEAD_PATHS, "lm_head")
        except RuntimeError:
            # Tied embeddings: lm_head shares embed_tokens weight and isn't stored separately.
            if not self.config.tie_word_embeddings:
                raise
            lm_head_key = embed_tokens_key

        # Pull only the tensor we need; avoids materializing the whole file.
        def _read(key: str) -> torch.Tensor:
            (path,) = self._resolve_shard_paths(source, [weight_map[key]])
            with safe_open(path, framework="pt", device="cpu") as h:
                return h.get_tensor(key)

        # Explicit shape checks: copy_ would broadcast a wrong-but-compatible shape silently.
        hidden, vocab = self.config.hidden_size, self.config.vocab_size
        embed_tokens_w = _read(embed_tokens_key)
        lm_head_w = _read(lm_head_key)
        assert embed_tokens_w.shape == (vocab, hidden)
        assert lm_head_w.shape == (vocab, hidden)
        self.embed_tokens.weight.data.copy_(embed_tokens_w)
        self.lm_head.weight.data.copy_(lm_head_w)

        # Final norm only for models whose norm type we know (final_norm_type set); when known it
        # MUST be present — a missing key is a hard error, never a silent skip. Unknown: no norm.
        if self.config.final_norm_type is not None:
            norm_w = _read(self._find_weight_key(weight_map, _FINAL_NORM_PATHS, "final_norm"))
            assert norm_w.shape == (hidden,), (
                f"final-norm weight shape {tuple(norm_w.shape)} != ({hidden},)"
            )
            self.norm.weight.data.copy_(norm_w)

    def forward(self, *args, **kwargs):
        """Not implemented: FakeBaseModel omits full model weights and cannot run inference."""
        raise NotImplementedError("FakeBaseModel forward is not implemented.")


# Register so that AutoConfig / AutoModel / AutoModelForCausalLM can resolve "fake_base_model".
AutoConfig.register("fake_base_model", FakeBaseConfig)
AutoModel.register(FakeBaseConfig, FakeBaseModel)
AutoModelForCausalLM.register(FakeBaseConfig, FakeBaseModel)
