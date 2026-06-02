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

# Candidate module paths searched in order — shared with HFEagleModel._find_base_model_parts
_EMBED_TOKENS_PATHS = [
    "embed_tokens",
    "language_model.model.embed_tokens",
    "model.embed_tokens",
    "backbone.embeddings",
    "language_model.backbone.embeddings",
    "model.language_model.embed_tokens",
]
_LM_HEAD_PATHS = ["lm_head", "language_model.lm_head"]
_BASE_MODEL_PATHS = [
    "language_model.model",
    "model.language_model",
    "model",
    "backbone",
    "language_model.backbone",
]
_VLM_CONFIG_ATTRS = ["text_config", "llm_config"]
_SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"
_SAFETENSORS_SINGLE_FILENAME = "model.safetensors"


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
        **kwargs,
    ):
        """Initialize FakeBaseConfig with minimal model configuration parameters."""
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
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
        )
        model = cls(config)
        # Load lm_head and embed_tokens only from checkpoint
        lm_head_w, embed_tokens_w = model._load_weights(source)
        assert lm_head_w.shape == (config.vocab_size, config.hidden_size)
        assert embed_tokens_w.shape == (config.vocab_size, config.hidden_size)
        model.lm_head.weight.data.copy_(lm_head_w)
        model.embed_tokens.weight.data.copy_(embed_tokens_w)
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
        if (single_path := _try_fetch(_SAFETENSORS_SINGLE_FILENAME)) is not None:
            with safe_open(single_path, framework="pt") as h:
                return dict.fromkeys(h.keys(), _SAFETENSORS_SINGLE_FILENAME)
        raise FileNotFoundError(
            f"No {_SAFETENSORS_INDEX_FILENAME} or {_SAFETENSORS_SINGLE_FILENAME} found at "
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

    def _load_weights(self, source: str):
        """Load lm_head and embed_tokens weights from a local directory or HuggingFace Hub repo."""
        weight_map = self._load_index(source)

        embed_tokens_key = self._find_weight_key(weight_map, _EMBED_TOKENS_PATHS, "embed_tokens")
        try:
            lm_head_key = self._find_weight_key(weight_map, _LM_HEAD_PATHS, "lm_head")
        except RuntimeError:
            # Tied embeddings: lm_head shares embed_tokens weight and isn't stored separately.
            if not self.config.tie_word_embeddings:
                raise
            lm_head_key = embed_tokens_key

        lm_head_path, embed_tokens_path = self._resolve_shard_paths(
            source, [weight_map[lm_head_key], weight_map[embed_tokens_key]]
        )

        def _read(path: str, key: str, role: str = "") -> torch.Tensor:
            """Pull a single tensor; falls back to consolidated.safetensors for Mistral."""
            try:
                with safe_open(path, framework="pt", device="cpu") as h:
                    return h.get_tensor(key)
            except FileNotFoundError:
                _aliases = {
                    "embed_tokens": ["tok_embeddings.weight"],
                    "lm_head": ["output.weight"],
                }
                consolidated = os.path.join(os.path.dirname(path), "consolidated.safetensors")
                if os.path.isfile(consolidated):
                    with safe_open(consolidated, framework="pt", device="cpu") as h:
                        for alias in _aliases.get(role, []):
                            if alias in h.keys():
                                return h.get_tensor(alias)
                raise

        return (
            _read(lm_head_path, lm_head_key, "lm_head"),
            _read(embed_tokens_path, embed_tokens_key, "embed_tokens"),
        )

    def forward(self, *args, **kwargs):
        """Not implemented: FakeBaseModel omits full model weights and cannot run inference."""
        raise NotImplementedError("FakeBaseModel forward is not implemented.")


# Register so that AutoConfig / AutoModel / AutoModelForCausalLM can resolve "fake_base_model".
AutoConfig.register("fake_base_model", FakeBaseConfig)
AutoModel.register(FakeBaseConfig, FakeBaseModel)
AutoModelForCausalLM.register(FakeBaseConfig, FakeBaseModel)
