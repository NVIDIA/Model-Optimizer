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
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from modelopt.torch.export.plugins.hf_spec_export import _get_rope_theta
from modelopt.torch.utils.plugins.hf_checkpoint_utils import (
    copy_non_model_files,
    resolve_checkpoint_file,
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


def find_final_norm_path(model) -> str | None:
    """Return the attribute path of ``model``'s final norm, or ``None`` if it has none.

    The norm is not always at ``model.norm``: a real causal LM keeps it there while
    ``FakeBaseModel`` holds it at ``.norm``. Returns the path rather than the module
    because the speculative plugins store the path and resolve it later, while callers
    that want the module can ``get_submodule`` it.

    Resolves each path with ``get_submodule`` rather than a ``named_modules()`` dict:
    ``named_modules`` de-duplicates by default, so a norm instance also registered
    under an earlier path would be missing from the dict and the lookup would return
    ``None`` for a model that has one. ``get_submodule`` walks the attribute path and
    is indifferent to aliasing.
    """
    for path in _FINAL_NORM_PATHS:
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(module, torch.nn.Module):
            return path
    return None


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
        # Published in both shapes: Transformers 5 consumers read the dict first, and it has
        # to be well-formed -- the EAGLE draft config is built from this class too, and
        # rotary embeddings index rope_parameters["rope_type"] unconditionally.
        self.rope_theta = rope_theta
        if rope_theta is not None:
            self.rope_parameters = {"rope_theta": rope_theta, "rope_type": "default"}
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
        model = cls(cls._config_from_source(source, trust_remote_code))
        model._load_weights(source)
        return model

    @staticmethod
    def _config_from_source(source: str, trust_remote_code: bool) -> FakeBaseConfig:
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
        return FakeBaseConfig(
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
            # Shared with the exporter: where a config keeps rope_theta depends on the
            # transformers version, and reading it wrong is silent until serve time.
            rope_theta=_get_rope_theta(base_cfg),
            final_norm_type=_select_final_norm_type(
                getattr(base_cfg, "model_type", None), base_cfg
            ),
        )

    @classmethod
    def export_source_slice(
        cls,
        source: str | os.PathLike,
        export_dir: str | os.PathLike,
        *,
        trust_remote_code: bool = False,
    ) -> Path:
        """Export only the source weights used by fake-base training, without changing their dtype.

        ``source`` must be a local Hugging Face checkpoint. Its original config, tokenizer and
        custom code are retained; the result is an input to ``from_source``, not a serving model.
        ``export_dir`` must not exist. Prepare the slice once and share it between trainer workers;
        the streaming teacher still needs the complete checkpoint.
        """
        source, destination = Path(source), Path(export_dir)
        if not source.is_dir():
            raise ValueError("source must be a local Hugging Face checkpoint directory")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        config = cls._config_from_source(str(source), trust_remote_code)
        weights = cls._source_weights(str(source), config, validate_local_paths=True)
        config_path = resolve_checkpoint_file(source, "config.json", max_bytes=None)
        tokenizer_files = cls._source_tokenizer_files(source, trust_remote_code)
        model_files = {
            (source / name).resolve()
            for name in set(cls._load_index(str(source), validate_local_paths=True).values())
        }
        # Alternate full weights can coexist with the safetensors selected by the loader.
        model_files.update(
            path.resolve()
            for pattern in (
                "pytorch_model*.bin",
                "tf_model.h5",
                "flax_model.msgpack",
                "consolidated*.safetensors",
                "consolidated*.pth",
                "adapter_model.*",
            )
            for path in source.rglob(pattern)
        )
        if model_files.intersection((source / name).resolve() for name in tokenizer_files):
            raise ValueError("Tokenizer assets must not refer to model weight files")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".base-slice-", dir=destination.parent
        ) as temporary:
            output = Path(temporary) / "slice"
            output.mkdir()
            copy_non_model_files(source, output, strict=True, tokenizer_files=tokenizer_files)
            shutil.copy2(config_path, output / "config.json")
            save_file(
                dict(weights.values()),
                output / "model.safetensors",
                metadata={"format": "pt"},
            )
            shutil.copymode(source, output)
            output.rename(destination)
        return destination

    @staticmethod
    def _source_tokenizer_files(source: Path, trust_remote_code: bool) -> tuple[str, ...]:
        """Read declared tokenizer filenames without loading tokenizer data."""
        if not (source / "tokenizer_config.json").exists():
            return ()
        config_path = resolve_checkpoint_file(source, "tokenizer_config.json")
        metadata = json.loads(config_path.read_text())
        auto_map = metadata.get("auto_map", {})
        custom = auto_map if isinstance(auto_map, (tuple, list)) else auto_map.get("AutoTokenizer")
        tokenizer_class = tokenizer_class_from_name(metadata.get("tokenizer_class") or "")
        if custom and trust_remote_code:
            tokenizer_class = get_class_from_dynamic_module(
                custom[1] or custom[0], str(source), local_files_only=True
            )
        elif custom and tokenizer_class is None:
            raise ValueError("Custom tokenizer code requires trust_remote_code=True")
        return tuple(tokenizer_class.vocab_files_names.values()) if tokenizer_class else ()

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
    def _load_index(source: str, *, validate_local_paths: bool = False) -> dict:
        """Load weight_map from a sharded index, or synthesize one from a single safetensors file.

        Sharded checkpoints ship ``model.safetensors.index.json`` mapping every key to its shard;
        small checkpoints ship a single ``model.safetensors`` with no index — we read its keys
        and synthesize the equivalent weight_map so downstream code stays the same.
        """

        def _try_fetch(name: str) -> str | None:
            if os.path.isdir(source):
                path = os.path.join(source, name)
                if not os.path.isfile(path):
                    return None
                if validate_local_paths:
                    return str(resolve_checkpoint_file(Path(source), name, max_bytes=None))
                return path
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
    def _resolve_shard_paths(
        source: str, shard_filenames: list[str], *, validate_local_paths: bool = False
    ) -> list[str]:
        """Return local filesystem paths for each shard filename.

        Export validates local paths; ordinary loading also permits user-created cache views.
        Hub repo shards are downloaded via ``hf_hub_download`` (cached on subsequent calls).
        """
        if os.path.isdir(source):
            if validate_local_paths:
                if any(
                    Path(name).is_absolute() or ".." in Path(name).parts for name in shard_filenames
                ):
                    raise ValueError(
                        "Checkpoint shard filenames must be relative paths without '..'"
                    )
                return [
                    str(resolve_checkpoint_file(Path(source), name, max_bytes=None))
                    for name in shard_filenames
                ]
            return [os.path.join(source, name) for name in shard_filenames]
        return [hf_hub_download(repo_id=source, filename=name) for name in shard_filenames]

    @classmethod
    def _source_weights(
        cls, source: str, config: FakeBaseConfig, *, validate_local_paths: bool = False
    ) -> dict[str, tuple[str, torch.Tensor]]:
        weight_map = cls._load_index(source, validate_local_paths=validate_local_paths)

        embed_tokens_key = cls._find_weight_key(weight_map, _EMBED_TOKENS_PATHS, "embed_tokens")
        try:
            lm_head_key = cls._find_weight_key(weight_map, _LM_HEAD_PATHS, "lm_head")
        except RuntimeError:
            # Tied embeddings: lm_head shares embed_tokens weight and isn't stored separately.
            if not config.tie_word_embeddings:
                raise
            lm_head_key = embed_tokens_key
        keys = {"embed_tokens": embed_tokens_key, "lm_head": lm_head_key}
        if config.final_norm_type is not None:
            keys["norm"] = cls._find_weight_key(weight_map, _FINAL_NORM_PATHS, "final_norm")
        tensors = {}
        for key in dict.fromkeys(keys.values()):
            (path,) = cls._resolve_shard_paths(
                source, [weight_map[key]], validate_local_paths=validate_local_paths
            )
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensors[key] = handle.get_tensor(key)
        result = {}
        for role, key in keys.items():
            tensor = tensors[key]
            shape = (
                (config.hidden_size,) if role == "norm" else (config.vocab_size, config.hidden_size)
            )
            # copy_ would broadcast a wrong-but-compatible shape silently.
            if tensor.shape != shape:
                raise ValueError(f"{role} weight shape {tuple(tensor.shape)} != {shape}")
            result[role] = (key, tensor)
        return result

    def _load_weights(self, source: str) -> None:
        """Load only the embeddings, output head and known final norm from the checkpoint."""
        for role, (_, tensor) in self._source_weights(source, self.config).items():
            self.get_submodule(role).weight.data.copy_(tensor)

    def forward(self, *args, **kwargs):
        """Not implemented: FakeBaseModel omits full model weights and cannot run inference."""
        raise NotImplementedError("FakeBaseModel forward is not implemented.")


# Register so that AutoConfig / AutoModel / AutoModelForCausalLM can resolve "fake_base_model".
AutoConfig.register("fake_base_model", FakeBaseConfig)
AutoModel.register(FakeBaseConfig, FakeBaseModel)
AutoModelForCausalLM.register(FakeBaseConfig, FakeBaseModel)
