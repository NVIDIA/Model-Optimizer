# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utility functions for model type detection and classification."""

import re
import warnings
from collections import defaultdict

import torch.nn as nn

from modelopt.torch.quantization.utils.core_utils import has_accelerate_offload
from modelopt.torch.utils.distributed import is_fsdp2_model

MODEL_NAME_TO_TYPE = {
    "GPT2": "gpt",
    "Mllama": "mllama",
    "Llama4": "llama4",
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "Qwen3Moe": "qwen3moe",
    "Qwen3Next": "qwen3next",
    "QWen": "qwen",
    "RecurrentGemma": "recurrentgemma",
    # DiffusionGemma must come before "Gemma" — get_model_type substring-matches
    # in order, and "gemma" is a substring of "diffusiongemma".
    "DiffusionGemma": "diffusion_gemma",
    "Gemma3": "gemma3",
    "Gemma2": "gemma2",
    "Gemma": "gemma",
    "phi3small": "phi3small",
    "phi3": "phi3",
    "PhiMoEForCausalLM": "phi3",
    "phi": "phi",
    "TLGv4ForCausalLM": "phi",
    "MixtralForCausalLM": "llama",
    "ArcticForCausalLM": "llama",
    "StarCoder": "gpt",
    "Dbrx": "dbrx",
    "T5": "t5",
    "Bart": "bart",
    "GLM": "glm",
    "InternLM2ForCausalLM": "internlm",
    "ExaoneForCausalLM": "exaone",
    "NemotronH": "nemotron_h",
    "Nemotron": "gpt",
    "Deepseek": "deepseek",
    "Whisper": "whisper",
    "gptoss": "gptoss",
    "MiniMax": "minimax",
}

__doc__ = f"""Utility functions for model type detection and classification.

    .. code-block:: python

        {MODEL_NAME_TO_TYPE=}
"""

__all__ = ["get_language_model_from_vl", "get_model_type", "is_multimodal_model"]


def get_model_type(model):
    """Try get the model type from the model name. If not found, return None."""
    for k, v in MODEL_NAME_TO_TYPE.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def is_multimodal_model(model):
    """Check if a model is a Vision-Language Model (VLM) or multimodal model.

    This function detects various multimodal model architectures by checking for:
    - Standard vision configurations (vision_config)
    - Language model attributes (language_model)
    - Nemotron-Parse conditional generation models

    Args:
        model: The HuggingFace model instance to check

    Returns:
        bool: True if the model is detected as multimodal, False otherwise

    Examples:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> is_multimodal_model(model)
        True
    """
    config = model.config

    # Check for Nemotron-Parse encoder-decoder architecture
    architectures = getattr(config, "architectures", [])
    is_nemotron_parse = any("nemotronparse" in arch.lower() for arch in architectures)

    return (
        hasattr(config, "vision_config")  # Standard vision config (e.g., Qwen2.5-VL)
        or hasattr(model, "language_model")  # Language model attribute (e.g., LLaVA)
        or is_nemotron_parse  # Nemotron-Parse conditional generation model
    )


def get_language_model_from_vl(model) -> list[nn.Module] | None:
    """Extract the language model lineage from a Vision-Language Model (VLM).

    This function handles the common patterns for accessing the language model component
    in various VLM architectures. It checks multiple possible locations where the
    language model might be stored.

    Args:
        model: The VLM model instance to extract the language model from

    Returns:
        list: the lineage path towards the language model

    Examples:
        >>> # For LLaVA-style models
        >>> lineage = get_language_model_from_vl(vlm_model)
        >>> # lineage[0] is vlm_model
        >>> # lineage[1] is vlm_model.language_model
    """
    # always prioritize model.model.langauge_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return [model, model.model, model.model.language_model]

    if hasattr(model, "language_model"):
        return [model, model.language_model]

    # Pattern 3: For encoder-decoder VL models (e.g., Nemotron-Parse), the decoder is the language model.
    # Only match if the model is detected as multimodal to avoid matching non-VLM encoder-decoder
    # models like T5, Bart, Whisper which also have .decoder.
    if hasattr(model, "decoder") and is_multimodal_model(model):
        return [model, model.decoder]

    # Pattern 4: No language_model found
    return None


def _build_tied_alias_map(model: nn.Module) -> dict[str, str]:
    r"""Map each tied *alias* parameter name to its *canonical* name.

    Ties are detected by **object identity** — parameters registered under more than one
    name that are the same live :class:`~torch.nn.Parameter`. Built before export packs or
    splits weights (which break the shared object), while all params are resident, so
    identity is reliable; the map is keyed by *name*, so it survives the later packing /
    FSDP gather / offload when the drop actually runs.

    Declarations do not *create* ties — they only pick the canonical (kept) member of a
    shared group:

    - dict-style ``_tied_weights_keys``: a name matching an alias key (relative to the
      declaring submodule) is an alias side; the canonical is a non-alias group member.
      Only the alias pattern is matched — the canonical-side value is never parsed, so
      parallel-pattern / backref declarations (DiffusionGemma) need no special handling.
    - ``tie_word_embeddings=True``: the input-embedding name is canonical.

    Groups left untouched (both sides kept): a declared-but-unapplied tie can't form
    (distinct objects never group); an *undeclared* share or an all-alias group has no
    canonical the loader could re-tie from, so it is not dropped. An undeclared share that
    still aliases storage is caught by the address backstop in :func:`postprocess_state_dict`.
    """
    # Group names by object identity. remove_duplicate=False so a shared Parameter appears
    # under every name -- that gives both the tie signal and the canonical/alias names.
    groups: dict[int, list[str]] = defaultdict(list)
    param_names: list[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups[id(parameter)].append(name)
        param_names.append(name)

    # Mark declared alias names (the drop side). Only the alias pattern (dict key) is
    # matched; the canonical-side value is never parsed.
    declared_aliases: set[str] = set()
    for mod_name, submodule in model.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", None)
        if not isinstance(tied, dict) or not tied:
            continue
        prefix = f"{mod_name}." if mod_name else ""
        plen = len(prefix)
        for alias_pat in tied:
            try:
                alias_re = re.compile(alias_pat)
            except re.error:
                continue
            for full_name in param_names:
                if prefix and not full_name.startswith(prefix):
                    continue
                if alias_re.search(full_name[plen:]):
                    declared_aliases.add(full_name)

    # tie_word_embeddings: the input-embedding name is canonical for the shared weight.
    embedding_canonical: dict[int, str] = {}
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        try:
            in_emb = model.get_input_embeddings()
            out_emb = model.get_output_embeddings()
        except (AttributeError, NotImplementedError):
            in_emb = out_emb = None
        in_weight = getattr(in_emb, "weight", None)
        # Confirm the tie is actually applied: input and output share the same weight object.
        if in_weight is not None and in_weight is getattr(out_emb, "weight", None):
            in_name = {m: n for n, m in model.named_modules()}.get(in_emb)
            if in_name is not None:
                embedding_canonical[id(in_weight)] = f"{in_name}.weight" if in_name else "weight"

    # Emit {alias -> canonical} for shared groups that have a clear, declared canonical.
    alias_to_canonical: dict[str, str] = {}
    for parameter_id, names in groups.items():
        if len(names) < 2:
            continue
        canonical = embedding_canonical.get(parameter_id)
        if canonical is None or canonical not in names:
            non_aliases = [name for name in names if name not in declared_aliases]
            # No canonical to keep: undeclared share (all non-alias) or all-alias group.
            if len(non_aliases) == len(names) or not non_aliases:
                continue
            canonical = non_aliases[0]
        for name in names:
            if name != canonical:
                alias_to_canonical[name] = canonical

    # Fail loud, don't fail silent under sharding: a declared alias whose param did NOT join a
    # shared id-group is either an unapplied tie (fine) or -- under FSDP2 / offload -- a tie
    # whose object identity the wrapper may have split into distinct sharded params, which would
    # silently skip dedup. Detection here rests on id(parameter); warn only in that context so a
    # future torch change can't disable dedup quietly.
    unrealized = declared_aliases - set(alias_to_canonical)
    if unrealized:
        if is_fsdp2_model(model) or has_accelerate_offload(model):
            warnings.warn(
                f"{len(unrealized)} declared tied-weight alias(es) did not form a shared-parameter "
                f"group under FSDP2/offload (e.g. {sorted(unrealized)[0]!r}); their duplicates may "
                f"not be deduplicated. If these are genuinely tied, gather/unshard before export."
            )
    return alias_to_canonical


class TiedWeightMap:
    """Name-based lookups over the ``{alias: canonical}`` map from :func:`_build_tied_alias_map`.

    A thin, immutable view built once per export. Export sites that once keyed dedup on
    ``data_ptr`` ask for a *group key* instead: both sides of a tie share one key, an untied
    parameter returns ``None``. The key is a name, so it survives packing, FSDP resharding,
    offload, and allocator reuse — where ``data_ptr`` does not.
    """

    def __init__(self, model: nn.Module) -> None:
        self.alias_to_canonical: dict[str, str] = _build_tied_alias_map(model)
        self.canonical_names: set[str] = set(self.alias_to_canonical.values())

    def group_key(self, param_full_name: str) -> str | None:
        """Return the canonical group key for a parameter, or ``None`` if untied.

        Both sides of a declared tie map to the same canonical name, so the key is
        independent of which side the export walk visits first.
        """
        if param_full_name in self.alias_to_canonical:
            return self.alias_to_canonical[param_full_name]
        if param_full_name in self.canonical_names:
            return param_full_name
        return None

    def container_group_key(self, container_name: str, first_proj_attr: str) -> str | None:
        """Return a group key for a fused-experts container, or ``None`` if untied.

        The tie is on the container's 3-D projection Parameter (e.g.
        ``…experts.gate_up_proj``); the canonical is the *same* projection on another
        container (they are one id-group), so it carries the same suffix. Stripping that
        suffix yields one key shared by all of the container's projections.
        """
        gk = self.group_key(f"{container_name}.{first_proj_attr}")
        if gk is None:
            return None
        return gk.removesuffix(f".{first_proj_attr}")
