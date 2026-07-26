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

"""Resolve which modules to score and build the matching parallel-aware scorers.

Reuses the puzzletron ``pruning_mixin.get_module_names_to_hook(model)`` — the same
function the legacy scorer uses — so the per-module **canonical keys** (e.g.
``model.layers.5.mlp.down_proj``) are identical to what the pruning step looks up.
For each hooked module it instantiates the AutoModel scorer for the configured
``method`` with the kwargs that method needs (derived from the model config +
per-layer block config), exactly mirroring ``register_activation_hooks``.
"""

import logging
import os
import re

from modelopt.torch.puzzletron.anymodel.capabilities import MagnitudeFallbackSpec
from modelopt.torch.puzzletron.block_config import maybe_cast_block_configs
from modelopt.torch.puzzletron.utils.dummy_modules import DummyBlock, DummyModule

from .hooks import (
    ActivationMagnitudeScorer,
    FFNIndependentScorer,
    FFNIterativeScorer,
    GatedDeltaNetActivationScorer,
    GroupedAttentionScorer,
    MambaInProjContributionScorer,
    MoEExpertRemovalDiffScorer,
    MoEGroupedExpertChannelScorer,
    MoELatentCalibrationScorer,
    MoESharedExpertChannelScorer,
)
from .reduction import MeshGroups

logger = logging.getLogger(__name__)

__all__ = ["build_magnitude_scorers", "build_scorers"]


def build_magnitude_scorers(
    model,
    groups: MeshGroups,
    targets: list[MagnitudeFallbackSpec] | tuple[MagnitudeFallbackSpec, ...],
    *,
    model_descriptor=None,
    target_type: str = "generic",
    register: bool = True,
) -> list[ActivationMagnitudeScorer]:
    """Resolve descriptor-owned observation templates on the local model/PP shard."""

    scorers: list[ActivationMagnitudeScorer] = []
    seen: set[tuple[str, str]] = set()
    named_modules = dict(model.named_modules())
    for raw_target in targets:
        target = (
            raw_target
            if isinstance(raw_target, MagnitudeFallbackSpec)
            else MagnitudeFallbackSpec(**raw_target)
        )
        template = target.observation_module
        if "{layer_idx}" in template:
            escaped = re.escape(template).replace(re.escape("{layer_idx}"), r"(?P<layer_idx>\d+)")
            matches = []
            for module_name, module in named_modules.items():
                match = re.fullmatch(escaped, module_name)
                if match is not None:
                    matches.append((int(match.group("layer_idx")), module_name, module))
            matches.sort(key=lambda value: (value[0], value[1]))
        else:
            module_name = template
            adapted_name = (
                model_descriptor.adapt_module_name_for_model(module_name, model)
                if model_descriptor is not None
                else module_name
            )
            module = named_modules.get(adapted_name)
            matches = [] if module is None else [(None, module_name, module)]

        for block_idx, canonical_name, module in matches:
            identity = (canonical_name, target.output_field)
            if identity in seen:
                raise ValueError(
                    f"duplicate magnitude fallback target {canonical_name!r} "
                    f"field={target.output_field!r}"
                )
            seen.add(identity)
            scorer = ActivationMagnitudeScorer(
                module,
                groups,
                tensor_selector=target.tensor_selector,
                scored_dim=target.scored_dim,
                output_field=target.output_field,
                expected_size=target.expected_size,
                target_type=target_type,
                block_idx=block_idx,
                name=canonical_name,
            )
            if register:
                scorer.register()
            scorers.append(scorer)
    return scorers


def _canonical_output_name(pruning_mixin, method, block_idx, fallback):
    """Output key for a scored module — the *descriptor* name the pruning step looks up.

    The pruning step keys activation scores by ``layer_descriptor.ffn_prefix(i).down_proj`` /
    ``attn_prefix(i).o_proj`` (``pruning_utils._load_activations_log``), derived from the
    checkpoint's module structure. The model loaded for scoring can be structured differently
    — e.g. NeMo loads a VL checkpoint as a text-only causal LM, so its modules are
    ``model.layers.i...`` while the checkpoint/descriptor use ``model.language_model.layers.i...``.
    Keying the output by the descriptor (not the loaded module path) makes the scores match by
    construction for any descriptor; for an already-aligned model both are identical. Falls back
    to the loaded module path when the descriptor lacks the relevant prefix (e.g. test doubles).
    """
    descriptor = getattr(pruning_mixin, "layer_descriptor", None)
    if descriptor is None or block_idx is None:
        return fallback
    if method in ("independent", "iterative") and hasattr(descriptor, "ffn_prefix"):
        return f"{descriptor.ffn_prefix(block_idx)}.down_proj"
    if method == "ple_channel_contribution" and hasattr(descriptor, "ffn_prefix"):
        return f"{descriptor.ffn_prefix(block_idx)}.{descriptor.down_proj_name}"
    if method in {"grouped_attention_contribution", "mla_head_contribution"} and hasattr(
        descriptor, "attn_prefix"
    ):
        # Qwen and Nemotron share one KV-group/query-head iterative payload.
        return f"{descriptor.attn_prefix(block_idx)}.o_proj"
    if hasattr(descriptor, "canonical_score_key"):
        return descriptor.canonical_score_key(method, block_idx, fallback)
    # MoE/Mamba descriptors can also expose canonical keys through their layer descriptor.
    layer_descriptor = getattr(pruning_mixin, "layer_descriptor", None)
    if hasattr(layer_descriptor, "canonical_score_key"):
        return layer_descriptor.canonical_score_key(method, block_idx, fallback)
    return fallback


def _moe_dims(model_config, block_config):
    top_k = num_experts = expert_intermediate = shared_intermediate = latent_dim = None
    if block_config is not None:
        moe = block_config.get_subblock("moe")
        if moe is not None:
            top_k = moe.top_k
            num_experts = moe.num_experts
            expert_intermediate = moe.expert_intermediate_size
            shared_intermediate = moe.shared_expert_intermediate_size
            latent_dim = moe.latent_dim
    if top_k is None:
        top_k = getattr(model_config, "num_experts_per_tok", None) or getattr(model_config, "moe_top_k", None)
    if num_experts is None:
        num_experts = (
            getattr(model_config, "num_local_experts", None)
            or getattr(model_config, "num_experts", None)
            or getattr(model_config, "n_routed_experts", None)
        )
    if expert_intermediate is None:
        expert_intermediate = getattr(model_config, "moe_intermediate_size", None)
    if shared_intermediate is None:
        shared_intermediate = getattr(model_config, "moe_shared_expert_intermediate_size", None)
    if latent_dim is None:
        latent_dim = getattr(model_config, "moe_latent_size", None)
    return top_k, num_experts, expert_intermediate, shared_intermediate, latent_dim


def _mamba_dims(model_config, block_config):
    num_heads = head_dim = num_groups = state_dim = None
    if block_config is not None:
        mamba = block_config.get_subblock("mamba")
        if mamba is not None:
            num_heads = mamba.num_heads
            head_dim = mamba.head_dim
            num_groups = mamba.num_groups
            state_dim = mamba.state_dim
    num_heads = num_heads or getattr(model_config, "mamba_num_heads", None)
    head_dim = head_dim or getattr(model_config, "mamba_head_dim", None)
    num_groups = num_groups or getattr(model_config, "n_groups", None)
    state_dim = state_dim or getattr(model_config, "ssm_state_size", None)
    if None in (num_heads, head_dim, num_groups, state_dim):
        raise ValueError(
            "Mamba scoring requires num_heads, head_dim, num_groups, and state_dim "
            "from the typed block config or model config."
        )
    return int(num_heads), int(head_dim), int(num_groups), int(state_dim)


def _make_scorer(
    method,
    module,
    groups,
    model_config,
    block_config,
    block_idx,
    module_name,
    *,
    optimize_for="memory",
    validation_full_iters=None,
    calibration_method=None,
    clear_gpu_memory=False,
    scored_axes=None,
    token_chunk_size=None,
    canonicalize_name=None,
):
    """Instantiate the scorer for ``method`` with method-specific kwargs."""
    def out_name(raw_name: str) -> str:
        return canonicalize_name(raw_name) if canonicalize_name is not None else raw_name

    if method == "independent":
        return FFNIndependentScorer(module, groups, block_idx=block_idx, name=out_name(module_name))

    if method == "iterative":
        if validation_full_iters is None:
            raise ValueError("iterative scoring requires validation_full_iters.")
        return FFNIterativeScorer(
            module,
            groups,
            validation_full_iters=validation_full_iters,
            calibration_method=calibration_method,
            clear_gpu_memory=clear_gpu_memory,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method == "ple_channel_contribution":
        if validation_full_iters is None:
            raise ValueError("PLE iterative scoring requires validation_full_iters.")
        return FFNIterativeScorer(
            module,
            groups,
            validation_full_iters=validation_full_iters,
            calibration_method=calibration_method,
            clear_gpu_memory=clear_gpu_memory,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method == "grouped_attention_contribution":
        num_q_heads = model_config.num_attention_heads
        attention = None
        # block_config carries per-layer KV heads for heterogeneous (AnyModel) checkpoints;
        # under force_hf the HF config has no block_configs, so fall back to the (homogeneous)
        # model config.
        if block_config is not None:
            attention = block_config.get_subblock("attention")
            if attention is None or attention.num_kv_heads is None:
                num_kv_heads = model_config.num_key_value_heads
            else:
                num_kv_heads = attention.num_kv_heads
                num_q_heads = attention.num_query_heads or num_q_heads
        else:
            num_kv_heads = model_config.num_key_value_heads
        head_dim = (
            getattr(attention, "qk_head_dim", None)
            if attention is not None
            else None
        ) or getattr(model_config, "head_dim", None) or (
            model_config.hidden_size // num_q_heads
        )
        return GroupedAttentionScorer(
            module,
            groups,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            optimize_for=optimize_for,
            validation_full_iters=validation_full_iters,
            calibration_method=calibration_method,
            clear_gpu_memory=clear_gpu_memory,
            scored_axes=scored_axes,
            token_chunk_size=token_chunk_size,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method == "mla_head_contribution":
        mla = block_config.get_subblock("mla") if block_config is not None else None
        num_heads = int(
            getattr(mla, "num_heads", None)
            or getattr(model_config, "num_attention_heads")
        )
        v_head_dim = int(getattr(model_config, "v_head_dim"))
        return GroupedAttentionScorer(
            module,
            groups,
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=v_head_dim,
            optimize_for=optimize_for,
            validation_full_iters=validation_full_iters,
            calibration_method=calibration_method,
            clear_gpu_memory=clear_gpu_memory,
            scored_axes=("kv_groups",),
            token_chunk_size=token_chunk_size,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method == "gdn_activation_contribution":
        return GatedDeltaNetActivationScorer(
            module,
            groups,
            token_chunk_size=token_chunk_size or 128,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method == "removed_expert_diff":
        top_k, num_experts, _, _, _ = _moe_dims(model_config, block_config)
        if top_k is None or num_experts is None:
            raise ValueError(f"removed_expert_diff needs top_k and num_experts for {module_name!r}")
        return MoEExpertRemovalDiffScorer(
            module,
            groups,
            top_k=top_k,
            num_experts=num_experts,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method in ("moe_channel", "moe_cett", "expert_intermediate_contribution"):
        _, num_experts, _, _, _ = _moe_dims(model_config, block_config)
        if validation_full_iters is None:
            raise ValueError("coupled MoE channel scoring requires validation_full_iters")
        if hasattr(module, "gate_and_up_projs") and hasattr(module, "down_projs"):
            return MoEGroupedExpertChannelScorer(
                module,
                groups,
                num_experts=num_experts,
                validation_full_iters=validation_full_iters,
                block_idx=block_idx,
                name=out_name(module_name),
            )
        raise ValueError(
            f"{method} requires native grouped experts so every expert's prior removals "
            f"share one EP-global residual; got {type(module).__name__} at {module_name!r}"
        )

    if method in ("shared_expert_intermediate_contribution", "moe_shared_channel"):
        if validation_full_iters is None:
            raise ValueError("shared-expert iterative scoring requires validation_full_iters")
        return MoESharedExpertChannelScorer(
            module,
            groups,
            validation_full_iters=validation_full_iters,
            calibration_method=calibration_method,
            clear_gpu_memory=clear_gpu_memory,
            token_sample_cap=max(
                1, int(os.environ.get("MOE_SHARED_SCORING_TOKENS_PER_BATCH", "128"))
            ),
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method == "moe_latent":
        _, num_experts, _, _, latent_dim = _moe_dims(model_config, block_config)
        if num_experts is None:
            raise ValueError(f"moe_latent needs num_experts for {module_name!r}")
        fc1 = getattr(module, "fc1_latent_proj", None)
        fc2 = getattr(module, "fc2_latent_proj", None)
        if latent_dim is None or not all(hasattr(proj, "weight") for proj in (fc1, fc2)):
            raise ValueError(f"moe_latent requested for non-latent MoE module {module_name!r}")
        return MoELatentCalibrationScorer(
            module,
            groups,
            num_experts=num_experts,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    if method == "mamba_head_and_dim":
        num_heads, head_dim, num_groups, state_dim = _mamba_dims(model_config, block_config)
        return MambaInProjContributionScorer(
            module,
            groups,
            num_heads=num_heads,
            head_dim=head_dim,
            num_groups=num_groups,
            state_dim=state_dim,
            block_idx=block_idx,
            name=out_name(module_name),
        )

    raise NotImplementedError(
        f"AutoModel scoring method {method!r} is not implemented yet. Supported: "
        "'independent' + 'iterative' (FFN), 'grouped_attention_contribution' "
        "and 'ple_channel_contribution' (per-layer embeddings), "
        "(attention), 'mla_head_contribution' (coupled MLA heads), "
        "'gdn_activation_contribution' (Qwen GatedDeltaNet), "
        "exact MoE removal/channel/latent methods, and MiniTron Mamba head/head-dim."
    )


def build_scorers(
    model,
    groups: MeshGroups,
    pruning_mixin,
    method: str,
    *,
    model_descriptor=None,
    block_configs=None,
    optimize_for: str = "memory",
    validation_full_iters: int | None = None,
    calibration_method: str | None = None,
    clear_gpu_memory: bool = False,
    scored_axes=None,
    token_chunk_size: int | None = None,
    register: bool = True,
) -> list:
    """Build (and optionally register) scorers for every target module on this rank.

    Args:
        model: The parallelized AutoModel.
        groups: Resolved mesh process groups.
        pruning_mixin: The puzzletron pruning mixin selecting the target modules.
        method: The scoring method (selects the scorer class).
        model_descriptor: Descriptor used to select the language-model config from
            native conditional/VL model configs.
        block_configs: Per-layer AnyModel block configs (list of dict/BlockConfig), read
            from the checkpoint (config.json / block_configs.json). Used for per-layer dims
            (e.g. KV heads) on heterogeneous teachers; ``None`` for a homogeneous model.
        optimize_for: Forwarded to scorers that accept it (attention).
        validation_full_iters: Number of iterations for the iterative FFN scorer
            (= eval_samples // micro_batch_size).
        calibration_method: Optional calibration for the iterative scorer.
        clear_gpu_memory: Aggressively free intermediates in the iterative scorer.
        register: Register each scorer as a forward hook on its module.

    Returns:
        The list of constructed scorers (each ``.name`` is the canonical module key).
    """
    # Per-layer configs come from the checkpoint (passed in explicitly), not from
    # model.config: under force_hf the loaded HF config does not carry block_configs (and
    # for VL models it is the nested text config). This mirrors the reference KD recipes,
    # which read block_configs from the checkpoint dir and pass them around explicitly.
    cast_block_configs = maybe_cast_block_configs(block_configs) if block_configs else None
    model_config = (
        model_descriptor.get_language_model_config(model.config)
        if model_descriptor is not None
        else model.config
    )

    scorers = []
    for block_idx, module_name in pruning_mixin.get_module_names_to_hook(model):
        # Legacy layer descriptors select modules by suffix.  Native models can
        # contain auxiliary decoder-shaped trees (for example MTP layers) whose
        # ``*.mlp.down_proj``/``*.self_attn.o_proj`` suffixes also match.  Those
        # modules are not teacher block configs and must never contribute to
        # pruning rankings.  Resolve the canonical decoder prefix through the
        # descriptor's model adapter and reject every auxiliary match.
        if model_descriptor is not None and block_idx is not None:
            decoder_prefix = model_descriptor.layer_block_name(int(block_idx))
            adapt_name = getattr(model_descriptor, "adapt_module_name_for_model", None)
            if callable(adapt_name):
                decoder_prefix = adapt_name(decoder_prefix, model)
            if module_name != decoder_prefix and not module_name.startswith(
                f"{decoder_prefix}."
            ):
                logger.debug(
                    "Skipping auxiliary activation target %s for block %s; "
                    "descriptor decoder prefix is %s",
                    module_name,
                    block_idx,
                    decoder_prefix,
                )
                continue
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            # Module not on this rank's shard (e.g. owned by another pipeline stage).
            continue
        if isinstance(module, (DummyModule, DummyBlock)):
            continue
        block_config = (
            cast_block_configs[block_idx]
            if (cast_block_configs is not None and block_idx is not None)
            else None
        )
        # The hook attaches to the loaded module by its real path (``module_name``); the score is
        # written under the descriptor's canonical name so the pruning step finds it regardless of
        # how the scoring model is structured (e.g. text-only vs VL builds of the same checkpoint).
        def canonicalize_name(raw_name: str) -> str:
            return _canonical_output_name(pruning_mixin, method, block_idx, raw_name)

        scorer = _make_scorer(
            method,
            module,
            groups,
            model_config,
            block_config,
            block_idx,
            module_name,
            optimize_for=optimize_for,
            validation_full_iters=validation_full_iters,
            calibration_method=calibration_method,
            clear_gpu_memory=clear_gpu_memory,
            scored_axes=scored_axes,
            token_chunk_size=token_chunk_size,
            canonicalize_name=canonicalize_name,
        )
        new_scorers = scorer if isinstance(scorer, list) else [scorer]
        for item in new_scorers:
            if register:
                item.register()
            scorers.append(item)

    if scorers:
        logger.info("Built %d %s scorers on this rank", len(scorers), method)
    return scorers
