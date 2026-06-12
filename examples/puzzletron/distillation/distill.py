#!/usr/bin/env python3
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

"""Knowledge distillation with heterogeneous AnyModel/Puzzletron models via Megatron Bridge.

Overview
--------
This script runs knowledge distillation (KD) where either or both of the student and teacher
may be heterogeneous AnyModel/Puzzletron checkpoints — models in which each decoder layer can
have a different architecture (Mamba, MoE, dense attention, etc.).

The heterogeneous support is provided by the ``mbridge_patcher`` context manager from
``layer_patchers.py``, which intercepts Megatron-Core's per-layer construction and injects
per-layer config overrides from ``block_configs``.  Two class-level patches are applied
before training:

- ``apply_patch()`` — patches ``ModelProviderMixin.provide()`` so teacher's ``provide()``
  automatically activates ``mbridge_patcher``.
- ``apply_distillation_patch()`` — patches ``DistillationProvider.provide()`` so the
  student model build (which goes through ``self._super_class.provide(self, ...)``, bypassing
  the standard provider patch) is also wrapped in ``mbridge_patcher``.

After both patches are applied, ``set_student_block_configs()`` and
``set_provider_block_configs()`` attach per-layer configs to the student and teacher providers
respectively.

Pipeline
--------
1.  Load HF configs for both student and teacher (config.json only, no weights).
2.  Read ``block_configs`` from each HF config (set by AnyModel when saving a heterogeneous
    checkpoint).  Fall back to generating them via the AnyModel ``ConverterFactory`` for
    models not yet saved with AnyModel.
3.  Apply class-level provider patches (``apply_patch``, ``apply_distillation_patch``).
4.  Load each model's HF weights via ``AutoBridge.from_hf_pretrained()`` (inside
    ``deci_x_patcher`` to handle heterogeneous HF model construction).
5.  Convert each bridge to a Megatron provider via ``bridge.to_megatron_provider()``.
6.  Configure parallelism on both providers (defaults; YAML can override).
7.  Wrap student + teacher into a ``DistillationProvider`` via
    ``convert_to_distillation_provider()``.
8.  Attach per-layer ``block_configs`` to student and teacher.
9.  Build a Bridge ``ConfigContainer`` (``_pretrain_common()`` + ``model = distill_provider``),
    merge ``--config-file`` YAML and Hydra-style CLI overrides onto the **full** container,
    then sync shared fields to the teacher provider.
10. Call Bridge's ``distill()`` to run training.

Block config sources (priority order)
--------------------------------------
1. ``hf_config.block_configs`` — canonical source, set by AnyModel when saving.
2. ``ConverterFactory`` — generates default block_configs from global model config.
3. ``None`` — homogeneous model; global TransformerConfig used for all layers.

Supported models (``--student`` / ``--teacher``)
-------------------------------------------------
    gpt    → openai/gpt-oss-20b                          (GPT-OSS, all-MoE)
    nemo   → nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  (Nemotron-H, Mamba+MoE hybrid)
    llama  → meta-llama/Llama-3.2-3B-Instruct            (dense GPT)
    qwen   → Qwen/Qwen3-8B                               (dense GPT with GQA)

Usage
-----
    # Minimal: Nemotron-H teacher → Llama student (1 GPU)
    torchrun --nproc_per_node=1 distill.py \\
        --student llama --teacher nemo \\
        --teacher-checkpoint /path/to/nemotronh

    # From local checkpoints, with custom YAML config:
    torchrun --nproc_per_node=8 distill.py \\
        --student llama --student-checkpoint /path/to/student \\
        --teacher nemo  --teacher-checkpoint /path/to/teacher \\
        --config-file kd.yaml

    # CLI overrides (Hydra-style), overriding YAML defaults:
    torchrun --nproc_per_node=8 distill.py \\
        --student llama --teacher nemo \\
        model.tensor_model_parallel_size=4 \\
        train.train_iters=50000 \\
        optimizer.lr=1e-4

    # Parallelism flags (convenience; same fields can be set via YAML or Hydra dotlist):
    torchrun --nproc_per_node=8 distill.py \\
        --student llama --teacher nemo \\
        --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \\
        --expert-model-parallel-size 4 --expert-tensor-parallel-size 1

    # All-heterogeneous: GPT-OSS teacher → Nemotron-H student
    torchrun --nproc_per_node=8 distill.py \\
        --student nemo  --student-checkpoint /path/to/student \\
        --teacher gpt   --teacher-checkpoint /path/to/teacher \\
        model.tensor_model_parallel_size=4 \\
        model.expert_model_parallel_size=4

Configuration precedence
------------------------
1. Defaults from ``_pretrain_common()`` embedded in ``_build_distill_config_container()``.
2. YAML file from ``--config-file`` (defaults to ``kd-container-default.yaml`` if present).
3. Hydra-style CLI overrides (highest priority).

All sections of the YAML (``model``, ``train``, ``checkpoint``, ``dataset``, …) are applied to
the :class:`~megatron.bridge.training.config.ConfigContainer`.

KD-specific settings (in YAML or as CLI ``model.kd_config.*``):
    logit_layers, intermediate_layer_pairs, skip_lm_loss,
    kd_loss_scale, logit_kl_temperature.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import torch
from _common import (
    DEFAULT_CONFIG_FILE,
    MODEL_REGISTRY,
    _build_provider,
    _get_block_configs,
    _get_model_descriptor,
    _load_bridge,
    _load_hf_config,
    configure_logging,
    run_entrypoint,
)
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.anymodel.converter import *  # noqa: F403
from modelopt.torch.puzzletron.anymodel.model_descriptor import *  # noqa: F403

configure_logging()
logger = logging.getLogger(__name__)

if os.environ.get("MBRIDGE_PATCHER_DEBUG", "0").lower() in ("1", "true", "yes", "on"):
    logging.getLogger("layer_patchers").setLevel(logging.DEBUG)
    logging.getLogger("provider_patch").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# ConfigContainer construction
# ---------------------------------------------------------------------------


def _build_distill_config_container(
    distill_provider, student_checkpoint_path: str
) -> ConfigContainer:
    """Build a :class:`ConfigContainer` for distillation using Bridge ``_pretrain_common()`` defaults.

    Sets ``model`` to the :class:`DistillationProvider`, points the HF tokenizer at the
    student checkpoint when using ``HuggingFaceTokenizer``, and aligns dataset ``seq_length``
    with the provider. YAML / CLI overrides are applied in :func:`main` via
    :func:`apply_overrides` on the full container.
    """
    cfg = _pretrain_common()
    cfg.model = distill_provider

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = student_checkpoint_path

    provider_seq_len = getattr(distill_provider, "seq_length", None)
    if provider_seq_len:
        cfg.dataset.seq_length = provider_seq_len

    return cfg


def _sync_teacher_config_from_student(distill_provider) -> None:
    """Copy shared parallelism / checkpoint layout fields from student to teacher provider."""
    shared_parallel_attrs = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "pipeline_dtype",
        "seq_length",
        "hetereogenous_dist_checkpoint",
    )
    teacher = getattr(distill_provider, "teacher", None)
    if teacher is None:
        return
    for attr in shared_parallel_attrs:
        if hasattr(teacher, attr):
            setattr(teacher, attr, getattr(distill_provider, attr))


# ---------------------------------------------------------------------------
# Hybrid MoE aux-loss tracker fix
# ---------------------------------------------------------------------------
# Bridge's `training_log` passes ``num_layers = hybrid_layer_pattern.count("E")``
# (e.g. 13) to ``track_moe_metrics`` for hybrid models, but Megatron-Core MoE
# routers write into ``tracker[name]["values"]`` of size ``self.config.num_layers``
# (e.g. 56 — the *full* pattern length, see core/transformer/moe/router.py).
#
# When ``_apply_no_ops`` (in layer_patchers.py) replaces the MLP of a fully
# pruned ``E`` slot with ``NoOpWithBias`` and resets ``is_moe_layer=False``, the
# router for that layer never fires.  If a particular PP stage ends up with
# *zero* surviving MoE layers, no router on that stage writes to the tracker,
# so ``track_moe_metrics(..., force_initialize=True)`` creates the entry with
# size 13 on those ranks while ranks on stages with surviving MoE layers still
# carry size-56 tensors from their routers.
#
# The first cross-PP ``all_reduce`` inside
# ``reduce_aux_losses_tracker_across_ranks`` then deadlocks (NCCL silently
# hangs on mismatched buffer sizes).  py-spy stack:
#     all_reduce -> reduce_aux_losses_tracker_across_ranks
#                -> track_moe_metrics -> training_log
#
# Fix: monkey-patch ``track_moe_metrics`` to override the ``num_layers``
# argument with ``config.num_layers`` (matching the router) and synthesise
# ``moe_layer_freq`` from ``hybrid_layer_pattern`` so the per-MoE-layer
# averaging denominator (``num_moe_layers`` at moe_utils.py:1090) stays
# correct (= count of "E" slots).
def _install_hybrid_moe_aux_loss_size_fix(config: ConfigContainer) -> None:
    """Align track_moe_metrics's tracker size with what MoE routers actually use.

    No-op for non-hybrid models or when no MoE experts are configured.
    """
    model_cfg = config.model
    rank0 = get_rank_safe() == 0

    def _info(msg: str, *args: Any) -> None:
        # Use print (not logger) so the message survives even if the logger is
        # not yet configured / muted on rank 0 at this point in startup.
        if rank0:
            print("[MoEAuxFix-install] " + (msg % args if args else msg), flush=True)

    is_hybrid = getattr(model_cfg, "is_hybrid_model", False)
    num_experts = getattr(model_cfg, "num_moe_experts", None)
    pattern: str = getattr(model_cfg, "hybrid_layer_pattern", "") or ""
    full_num_layers: int | None = getattr(model_cfg, "num_layers", None)
    block_configs = getattr(model_cfg, "block_configs", None)

    # For Puzzletron heterogeneous students, ``hybrid_layer_pattern`` is empty
    # because layer types are encoded per-layer in ``block_configs`` instead.
    # We therefore derive ``moe_layer_freq`` from ``block_configs`` when
    # available, and otherwise fall back to None (only affects the per-layer
    # logging denominator, not the all_reduce size).
    moe_layer_freq: list[int] | None = None
    if pattern:
        moe_layer_freq = [1 if c == "E" else 0 for c in pattern]
    elif block_configs:
        derived: list[int] = []
        for bc in block_configs:
            ffn = getattr(bc, "ffn", None) if not isinstance(bc, dict) else bc.get("ffn")
            if ffn is None:
                derived.append(0)
                continue
            ffn_no_op = (
                getattr(ffn, "no_op", False)
                if not isinstance(ffn, dict)
                else ffn.get("no_op", False)
            )
            ffn_type = (
                getattr(ffn, "block_type", None)
                if not isinstance(ffn, dict)
                else ffn.get("block_type")
            )
            ffn_type_str = str(ffn_type).lower() if ffn_type is not None else ""
            is_moe = (not ffn_no_op) and ("moe" in ffn_type_str or "expert" in ffn_type_str)
            derived.append(1 if is_moe else 0)
        if any(derived):
            moe_layer_freq = derived

    moe_count = sum(moe_layer_freq) if moe_layer_freq else 0
    _info(
        "called: model_cls=%s is_hybrid_model=%s num_moe_experts=%s "
        "num_layers=%s hybrid_layer_pattern_len=%d count_E=%d "
        "block_configs_len=%s moe_layer_freq_sum=%s",
        type(model_cfg).__name__,
        is_hybrid,
        num_experts,
        full_num_layers,
        len(pattern),
        pattern.count("E"),
        len(block_configs) if block_configs is not None else None,
        moe_count if moe_layer_freq is not None else None,
    )

    if not is_hybrid:
        _info("skip: is_hybrid_model is False")
        return
    if num_experts is None:
        _info("skip: num_moe_experts is None")
        return
    if full_num_layers is None:
        _info("skip: num_layers missing")
        return

    import megatron.core.transformer.moe.moe_utils as _moe_utils

    # Bridge does `from ...moe_utils import track_moe_metrics` (named import),
    # so train_utils binds the original at import time and patching only
    # _moe_utils.track_moe_metrics has NO effect on the binding train_utils
    # actually calls. We must patch BOTH module dicts: the canonical source
    # AND every importer that already grabbed the name.
    targets: list[tuple[Any, str]] = [(_moe_utils, "track_moe_metrics")]
    try:
        import megatron.bridge.training.utils.train_utils as _bridge_train_utils

        if hasattr(_bridge_train_utils, "track_moe_metrics"):
            targets.append((_bridge_train_utils, "track_moe_metrics"))
    except ImportError:
        _info("skip: megatron.bridge.training.utils.train_utils not importable")

    if getattr(_moe_utils.track_moe_metrics, "_hybrid_size_fix_applied", False):
        _info("skip: already applied (idempotent)")
        return

    orig_track_moe_metrics = _moe_utils.track_moe_metrics

    def _describe_moe_layer_freq(value: Any) -> str:
        # Megatron-Core overloads moe_layer_freq as either an int (periodicity),
        # a list[int] (per-layer 0/1 mask), or None. The diagnostic must handle
        # all three or it will explode (e.g. sum(int) -> TypeError).
        if value is None:
            return "None"
        if isinstance(value, (list, tuple)):
            return f"list(len={len(value)}, sum={sum(value)})"
        return f"scalar({value!r})"

    def _patched_track_moe_metrics(*args: Any, **kwargs: Any) -> Any:
        kwargs["num_layers"] = full_num_layers
        kwargs.setdefault("moe_layer_freq", moe_layer_freq)
        if kwargs.get("moe_layer_freq") is None:
            kwargs["moe_layer_freq"] = moe_layer_freq

        # Per-rank diagnostics around the cross-rank all_reduces inside
        # track_moe_metrics → reduce_aux_losses_tracker_across_ranks.  If a
        # rank prints "[MoEAuxFix] enter" but never prints "[MoEAuxFix] exit",
        # that rank is the one stuck in the all_reduce.  Compare the printed
        # tracker buffer sizes across ranks — they MUST all match (= full_num_layers
        # + (mtp_num_layers or 0)) for the PP all_reduce to succeed.
        rank = get_rank_safe()
        iteration = kwargs.get("iteration", "?")
        if logger.isEnabledFor(logging.DEBUG):
            tracker = _moe_utils.get_moe_layer_wise_logging_tracker()
            sizes_before = {k: tuple(v["values"].shape) for k, v in tracker.items()}
            keys_before = sorted(tracker.keys())
            logger.debug(
                "[MoEAuxFix] enter rank=%s iter=%s num_layers_arg=%s "
                "moe_layer_freq=%s tracker_keys=%s tracker_sizes=%s",
                rank,
                iteration,
                kwargs["num_layers"],
                _describe_moe_layer_freq(kwargs["moe_layer_freq"]),
                keys_before,
                sizes_before,
            )
        try:
            ret = orig_track_moe_metrics(*args, **kwargs)
        except Exception as exc:
            logger.error("[MoEAuxFix] EXCEPTION rank=%s iter=%s: %r", rank, iteration, exc)
            raise
        logger.debug("[MoEAuxFix] exit rank=%s iter=%s", rank, iteration)
        return ret

    _patched_track_moe_metrics._hybrid_size_fix_applied = True

    patched_locations = []
    for mod, attr in targets:
        try:
            setattr(mod, attr, _patched_track_moe_metrics)
            patched_locations.append(f"{mod.__name__}.{attr}")
        except Exception as exc:
            _info("warn: failed to patch %s.%s: %r", mod.__name__, attr, exc)

    moe_slots = sum(moe_layer_freq) if moe_layer_freq else 0
    _info(
        "INSTALLED: num_layers=%d MoE slots=%s (out of %d) patched=%s",
        full_num_layers,
        moe_slots if moe_layer_freq is not None else "unknown",
        len(pattern),
        patched_locations,
    )
    logger.info(
        "Installed hybrid MoE aux-loss tracker size fix: num_layers=%d, "
        "MoE slots=%s (out of %d), patched=%s",
        full_num_layers,
        moe_slots if moe_layer_freq is not None else "unknown",
        len(pattern),
        patched_locations,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    _validate_registry_keys(args)

    student_hf_id, student_converter = MODEL_REGISTRY[args.student]
    teacher_hf_id, teacher_converter = MODEL_REGISTRY[args.teacher]
    student_path = args.student_checkpoint or student_hf_id
    teacher_path = args.teacher_checkpoint or teacher_hf_id

    _log_config(args, student_path, teacher_path, student_converter, teacher_converter)

    # ------------------------------------------------------------------
    # Step 1: Load HF configs (no weights, just config.json)
    #   The HF config is the canonical source for block_configs.
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading HF configs")
    student_hf_cfg = _load_hf_config(student_path, args.trust_remote_code)
    teacher_hf_cfg = _load_hf_config(teacher_path, args.trust_remote_code)

    # ------------------------------------------------------------------
    # Step 2: Obtain block_configs for student and teacher
    #   None → homogeneous model (no per-layer overrides needed).
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading block_configs")
    student_block_configs = _get_block_configs(student_hf_cfg, student_converter)
    teacher_block_configs = _get_block_configs(teacher_hf_cfg, teacher_converter)
    logger.info(
        "  student block_configs: %s",
        f"{len(student_block_configs)} layers" if student_block_configs else "None (homogeneous)",
    )
    logger.info(
        "  teacher block_configs: %s",
        f"{len(teacher_block_configs)} layers" if teacher_block_configs else "None (homogeneous)",
    )

    # ------------------------------------------------------------------
    # Step 3: Get AnyModel descriptors (needed for deci_x_patcher)
    # ------------------------------------------------------------------
    logger.info("Step 3: Getting AnyModel descriptors")
    student_descriptor = _get_model_descriptor(student_converter)
    teacher_descriptor = _get_model_descriptor(teacher_converter)

    # ------------------------------------------------------------------
    # Step 4: Apply class-level provider patches (one-time setup)
    #
    #   apply_patch():             Patches ModelProviderMixin.provide() so that
    #                              teacher.provide() activates mbridge_patcher.
    #   apply_distillation_patch(): Patches DistillationProvider.provide() so that
    #                              the student build (which calls _super_class.provide
    #                              directly, bypassing the class-level patch) is also
    #                              wrapped in mbridge_patcher.
    # ------------------------------------------------------------------
    logger.info("Step 4: Applying provider patches")
    from model_bridge_patch import apply_patch as apply_model_bridge_patch
    from provider_patch import (
        apply_distillation_patch,
        apply_patch,
        set_provider_block_configs,
        set_student_block_configs,
    )

    # Patches the upstream Megatron-Bridge model_bridge module to (a) make
    # _megatron_local_name_to_global heterogeneous-EP-aware and (b) report
    # any Megatron parameter the HF->Megatron load left untouched (which is
    # the iter-1 NaN root cause for heterogeneous students).
    # Replaces the previous bind-mount of a full upstream model_bridge.py copy.
    apply_model_bridge_patch()
    apply_patch()
    apply_distillation_patch()

    # ------------------------------------------------------------------
    # Step 5: Load HF models into Bridge objects
    #   deci_x_patcher patches the HF model's __init__ to construct
    #   heterogeneous layers (different sub-layer types per slot).
    # ------------------------------------------------------------------
    logger.info("Step 5: Loading HF models into Megatron Bridge")
    student_bridge = _load_bridge(student_path, args.trust_remote_code, student_descriptor)
    teacher_bridge = _load_bridge(teacher_path, args.trust_remote_code, teacher_descriptor)

    logger.info("  student bridge: %s", type(student_bridge).__name__)
    logger.info("  teacher bridge: %s", type(teacher_bridge).__name__)

    # ------------------------------------------------------------------
    # Step 6: Convert bridges to Megatron providers
    #   to_megatron_provider(load_weights=True) registers a pre_wrap_hook
    #   that loads HF weights into the MCore model before DDP wrapping.
    # ------------------------------------------------------------------
    logger.info("Step 6: Converting bridges to Megatron providers")
    student_provider = _build_provider(student_bridge)
    teacher_provider = _build_provider(teacher_bridge)
    logger.info("  student provider: %s", type(student_provider).__name__)
    logger.info("  teacher provider: %s", type(teacher_provider).__name__)

    student_provider.tensor_model_parallel_size = args.tensor_model_parallel_size
    student_provider.sequence_parallel = student_provider.tensor_model_parallel_size > 1
    student_provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    student_provider.pipeline_dtype = torch.bfloat16
    student_provider.context_parallel_size = 1
    student_provider.expert_model_parallel_size = args.expert_model_parallel_size
    student_provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    student_provider.hetereogenous_dist_checkpoint = True

    # Teacher is always aligned to student parallelism settings.
    teacher_provider.tensor_model_parallel_size = student_provider.tensor_model_parallel_size
    teacher_provider.sequence_parallel = student_provider.sequence_parallel
    teacher_provider.pipeline_model_parallel_size = student_provider.pipeline_model_parallel_size
    teacher_provider.pipeline_dtype = student_provider.pipeline_dtype
    teacher_provider.context_parallel_size = student_provider.context_parallel_size
    teacher_provider.expert_model_parallel_size = student_provider.expert_model_parallel_size
    teacher_provider.expert_tensor_parallel_size = student_provider.expert_tensor_parallel_size
    teacher_provider.hetereogenous_dist_checkpoint = student_provider.hetereogenous_dist_checkpoint

    # ------------------------------------------------------------------
    # Step 7: Create DistillationProvider
    #   convert_to_distillation_provider() mutates student_provider's
    #   __class__ to DistillationProvider and attaches teacher + kd_config.
    # ------------------------------------------------------------------
    logger.info("Step 7: Creating DistillationProvider")
    kd_config = ModelOptDistillConfig()

    distill_provider = convert_to_distillation_provider(
        student_provider, teacher_provider, kd_config
    )

    logger.info(
        "  DistillationProvider created (student=%s, teacher=%s)",
        distill_provider._super_class.__name__,
        type(teacher_provider).__name__,
    )

    # ------------------------------------------------------------------
    # Step 8: Attach block_configs to student and teacher
    #
    #   set_student_block_configs(): stores student_block_configs on the
    #     DistillationProvider; apply_distillation_patch() (Step 4) ensures
    #     it is activated during DistillationProvider.provide().
    #
    #   set_provider_block_configs(teacher, ...): patches teacher.provide()
    #     at the instance level to activate mbridge_patcher automatically.
    # ------------------------------------------------------------------
    logger.info("Step 8: Attaching block_configs")
    set_student_block_configs(distill_provider, student_block_configs)
    set_provider_block_configs(distill_provider.teacher, teacher_block_configs)

    # ------------------------------------------------------------------
    # Step 9: ConfigContainer + YAML / CLI (full Bridge container, not model-only)
    # ------------------------------------------------------------------
    logger.info("Step 9: Building ConfigContainer and applying YAML / CLI overrides")
    config = _build_distill_config_container(distill_provider, student_path)

    merged_cfg, excluded_fields = create_omegaconf_dict_config(config)

    config_file = args.config_file
    if config_file and os.path.exists(config_file):
        logger.info("  Loading YAML overrides from: %s", config_file)
        yaml_overrides = OmegaConf.load(config_file)
        merged_cfg = OmegaConf.merge(merged_cfg, yaml_overrides)
    elif config_file and not os.path.exists(config_file):
        logger.warning("  Config file not found: %s — skipping YAML overrides", config_file)

    if args.overrides:
        logger.info("  Applying CLI overrides: %s", args.overrides)
        merged_cfg = parse_hydra_overrides(merged_cfg, args.overrides)

    final_cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)

    # Bridge's apply_overrides(...) re-binds the teacher's hybrid_override_pattern
    # / mtp_hybrid_override_pattern fields when applying the merged config dict
    # (they live on the student, not the teacher, in the merged dict). Snapshot
    # them so we can put them back on distill_provider.teacher afterwards.
    teacher_hop = getattr(distill_provider.teacher, "hybrid_override_pattern", None)
    teacher_mtp_hop = getattr(distill_provider.teacher, "mtp_hybrid_override_pattern", None)
    apply_overrides(config, final_cfg_dict, excluded_fields)
    object.__setattr__(distill_provider.teacher, "hybrid_override_pattern", teacher_hop)
    object.__setattr__(distill_provider.teacher, "mtp_hybrid_override_pattern", teacher_mtp_hop)

    _sync_teacher_config_from_student(config.model)

    # ------------------------------------------------------------------
    # Step 10: Run distillation
    #
    #   distill(cfg) → pretrain(cfg, forward_step_modelopt)
    #
    #   During the first forward pass, Bridge calls:
    #     cfg.model.provide_distributed_model()
    #       → DistillationProvider.provide() [patched by apply_distillation_patch]
    #         → mbridge_patcher(student_block_configs) wraps _super_class.provide
    #           → student MCore model is built with per-layer overrides
    #         → teacher.provide() [patched by set_provider_block_configs]
    #           → mbridge_patcher(teacher_block_configs) wraps original provide
    #             → teacher MCore model is built with per-layer overrides
    #         → mtd.convert(student, kd_loss_mode) → DistillationModel
    # ------------------------------------------------------------------
    logger.info("Step 10: Starting distillation")
    logger.info("Rank: %s", get_rank_safe())

    config.model.teacher.finalize()

    # Fix Bridge↔MCore tracker-size mismatch that deadlocks training_log when a
    # PP stage has zero surviving MoE layers (see helper docstring above).
    _install_hybrid_moe_aux_loss_size_fix(config)

    try:
        distill(config)
    except Exception as e:
        logger.error("Error during distillation: %s", e)
        raise e
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_registry_keys(args: argparse.Namespace) -> None:
    valid = sorted(MODEL_REGISTRY)
    if args.student not in MODEL_REGISTRY:
        raise ValueError(f"Unknown --student '{args.student}'. Valid choices: {valid}")
    if args.teacher not in MODEL_REGISTRY:
        raise ValueError(f"Unknown --teacher '{args.teacher}'. Valid choices: {valid}")


def _log_config(
    args: argparse.Namespace,
    student_path: str,
    teacher_path: str,
    student_converter: str,
    teacher_converter: str,
) -> None:
    logger.info("=== mbridge_distillation_v2/distill ===")
    logger.info("  student key:        %s  (converter=%s)", args.student, student_converter)
    logger.info("  student load path:  %s", student_path)
    logger.info("  teacher key:        %s  (converter=%s)", args.teacher, teacher_converter)
    logger.info("  teacher load path:  %s", teacher_path)
    logger.info("  config file:        %s", args.config_file)
    logger.info(
        "  parallelism:        TP=%s PP=%s EP=%s ETP=%s",
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.expert_tensor_parallel_size,
    )
    logger.info("  CLI overrides:      %s", args.overrides)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --student, --trust-remote-code ---
    parser.add_argument(
        "--student",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help=(
            "Student model key. Determines the HuggingFace model ID (used when "
            "--student-checkpoint is omitted) and the AnyModel converter for block_configs."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace config/model loading.",
    )

    # --- Student checkpoint ---
    parser.add_argument(
        "--student-checkpoint",
        default=None,
        metavar="PATH",
        dest="student_checkpoint",
        help=(
            "Local directory of an HF-format student checkpoint. "
            "If omitted, the model is loaded from HuggingFace Hub."
        ),
    )

    # --- Teacher ---
    parser.add_argument(
        "--teacher",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help=(
            "Teacher model key. Determines the HuggingFace model ID (used when "
            "--teacher-checkpoint is omitted) and the AnyModel converter for block_configs."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        metavar="PATH",
        dest="teacher_checkpoint",
        help=(
            "Local directory of an HF-format teacher checkpoint. "
            "If omitted, the model is loaded from HuggingFace Hub."
        ),
    )

    # --- Configuration file ---
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE) if DEFAULT_CONFIG_FILE.exists() else None,
        metavar="YAML",
        help=(
            "Path to a YAML override file (OmegaConf format). "
            "Merged on top of the base config before CLI overrides are applied. "
            "Defaults to kd-container-default.yaml in the script directory (if it exists)."
        ),
    )

    # --- Megatron parallelism (student + teacher providers; YAML / Hydra can still override) ---
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="tensor_model_parallel_size",
        help="Tensor model parallel size (TP). Also enables sequence parallel when > 1.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="pipeline_model_parallel_size",
        help="Pipeline model parallel size (PP).",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="expert_model_parallel_size",
        help="Expert model parallel size (EP) for MoE.",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        metavar="N",
        dest="expert_tensor_parallel_size",
        help="Expert tensor parallel size (ETP) for MoE.",
    )

    # --- Hydra-style pass-through overrides ---
    # Everything that doesn't match a known flag is treated as a Hydra dotlist override.
    # Example: model.tensor_model_parallel_size=4 train.train_iters=100000
    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    return args


if __name__ == "__main__":
    run_entrypoint(main, _parse_args)
