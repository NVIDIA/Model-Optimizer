#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from megatron.bridge import AutoBridge
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
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0
from modelopt.torch.puzzletron.anymodel.converter import *
from modelopt.torch.puzzletron.anymodel.model_descriptor import *

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps --student / --teacher key → (hf_model_id, anymodel_converter_name)
#
#   hf_model_id:          Default load path when --{student,teacher}-checkpoint is omitted.
#   anymodel_converter:   Key for ConverterFactory and ModelDescriptorFactory.
#                         Also determines block_configs generation fallback.
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gptoss": ("openai/gpt-oss-20b",                         "gpt_oss"),
    "nemo":   ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron_h_v2"),
    "llama":  ("meta-llama/Llama-3.2-3B-Instruct",           "llama"),
    "qwen":   ("Qwen/Qwen3-8B",                              "qwen3"),
}

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILE: Path = SCRIPT_DIR / "kd-container-default.yaml"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_hf_config(load_path: str, trust_remote_code: bool):
    """Load only the HuggingFace config (no weights) from a path or Hub model ID."""
    from transformers import AutoConfig
    logger.info("Loading HF config from %r", load_path)
    return AutoConfig.from_pretrained(load_path, trust_remote_code=trust_remote_code)


def _get_block_configs(hf_config, converter_name: str) -> Optional[list]:
    """Load or generate block_configs for a model.

    Priority:
        1. ``hf_config.block_configs`` — set by AnyModel when saving (canonical source).
        2. ``ConverterFactory`` — generated from global model config (fallback).
        3. ``None`` — homogeneous model (no per-layer overrides).
    """
    from block_config_utils import load_block_configs
    return load_block_configs(hf_config, converter_name)


def _get_model_descriptor(converter_name: str):
    """Return the AnyModel ModelDescriptor for the given converter name, or None."""
    try:
        print(f"### converter_name: {converter_name}")
        from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
        print(f"### ModelDescriptorFactory: {ModelDescriptorFactory}")
        descriptor = ModelDescriptorFactory.get(converter_name)
        if descriptor is None:
            logger.warning("No AnyModel descriptor found for converter '%s'", converter_name)
        return descriptor
    except ImportError:
        logger.warning("ModelOpt AnyModel not installed; cannot obtain model descriptor")
        return None


def _load_bridge(load_path: str, trust_remote_code: bool, descriptor) -> "AutoBridge":
    """Load an HF model into a Megatron Bridge object.

    If an AnyModel descriptor is available, the model is loaded inside ``deci_x_patcher``,
    which patches the HF model's ``from_pretrained`` path to correctly construct
    heterogeneous layers (different sub-layer types per slot).  For standard homogeneous
    models the patcher is a no-op, so using it unconditionally is safe.
    """
    if descriptor is not None:
        from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
        logger.info("Loading HF model via deci_x_patcher (descriptor=%s)", type(descriptor).__name__)
        with deci_x_patcher(model_descriptor=descriptor):
            return AutoBridge.from_hf_pretrained(load_path, trust_remote_code=trust_remote_code)
    else:
        logger.info("Loading HF model without deci_x_patcher (AnyModel not available)")
        return AutoBridge.from_hf_pretrained(load_path, trust_remote_code=trust_remote_code)


def _build_provider(bridge: "AutoBridge") -> "ModelProviderMixin":
    """Convert a Bridge to a Megatron model provider with weight loading registered."""
    return bridge.to_megatron_provider(load_weights=True)


# ---------------------------------------------------------------------------
# ConfigContainer construction
# ---------------------------------------------------------------------------


def _build_distill_config_container(distill_provider, student_checkpoint_path: str) -> ConfigContainer:
    """Build a :class:`ConfigContainer` for distillation using Bridge ``_pretrain_common()`` defaults.

    Sets ``model`` to the :class:`DistillationProvider`, points the HF tokenizer at the
    student checkpoint when using ``HuggingFaceTokenizer``, and aligns dataset ``seq_length``
    with the provider. YAML / CLI overrides are applied in :func:`main` via
    :func:`apply_overrides` on the full container.
    """
    cfg = _pretrain_common()
    cfg.model = distill_provider  # type: ignore[assignment]

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = student_checkpoint_path

    provider_seq_len = getattr(distill_provider, "seq_length", None)
    if provider_seq_len:
        cfg.dataset.seq_length = provider_seq_len

    return cfg


def _sync_teacher_config_from_student(distill_provider) -> None:
    """Copy shared parallelism / checkpoint layout fields from student to teacher provider."""
    _SHARED_PARALLEL_ATTRS = (
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
    for attr in _SHARED_PARALLEL_ATTRS:
        if hasattr(teacher, attr):
            setattr(teacher, attr, getattr(distill_provider, attr))
    for attr in ("tensor_model_parallel_size", "pipeline_model_parallel_size"):
        if getattr(distill_provider, attr) != getattr(teacher, attr):
            raise RuntimeError(
                f"Teacher/student {attr} mismatch after re-sync: "
                f"student={getattr(distill_provider, attr)}, teacher={getattr(teacher, attr)}."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:  # noqa: C901 (complexity OK for an orchestration fn)
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
    from provider_patch import (
        apply_distillation_patch,
        apply_patch,
        set_provider_block_configs,
        set_student_block_configs,
    )
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

    student_provider.seq_length = 4096
    student_provider.tensor_model_parallel_size = 1
    student_provider.sequence_parallel = student_provider.tensor_model_parallel_size > 1
    student_provider.pipeline_model_parallel_size = 2
    student_provider.pipeline_dtype = torch.bfloat16
    student_provider.context_parallel_size = 1
    student_provider.expert_model_parallel_size = 1
    student_provider.expert_tensor_parallel_size = 1
    student_provider.hetereogenous_dist_checkpoint = True

    # Fix teacher to match student
    teacher_provider.seq_length = student_provider.seq_length
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
    apply_overrides(config, final_cfg_dict, excluded_fields)

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

    try:
        distill(config=config)
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
    logger.info("  CLI overrides:      %s", args.overrides)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Model selection ---
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
        "--student-checkpoint",
        default=None,
        metavar="PATH",
        dest="student_checkpoint",
        help=(
            "Local directory of an HF-format student checkpoint. "
            "If omitted, the model is loaded from HuggingFace Hub."
        ),
    )
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

    # --- Configuration ---
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
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace config/model loading.",
    )

    # --- Hydra-style pass-through overrides ---
    # Everything that doesn't match a known flag is treated as a Hydra dotlist override.
    # Example: model.tensor_model_parallel_size=4 train.train_iters=100000
    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    return args


if __name__ == "__main__":
    args = _parse_args()
    try:
        main(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
