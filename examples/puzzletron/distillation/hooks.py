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

"""PuzzletronHooks — heterogeneous AnyModel/Puzzletron extension for mbridge distillation.

Plugs into the ``DistillHooks`` extension points in ``megatron_bridge/distill.py`` to add:

- Per-layer ``block_configs`` support (heterogeneous Mamba/MoE/dense-attention layers).
- Provider patching (``apply_patch``, ``apply_distillation_patch``) so that ``provide()``
  automatically activates ``mbridge_patcher``.
- YAML / Hydra-style ``ConfigContainer`` overrides via ``_pretrain_common()`` + OmegaConf.
- Hybrid MoE aux-loss tracker size fix that prevents NCCL deadlocks when a PP stage has zero
  surviving MoE layers.
- ``teacher.finalize()`` call before training starts.

The reusable, ``DistillHooks``-independent logic lives in the library under
``modelopt.torch.puzzletron.plugins.mbridge`` (the layer/provider patchers,
``block_configs`` translation, and the distillation helpers imported below). This
module is the thin example-side glue that wires those into the mbridge
``DistillHooks`` lifecycle.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from _common import _get_block_configs, _get_model_descriptor, _load_hf_config
from _common import _load_bridge as _common_load_bridge
from megatron.bridge.training.config import ConfigContainer
from omegaconf import OmegaConf

from modelopt.torch.puzzletron.plugins.mbridge import (
    build_distill_config_container,
    install_hybrid_moe_aux_loss_size_fix,
    sync_teacher_config_from_student,
    sync_teacher_from_student,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "megatron_bridge"))
from distill import DistillHooks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PuzzletronHooks
# ---------------------------------------------------------------------------


class PuzzletronHooks(DistillHooks):
    """Extension hooks for heterogeneous AnyModel / Puzzletron distillation.

    Constructed before ``mbridge_main()`` is called. Eagerly loads HF configs
    and derives block_configs + AnyModel descriptors in ``__init__`` so all
    expensive pre-work happens before the first model load.

    Args:
        student_path: HF checkpoint path (or Hub model ID) for the student.
        student_converter: AnyModel converter key (from MODEL_REGISTRY) for the student.
        teacher_path: HF checkpoint path (or Hub model ID) for the teacher.
        teacher_converter: AnyModel converter key for the teacher.
        config_file: Optional path to a YAML override file (OmegaConf format).
        overrides: List of Hydra-style CLI overrides (e.g. ``["train.train_iters=50000"]``).
        trust_remote_code: Passed through to HF loading calls.
        patch_router_expert_bias: When True, patch Megatron router expert-bias
            helpers for heterogeneous MoE (see ``router_expert_bias_patch``).
    """

    def __init__(
        self,
        student_path: str,
        student_converter: str,
        teacher_path: str,
        teacher_converter: str,
        config_file: str | None,
        overrides: list[str],
        trust_remote_code: bool,
        patch_router_expert_bias: bool = False,
    ) -> None:
        self._student_path = student_path
        self._teacher_path = teacher_path
        self._config_file = config_file
        self._overrides = overrides
        self._patch_router_expert_bias = patch_router_expert_bias

        logger.info("PuzzletronHooks: loading HF configs to derive block_configs and descriptors")
        student_hf_cfg = _load_hf_config(student_path, trust_remote_code)
        teacher_hf_cfg = _load_hf_config(teacher_path, trust_remote_code)

        self._block_configs: dict[str, list | None] = {
            student_path: _get_block_configs(student_hf_cfg, student_converter),
            teacher_path: _get_block_configs(teacher_hf_cfg, teacher_converter),
        }
        self._descriptors: dict[str, Any] = {
            student_path: _get_model_descriptor(student_converter),
            teacher_path: _get_model_descriptor(teacher_converter),
        }
        student_blocks = self._block_configs[student_path]
        teacher_blocks = self._block_configs[teacher_path]
        logger.info(
            "  student block_configs: %s",
            f"{len(student_blocks)} layers" if student_blocks else "None (homogeneous)",
        )
        logger.info(
            "  teacher block_configs: %s",
            f"{len(teacher_blocks)} layers" if teacher_blocks else "None (homogeneous)",
        )

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def before_load_models(self, args: argparse.Namespace) -> None:
        """Apply class-level provider patches required for heterogeneous models."""
        from modelopt.torch.puzzletron.plugins.mbridge import apply_distillation_patch, apply_patch

        logger.info("PuzzletronHooks: applying provider patches")
        # Patches ModelProviderMixin.provide() so teacher's provide() activates mbridge_patcher.
        apply_patch()
        # Patches DistillationProvider.provide() so the student build (which calls
        # _super_class.provide directly, bypassing the class-level patch) is also wrapped.
        apply_distillation_patch()

    def load_bridge(self, path: str, trust_remote_code: bool):
        """Load an HF checkpoint via deci_x_patcher when an AnyModel descriptor is available."""
        return _common_load_bridge(path, trust_remote_code, self._descriptors[path])

    def after_providers_created(self, student_provider, teacher_provider) -> None:
        """Set hetereogenous_dist_checkpoint on student and sync all parallelism to teacher."""
        student_provider.hetereogenous_dist_checkpoint = True
        sync_teacher_from_student(student_provider, teacher_provider)

    def after_distill_provider_created(self, distill_provider) -> None:
        """Attach per-layer block_configs to student and teacher providers."""
        from modelopt.torch.puzzletron.plugins.mbridge import (
            set_provider_block_configs,
            set_student_block_configs,
        )

        logger.info("PuzzletronHooks: attaching block_configs to student and teacher")
        set_student_block_configs(distill_provider, self._block_configs[self._student_path])
        set_provider_block_configs(
            distill_provider.teacher, self._block_configs[self._teacher_path]
        )

    def build_config(
        self,
        distill_provider,
        args: argparse.Namespace,
        checkpoint_dir: str | None,
        tensorboard_dir: str | None,
    ) -> ConfigContainer:
        """Build ConfigContainer via _pretrain_common() and apply YAML / Hydra overrides."""
        from megatron.bridge.training.utils.omegaconf_utils import (
            apply_overrides,
            create_omegaconf_dict_config,
            parse_hydra_overrides,
        )

        logger.info("PuzzletronHooks: building ConfigContainer via _pretrain_common()")
        config = build_distill_config_container(distill_provider, self._student_path)

        merged_cfg, excluded_fields = create_omegaconf_dict_config(config)

        if self._config_file and os.path.exists(self._config_file):
            logger.info("  Loading YAML overrides from: %s", self._config_file)
            yaml_overrides = OmegaConf.load(self._config_file)
            merged_cfg = OmegaConf.merge(merged_cfg, yaml_overrides)
        elif self._config_file:
            logger.warning(
                "  Config file not found: %s — skipping YAML overrides", self._config_file
            )

        if self._overrides:
            logger.info("  Applying CLI overrides: %s", self._overrides)
            merged_cfg = parse_hydra_overrides(merged_cfg, self._overrides)

        final_cfg_dict = OmegaConf.to_container(merged_cfg, resolve=True)

        # Bridge's apply_overrides re-binds the teacher's hybrid_override_pattern /
        # mtp_hybrid_override_pattern fields when applying the merged config dict
        # (they live on the student, not the teacher, in the merged dict). Snapshot
        # them so we can put them back on distill_provider.teacher afterwards.
        teacher_hop = getattr(distill_provider.teacher, "hybrid_override_pattern", None)
        teacher_mtp_hop = getattr(distill_provider.teacher, "mtp_hybrid_override_pattern", None)
        apply_overrides(config, final_cfg_dict, excluded_fields)
        object.__setattr__(distill_provider.teacher, "hybrid_override_pattern", teacher_hop)
        object.__setattr__(distill_provider.teacher, "mtp_hybrid_override_pattern", teacher_mtp_hop)

        sync_teacher_config_from_student(config.model)
        return config

    def after_config_built(self, config: ConfigContainer) -> None:
        """Install the hybrid MoE aux-loss tracker size fix."""
        install_hybrid_moe_aux_loss_size_fix(config)

    def before_distill(self, config: ConfigContainer) -> None:
        """Finalize the teacher provider before training starts."""
        config.model.teacher.finalize()
