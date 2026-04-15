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

"""Configurations for speculative decoding modes."""

import warnings
from copy import deepcopy
from typing import Any

from pydantic import ValidationInfo, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

from .eagle.default_config import default_eagle_config, default_kimik2_eagle_config

kimik2_eagle_default_config = deepcopy(default_kimik2_eagle_config)

eagle3_default_config = deepcopy(default_eagle_config)
eagle_mtp_default_config = deepcopy(default_eagle_config)

eagle3_default_config.update({"use_aux_hidden_state": True, "use_last_layernorm": True})
eagle_mtp_default_config.update({"use_last_layernorm": True, "use_mtp_layernorm": True})


EAGLE3_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_architecture_config": eagle3_default_config,
    },
}

EAGLE_MTP_DEFAULT_CFG = {
    "algorithm": "eagle",
    "config": {
        "eagle_reuse_base_decoder": True,
        "eagle_architecture_config": eagle_mtp_default_config,
    },
}


def _get_dflash_default_config():
    from .dflash.default_config import default_dflash_config

    return default_dflash_config


DFLASH_DEFAULT_CFG = {
    "algorithm": "dflash",
    "config": {
        "dflash_architecture_config": {},  # merged with default at convert time
    },
}


class DFlashConfig(ModeloptBaseConfig):
    """DFlash config for block-wise parallel speculative decoding."""

    dflash_block_size: int = ModeloptField(
        default=8,
        description="Block size for parallel prediction. Draft predicts this many tokens per block.",
    )

    dflash_freeze_base_model: bool = ModeloptField(
        default=True, description="Whether to freeze base model during DFlash module training."
    )

    dflash_self_logit_distillation: bool = ModeloptField(
        default=True, description="Whether to use logit distillation from base model."
    )

    dflash_loss_decay_factor: float = ModeloptField(
        default=0.0,
        description="Gamma for exponential loss decay weighting (paper Eq.4). "
        "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. 0 disables.",
    )

    dflash_num_anchors: int = ModeloptField(
        default=512,
        description="Number of random anchor positions sampled per sequence during training.",
    )

    dflash_report_acc: bool = ModeloptField(
        default=True, description="Whether to report eval accuracy."
    )

    dflash_mask_token_id: int = ModeloptField(
        default=None,
        description="Token ID used for masked (unknown) positions. "
        "Set explicitly or auto-detected from tokenizer.mask_token_id in main.py.",
    )

    dflash_architecture_config: dict = ModeloptField(
        default={}, description="Config for the DFlash draft module architecture."
    )

    dflash_use_torch_compile: bool = ModeloptField(
        default=True,
        description="Whether to use torch.compile on DFlash forward/loss methods.",
    )


class MedusaConfig(ModeloptBaseConfig):
    """Medusa config."""

    medusa_num_heads: int = ModeloptField(
        default=2,
        description=("The number of medusa heads added to the model."),
    )

    medusa_num_layers: int = ModeloptField(
        default=1,
        description=("The number of ResBlocks used in medusa head."),
    )


class EagleConfig(ModeloptBaseConfig):
    """Eagle config."""

    eagle_offline: bool = ModeloptField(
        default=False, description=("Whether to use detached Eagle.")
    )

    eagle_hidden_state_distillation: bool = ModeloptField(
        default=False, description=("Whether to use feature hidden states distillation.")
    )

    eagle_self_logit_distillation: bool = ModeloptField(
        default=True, description=("Whether to use logit distillation.")
    )

    eagle_freeze_base_model: bool = ModeloptField(
        default=True, description=("Whether to freeze base model during eagle module training.")
    )

    eagle_report_acc: bool = ModeloptField(
        default=True, description=("Whether to report eval accuracy.")
    )

    eagle_reuse_base_decoder: bool = ModeloptField(
        default=False, description=("Whether to reuse base model decoder in eagle module.")
    )

    eagle_loss_decay_factor: float = ModeloptField(
        default=0.9, description=("The decay factor for multiple eagle_loss.")
    )

    eagle_architecture_config: dict = ModeloptField(
        default={}, description=("The config for eagle module architecture.")
    )

    eagle_decoder_type: str = ModeloptField(
        default="llama",
        description=("The class of eagle decoder to use. Available options: llama, kimik2"),
    )

    eagle_ttt_steps: int = ModeloptField(
        default=3, description=("The number of train-time-test steps in training.")
    )

    eagle_mix_hidden_states: bool = ModeloptField(
        default=False,
        description=(
            "Whether to mix hidden states of multiple TTT steps. It is a technique to reduce training cost."
        ),
    )

    eagle_use_torch_compile: bool = ModeloptField(
        default=True,
        description="Whether to use torch.compile on eagle forward/loss methods for faster training.",
    )

    eagle_enable_nvtx: bool = ModeloptField(
        default=False,
        description="Whether to enable NVTX ranges for profiling eagle forward/loss methods.",
    )

    eagle_export_rope_scaling: dict = ModeloptField(
        default={"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 2048},
        description=(
            "The rope_scaling config to inject into the exported HuggingFace model config. "
            "Applied when the training rope_type is 'default' (no scaling). "
            "Set to empty dict {} to disable rope scaling injection at export."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_eagle_offline(cls, data: Any, info: ValidationInfo) -> Any:
        """Derive ``eagle_offline`` from ``data_args.offline_data_path`` when provided in context."""
        ctx = info.context if info.context else {}
        data_args = ctx.get("data_args")
        if data_args is not None and isinstance(data, dict):
            data["eagle_offline"] = data_args.offline_data_path is not None
        return data

    @model_validator(mode="after")
    def _check_rope_scaling_consistency(self) -> "EagleConfig":
        if not self.eagle_export_rope_scaling:
            return self
        rope_cfg = self.eagle_architecture_config.get("rope_scaling", {}) or {}
        rope_type = rope_cfg.get("rope_type") or rope_cfg.get("type")
        if rope_type is not None and rope_type != "default":
            raise ValueError(
                f"eagle_export_rope_scaling is set but eagle_architecture_config has "
                f"rope_type='{rope_type}'. Export rope overwrite is only valid when the "
                f"training rope_type is 'default' (no scaling)."
            )
        return self

    @model_validator(mode="after")
    def _warn_rope_vs_training_seq_len(self, info: ValidationInfo) -> "EagleConfig":
        ctx = info.context if info.context else {}
        training_args = ctx.get("training_args")
        if training_args is None:
            return self
        orig_max_pos = self.eagle_export_rope_scaling.get("original_max_position_embeddings")
        if orig_max_pos is not None and orig_max_pos != training_args.training_seq_len:
            warnings.warn(
                f"eagle_export_rope_scaling.original_max_position_embeddings ({orig_max_pos}) "
                f"differs from training_seq_len ({training_args.training_seq_len}). "
                f"This may affect long-context inference quality."
            )
        return self
