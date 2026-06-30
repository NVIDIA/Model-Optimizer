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

from copy import deepcopy

from pydantic import model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

from .eagle.default_config import default_eagle_config, default_kimik2_eagle_config

kimik2_eagle_default_config = deepcopy(default_kimik2_eagle_config)

eagle3_default_config = deepcopy(default_eagle_config)
eagle_mtp_default_config = deepcopy(default_eagle_config)

eagle3_default_config.update({"use_aux_hidden_state": True, "use_last_layernorm": True})
eagle_mtp_default_config.update({"use_last_layernorm": True, "use_mtp_layernorm": True})

_EAGLE_HSM_MODES = ("sparse_replace", "uniform_layer_sample", "no_replace")


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

    dflash_offline: bool = ModeloptField(
        default=False,
        description=(
            "Whether to use detached DFlash (offline training from pre-computed hidden states). "
            "Derived by ModelOptDFlashRecipe from data.offline_data_path; not user-configurable."
        ),
    )

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

    dflash_mask_token_id: int | None = ModeloptField(
        default=None,
        description=(
            "Token ID used for masked (unknown) positions. Set explicitly in the recipe YAML, "
            "or left unset to fall back to ``tokenizer.mask_token_id`` at training time."
        ),
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
        default=False,
        description=(
            "Whether the Eagle module consumes pre-computed hidden states (offline or streaming) "
            "instead of running the base model in-process. Derived by ModelOptEagleRecipe from "
            "``data.mode``; not user-configurable."
        ),
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

    eagle_hsm_mode: str = ModeloptField(
        default="sparse_replace",
        description=(
            "Hidden state mixing variant used when eagle_mix_hidden_states=True. "
            "Available options: sparse_replace, uniform_layer_sample."
        ),
    )

    eagle_share_kv: bool = ModeloptField(
        default=False,
        description=(
            "Whether to share K/V across TTT steps: each draft self-attention captures its "
            "post-RoPE (K, V) at step 0 and reuses them in subsequent steps instead of "
            "recomputing. This narrows the train-infer gap by mimicking the inference-time "
            "KV-cache reuse. Disables the TTT triangular mask (the original eagle_attn_mask_0 "
            "is reused, like with eagle_mix_hidden_states). Composable with eagle_mix_hidden_states. "
            "Memory cost: each TTT step retains the prior step's (K, V) activations with gradients. "
            "Not supported with flex_attention or torch.compile."
        ),
    )

    eagle_share_kv_roll_query: bool = ModeloptField(
        default=False,
        description=(
            "When eagle_share_kv is on, roll Q right by +1 inside the draft attention so "
            "Q'_i attends to K_0..K_{i+1}. Preserves K_0 in the context and gives correct "
            "RoPE relative positions, at the cost of monkey-patching the HF attention "
            "function. Requires eagle_share_kv=True."
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

    eagle_base_lora: bool = ModeloptField(
        default=False,
        description=(
            "Whether to add LoRA adapters to the base model for co-training with the EAGLE module. "
            "Requires the `peft` library. Incompatible with eagle_offline=True."
        ),
    )

    eagle_base_lora_rank: int = ModeloptField(
        default=64,
        description="LoRA rank for the base model adapters.",
    )

    eagle_base_lora_alpha: float = ModeloptField(
        default=16.0,
        description="LoRA alpha (scaling) for the base model adapters.",
    )

    eagle_base_lora_target_modules: list | None = ModeloptField(
        default=None,
        description=(
            "List of module name patterns to apply LoRA to in the base model "
            "(e.g. ['q_proj', 'v_proj']). None uses peft defaults."
        ),
    )

    eagle_base_lora_preservation_loss_weight: float = ModeloptField(
        default=0.1,
        description=(
            "Weight for the preservation loss that minimizes the KL divergence between "
            "the LoRA-adapted base model output and the original base model output."
        ),
    )

    eagle_base_lora_warmup_steps: int = ModeloptField(
        default=0,
        description=(
            "Number of warmup steps where LoRA is frozen and only the EAGLE draft head trains. "
            "After warmup, LoRA is enabled for co-training."
        ),
    )

    eagle_base_lora_logits_detach_prob: float = ModeloptField(
        default=0.5,
        description=(
            "After warmup, probability of detaching base_output_softmax_logits each step. "
            "Acts as dropout regularization on the eagle-loss-to-LoRA gradient path through "
            "logits, preventing LoRA from degenerating to maximize EAGLE accuracy at the cost "
            "of base model quality. 1.0 = always detach (no logits gradient), 0.0 = never detach."
        ),
    )

    @model_validator(mode="after")
    def _check_hsm_mode_consistency(self) -> "EagleConfig":
        if self.eagle_hsm_mode not in _EAGLE_HSM_MODES:
            raise ValueError(
                f"eagle_hsm_mode must be one of {_EAGLE_HSM_MODES}, got {self.eagle_hsm_mode!r}."
            )
        if self.eagle_hsm_mode != "sparse_replace" and not self.eagle_mix_hidden_states:
            raise ValueError(
                f"eagle_hsm_mode={self.eagle_hsm_mode!r} requires eagle_mix_hidden_states=True."
            )
        return self

    @model_validator(mode="after")
    def _check_share_kv_consistency(self) -> "EagleConfig":
        if self.eagle_share_kv_roll_query and not self.eagle_share_kv:
            raise ValueError("eagle_share_kv_roll_query=True requires eagle_share_kv=True.")
        return self

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
