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

"""Modify state_dict and config for exporting speculative decoding in official format."""

import json
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .hf_spec_configs import kimik2_eagle_template_config, llama_eagle_template_config

ALL_SPEC_MODES = ["eagle", "dflash"]

LLAMA_EAGLE_SINGLE_LAYER = {
    "required": {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
        "layers.0.hidden_norm",
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "norm",
        "fc",
    },
    "optional": {"d2t", "lm_head"},
}

KIMIK2_EAGLE_SINGLE_LAYER = {
    "required": {
        "layers.0.self_attn.kv_a_layernorm",
        "layers.0.self_attn.q_a_layernorm",
        "layers.0.self_attn.q_a_proj",
        "layers.0.self_attn.q_b_proj",
        "layers.0.self_attn.kv_a_proj_with_mqa",
        "layers.0.self_attn.kv_b_proj",
        "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
        "layers.0.hidden_norm",
        "layers.0.input_layernorm",
        "layers.0.post_attention_layernorm",
        "norm",
        "fc",
    },
    "optional": {
        "d2t",
        "lm_head",
    },
}


def has_spec_opt(model: nn.Module):
    """Check if the model has speculative decoding optimization."""
    opt_modes = getattr(model, "_modelopt_state", [])
    return any(mode[0] in ALL_SPEC_MODES for mode in opt_modes)


def has_quant_opt(model: nn.Module):
    """Check if the model has quantization optimization."""
    opt_modes = getattr(model, "_modelopt_state", [])
    return any(mode[0] == "quantize" for mode in opt_modes)


class SpeculativeDecodingExporter(ABC):
    """Export a modelopt speculative decoding checkpoint to deployment format."""

    def __init__(self, model: nn.Module):
        """Initialize the SpeculativeDecodingExporter."""
        self.model = model

    @abstractmethod
    def export(
        self,
        export_dir: Path | str,
        dtype: torch.dtype | None = None,
    ):
        """Export the model to the deployment format."""
        raise NotImplementedError("Subclasses must implement this method.")


class EagleExporter(SpeculativeDecodingExporter):
    """Draft model exporter for Eagle."""

    def __init__(self, model: nn.Module):
        """Initialize the EagleExporter."""
        super().__init__(model)
        self.eagle_decoder_type = model.eagle_config.eagle_decoder_type
        self.num_hidden_layers = model.eagle_config.num_hidden_layers

    def _check_valid_sd(self, export_sd: dict):
        """Check the export state dict is valid, otherwise raise Exception."""
        expected_keys_single_layer = {
            "llama": LLAMA_EAGLE_SINGLE_LAYER,
            "kimik2": KIMIK2_EAGLE_SINGLE_LAYER,
        }[self.eagle_decoder_type]
        # fc and hidden_norm are only present when use_aux_hidden_state=True
        use_aux = getattr(self.model.eagle_config, "use_aux_hidden_state", False)
        aux_only_keys = {"fc", "layers.0.hidden_norm"}
        required_keys = set(expected_keys_single_layer["required"])
        if not use_aux:
            required_keys -= aux_only_keys
        # Check that export sd has required keys
        for key in required_keys:
            assert f"{key}.weight" in export_sd, f"Missing required key: {key}.weight"
        for i in range(1, self.num_hidden_layers):
            for key in required_keys - {
                "layers.0.hidden_norm",
                "layers.0.input_layernorm",
                "norm",
                "fc",
            }:
                assert f"{key}.weight".replace("layers.0", f"layers.{i}") in export_sd, (
                    f"Missing required key: {key}.weight"
                )

        # Check that export sd has no unexpected keys
        # Note that quantized eagle are allowed to have scales
        allowed_keys_single_layer = (
            expected_keys_single_layer["required"] | expected_keys_single_layer["optional"]
        )
        for key in export_sd:
            assert (
                re.sub(r"layers\.\d+\.", "layers.0.", key.rsplit(".", 1)[0])
                in allowed_keys_single_layer
            ), f"Unexpected key: {key}"

    def _extract_state_dict(self, full_state_dict: dict):
        """Extract and return eagle state dict in deployment format."""
        export_sd = {}
        for key in full_state_dict:
            if "eagle_module" in key:
                export_key = key.replace("eagle_module.", "")
                export_sd[export_key] = full_state_dict[key].clone()
        # Use base model's lm head if draft model doesn't have one
        if "lm_head.weight" not in export_sd:
            export_sd["lm_head.weight"] = full_state_dict["lm_head.weight"]

        self._check_valid_sd(export_sd)

        return export_sd

    def _export_config(self):
        """Export config.json in deployment format."""
        template_config: dict = {
            "llama": llama_eagle_template_config,
            "kimik2": kimik2_eagle_template_config,
        }[self.model.eagle_config.eagle_decoder_type]
        template_config = deepcopy(template_config)

        def _get_config_from_draft_or_base(key: str, model: nn.Module):
            if getattr(model.eagle_config, key, None) is not None:
                return getattr(model.eagle_config, key)
            elif getattr(model.config, key, None) is not None:
                return getattr(model.config, key)
            else:
                return None

        for key in template_config:
            value = template_config[key]
            if isinstance(value, dict):
                # for eagle config, we find it in model.eagle_config
                for sub_key in value:
                    if value[sub_key] is None:
                        value[sub_key] = _get_config_from_draft_or_base(sub_key, self.model)
            elif value is None:
                # First, we try to load from eagle config.
                new_value = _get_config_from_draft_or_base(key, self.model)
                # If the value is a torch.dtype, we convert to string for serialization.
                if isinstance(new_value, torch.dtype):
                    new_value = str(new_value).replace("torch.", "")
                template_config[key] = new_value

        # Inject export rope scaling override when training rope_type is "default".
        rope_cfg = self.model.eagle_config.rope_scaling or {}
        training_rope_type = rope_cfg.get("rope_type") or rope_cfg.get("type")
        eagle_export_rope_scaling = getattr(self.model, "eagle_export_rope_scaling", None)
        if eagle_export_rope_scaling and training_rope_type == "default":
            template_config["rope_scaling"] = eagle_export_rope_scaling

        # In transformers 5.x, rope_theta is under rope_scaling, not the main config.
        # Always source from the training rope config (rope_theta is not in export overrides).
        if template_config.get("rope_theta") is None and rope_cfg:
            template_config["rope_theta"] = rope_cfg.get("rope_theta")

        return template_config

    def _export_lora(self, export_dir: Path, full_sd: dict):
        """Export base model LoRA adapter weights alongside the eagle module artifacts."""
        from peft import LoraConfig

        lora_sd = {k: v for k, v in full_sd.items() if ".lora_A." in k or ".lora_B." in k}
        if not lora_sd:
            raise RuntimeError(
                "No LoRA adapter tensors found in the model state dict. "
                "Ensure eagle_base_lora=True and the model was converted with LoRA adapters."
            )
        # Reformat keys to match peft's standard adapter file format:
        # 1. Strip the adapter name (e.g. ".default") — peft re-inserts it on load.
        # 2. Add "base_model.model." prefix — peft expects this in saved adapters.
        lora_sd = {
            re.sub(r"(lora_[AB])\.\w+\.weight$", r"\1.weight", k): v for k, v in lora_sd.items()
        }
        lora_sd = {f"base_model.model.{k}": v for k, v in lora_sd.items()}
        save_file(lora_sd, export_dir / "adapter_model.safetensors")

        # Infer target modules from the exported LoRA keys (e.g., "q_proj", "v_proj")
        # Keys are like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        target_modules = sorted({k.split(".")[-3] for k in lora_sd if ".lora_A." in k})
        lora_config = LoraConfig(
            r=self.model.eagle_base_lora_rank,
            lora_alpha=self.model.eagle_base_lora_alpha,
            target_modules=target_modules,
            bias="none",
        )
        with open(export_dir / "adapter_config.json", "w") as f:
            json.dump(
                lora_config.to_dict(),
                f,
                indent=4,
                default=lambda o: sorted(o) if isinstance(o, set) else o,
            )

    def export(
        self,
        export_dir: Path | str,
        dtype: torch.dtype | None = None,
    ):
        """Export the model to the deployment format."""
        # Make export dir
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export quantized modules
        if has_quant_opt(self.model):
            from ..unified_export_hf import _export_transformers_checkpoint

            full_sd, hf_quant_config = _export_transformers_checkpoint(self.model, dtype)
        else:
            full_sd, hf_quant_config = self.model.state_dict(), None

        # Export state dict
        drafter_sd = self._extract_state_dict(full_sd)
        save_file(drafter_sd, f"{export_dir}/model.safetensors")

        # Export config
        drafter_config = self._export_config()
        if hf_quant_config is not None:
            drafter_config["quantization_config"] = hf_quant_config
        with open(f"{export_dir}/config.json", "w") as file:
            json.dump(drafter_config, file, indent=4)

        # Export hf_quant_config for backward compatibility
        if hf_quant_config is not None:
            with open(f"{export_dir}/hf_quant_config.json", "w") as file:
                json.dump(hf_quant_config, file, indent=4)

        # Export LoRA adapter weights separately
        if getattr(self.model, "eagle_base_lora", False):
            self._export_lora(export_dir, full_sd)


class EagleMedusaExporter(EagleExporter):
    """Draft model exporter for EagleMedusa."""

    def __init__(self, model: nn.Module):
        """Initialize the EagleMedusaExporter."""
        super().__init__(model)
        self.parallel_draft_step = model.eagle_config.parallel_draft_step
        self.parallel_draft_heads_num_layers = model.eagle_config.parallel_draft_heads_num_layers
        # NOTE: tmp: bypassing format check for parallel draft
        self._check_valid_sd = lambda *args, **kwargs: None

    def _extract_state_dict(self, full_state_dict: dict):
        """Extract the state dict of the draft model in deployment format."""
        export_sd = super()._extract_state_dict(full_state_dict)
        if self.parallel_draft_step <= 1:
            return export_sd

        for i in range(self.parallel_draft_step - 1):
            for j in range(self.parallel_draft_heads_num_layers):
                export_sd[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.weight"] = (
                    export_sd.pop(f"parallel_draft_heads.medusa_heads.{i}.{j}.linear.weight")
                )
                if f"parallel_draft_heads.medusa_heads.{i}.{j}.linear.bias" in export_sd:
                    export_sd[f"parallel_draft_heads.{i}.medusa_layers.{j}.linear.bias"] = (
                        export_sd.pop(f"parallel_draft_heads.medusa_heads.{i}.{j}.linear.bias")
                    )
        return export_sd


class DFlashExporter(SpeculativeDecodingExporter):
    """Draft model exporter for DFlash.

    Exports in z-lab compatible format:
    - model.safetensors: draft module weights (no prefix)
    - config.json: Qwen3-style config with dflash_config field
    """

    def _extract_state_dict(self, full_state_dict: dict):
        """Extract DFlash module weights, stripping the dflash_module prefix."""
        export_sd = {}
        for key, value in full_state_dict.items():
            if "dflash_module." in key:
                export_key = key.split("dflash_module.", 1)[1]
                # Skip rotary embedding buffers (not needed, recomputed)
                if "rotary_emb" in export_key:
                    continue
                export_sd[export_key] = value.clone()
        return export_sd

    def _export_config(self):
        """Build config.json matching z-lab DFlash format."""
        model = self.model
        base_config = (
            getattr(model.config, "text_config", None)
            or getattr(model.config, "llm_config", None)
            or model.config
        )
        draft_config = model.dflash_config

        config = {
            "architectures": ["DFlashDraftModel"],
            # DFlash draft uses Qwen3 architecture regardless of base model type.
            # Use "qwen3" as default; base model types like "qwen3_5_text" are not
            # recognized by vLLM for draft model loading.
            "model_type": "qwen3",
            "block_size": model.dflash_block_size,
            "dflash_config": {
                "mask_token_id": model.mask_token_id,
                "target_layer_ids": list(model.target_layer_ids),
            },
            # Architecture dimensions
            "hidden_size": getattr(draft_config, "hidden_size", base_config.hidden_size),
            "num_hidden_layers": draft_config.num_hidden_layers,
            "num_attention_heads": getattr(
                draft_config, "num_attention_heads", base_config.num_attention_heads
            ),
            "num_key_value_heads": getattr(
                draft_config, "num_key_value_heads", base_config.num_key_value_heads
            ),
            "head_dim": getattr(
                draft_config,
                "head_dim",
                base_config.hidden_size // base_config.num_attention_heads,
            ),
            "intermediate_size": getattr(
                draft_config, "intermediate_size", base_config.intermediate_size
            ),
            "hidden_act": getattr(draft_config, "hidden_act", "silu"),
            "rms_norm_eps": getattr(draft_config, "rms_norm_eps", 1e-6),
            "vocab_size": base_config.vocab_size,
            "max_position_embeddings": getattr(base_config, "max_position_embeddings", 32768),
            "initializer_range": getattr(base_config, "initializer_range", 0.02),
            "attention_bias": getattr(draft_config, "attention_bias", False),
            "attention_dropout": getattr(draft_config, "attention_dropout", 0.0),
            "rope_theta": getattr(
                draft_config, "rope_theta", getattr(base_config, "rope_theta", 1000000.0)
            ),
            # DFlash draft uses standard Qwen3 RoPE, not M-RoPE from multimodal models.
            # z-lab uses null; vLLM handles null rope_scaling correctly.
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": str(getattr(base_config, "torch_dtype", torch.bfloat16)).replace(
                "torch.", ""
            ),
            "num_target_layers": base_config.num_hidden_layers,
        }

        # Add layer_types if present (Qwen3-style)
        if hasattr(draft_config, "layer_types"):
            config["layer_types"] = draft_config.layer_types
        else:
            config["layer_types"] = ["full_attention"] * draft_config.num_hidden_layers

        return config

    def export(self, export_dir: Path | str, dtype: torch.dtype | None = None):
        """Export the DFlash draft model to deployment format."""
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export quantized modules if applicable
        if has_quant_opt(self.model):
            from ..unified_export_hf import _export_transformers_checkpoint

            full_sd, hf_quant_config = _export_transformers_checkpoint(self.model, dtype)
        else:
            full_sd, hf_quant_config = self.model.state_dict(), None

        # Export state dict
        drafter_sd = self._extract_state_dict(full_sd)
        assert drafter_sd, "No dflash_module weights found in state dict"
        if dtype is not None and hf_quant_config is None:
            drafter_sd = {k: v.to(dtype) for k, v in drafter_sd.items()}
        save_file(drafter_sd, f"{export_dir}/model.safetensors")

        # Export config
        drafter_config = self._export_config()
        # Update torch_dtype to match actual exported weights
        if dtype is not None:
            drafter_config["torch_dtype"] = str(dtype).replace("torch.", "")
        if hf_quant_config is not None:
            drafter_config["quantization_config"] = hf_quant_config
        with open(f"{export_dir}/config.json", "w") as f:
            json.dump(drafter_config, f, indent=2)

        if hf_quant_config is not None:
            with open(f"{export_dir}/hf_quant_config.json", "w") as f:
                json.dump(hf_quant_config, f, indent=2)

        print(
            f"Exported DFlash draft model: {len(drafter_sd)} tensors, "
            f"config keys: {list(drafter_config.keys())[:5]}..."
        )
