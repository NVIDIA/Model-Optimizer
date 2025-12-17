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

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


"""Code that export quantized Megatron Core models for deployment."""

import json
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any
from warnings import warn

import torch
import torch.distributed
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

from modelopt import __version__
from modelopt.torch.utils import import_plugin

from .model_config import (
    KV_CACHE_FP8,
    KV_CACHE_NVFP4,
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_FP8_PB_WO,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
)
from .plugins.mcore_common import all_mcore_hf_export_mapping
from .plugins.mcore_custom import CustomModuleMapping, save_safetensors
from .plugins.megatron_importer import GPTModelImporter
from .quant_utils import (
    get_activation_scaling_factor,
    get_kv_cache_dtype,
    get_quantization_format,
    get_scaling_factor,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    to_quantized_weight,
)

with import_plugin("transformers", verbose=False):
    import transformers
    from transformers import AutoProcessor

has_mcore = False
with import_plugin("megatron"):
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.mamba import MambaModel
    from megatron.core.models.multimodal.llava_model import LLaVAModel
    from megatron.core.parallel_state import (
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
    )
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.torch_norm import L2Norm
    from megatron.core.transformer.transformer_layer import TransformerLayer

    has_mcore = True

__all__ = [
    "export_mcore_gpt_to_hf",
    "import_mcore_gpt_from_hf",
]


# This path uses output_quantizer for KV cache quantization.
# The function below is the old version of get_kv_cache_scaling_factor which is now refactored to handle bmm_quantizer.
def get_kv_cache_scaling_factor(kv_module: nn.Module) -> torch.Tensor:
    """Returns the kv_cache scaling factor if output quantizer is set. Else returns None by default."""
    scaling_factor = (
        get_scaling_factor(kv_module.output_quantizer)
        if hasattr(kv_module, "output_quantizer")
        else None
    )

    if not scaling_factor:
        return None

    # For FP8, we recommend default kv cache scaling factor to be 1.
    if get_kv_cache_dtype(kv_module) == KV_CACHE_FP8:
        if scaling_factor.item() > 0.5:
            warn(
                f"!!!!Large KV activations detected: {scaling_factor.item()}, "
                "Quantized KV cache may lead to higher accuracy drop.\n!!!!"
            )
        scaling_factor = torch.max(
            scaling_factor,
            torch.tensor([1.0], dtype=torch.float, device=scaling_factor.device),
        )
    return scaling_factor


class GPTModelExporter:
    """Megatron Core GPTModel Exporter.

    The Exporter is created by `export_mcore_gpt_to_hf` to host attributes
    and methods that export a quantized Megatron Core GPTModel to the Hugging
    Face unified checkpoint.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        export_extra_modules: If True, export extra modules like medusa_heads or
            eagle_module. Otherwise, only export the base model.
        dtype: The weights data type to export the unquantized layers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        export_extra_modules: bool = False,
        dtype=torch.bfloat16,
        trust_remote_code: bool = True,
        moe_router_dtype: torch.dtype | None = None,
    ):
        """Create a GPTModel exporter instance."""
        if not isinstance(model, (GPTModel, MambaModel, LLaVAModel)):
            raise ValueError("Input to GPTModelExport must be a megatron.core.models.GPTModel!")

        self._state_dict = OrderedDict()
        self._hf_pretrained_model_name = pretrained_model_name_or_path
        self._hf_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.moe_router_dtype = None
        if moe_router_dtype == "fp32":
            self.moe_router_dtype = torch.float32
        elif moe_router_dtype == "fp64":
            self.moe_router_dtype = torch.float64

        # If multimodal, extra the text_config
        self._hf_text_config = getattr(self._hf_config, "text_config", self._hf_config)

        # Update hf_config
        self._hf_text_config.num_hidden_layers = model.config.num_layers
        self._hf_text_config.hidden_size = model.config.hidden_size
        self._hf_text_config.head_dim = model.config.kv_channels
        self._hf_text_config.num_attention_heads = model.config.num_attention_heads
        self._hf_text_config.num_key_value_heads = model.config.num_query_groups
        self.is_multimodal = isinstance(model, LLaVAModel)
        if not self.is_multimodal:
            self._hf_text_config.intermediate_size = model.config.ffn_hidden_size
        self._hf_quant_config = None
        self._hf_extra_config = None
        self.export_extra_modules = export_extra_modules
        self.is_multimodal = isinstance(model, LLaVAModel)
        self.model = model.language_model if self.is_multimodal else model
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.arch = self._hf_config.architectures[0]
        # TODO: May modify this later according to what quantization exported ckpt is, currently only support BF16.
        if self.arch == "GptOssForCausalLM":
            if hasattr(self._hf_config, "quantization_config"):
                del self._hf_config.quantization_config
        self.all_rules = self._populate_rule_book()
        self.rules = self.all_rules[self.arch]

        if not hasattr(model, "_modelopt_state"):
            return

        for mode, mode_cfg in model._modelopt_state:
            if mode == "medusa" and export_extra_modules:
                medusa_config = {
                    "num_medusa_heads": mode_cfg["config"]["medusa_num_heads"],
                    "num_medusa_layers": mode_cfg["config"]["medusa_num_layers"],
                }
                self._hf_config.medusa = medusa_config
                self.rules = self.all_rules["MedusaLlamaForCausalLM"]

            if mode == "eagle" and export_extra_modules:
                if mode_cfg["config"]["eagle_architecture_config"]["use_aux_hidden_state"]:
                    if mode_cfg["config"]["eagle_architecture_config"]["num_hidden_layers"] > 1:
                        architectures = "LlamaForCausalLMEagle3Deep"
                    else:
                        architectures = "LlamaForCausalLMEagle3"
                else:
                    architectures = "LlamaForCausalLMEagle"

                self.rules = self.all_rules[architectures]

                if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
                    # By default, we use Llama-3.1
                    self._hf_extra_config = transformers.AutoConfig.from_pretrained(
                        "nvidia/Llama-3.1-8B-Instruct-FP8", trust_remote_code=self.trust_remote_code
                    )

                    eagle_config = {
                        "use_input_layernorm_in_first_layer": model.eagle_config.use_input_layernorm_in_first_layer,
                        "use_last_layernorm": model.eagle_config.use_last_layernorm,
                        "use_mtp_layernorm": model.eagle_config.use_mtp_layernorm,
                        "use_aux_hidden_state": model.eagle_config.use_aux_hidden_state,
                        "eagle_aux_hidden_state_layer_ids": model.eagle_config.eagle_aux_hidden_state_layer_ids,
                        "next_layer_regular": True,
                        "parallel_draft_step": model.eagle_config.parallel_draft_step,
                        "parallel_draft_heads_num_layers": model.eagle_config.parallel_draft_heads_num_layers,
                    }

                    eagle_config_update = {
                        "architectures": [architectures],
                        "head_dim": model.eagle_module.config.kv_channels,
                        "hidden_act": self._hf_text_config.hidden_act,
                        "hidden_size": self._hf_text_config.hidden_size,
                        "intermediate_size": model.eagle_module.config.ffn_hidden_size,
                        "max_position_embeddings": self._hf_text_config.max_position_embeddings,
                        "num_attention_heads": model.eagle_module.config.num_attention_heads,
                        "num_key_value_heads": model.eagle_module.config.num_query_groups,
                        "num_hidden_layers": model.eagle_config.num_layers,
                        "vocab_size": self._hf_text_config.vocab_size,
                        # Unset any special token ids given that the tokenizer can change here.
                        "bos_token_id": None,
                        "eos_token_id": None,
                        "pad_token_id": None,
                        "sep_token_id": None,
                        # The following attributes are EAGLE specific
                        "eagle_config": eagle_config,
                        "draft_vocab_size": model.eagle_config.draft_vocab_size,
                    }

                    self._hf_extra_config.update(eagle_config_update)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
    ):
        """Save a unified checkpoint which can be deployed by vLLM and TensorRT-LLM.

        Args:
            save_directory: Directory to which to save. Will be created if it doesn't exist.
        """
        pp_rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()

        # We use the 1st PP rank to handle VLM because vision_models
        # and vision_proj only exist in the first stage.
        is_first_stage_main_rank = pp_rank == 0
        # We use the last PP rank to write the config because
        # medusa_heads and eagle_module only exist in the last stage.
        is_last_stage_main_rank = pp_rank == pp_size - 1

        # Main export process
        state_dict = self.extra_state_dict if self.export_extra_modules else self.state_dict
        quantization_format = self._get_quantization_format(self.model)

        quantization = None

        if quantization_format in (
            QUANTIZATION_FP8_PB_REAL,
            QUANTIZATION_FP8_PB_WO,
        ):
            quantization = quantization_format
        elif quantization_format == QUANTIZATION_FP8:
            quantization = "FP8"
        elif quantization_format == QUANTIZATION_NVFP4:
            quantization = "NVFP4"

        kv_cache_quantization = None
        kv_cache_dtype = get_kv_cache_dtype(self.model)
        if kv_cache_dtype in (KV_CACHE_FP8, KV_CACHE_NVFP4):
            # FP8 KV Cache is supported in VLLM; NVFP4 supported in TRTLLM
            kv_cache_quantization = kv_cache_dtype
        # We use the last PP rank and the 1st EP rank to write the config because
        # medusa_heads and eagle_module only exist in the last stage.
        if is_last_stage_main_rank:
            if self.export_extra_modules and self._hf_extra_config is not None:
                self._hf_extra_config.save_pretrained(save_directory)
            else:
                self._hf_config.save_pretrained(save_directory)
                try:
                    generation_config = transformers.GenerationConfig.from_pretrained(
                        self._hf_pretrained_model_name
                    )
                    generation_config.save_pretrained(save_directory)
                except OSError:
                    pass
                try:
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        self._hf_pretrained_model_name
                    )
                    tokenizer.save_pretrained(save_directory)
                except OSError:
                    pass
                except TypeError:
                    pass
                try:
                    # Load and save preprocessor config from the original model
                    processor = AutoProcessor.from_pretrained(
                        self._hf_pretrained_model_name, trust_remote_code=self.trust_remote_code
                    )
                    if hasattr(processor, "image_processor"):
                        processor.image_processor.save_pretrained(save_directory)
                except (OSError, ValueError, ImportError):
                    pass

        if is_last_stage_main_rank and quantization is not None:
            hf_quant_config = {
                "producer": {
                    "name": "modelopt",
                    "version": __version__,
                },
                "quantization": {
                    "quant_algo": quantization,
                    "kv_cache_quant_algo": kv_cache_quantization,
                    "exclude_modules": ["lm_head"],
                },
            }
            with open(save_directory + "/hf_quant_config.json", "w") as f:
                json.dump(hf_quant_config, f, indent=4)

        if (
            is_first_stage_main_rank
            and self.is_multimodal
            and pretrained_model_name_or_path is not None
        ):
            hf_checkpoint_path = Path(pretrained_model_name_or_path)
            if not hf_checkpoint_path.is_dir():
                hf_checkpoint_path = tempfile.gettempdir() + "/" + pretrained_model_name_or_path
                if not Path(hf_checkpoint_path).exists():
                    snapshot_download(
                        repo_id=pretrained_model_name_or_path,
                        local_dir=hf_checkpoint_path,
                    )

            safetensors_file = Path(hf_checkpoint_path) / "model.safetensors"
            safetensors_index_file = Path(hf_checkpoint_path) / "model.safetensors.index.json"

            multimodal_state_dict = {}

            if safetensors_file.is_file():
                print(f"Loading multimodal components from single file: {safetensors_file}")
                with safe_open(safetensors_file, framework="pt") as f:
                    multimodal_keys = [
                        key
                        for key in f.keys()  # noqa: SIM118
                        if key.startswith(("multi_modal_projector", "vision_model"))
                    ]
                    for key in tqdm(multimodal_keys, desc="Loading multimodal tensors"):
                        multimodal_state_dict[key] = f.get_tensor(key)

            elif safetensors_index_file.is_file():
                print(f"Loading multimodal components from sharded model: {hf_checkpoint_path}")
                with open(safetensors_index_file) as f:
                    safetensors_index = json.load(f)

                # For multimodal models, vision_model and multi_modal_projector are in the first shard
                all_shard_files = sorted(set(safetensors_index["weight_map"].values()))
                first_shard_file = all_shard_files[0]  # e.g., "model-00001-of-00050.safetensors"

                # Load multimodal components from the first shard file
                safetensors_filepath = Path(hf_checkpoint_path) / first_shard_file
                print(f"Loading multimodal components from {first_shard_file}")

                with safe_open(safetensors_filepath, framework="pt") as f:
                    shard_keys = list(f.keys())
                    multimodal_keys_in_shard = [
                        k
                        for k in shard_keys
                        if k.startswith(("multi_modal_projector", "vision_model"))
                    ]

                    if multimodal_keys_in_shard:
                        print(
                            f"Found {len(multimodal_keys_in_shard)} multimodal tensors in {first_shard_file}"
                        )
                        for key in tqdm(
                            multimodal_keys_in_shard, desc="Loading multimodal tensors"
                        ):
                            multimodal_state_dict[key] = f.get_tensor(key)
                    else:
                        print(f"No multimodal components found in {first_shard_file}")

            else:
                print(f"Warning: No safetensors files found in {hf_checkpoint_path}")

            print(f"Successfully loaded {len(multimodal_state_dict)} multimodal tensors")
            # Add multimodal components to state_dict
            state_dict.update(multimodal_state_dict)

        # Barrier to ensure the export_dir has been created.
        torch.distributed.barrier()

        if self.export_extra_modules:
            if is_last_stage_main_rank:
                save_file(
                    state_dict, save_directory + "/model.safetensors", metadata={"format": "pt"}
                )
            torch.distributed.barrier()
            return

        if (
            is_last_stage_main_rank
            and self._hf_config is not None
            and pretrained_model_name_or_path is not None
        ):
            # For models that keep configuration and modeling files as part of the checkpoint,
            # we need to copy them to the export directory for seamless integration with inference
            # frameworks.
            hf_checkpoint_path = Path(pretrained_model_name_or_path)
            model_type = getattr(self._hf_config, "model_type", None)

            if hf_checkpoint_path.is_dir():
                # Local directory - files should be there
                config_file = hf_checkpoint_path / f"configuration_{model_type}.py"
                modeling_file = hf_checkpoint_path / f"modeling_{model_type}.py"
            else:
                # Remote model ID - download from HuggingFace Hub (cached automatically)
                try:
                    config_file = hf_hub_download(
                        repo_id=pretrained_model_name_or_path,
                        filename=f"configuration_{model_type}.py",
                    )
                except Exception:
                    config_file = ""
                try:
                    modeling_file = hf_hub_download(
                        repo_id=pretrained_model_name_or_path, filename=f"modeling_{model_type}.py"
                    )
                except Exception:
                    modeling_file = ""

            if config_file and os.path.exists(config_file):
                shutil.copy(config_file, f"{save_directory}/configuration_{model_type}.py")
            if modeling_file and os.path.exists(modeling_file):
                shutil.copy(modeling_file, f"{save_directory}/modeling_{model_type}.py")

        save_safetensors(state_dict, save_directory)

    @property
    def state_dict(self):
        """Return the real quantized state_dict of the base model."""
        if len(self._state_dict) == 0:
            self._get_state_dict()
        return self._state_dict

    @property
    def extra_state_dict(self):
        if len(self._state_dict) == 0:
            self._get_medusa_heads_state_dict()
            self._get_eagle_module_state_dict()
        return self._state_dict

    def _populate_rule_book(self):
        all_rules = {}

        def _custom_mapping_to_lambda(mapping):
            method_map = {
                "name_remapping": self._name_remapping,
                "qkv_slicing": self._qkv_slicing,
                "gated_mlp_slicing": self._gated_mlp_slicing,
                "pack_name_remapping": self._pack_name_remapping,
                "pack_name_remapping_gpt_oss": self._pack_name_remapping_gpt_oss,
            }
            func = method_map[mapping.func_name]
            prefix = mapping.target_name_or_prefix
            func_kwargs = mapping.func_kwargs
            return lambda m, *args, **kwargs: func(
                m, prefix.format(*args), **{**func_kwargs, **kwargs}
            )

        for arch, mappings in all_mcore_hf_export_mapping.items():
            all_rules[arch] = {
                k: _custom_mapping_to_lambda(v) if isinstance(v, CustomModuleMapping) else v
                for (k, v) in mappings.items()
                if isinstance(v, (CustomModuleMapping, bool))
            }

        return all_rules

    def _get_quantized_state(
        self,
        module: torch.nn.Module,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[dict[str, torch.Tensor], str, int]:
        """Return a state_dict, quantization format, and block_size of the module.

        Args:
            module: The target module to perform real quantization.
            dtype: The default data type.

        Returns:
            Tuple: state_dict, quantization format, and block_size of the module.
        """
        name_to_value = {}
        qformat: str = self._get_quantization_format(module)
        block_size = get_weight_block_size(module)

        if hasattr(module, "weight") and module.weight is not None:
            weight = module.weight.to(dtype).cpu()
            name_to_value["weight"] = weight
        else:
            return name_to_value, qformat, block_size

        if hasattr(module, "bias") and module.bias is not None:
            name_to_value["bias"] = module.bias.to(dtype).cpu()

        if hasattr(module, "expert_bias") and module.expert_bias is not None:
            name_to_value["expert_bias"] = module.expert_bias.to(dtype).cpu()

        if qformat == QUANTIZATION_NONE:
            return name_to_value, qformat, block_size
        # Getting the weight scales
        weight_scale = get_weight_scaling_factor(module)
        weight_scale_2 = get_weight_scaling_factor_2(module)
        if weight_scale is not None:
            name_to_value["weight_scale"] = weight_scale

        if weight_scale_2 is not None:
            name_to_value["weight_scale_2"] = weight_scale_2

        # Getting the input scale
        input_scale = get_activation_scaling_factor(module)
        if input_scale is not None:
            name_to_value["input_scale"] = input_scale
            # TODO (chenhany): support AWQ with pre_quant_scale
            if hasattr(module.input_quantizer, "_pre_quant_scale"):
                raise ValueError("Detect pre_quant_scale! SmoothQuant/AWQ are not yet supported!")

        if hasattr(module, "output_quantizer"):
            output_scale = get_kv_cache_scaling_factor(module)
            if output_scale is not None:
                name_to_value["output_scale"] = output_scale

        return name_to_value, qformat, block_size

    def _get_quantization_format(self, module: torch.nn.Module):
        return get_quantization_format(module)

    def _get_weight_scales(self, quantized_state: dict[str, Any], qformat: str):
        weight_scale = quantized_state.pop("weight_scale", None)
        weight_scale_2 = quantized_state.pop("weight_scale_2", None)

        if weight_scale is not None:
            weight_scale = weight_scale.clone().detach()
            if qformat == QUANTIZATION_FP8 and weight_scale.numel() == 1:
                weight_scale = weight_scale.squeeze()
        if weight_scale_2 is not None:
            weight_scale_2 = weight_scale_2.clone().detach()

        return weight_scale, weight_scale_2

    def _name_remapping(
        self,
        module: torch.nn.Module | torch.Tensor,
        prefix: str,
        skip_output_scale: bool = True,
        mapping={},
        dtype: torch.dtype | None = None,
    ):
        if dtype is None:
            dtype = self.dtype

        if isinstance(module, torch.Tensor):
            self._state_dict[prefix] = module
            return

        name_to_value, qformat, block_size = self._get_quantized_state(module, dtype)

        weight = name_to_value.pop("weight")
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        if weight_scale is None:
            self._state_dict[prefix + "weight"] = weight
        else:
            self._state_dict[prefix + "weight"] = to_quantized_weight(
                weight,
                weight_scale,
                qformat,
                weight_scale_2,
                block_size,
            )
            self._state_dict[prefix + "weight_scale"] = weight_scale.detach().clone()

        if weight_scale_2 is not None:
            if len(weight_scale_2.shape) > 0:
                raise ValueError("weight_scale_2 must be a scalar!")
            self._state_dict[prefix + "weight_scale_2"] = weight_scale_2.detach().clone()

        for key, val in name_to_value.items():
            if key == "output_scale" and skip_output_scale:
                continue
            else:
                source_key = mapping.get(key, key)
                self._state_dict[prefix + source_key] = val

    def _gated_mlp_slicing(
        self, module, prefix, gate_proj_name="gate_proj", up_proj_name="up_proj"
    ):
        name_to_value, qformat, block_size = self._get_quantized_state(module, self.dtype)

        weight = name_to_value.pop("weight")
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        gate_proj_prefix = prefix + gate_proj_name + "."
        up_proj_prefix = prefix + up_proj_name + "."

        ffn_hidden_size = module.config.ffn_hidden_size
        gate_proj_weight = weight[:ffn_hidden_size, :]
        up_proj_weight = weight[ffn_hidden_size:, :]

        if weight_scale is None:
            self._state_dict[gate_proj_prefix + "weight"] = gate_proj_weight
            self._state_dict[up_proj_prefix + "weight"] = up_proj_weight
        else:
            if len(weight_scale.shape) == 0:
                gate_proj_weight_scale = weight_scale.detach().clone()
                up_proj_weight_scale = weight_scale.detach().clone()
            else:
                gate_proj_weight_scale = weight_scale[:ffn_hidden_size]
                up_proj_weight_scale = weight_scale[ffn_hidden_size:]
            self._state_dict[gate_proj_prefix + "weight"] = to_quantized_weight(
                gate_proj_weight,
                gate_proj_weight_scale,
                qformat,
                weight_scale_2,
                block_size,
            )
            self._state_dict[up_proj_prefix + "weight"] = to_quantized_weight(
                up_proj_weight,
                up_proj_weight_scale,
                qformat,
                weight_scale_2,
                block_size,
            )
            self._state_dict[gate_proj_prefix + "weight_scale"] = gate_proj_weight_scale
            self._state_dict[up_proj_prefix + "weight_scale"] = up_proj_weight_scale

        if weight_scale_2 is not None:
            if len(weight_scale_2.shape) > 0:
                raise ValueError("weight_scale_2 must be a scalar!")
            self._state_dict[gate_proj_prefix + "weight_scale_2"] = weight_scale_2.detach().clone()
            self._state_dict[up_proj_prefix + "weight_scale_2"] = weight_scale_2.detach().clone()

        # weight and weight_scale have been pop out.
        for key, val in name_to_value.items():
            gate_proj_key = gate_proj_prefix + key
            up_proj_key = up_proj_prefix + key
            if key == "output_scale":
                continue
            else:
                self._state_dict[gate_proj_key] = val.detach().clone()
                self._state_dict[up_proj_key] = val.detach().clone()

    def _qkv_slicing(
        self,
        module,
        prefix,
        q_proj_name="q_proj",
        k_proj_name="k_proj",
        v_proj_name="v_proj",
        k_scale_name="k_scale",
        v_scale_name="v_scale",
    ):
        name_to_value, qformat, block_size = self._get_quantized_state(module, self.dtype)

        q_proj_prefix = prefix + q_proj_name + "."
        k_proj_prefix = prefix + k_proj_name + "."
        v_proj_prefix = prefix + v_proj_name + "."

        config = module.config
        hidden_size = config.hidden_size
        num_query_groups = config.num_query_groups
        head_num = config.num_attention_heads
        head_size = config.kv_channels
        heads_per_group = head_num // num_query_groups
        qkv_total_dim = head_num + 2 * num_query_groups

        weight = name_to_value.pop("weight")

        if weight.shape[-1] == 2 * hidden_size:
            print(
                "Parameter linear_qkv.weight has 2x the hidden_size."
                "Set hidden_size to 2x the hidden_size. EAGLE3 is the only known"
                "use case which has this behavior."
            )
            hidden_size = 2 * hidden_size

        weight = weight.reshape([qkv_total_dim, head_size, hidden_size])
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
        ## Example of slices
        ## 7b: num_query_groups = head_num = 32,
        ## q_slice = [0, 3, 6, 9 , ... 90, 93]
        ## k_slice = [1, 4, 7, 10, ... 91, 94]
        ## v_slice = [2, 5, 8, 11, ... 92, 95]
        ## 70b (with GQA): num_query_groups = 8, head_num = 64
        ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
        ## k_slice = [8, 18, 28, ... 68, 78]
        ## v_slice = [9, 19, 29, ... 69, 79]
        slices = [q_slice, k_slice, v_slice]
        prefixes = [q_proj_prefix, k_proj_prefix, v_proj_prefix]

        proj_weights = [weight[s].reshape(-1, hidden_size) for s in slices]
        proj_keys = [p + "weight" for p in prefixes]

        if weight_scale is None:
            for key, weight in zip(proj_keys, proj_weights):
                self._state_dict[key] = weight
        else:
            if len(weight_scale.shape) > 0:
                # AWQ per-block or per-channel scaling
                weight_scale_dtype = weight_scale.dtype
                weight_scale_hidden_size = weight_scale.shape[-1]
                weight_scale = weight_scale.to(dtype=float).reshape(
                    [qkv_total_dim, head_size, weight_scale_hidden_size]
                )
                proj_weight_scales = [
                    weight_scale[s]
                    .reshape(-1, weight_scale_hidden_size)
                    .to(dtype=weight_scale_dtype)
                    for s in slices
                ]
            else:
                # per-tensor scaling
                proj_weight_scales = [
                    weight_scale.detach().clone(),
                    weight_scale.detach().clone(),
                    weight_scale.detach().clone(),
                ]

            for weight, scale, key in zip(proj_weights, proj_weight_scales, proj_keys):
                quantized_weight = to_quantized_weight(
                    weight,
                    scale,
                    qformat,
                    weight_scale_2,
                    block_size,
                )
                self._state_dict[key] = quantized_weight
                self._state_dict[key + "_scale"] = scale

        if weight_scale_2 is not None:
            if len(weight_scale_2.shape) > 0:
                raise ValueError("weight_scale_2 must be a scalar!")
            for weight, scale, key in zip(proj_weights, proj_weight_scales, proj_keys):
                self._state_dict[key + "_scale_2"] = weight_scale_2.detach().clone()

        # weight and weight_scale have been pop out.
        for key, val in name_to_value.items():
            q_proj_key = q_proj_prefix + key
            k_proj_key = k_proj_prefix + key
            v_proj_key = v_proj_prefix + key
            if key == "output_scale":
                self._state_dict[prefix + k_scale_name] = val.detach().clone()
                self._state_dict[prefix + v_scale_name] = val.detach().clone()
            elif key == "bias":
                # Slice bias similar to weight
                bias = val.detach().clone()
                bias = bias.reshape([qkv_total_dim, head_size])
                proj_biases = [bias[s].reshape(-1) for s in slices]
                proj_bias_keys = [q_proj_prefix + key, k_proj_prefix + key, v_proj_prefix + key]
                for bias_tensor, bias_key in zip(proj_biases, proj_bias_keys):
                    self._state_dict[bias_key] = bias_tensor
            else:
                self._state_dict[q_proj_key] = val.detach().clone()
                self._state_dict[k_proj_key] = val.detach().clone()
                self._state_dict[v_proj_key] = val.detach().clone()

    def _pack_name_remapping(self, module, prefix, layer_type=None):
        """Pack name remapping into one tensor."""
        weight_list = []
        weight_scale_list = []
        weight_scale_2_list = []
        input_scale_list = []

        for expert in module:
            assert layer_type is not None, "layer_type is required for pack_name_remapping"
            name_to_value, qformat, block_size = self._get_quantized_state(
                getattr(expert, layer_type), self.dtype
            )
            weight = name_to_value.pop("weight")
            weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)
            input_scale = (
                name_to_value.pop("input_scale") if "input_scale" in name_to_value else None
            )

            weight_list.append(weight)
            weight_scale_list.append(weight_scale)
            weight_scale_2_list.append(weight_scale_2)
            input_scale_list.append(input_scale)

        merged_weight = torch.stack(weight_list, dim=0)

        # Transpose the last two dimensions to match HuggingFace format
        # NeMo format: [num_experts, out_features, in_features]
        # HF format: [num_experts, in_features, out_features]
        merged_weight = merged_weight.transpose(-2, -1).contiguous()

        if weight_scale_2_list[0] is None:
            merged_weight_scale_2 = None
            if weight_scale_list[0] is not None:
                merged_weight_scale = torch.max(torch.stack(weight_scale_list, dim=0), dim=0)[0]
            else:
                merged_weight_scale = None
        else:
            # NVFP4
            merged_weight_scale_2 = torch.max(torch.stack(weight_scale_2_list, dim=0), dim=0)[0]
            merged_weight_scale = torch.stack(weight_scale_list, dim=0)
            # Transpose the scaling factors to match the transposed weights
            merged_weight_scale = merged_weight_scale.transpose(-2, -1).contiguous()

        if input_scale_list[0] is not None:
            merged_input_scale = torch.max(torch.stack(input_scale_list, dim=0), dim=0)[0]
        else:
            merged_input_scale = None

        # Save the merged weights
        if merged_weight_scale is None:
            self._state_dict[prefix] = merged_weight
        else:
            self._state_dict[prefix] = to_quantized_weight(
                merged_weight,
                merged_weight_scale,
                qformat,
                merged_weight_scale_2,
                block_size,
            )
            self._state_dict[prefix + "_weight_scale"] = merged_weight_scale
            if merged_weight_scale_2 is not None:
                self._state_dict[prefix + "_weight_scale_2"] = merged_weight_scale_2
        if merged_input_scale is not None:
            self._state_dict[prefix + "_input_scale"] = merged_input_scale

    def _pack_name_remapping_gpt_oss(self, module, prefix, layer_type=None):
        """Pack name remapping into one tensor."""
        weight_list = []
        weight_scale_list = []
        weight_scale_2_list = []
        input_scale_list = []
        bias_list = []

        for expert in module:
            assert layer_type is not None, "layer_type is required for pack_name_remapping"
            name_to_value, qformat, block_size = self._get_quantized_state(
                getattr(expert, layer_type), self.dtype
            )
            weight = name_to_value.pop("weight")
            bias = name_to_value.pop("bias", None)
            weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)
            input_scale = (
                name_to_value.pop("input_scale") if "input_scale" in name_to_value else None
            )

            weight_list.append(weight)
            weight_scale_list.append(weight_scale)
            weight_scale_2_list.append(weight_scale_2)
            input_scale_list.append(input_scale)
            bias_list.append(bias)

        merged_weight = torch.stack(weight_list, dim=0)

        # Transpose the last two dimensions to match HuggingFace format (except for GptOssForCausalLM)
        # NeMo format: [num_experts, out_features, in_features]
        # HF format: [num_experts, in_features, out_features]

        # TODO: Need to decide if we want to transpose the weight or not.
        merged_weight = merged_weight.transpose(-2, -1).contiguous()

        # Apply interleaving for GptOssForCausalLM linear_fc1 to match HF format
        if layer_type == "linear_fc1":
            # Megatron has de-interleaved format, need to interleave for HF
            # Pattern: first half -> even indices, second half -> odd indices
            num_experts, in_features, out_features = merged_weight.shape
            half_out = out_features // 2

            # Create interleaved tensor
            interleaved_weight = torch.zeros_like(merged_weight)
            interleaved_weight[:, :, ::2] = merged_weight[
                :, :, :half_out
            ]  # First half -> even indices
            interleaved_weight[:, :, 1::2] = merged_weight[
                :, :, half_out:
            ]  # Second half -> odd indices
            merged_weight = interleaved_weight

        # Handle bias tensors
        merged_bias = None
        if bias_list[0] is not None:
            merged_bias = torch.stack(bias_list, dim=0)

            # Apply interleaving for GptOssForCausalLM linear_fc1 bias to match HF format
            if layer_type == "linear_fc1":
                num_experts, bias_len = merged_bias.shape
                half_bias_len = bias_len // 2

                # Create interleaved bias tensor
                interleaved_bias = torch.zeros_like(merged_bias)
                interleaved_bias[:, ::2] = merged_bias[
                    :, :half_bias_len
                ]  # First half -> even indices
                interleaved_bias[:, 1::2] = merged_bias[
                    :, half_bias_len:
                ]  # Second half -> odd indices
                merged_bias = interleaved_bias

        if weight_scale_2_list[0] is None:
            merged_weight_scale_2 = None
            if weight_scale_list[0] is not None:
                merged_weight_scale = torch.max(torch.stack(weight_scale_list, dim=0), dim=0)[0]
            else:
                merged_weight_scale = None
        else:
            # NVFP4
            merged_weight_scale_2 = torch.max(torch.stack(weight_scale_2_list, dim=0), dim=0)[0]
            merged_weight_scale = torch.stack(weight_scale_list, dim=0)
            # Transpose the scaling factors to match the transposed weights
            # TODO: Need to decide if we want to transpose the weight or not.
            merged_weight_scale = merged_weight_scale.transpose(-2, -1).contiguous()

        if input_scale_list[0] is not None:
            merged_input_scale = torch.max(torch.stack(input_scale_list, dim=0), dim=0)[0]
        else:
            merged_input_scale = None

        # Save the merged weights
        if merged_weight_scale is None:
            # TODO: May need to modify the key name later.
            self._state_dict[prefix] = merged_weight
        else:
            self._state_dict[prefix] = to_quantized_weight(
                merged_weight,
                merged_weight_scale,
                qformat,
                merged_weight_scale_2,
                block_size,
            )
            self._state_dict[prefix + "_weight_scale"] = merged_weight_scale
            if merged_weight_scale_2 is not None:
                self._state_dict[prefix + "_weight_scale_2"] = merged_weight_scale_2
        if merged_input_scale is not None:
            self._state_dict[prefix + "_input_scale"] = merged_input_scale

        # Save bias tensors if they exist
        if merged_bias is not None:
            # TODO: May need to modify the key name later.
            self._state_dict[prefix + "_bias"] = merged_bias

    def _get_medusa_heads_state_dict(self):
        medusa_heads = getattr(self.model, "medusa_heads", None)
        if medusa_heads is None:
            return

        for head_id, head in enumerate(medusa_heads):
            self.rules["medusa_heads.lm_head"](head.lm_head, head_id)
            for layer_id, layer in enumerate(head.medusa_layers):
                self.rules["medusa_heads.medusa_layers.linear"](layer.linear, head_id, layer_id)

    def _get_eagle_module_state_dict(self):
        eagle_module = getattr(self.model, "eagle_module", None)

        if eagle_module is None:
            return

        # if hasattr(self.model, "embedding"):
        #    self.rules["word_embeddings"](self.model.embedding.word_embeddings)

        self.rules["fc"](eagle_module.fc)
        if self.model.eagle_config.use_aux_hidden_state:
            self.rules["enorm"](eagle_module.enorm)
        elif self.model.eagle_config.use_mtp_layernorm:
            self.rules["enorm"](eagle_module.enorm)
            self.rules["hnorm"](eagle_module.hnorm)

        if self.model.eagle_config.use_last_layernorm:
            self.rules["final_layernorm"](eagle_module.decoder.final_layernorm)

        if hasattr(self.model.eagle_module, "eagle_output_layer"):
            self.rules["output_layer"](eagle_module.eagle_output_layer)
        if hasattr(self.model.eagle_module, "dt2"):
            self.rules["d2t"](eagle_module.d2t)

        for layer in eagle_module.decoder.layers:
            layer_id = layer.layer_number - 1

            # The first layernorm needs special handling here. We have a dedicated mapping
            # for the first layernorm since in EAGLE3 it will be mapped to hidden_norm
            # instead of input_layernorm (due to the specialized transformer layer).
            # The remaining EAGLE3 layers (if more than 1) are normal transformer layers
            # where input_layernorm is mapped to input_layernorm.
            if layer_id == 0 and self.model.eagle_config.use_input_layernorm_in_first_layer:
                self.rules["first_input_layernorm"](layer.input_layernorm, layer_id)
            elif layer_id > 0:
                self.rules["input_layernorm"](layer.input_layernorm, layer_id)

            if "MLASelfAttention" in str(type(layer.self_attention)):
                if hasattr(layer.self_attention, "linear_q_proj"):
                    self.rules["eagle_module.linear_q_proj"](
                        layer.self_attention.linear_q_proj, layer_id
                    )
                else:
                    self.rules["eagle_module.linear_q_down_proj"](
                        layer.self_attention.linear_q_down_proj, layer_id
                    )
                    self.rules["eagle_module.linear_q_layernorm"](
                        layer.self_attention.q_layernorm, layer_id
                    )
                    self.rules["eagle_module.linear_q_up_proj"](
                        layer.self_attention.linear_q_up_proj, layer_id
                    )

                self.rules["eagle_module.linear_kv_down_proj"](
                    layer.self_attention.linear_kv_down_proj, layer_id
                )
                self.rules["eagle_module.linear_kv_layernorm"](
                    layer.self_attention.kv_layernorm, layer_id
                )
                self.rules["eagle_module.linear_kv_up_proj"](
                    layer.self_attention.linear_kv_up_proj, layer_id
                )
            else:
                self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)

            self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
            self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)

            if "MoE" in str(type(layer.mlp)):
                self.rules["eagle_module.router"](layer.mlp.router, layer_id)
                if hasattr(layer.mlp, "shared_experts") and layer.mlp.shared_experts is not None:
                    self.rules["eagle_module.shared_experts.linear_fc1"](
                        layer.mlp.shared_experts.linear_fc1, layer_id
                    )
                    self.rules["eagle_module.shared_experts.linear_fc2"](
                        layer.mlp.shared_experts.linear_fc2, layer_id
                    )
                for expert_id, expert in enumerate(layer.mlp.experts.local_experts):
                    self.rules["eagle_module.local_experts.linear_fc1"](
                        expert.linear_fc1, layer_id, expert_id
                    )
                    self.rules["eagle_module.local_experts.linear_fc2"](
                        expert.linear_fc2, layer_id, expert_id
                    )
            else:
                self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
                self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)

        parallel_draft_heads = getattr(eagle_module, "parallel_draft_heads", None)
        if parallel_draft_heads is not None:
            for head_id, head in enumerate(parallel_draft_heads.medusa_heads):
                for layer_id, layer in enumerate(head):
                    self.rules["parallel_draft_heads.medusa_layers"](
                        layer.linear, head_id, layer_id
                    )
            self.rules["parallel_draft_heads.lm_head"](parallel_draft_heads.lm_head)

    def _get_state_dict(self):
        model = self.model

        # Embedding
        if hasattr(model, "embedding"):
            self.rules["word_embeddings"](model.embedding.word_embeddings)

        # Final layernorm
        if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm:
            self.rules["final_layernorm"](model.decoder.final_layernorm)

        if hasattr(model.decoder, "final_norm") and model.decoder.final_norm:
            self.rules["final_norm"](model.decoder.final_norm)

        # Output layer
        if hasattr(model, "output_layer") and not model.share_embeddings_and_output_weights:
            self.rules["output_layer"](model.output_layer)

        # Decoder layers
        for layer in model.decoder.layers:
            layer_id = layer.layer_number - 1

            if isinstance(layer, MambaLayer):
                if not isinstance(layer.norm, IdentityOp):
                    self.rules["norm"](layer.norm, layer_id)

                self.rules["mixer_norm"](layer.mixer.norm, layer_id)
                self.rules["A_log"](layer.mixer.A_log, layer_id)
                self.rules["D"](layer.mixer.D, layer_id)
                self.rules["dt_bias"](layer.mixer.dt_bias, layer_id)

                self.rules["conv1d"](layer.mixer.conv1d, layer_id)
                self.rules["in_proj"](layer.mixer.in_proj, layer_id)
                self.rules["out_proj"](layer.mixer.out_proj, layer_id)

            elif isinstance(layer, TransformerLayer):
                if not isinstance(layer.input_layernorm, IdentityOp):
                    self.rules["input_layernorm"](layer.input_layernorm, layer_id)

                if not isinstance(layer.self_attention, IdentityOp):
                    if "MLASelfAttention" in str(type(layer.self_attention)):
                        if hasattr(layer.self_attention, "linear_q_proj"):
                            self.rules["linear_q_proj"](
                                layer.self_attention.linear_q_proj, layer_id
                            )
                        else:
                            self.rules["linear_q_down_proj"](
                                layer.self_attention.linear_q_down_proj, layer_id
                            )
                            self.rules["linear_q_layernorm"](
                                layer.self_attention.q_layernorm, layer_id
                            )
                            self.rules["linear_q_up_proj"](
                                layer.self_attention.linear_q_up_proj, layer_id
                            )

                        self.rules["linear_kv_down_proj"](
                            layer.self_attention.linear_kv_down_proj, layer_id
                        )
                        self.rules["linear_kv_layernorm"](
                            layer.self_attention.kv_layernorm, layer_id
                        )
                        self.rules["linear_kv_up_proj"](
                            layer.self_attention.linear_kv_up_proj, layer_id
                        )
                        self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
                    else:
                        if layer.self_attention.q_layernorm is not None and not isinstance(
                            layer.self_attention.q_layernorm, (IdentityOp, L2Norm)
                        ):
                            self.rules["q_layernorm"](layer.self_attention.q_layernorm, layer_id)
                            self.rules["k_layernorm"](layer.self_attention.k_layernorm, layer_id)
                        self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id)
                        self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id)
                        if (
                            getattr(layer.self_attention.core_attention, "softmax_offset", None)
                            is not None
                        ):
                            self.rules["softmax_offset"](
                                layer.self_attention.core_attention.softmax_offset, layer_id
                            )

                if not isinstance(layer.pre_mlp_layernorm, IdentityOp):
                    self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id)

                if not isinstance(layer.mlp, IdentityOp):
                    if "MoE" in str(type(layer.mlp)):
                        self.rules["router"](
                            layer.mlp.router, layer_id, dtype=self.moe_router_dtype
                        )
                        if (
                            hasattr(layer.mlp, "shared_experts")
                            and layer.mlp.shared_experts is not None
                        ):
                            self.rules["shared_experts.linear_fc1"](
                                layer.mlp.shared_experts.linear_fc1, layer_id
                            )
                            self.rules["shared_experts.linear_fc2"](
                                layer.mlp.shared_experts.linear_fc2, layer_id
                            )
                        if not self.rules.get("use_packed_local_experts", False):
                            for expert_id, expert in enumerate(layer.mlp.experts.local_experts):
                                self.rules["local_experts.linear_fc1"](
                                    expert.linear_fc1, layer_id, expert_id
                                )
                                self.rules["local_experts.linear_fc2"](
                                    expert.linear_fc2, layer_id, expert_id
                                )
                        else:
                            # For llama 4, in hf unified checkpoint, all local experts share one scale
                            self.rules["local_experts.linear_fc1"](
                                layer.mlp.experts.local_experts, layer_id
                            )
                            self.rules["local_experts.linear_fc2"](
                                layer.mlp.experts.local_experts, layer_id
                            )
                    else:
                        self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id)
                        self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id)
            else:
                raise ValueError("Only TransformerLayer or MambaLayer are supported.")


def export_mcore_gpt_to_hf(
    model: torch.nn.Module,
    pretrained_model_name_or_path: str | os.PathLike | None = None,
    export_extra_modules: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    export_dir: Path | str = tempfile.gettempdir(),
    moe_router_dtype: torch.dtype | None = None,
):
    """Export Megatron Core GPTModel to unified checkpoint and save to export_dir.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_name_or_path: Can be either: the *model id* of a
            pretrained model hosted inside a model repo on huggingface.co; or
            a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        export_extra_modules: If True, export extra modules like medusa_heads or
            eagle_module. Otherwise, only export the base model.
        dtype: The weights data type to export the unquantized layers.
        export_dir: The target export path.
    """
    exporter = GPTModelExporter(
        model,
        pretrained_model_name_or_path,
        export_extra_modules=export_extra_modules,
        dtype=dtype,
        moe_router_dtype=moe_router_dtype,
    )
    exporter.save_pretrained(export_dir, pretrained_model_name_or_path)


def import_mcore_gpt_from_hf(
    model: torch.nn.Module,
    pretrained_model_path: str,
    workspace_dir: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    moe_router_dtype: torch.dtype | None = None,
):
    """Import GPTModel state_dict from supported HuggingFace pretrained model path.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_path: A path to a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        dtype: The weights data type to import.
    """
    importer = GPTModelImporter(
        model,
        pretrained_model_path,
        workspace_dir=workspace_dir,
        dtype=dtype,
        moe_router_dtype=moe_router_dtype,
    )
    importer._import_state_dict()
