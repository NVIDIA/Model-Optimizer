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

import io
import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.distributed
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt import __version__
from modelopt.torch.quantization.nn.modules.tensor_quantizer import GroupedQuantizer
from modelopt.torch.utils import import_plugin, warn_rank_0

from .convert_hf_config import convert_hf_quant_config_format
from .model_config import (
    KV_CACHE_FP8,
    KV_CACHE_NVFP4,
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_REAL,
    QUANTIZATION_FP8_PB_WO,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_W4A16_NVFP4,
)
from .plugins.hf_checkpoint_utils import (
    _is_hf_hub_offline,
    copy_hf_ckpt_remote_code,
    copy_non_safetensor_files_from_ckpt,
    load_multimodal_components,
)
from .plugins.mcore_common import (
    all_mcore_hf_export_mapping,
    all_mcore_hf_vision_passthrough_mapping,
)
from .plugins.mcore_custom import (
    LLAVA_VISION_PREFIXES,
    CustomModuleMapping,
    get_safetensor,
    save_safetensors_by_layer_index,
)
from .plugins.megatron_importer import GPTModelImporter, _get_mamba_conv1d
from .quant_utils import (
    get_activation_scaling_factor,
    get_kv_cache_dtype,
    get_kv_cache_scaling_factor,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    process_layer_quant_config,
    to_quantized_weight,
)

with import_plugin("transformers", verbose=False):
    import transformers
    from transformers import AutoProcessor

has_mcore = False
with import_plugin("megatron"):
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.models.multimodal.llava_model import LLaVAModel
    from megatron.core.parallel_state import (
        get_data_parallel_rank,
        get_expert_model_parallel_group,
        get_expert_model_parallel_rank,
        get_expert_model_parallel_world_size,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_tensor_model_parallel_rank,
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
        trust_remote_code: Whether to trust remote code in the HuggingFace pretrained model.
        moe_router_dtype: The data type of the MoE router. Can be "fp32", "fp64", or None (default to the model dtype).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        export_extra_modules: bool = False,
        dtype=torch.bfloat16,
        trust_remote_code: bool = False,
        moe_router_dtype: str | None = None,
    ):
        """Create a GPTModel exporter instance."""
        # VLM wrappers keep the decoder under ``.language_model``; only that is exported, the
        # vision tower being copied from the HF checkpoint as-is.
        language_model = (
            model
            if isinstance(model, (GPTModel, HybridModel))
            else getattr(model, "language_model", None)
        )
        if not isinstance(language_model, (GPTModel, HybridModel)):
            raise ValueError("Input to GPTModelExport must be a megatron.core.models.GPTModel!")

        self._state_dict = OrderedDict()
        self._layer_state_dicts = OrderedDict()
        self._hf_pretrained_model_name = pretrained_model_name_or_path
        self._hf_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.moe_router_dtype = None
        if moe_router_dtype == "fp32":
            self.moe_router_dtype = torch.float32
        elif moe_router_dtype == "fp64":
            self.moe_router_dtype = torch.float64
        print(f"Exporting model with moe_router_dtype: {self.moe_router_dtype}")

        # If multimodal, extra the text_config
        self._hf_text_config = getattr(self._hf_config, "text_config", self._hf_config)

        # Update hf_config
        self._hf_text_config.num_hidden_layers = language_model.config.num_layers
        self._hf_text_config.hidden_size = language_model.config.hidden_size
        self._hf_text_config.head_dim = language_model.config.kv_channels
        self._hf_text_config.num_attention_heads = language_model.config.num_attention_heads
        self._hf_text_config.num_key_value_heads = language_model.config.num_query_groups
        self.is_multimodal = isinstance(model, LLaVAModel)
        if not self.is_multimodal:
            self._hf_text_config.intermediate_size = language_model.config.ffn_hidden_size
        self._hf_quant_config: dict = {}
        self._hf_extra_config = None
        self.export_extra_modules = export_extra_modules
        self.model = language_model
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.arch = self._hf_config.architectures[0]
        # ``None`` when there is no vision tower to copy through.
        self.vision_passthrough_prefixes = all_mcore_hf_vision_passthrough_mapping.get(
            self.arch, LLAVA_VISION_PREFIXES if self.is_multimodal else None
        )
        # TODO: May modify this later according to what quantization exported ckpt is, currently only support BF16.
        if self.arch == "GptOssForCausalLM":
            if hasattr(self._hf_config, "quantization_config"):
                del self._hf_config.quantization_config
        self.all_rules = self._populate_rule_book()
        self.rules = self.all_rules[self.arch]
        self.exclude_modules = []
        self.layer_config_dict = {}

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

    def save_pretrained_extra_modules(
        self,
        save_directory: str | os.PathLike,
    ):
        """Save a EAGLE or Medusa checkpoints which can be deployed by vLLM and TensorRT-LLM."""
        # We use the last PP rank to write the config because
        # medusa_heads and eagle_module only exist in the last stage.
        pp_rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()
        is_last_stage_main_rank = pp_rank == pp_size - 1

        state_dict = self.extra_state_dict

        if is_last_stage_main_rank and self._hf_extra_config is not None:
            self._hf_extra_config.save_pretrained(save_directory)
            save_file(state_dict, save_directory + "/model.safetensors", metadata={"format": "pt"})

        torch.distributed.barrier()

    @staticmethod
    def _is_sidecar_writer_rank(is_last_stage_main_rank: bool) -> bool:
        """True only for the DP0/EP0 last-stage-main rank (single writer for save_directory)."""
        return (
            is_last_stage_main_rank
            and get_data_parallel_rank() == 0
            and get_expert_model_parallel_rank() == 0
        )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        pretrained_model_name_or_path: str | os.PathLike,
    ):
        """Save a unified checkpoint which can be deployed by vLLM and TensorRT-LLM.

        Args:
            save_directory: Directory to which to save. Will be created if it doesn't exist.
        """
        pp_rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        # We use the 1st PP rank to handle VLM because vision_models
        # and vision_proj only exist in the first stage.
        is_first_stage_main_rank = pp_rank == 0 and tp_rank == 0
        # We use the last PP rank to write the config because
        # medusa_heads and eagle_module only exist in the last stage.
        is_last_stage_main_rank = pp_rank == pp_size - 1 and tp_rank == 0
        is_writer_rank = self._is_sidecar_writer_rank(is_last_stage_main_rank)

        # Main export process
        layer_state_dicts = self.layer_state_dicts

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
        elif quantization_format == QUANTIZATION_W4A16_NVFP4:
            quantization = "W4A16_NVFP4"

        if is_last_stage_main_rank:
            if is_writer_rank:
                if self._hf_pretrained_model_name is not None:
                    if os.path.isdir(self._hf_pretrained_model_name):
                        copy_non_safetensor_files_from_ckpt(
                            self._hf_pretrained_model_name, save_directory
                        )
                    else:
                        copy_hf_ckpt_remote_code(self._hf_pretrained_model_name, save_directory)
                self._hf_config.save_pretrained(save_directory)
                try:
                    generation_config = transformers.GenerationConfig.from_pretrained(
                        self._hf_pretrained_model_name,
                        trust_remote_code=self.trust_remote_code,
                    )
                    generation_config.save_pretrained(save_directory)
                except OSError:
                    pass
                # Hub-ID / None source: fetch tokenizer files via AutoTokenizer.
                if self._hf_pretrained_model_name is None or not os.path.isdir(
                    self._hf_pretrained_model_name
                ):
                    try:
                        tokenizer = transformers.AutoTokenizer.from_pretrained(
                            self._hf_pretrained_model_name,
                            trust_remote_code=self.trust_remote_code,
                        )
                        tokenizer.save_pretrained(save_directory)
                    except (OSError, TypeError, ValueError, ImportError):
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

            # MTP load mutates per-rank layer_state_dicts, so it runs on every last-stage main rank.
            mtp_state_dict = self._get_mtp_state_dict()
            if len(mtp_state_dict) > 0:
                layer_state_dicts[self.model.config.num_layers].update(mtp_state_dict)
                print(f"Successfully loaded {len(mtp_state_dict)} MTP tensors")

        combined_exclude_modules = self._gather_exclude_modules()
        combined_layer_config_dict = self._gather_layer_config_dict()
        # kv_cache_dtype is only set on attention-owning ranks; writer rank may not be one.
        gathered_kv_cache_dtype = self._gather_kv_cache_dtype()

        if is_writer_rank and quantization is not None:
            if combined_layer_config_dict:
                quantization_config = process_layer_quant_config(combined_layer_config_dict)
                quantization_config["exclude_modules"] = combined_exclude_modules
            else:
                quantization_config = {
                    "quant_algo": quantization,
                    "exclude_modules": combined_exclude_modules,
                }
                if quantization in ("NVFP4", "W4A16_NVFP4"):  # update block size
                    quantization_config["group_size"] = 16

            if gathered_kv_cache_dtype is not None:
                quantization_config["kv_cache_quant_algo"] = gathered_kv_cache_dtype

            self._hf_quant_config = {
                "producer": {
                    "name": "modelopt",
                    "version": __version__,
                },
                "quantization": quantization_config,
            }
            with open(save_directory + "/hf_quant_config.json", "w") as f:
                json.dump(self._hf_quant_config, f, indent=4)

        # Add multimodal components to state_dict. Since only support decoder model quantization,
        # no changes will be made to the multimodal components. We copy the multimodal components
        # from the pretrained model directly to the state_dict to avoid implementing the export logic.
        if is_first_stage_main_rank:
            # layer_state_dicts is keyed by layer_number (1-indexed), so the first
            # decoder layer on this (first) PP stage is the smallest key, not 0.
            # Merge the multimodal components into that shard so they land in a file
            # the index builder picks up (it scans shards 1..num_layers).
            first_layer_key = next(iter(layer_state_dicts))
            if self.vision_passthrough_prefixes is not None:
                layer_state_dicts[first_layer_key].update(
                    load_multimodal_components(
                        pretrained_model_name_or_path,
                        prefixes=self.vision_passthrough_prefixes,
                    )
                )

        # Bracket the writer's config.json read-modify-write with barriers so peers
        # never observe a truncated file (also ensures export_dir exists).
        torch.distributed.barrier()
        config_json_file = save_directory + "/config.json"
        if is_writer_rank and self._hf_quant_config and os.path.exists(config_json_file):
            with open(config_json_file) as f:
                config_dict = json.load(f)
            config_dict["quantization_config"] = convert_hf_quant_config_format(
                self._hf_quant_config
            )
            with open(config_json_file, "w") as f:
                json.dump(config_dict, f, indent=4)
        torch.distributed.barrier()

        # save_safetensors(state_dict, save_directory)
        save_safetensors_by_layer_index(
            layer_state_dicts=layer_state_dicts,
            total_layers=self.model.config.num_layers,
            save_directory=save_directory,
            name_template="model-{:05d}-of-{:05d}",
        )

        # Every rank has written its shards; one rank now checks nothing was dropped.
        torch.distributed.barrier()
        if is_writer_rank:
            self._verify_exported_keys(save_directory, pretrained_model_name_or_path)

    def _verify_exported_keys(self, save_directory, pretrained_model_name_or_path) -> None:
        """Raise if the export dropped tensors the source has: a missing rule emits nothing."""
        if pretrained_model_name_or_path is None:
            return
        source_dir = str(pretrained_model_name_or_path)
        if not os.path.isdir(source_dir):
            # A repo id is the documented invocation, so fetch just the index rather than skip.
            try:
                source_dir = snapshot_download(
                    repo_id=source_dir,
                    # Unsharded repos have no index, only a single model.safetensors.
                    allow_patterns=["*.safetensors.index.json", "model.safetensors"],
                    local_files_only=_is_hf_hub_offline(),
                )
            except Exception:
                warn_rank_0(
                    f"Export self-check skipped: cannot read {pretrained_model_name_or_path}."
                )
                return
        index_file = Path(save_directory) / "model.safetensors.index.json"
        if not index_file.exists():
            single = Path(save_directory) / "model.safetensors"
            if not single.exists():
                warn_rank_0("Export self-check skipped: no safetensors written.")
                return
            with safe_open(str(single), framework="pt", device="cpu") as f:
                exported = set(f.keys())
        else:
            with open(index_file) as f:
                exported = set(json.load(f)["weight_map"])
        source = _read_checkpoint_keys(source_dir)
        if not source:
            warn_rank_0(f"Export self-check skipped: no tensor index found in {source_dir}.")
            return

        # Narrow on purpose: compare module prefixes, not tensor names, since a quantized source
        # carries extras with no export counterpart, and only inside decoder layers, whose naming
        # is stable. A dropped decoder module is the case that loads fine and produces garbage.
        num_layers = self.model.config.num_layers
        exported_modules = {key.rsplit(".", 1)[0] for key in exported}
        missing = set()
        for key in source - exported:
            layer = re.search(r"\.layers\.(\d+)\.", key)
            if layer is None:
                continue  # see the note above: decoder layers only
            if int(layer.group(1)) >= num_layers:
                continue  # depth-pruned model: the source has layers this export does not
            if key.rsplit(".", 1)[0] in exported_modules:
                continue  # module is exported; this name is a source-side quantization artifact
            if "rotary_emb" in key:
                continue  # non-persistent buffer some conversions still ship
            missing.add(key)
        if missing:
            raise RuntimeError(
                f"Export dropped {len(missing)} tensor(s) present in "
                f"{pretrained_model_name_or_path}, e.g. {sorted(missing)[:8]}. The checkpoint "
                f"written to {save_directory} is incomplete -- the architecture has no export "
                "rule for one of its decoder modules."
            )

    @property
    def state_dict(self):
        """Return the real quantized state_dict of the base model."""
        if len(self._state_dict) == 0:
            self._get_state_dict()
        return self._state_dict

    @property
    def layer_state_dicts(self):
        if len(self._layer_state_dicts) == 0:
            self._get_state_dict()
        return self._layer_state_dicts

    @property
    def extra_state_dict(self):
        if len(self._state_dict) == 0:
            self._get_medusa_heads_state_dict()
            self._get_eagle_module_state_dict()
        return self._state_dict

    def _get_state_dict(self):
        model = self.model

        # Embedding
        if hasattr(model, "embedding"):
            self.rules["word_embeddings"](model.embedding.word_embeddings)

        # Decoder layers
        for layer in model.decoder.layers:
            layer_id = layer.layer_number - 1
            if isinstance(layer, MambaLayer):
                self._get_mamba_layer_state_dict(layer, layer_id)
            elif isinstance(layer, TransformerLayer):
                self._get_transformer_layer_state_dict(layer, layer_id)
            else:
                raise ValueError("Only TransformerLayer or MambaLayer are supported.")

            self._layer_state_dicts[layer.layer_number] = self._state_dict
            if layer.layer_number != self.model.config.num_layers:
                self._state_dict = OrderedDict()

        # Final layernorm
        if hasattr(model.decoder, "final_layernorm") and model.decoder.final_layernorm:
            self.rules["final_layernorm"](model.decoder.final_layernorm)

        if hasattr(model.decoder, "final_norm") and model.decoder.final_norm:
            self.rules["final_norm"](model.decoder.final_norm)

        # Output layer
        if hasattr(model, "output_layer") and not model.share_embeddings_and_output_weights:
            self.rules["output_layer"](model.output_layer)

    def _get_fused_norm_weight(self, module, primary_key: str = "fused_norm"):
        """Return ``(rule_key, layer_norm_weight)`` when TE fuses the norm into a linear layer.

        Mirrors the importer-side fallback chain: prefer the per-context key
        (``fused_input_layernorm`` for attention, ``fused_pre_mlp_layernorm`` for MLP) and
        fall back to the legacy ``fused_norm`` rule (Nemotron-H style, one norm shared
        across attention/mlp/mamba slots). Returns ``(None, None)`` when no rule is
        defined or the module has no ``layer_norm_weight``.
        """
        fused_key = primary_key if primary_key in self.rules else "fused_norm"
        if fused_key not in self.rules:
            return None, None
        weight = getattr(module, "layer_norm_weight", None)
        if weight is None:
            return None, None
        return fused_key, weight

    def _get_transformer_layer_state_dict(self, layer, layer_id, is_mtp=False):
        if not isinstance(layer.input_layernorm, IdentityOp):
            self.rules["input_layernorm"](layer.input_layernorm, layer_id, is_mtp=is_mtp)
        else:
            # GatedDeltaNet fuses the input layernorm into ``in_proj`` rather than ``linear_qkv``.
            qkv_module = getattr(layer.self_attention, "linear_qkv", None)
            if qkv_module is None:
                qkv_module = getattr(layer.self_attention, "in_proj", None)
            fused_key, norm_weight = self._get_fused_norm_weight(
                qkv_module,
                primary_key="fused_input_layernorm",
            )
            if norm_weight is not None:
                self.rules[fused_key](norm_weight, layer_id, is_mtp=is_mtp)

        if not isinstance(layer.self_attention, IdentityOp):
            if "MLASelfAttention" in str(type(layer.self_attention)):
                if hasattr(layer.self_attention, "linear_q_proj"):
                    self.rules["linear_q_proj"](
                        layer.self_attention.linear_q_proj, layer_id, is_mtp=is_mtp
                    )
                else:
                    self.rules["linear_q_down_proj"](
                        layer.self_attention.linear_q_down_proj, layer_id, is_mtp=is_mtp
                    )
                    self.rules["linear_q_layernorm"](
                        layer.self_attention.q_layernorm, layer_id, is_mtp=is_mtp
                    )
                    self.rules["linear_q_up_proj"](
                        layer.self_attention.linear_q_up_proj, layer_id, is_mtp=is_mtp
                    )

                self.rules["linear_kv_down_proj"](
                    layer.self_attention.linear_kv_down_proj, layer_id, is_mtp=is_mtp
                )
                self.rules["linear_kv_layernorm"](
                    layer.self_attention.kv_layernorm, layer_id, is_mtp=is_mtp
                )
                self.rules["linear_kv_up_proj"](
                    layer.self_attention.linear_kv_up_proj, layer_id, is_mtp=is_mtp
                )
                self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id, is_mtp=is_mtp)
            elif "linear_attn" in self.rules and hasattr(layer.self_attention, "in_proj"):
                # GatedDeltaNet (Qwen3.5 linear attention): no q/k layernorm, no core_attention.
                self._get_gated_delta_net_state_dict(layer, layer_id, is_mtp=is_mtp)
            else:
                if layer.self_attention.q_layernorm is not None and not isinstance(
                    layer.self_attention.q_layernorm, (IdentityOp, L2Norm)
                ):
                    self.rules["q_layernorm"](
                        layer.self_attention.q_layernorm, layer_id, is_mtp=is_mtp
                    )
                    self.rules["k_layernorm"](
                        layer.self_attention.k_layernorm, layer_id, is_mtp=is_mtp
                    )
                self.rules["linear_qkv"](layer.self_attention.linear_qkv, layer_id, is_mtp=is_mtp)
                if (
                    hasattr(layer.self_attention, "core_attention")
                    and "core_attention" in self.rules
                ):  # KV cache quant export
                    self.rules["core_attention"](
                        layer.self_attention.core_attention, layer_id, is_mtp=is_mtp
                    )
                self.rules["linear_proj"](layer.self_attention.linear_proj, layer_id, is_mtp=is_mtp)
                if getattr(layer.self_attention.core_attention, "softmax_offset", None) is not None:
                    self.rules["softmax_offset"](
                        layer.self_attention.core_attention.softmax_offset, layer_id, is_mtp=is_mtp
                    )

        if not isinstance(layer.pre_mlp_layernorm, IdentityOp):
            self.rules["pre_mlp_layernorm"](layer.pre_mlp_layernorm, layer_id, is_mtp=is_mtp)
        elif not isinstance(layer.mlp, IdentityOp) and "MoE" not in str(type(layer.mlp)):
            fused_key, norm_weight = self._get_fused_norm_weight(
                getattr(layer.mlp, "linear_fc1", None),
                primary_key="fused_pre_mlp_layernorm",
            )
            if norm_weight is not None:
                self.rules[fused_key](norm_weight, layer_id, is_mtp=is_mtp)

        if not isinstance(layer.mlp, IdentityOp):
            if "MoE" in str(type(layer.mlp)):
                self.rules["router"](
                    layer.mlp.router, layer_id, dtype=self.moe_router_dtype, is_mtp=is_mtp
                )
                if hasattr(layer.mlp, "fc1_latent_proj") and layer.mlp.fc1_latent_proj is not None:
                    self.rules["fc1_latent_proj"](
                        layer.mlp.fc1_latent_proj, layer_id, is_mtp=is_mtp
                    )
                if hasattr(layer.mlp, "fc2_latent_proj") and layer.mlp.fc2_latent_proj is not None:
                    self.rules["fc2_latent_proj"](
                        layer.mlp.fc2_latent_proj, layer_id, is_mtp=is_mtp
                    )
                if hasattr(layer.mlp, "shared_experts") and layer.mlp.shared_experts is not None:
                    self.rules["shared_experts.linear_fc1"](
                        layer.mlp.shared_experts.linear_fc1, layer_id, is_mtp=is_mtp
                    )
                    self.rules["shared_experts.linear_fc2"](
                        layer.mlp.shared_experts.linear_fc2, layer_id, is_mtp=is_mtp
                    )
                    if (
                        "shared_experts.gate_weight" in self.rules
                        and getattr(layer.mlp.shared_experts, "gate_weight", None) is not None
                    ):
                        self.rules["shared_experts.gate_weight"](
                            layer.mlp.shared_experts.gate_weight, layer_id, is_mtp=is_mtp
                        )
                if hasattr(layer.mlp.experts, "local_experts"):
                    if not self.rules.get("use_packed_local_experts", False):
                        for expert_id, expert in enumerate(layer.mlp.experts.local_experts):
                            self.rules["local_experts.linear_fc1"](
                                expert.linear_fc1, layer_id, expert_id, is_mtp=is_mtp
                            )
                            self.rules["local_experts.linear_fc2"](
                                expert.linear_fc2, layer_id, expert_id, is_mtp=is_mtp
                            )
                    else:
                        # For llama 4, in hf unified checkpoint, all local experts share one scale
                        self.rules["local_experts.linear_fc1"](
                            layer.mlp.experts.local_experts, layer_id, is_mtp=is_mtp
                        )
                        self.rules["local_experts.linear_fc2"](
                            layer.mlp.experts.local_experts, layer_id, is_mtp=is_mtp
                        )
                elif "experts.linear_fc1" in self.rules:
                    # TEGroupedMLP: experts use fused grouped GEMM with a single
                    # linear_fc1/linear_fc2 for all experts (no local_experts attribute).
                    # Uses "experts.linear_fc1" rule (GroupedMLPMerging) instead of
                    # "local_experts.linear_fc1" which expects per-expert iteration.
                    self.rules["experts.linear_fc1"](
                        layer.mlp.experts.linear_fc1, layer_id, is_mtp=is_mtp
                    )
                    self.rules["experts.linear_fc2"](
                        layer.mlp.experts.linear_fc2, layer_id, is_mtp=is_mtp
                    )
                else:
                    # Otherwise the routed experts are dropped and the checkpoint looks valid.
                    raise NotImplementedError(
                        f"No export rule for {type(layer.mlp.experts).__name__} experts of "
                        f"{self.arch}: fused (grouped GEMM) experts need an 'experts.linear_fc1' "
                        "rule. Re-run quantization and export with --no_moe_grouped_gemm to build "
                        "the experts as SequentialMLP instead."
                    )
            else:
                self.rules["linear_fc1"](layer.mlp.linear_fc1, layer_id, is_mtp=is_mtp)
                self.rules["linear_fc2"](layer.mlp.linear_fc2, layer_id, is_mtp=is_mtp)

    def _get_mtp_state_dict(self) -> dict[str, torch.Tensor]:
        """Export the live MTP module, or copy it from the pretrained model if absent."""
        model = getattr(self, "model", None)
        mtp = getattr(model, "mtp", None)
        if mtp is None or not hasattr(mtp, "layers") or len(mtp.layers) == 0:
            return self._copy_mtp_state_dict_from_pretrained()

        # Inner layers reuse the base walker with is_mtp=True (retargets backbone -> mtp).
        saved_state_dict = self._state_dict
        self._state_dict = OrderedDict()
        try:
            for mtp_layer in mtp.layers:
                # Some architectures (Qwen3.5) put a single TransformerLayer here, not a container.
                inner = mtp_layer.mtp_model_layer
                inner_layers = getattr(inner, "layers", None) or [inner]
                first_id = inner_layers[0].layer_number - 1
                last_id = inner_layers[-1].layer_number - 1

                # Outer predictor projections attach to the first inner HF index.
                if "mtp.enorm" in self.rules:
                    self.rules["mtp.enorm"](mtp_layer.enorm, first_id)
                if "mtp.hnorm" in self.rules:
                    self.rules["mtp.hnorm"](mtp_layer.hnorm, first_id)
                if "mtp.eh_proj" in self.rules:
                    self.rules["mtp.eh_proj"](mtp_layer.eh_proj, first_id)

                # Inner layers reuse the base decoder walker (is_mtp=True).
                for inner in inner_layers:
                    hf_layer_id = inner.layer_number - 1
                    if isinstance(inner, MambaLayer):
                        self._get_mamba_layer_state_dict(inner, hf_layer_id, is_mtp=True)
                    elif isinstance(inner, TransformerLayer):
                        self._get_transformer_layer_state_dict(inner, hf_layer_id, is_mtp=True)
                    else:
                        raise ValueError(
                            "Only TransformerLayer or MambaLayer are supported in the MTP block."
                        )

                # The MTP block's own final layernorm attaches to the last inner HF index.
                final_layernorm = getattr(mtp_layer, "final_layernorm", None)
                if (
                    "mtp.final_layernorm" in self.rules
                    and final_layernorm is not None
                    and not isinstance(final_layernorm, IdentityOp)
                ):
                    self.rules["mtp.final_layernorm"](final_layernorm, last_id)

            mtp_state_dict = self._state_dict
        finally:
            self._state_dict = saved_state_dict

        if len(mtp_state_dict) > 0:
            print(f"Exported {len(mtp_state_dict)} MTP tensors from the live model")
        return mtp_state_dict

    def _copy_mtp_state_dict_from_pretrained(self) -> dict[str, torch.Tensor]:
        """Copy the BF16 MTP weights from the pretrained model (used when there is no live MTP)."""
        mtp_state_dict = {}
        if not self._hf_pretrained_model_name:
            return mtp_state_dict

        mtp_exists = False

        if os.path.isdir(self._hf_pretrained_model_name):
            safetensors_index_file = (
                Path(self._hf_pretrained_model_name) / "model.safetensors.index.json"
            )
            single_safetensors_file = Path(self._hf_pretrained_model_name) / "model.safetensors"
        else:
            try:
                safetensors_index_file = Path(
                    hf_hub_download(
                        repo_id=self._hf_pretrained_model_name,
                        filename="model.safetensors.index.json",
                    )
                )
                single_safetensors_file = None
            except EntryNotFoundError:
                # Model uses a single unsharded safetensors file -- check it for MTP weights.
                safetensors_index_file = None
                try:
                    single_safetensors_file = Path(
                        hf_hub_download(
                            repo_id=self._hf_pretrained_model_name,
                            filename="model.safetensors",
                        )
                    )
                except EntryNotFoundError:
                    return mtp_state_dict

        if safetensors_index_file is not None and safetensors_index_file.exists():
            with open(safetensors_index_file) as f:
                safetensors_index = json.load(f)
            model_dir = safetensors_index_file.parent
            for key in safetensors_index["weight_map"]:
                if key.startswith("mtp.") and key not in self._state_dict:
                    mtp_state_dict[key] = get_safetensor(model_dir, key)
                    mtp_exists = True
            if mtp_exists:
                print(f"Exported MTP using {safetensors_index_file=}")
        elif single_safetensors_file is not None and single_safetensors_file.exists():
            with safe_open(str(single_safetensors_file), framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith("mtp.") and key not in self._state_dict:
                        mtp_state_dict[key] = f.get_tensor(key)
                        mtp_exists = True
            if mtp_exists:
                print(f"Exported MTP using {single_safetensors_file=}")

        if mtp_exists:
            self.exclude_modules.append("mtp*")
        return mtp_state_dict

    def _get_gated_delta_net_state_dict(self, layer, layer_id, is_mtp=False):
        """Export a GatedDeltaNet (Qwen3.5 linear-attention) layer's ``self_attention``."""
        gdn = layer.self_attention
        self.rules["linear_attn"](gdn, layer_id, is_mtp=is_mtp)
        self.rules["linear_attn.conv1d"](gdn.conv1d, layer_id, is_mtp=is_mtp)
        self.rules["linear_attn.A_log"](gdn.A_log, layer_id, is_mtp=is_mtp)
        self.rules["linear_attn.dt_bias"](gdn.dt_bias, layer_id, is_mtp=is_mtp)
        self.rules["linear_attn.out_norm"](gdn.out_norm, layer_id, is_mtp=is_mtp)
        self.rules["linear_attn.out_proj"](gdn.out_proj, layer_id, is_mtp=is_mtp)

    def _get_mamba_layer_state_dict(self, layer, layer_id, is_mtp=False):
        if not isinstance(layer.norm, IdentityOp):
            self.rules["norm"](layer.norm, layer_id, is_mtp=is_mtp)
        else:
            # TE spec: norm is fused into in_proj (QuantTELayerNormColumnParallelLinear).
            # Mamba uses the legacy single-key `fused_norm` rule (Nemotron-H style).
            fused_key, norm_weight = self._get_fused_norm_weight(layer.mixer.in_proj)
            if norm_weight is not None:
                self.rules[fused_key](norm_weight, layer_id, is_mtp=is_mtp)

        self.rules["mixer_norm"](layer.mixer.norm, layer_id, is_mtp=is_mtp)
        self.rules["A_log"](layer.mixer.A_log, layer_id, is_mtp=is_mtp)
        self.rules["D"](layer.mixer.D, layer_id, is_mtp=is_mtp)
        self.rules["dt_bias"](layer.mixer.dt_bias, layer_id, is_mtp=is_mtp)

        self.rules["conv1d"](_get_mamba_conv1d(layer.mixer), layer_id, is_mtp=is_mtp)
        self.rules["in_proj"](layer.mixer.in_proj, layer_id, is_mtp=is_mtp)
        self.rules["out_proj"](layer.mixer.out_proj, layer_id, is_mtp=is_mtp)

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

    def _populate_rule_book(self):
        all_rules = {}

        def _custom_mapping_to_lambda(mapping):
            method_map = {
                "name_remapping": self._name_remapping,
                "qkv_slicing": self._qkv_slicing,
                "self_attention_scaling": self._self_attention_scaling,
                "gated_mlp_slicing": self._gated_mlp_slicing,
                "gated_delta_net_slicing": self._gated_delta_net_slicing,
                "grouped_mlp_slicing": self._grouped_mlp_slicing,
                "grouped_mlp_packing": self._grouped_mlp_packing,
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

    def _get_weight_bias(
        self,
        module: torch.nn.Module,
        dtype: torch.dtype = torch.float16,
        name_to_value: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get the weight and bias of the module.

        Args:
            module: The target module to get the weight and bias.
            dtype: The data type of the weight and bias.
            name_to_value: The dictionary to store the weight and bias. A new dict is created
                if not provided.

        Returns:
            The dictionary containing the weight and bias.
        """
        if name_to_value is None:
            name_to_value = {}
        # numel() > 0 intentionally excludes zero-element weight tensors (e.g. MoE routing
        # layers whose weight is a placeholder) so callers can use "weight" in name_to_value
        # as a reliable guard without re-inspecting module.weight.
        if hasattr(module, "weight") and module.weight is not None and module.weight.numel() > 0:
            weight = module.weight.to(dtype).cpu()
            name_to_value["weight"] = weight

        if hasattr(module, "bias") and module.bias is not None and module.bias.numel() > 0:
            name_to_value["bias"] = module.bias.to(dtype).cpu()

        if (
            hasattr(module, "expert_bias")
            and module.expert_bias is not None
            and module.expert_bias.numel() > 0
        ):
            name_to_value["expert_bias"] = module.expert_bias.to(dtype).cpu()

        return name_to_value

    def _get_quantized_state(
        self,
        module: torch.nn.Module,
        dtype: torch.dtype = torch.float16,
        prefix: str = "",
    ) -> tuple[dict[str, torch.Tensor], str, int]:
        """Return a state_dict, quantization format, and block_size of the module.

        Args:
            module: The target module to perform real quantization.
            dtype: The default data type.
            prefix: The prefix of the layer.

        Returns:
            Tuple: state_dict, quantization format, and block_size of the module.
        """
        name_to_value = {}
        qformat: str = self._get_quantization_format(module)
        if qformat is None and "norm" not in prefix:
            self._record_excluded_module(prefix)
        block_size = get_weight_block_size(module)

        name_to_value = self._get_weight_bias(module, dtype, name_to_value)

        if "weight" not in name_to_value:
            return name_to_value, qformat, block_size

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

    def _record_layer_quant_config(self, prefix: str, qformat: str | None, block_size: int):
        """Record per-HF-layer quantization metadata for mixed precision exports."""
        if qformat in (None, QUANTIZATION_NONE):
            return

        layer_name = prefix.removesuffix(".")
        if "{" in layer_name or not layer_name:
            return

        self.layer_config_dict[layer_name + ".quantization"] = qformat
        self.layer_config_dict[layer_name + ".awq_block_size"] = block_size

    def _record_excluded_module(self, prefix: str):
        """Record an unquantized HF module prefix for hf_quant_config."""
        layer_name = prefix.removesuffix(".")
        if "{" in layer_name or not layer_name:
            return

        if layer_name not in self.exclude_modules:
            self.exclude_modules.append(layer_name)

    @staticmethod
    def _mtp_prefix(prefix: str) -> str:
        """Rewrite a base-model target prefix (backbone/model root) to its MTP counterpart."""
        if "backbone" in prefix:
            return prefix.replace("backbone", "mtp", 1)
        # Replace the root only: a VLM's "model.language_model." must not become "mtp.language_mtp.".
        if prefix.startswith("model.language_model."):
            return "mtp." + prefix[len("model.language_model.") :]
        return prefix.replace("model", "mtp", 1)

    def _name_remapping(
        self,
        module: torch.nn.Module | torch.Tensor,
        prefix: str,
        skip_output_scale: bool = True,
        mapping={},
        dtype: torch.dtype | None = None,
        is_mtp: bool = False,
        zero_centered_gamma: bool = False,
    ):
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        if dtype is None:
            dtype = self.dtype

        if isinstance(module, torch.Tensor):
            self._state_dict[prefix] = (module + 1.0) if zero_centered_gamma else module
            return

        name_to_value, qformat, block_size = self._get_quantized_state(module, dtype, prefix=prefix)
        self._record_layer_quant_config(prefix, qformat, block_size)

        weight = name_to_value.pop("weight")
        if zero_centered_gamma:
            # Megatron centres this gamma on 0, HF on 1. Assert, don't derive: the config flag
            # is model-wide while this one is per-norm.
            assert getattr(module, "zero_centered_gamma", False), (
                f"{prefix} is mapped as zero-centered gamma but the module reports otherwise; "
                "exporting would shift the weights by 1.0"
            )
            weight = weight + 1.0
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
        self, module, prefix, gate_proj_name="gate_proj", up_proj_name="up_proj", is_mtp=False
    ):
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        name_to_value, qformat, block_size = self._get_quantized_state(
            module, self.dtype, prefix=prefix
        )

        weight = name_to_value.pop("weight")
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        gate_proj_prefix = prefix + gate_proj_name + "."
        up_proj_prefix = prefix + up_proj_name + "."
        self._record_layer_quant_config(gate_proj_prefix, qformat, block_size)
        self._record_layer_quant_config(up_proj_prefix, qformat, block_size)

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

    def _grouped_mlp_packing(self, module, prefix, parallel_config=None, is_mtp=False):
        """Pack TEGroupedMLP experts into one ``[num_experts, out, in]`` tensor (Qwen3.5 layout).

        Reuses the per-expert path for the EP gather and per-expert quantizers, then stacks by
        global expert id and re-quantizes once with the max scale, as ``_pack_name_remapping`` does.
        """
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        marker = "\x00pack\x00"
        saved_state_dict = self._state_dict
        self._state_dict = OrderedDict()
        try:
            qformat, block_size = self._grouped_mlp_slicing(
                module,
                marker + "{}",
                parallel_config=parallel_config,
                is_mtp=False,
                quantize=False,
                record_quant_config=False,
            )
            per_expert = self._state_dict
        finally:
            self._state_dict = saved_state_dict

        def collect(suffix):
            found = {}
            for key, value in per_expert.items():
                if not key.startswith(marker) or not key.endswith(suffix):
                    continue
                found[int(key[len(marker) :].split(".", 1)[0])] = value
            return [found[i] for i in sorted(found)]

        weights = collect(".weight")
        if not weights:
            return
        handled = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale", ".output_scale")
        unhandled = {k.split(".", 1)[1] for k in per_expert if not k.endswith(handled)}
        assert not unhandled, (
            f"{prefix}: grouped-expert packing has no rule for {sorted(unhandled)}"
        )
        # Record against the packed prefix, as _pack_name_remapping does for the other packed path.
        if qformat in (None, QUANTIZATION_NONE):
            self._record_excluded_module(prefix)
        else:
            assert block_size is not None
            self._record_layer_quant_config(prefix, qformat, block_size)
        scales, scales_2 = collect(".weight_scale"), collect(".weight_scale_2")
        input_scales = collect(".input_scale")

        # Quantize once over the stack, exactly as _pack_name_remapping does.
        merged_weight = torch.stack(weights, dim=0)
        if not scales:
            self._state_dict[prefix] = merged_weight
        else:
            if scales_2:
                # NVFP4 keeps each expert's block scales, rescaled onto the merged global scale.
                merged_scale, merged_scale_2 = self._merge_nvfp4_expert_scales(scales, scales_2)
            else:
                merged_scale, merged_scale_2 = torch.max(torch.stack(scales, dim=0), dim=0)[0], None
            self._state_dict[prefix] = to_quantized_weight(
                merged_weight, merged_scale, qformat, merged_scale_2, block_size
            )
            # Same suffixes as _pack_name_remapping so both packed paths agree.
            self._state_dict[prefix + "_weight_scale"] = merged_scale
            if merged_scale_2 is not None:
                self._state_dict[prefix + "_weight_scale_2"] = merged_scale_2
        if input_scales:
            self._state_dict[prefix + "_input_scale"] = torch.max(
                torch.stack(input_scales, dim=0), dim=0
            )[0]

    def _grouped_mlp_slicing(
        self,
        module,
        prefix,
        parallel_config=None,
        is_mtp=False,
        quantize=True,
        record_quant_config=True,
    ):
        """Export TEGroupedMLP weight0..weight{N-1} as one HF-style entry per expert.

        ``quantize=False`` emits unquantized weights alongside the scales, which
        ``_grouped_mlp_packing`` needs so it can quantize once over the stacked tensor.

        At EP>1, local ids are mapped to global via ``module.local_expert_indices``
        and per-expert state is ``all_gather_object``-ed across the EP group. All EP ranks
        MUST enter this method for the same layer in lockstep or the gather hangs.

        Reverse of _grouped_mlp_merging in the importer.
        """
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        num_experts = module.num_gemms
        state_dict = module.state_dict()

        has_weight = hasattr(module, "weight")
        grouped_wq = getattr(module, "weight_quantizer", None)
        # Quantized TE grouped experts must be per-expert (GroupedQuantizer); None = unquantized MLP.
        assert grouped_wq is None or isinstance(grouped_wq, GroupedQuantizer), (
            f"TEGroupedLinear.weight_quantizer must be GroupedQuantizer or None, got "
            f"{type(grouped_wq).__name__}; pre-0.47 single-quantizer checkpoints are not supported."
        )
        if grouped_wq is not None and num_experts > len(grouped_wq):
            warn_rank_0(
                f"TEGroupedMLP has {num_experts} local experts but only {len(grouped_wq)} "
                f"per-expert weight quantizers; experts >= {len(grouped_wq)} reuse expert "
                f"{len(grouped_wq) - 1}'s scales (TP/EP-mismatch fallback)."
            )

        ep_size = (
            get_expert_model_parallel_world_size() if torch.distributed.is_initialized() else 1
        )
        ep_rank = get_expert_model_parallel_rank() if torch.distributed.is_initialized() else 0

        # Prefer module.local_expert_indices; fall back to Megatron's contiguous layout.
        # Normalize to list[int] since Megatron may expose this as a torch.Tensor.
        indices = getattr(module, "local_expert_indices", None)
        if indices is None and getattr(module, "experts", None) is not None:
            indices = getattr(module.experts, "local_expert_indices", None)
        if indices is None:
            local_expert_indices = [ep_rank * num_experts + i for i in range(num_experts)]
        elif isinstance(indices, torch.Tensor):
            local_expert_indices = indices.detach().cpu().tolist()
        else:
            local_expert_indices = [int(i) for i in indices]
        if len(local_expert_indices) != num_experts:
            raise ValueError(
                f"local_expert_indices length {len(local_expert_indices)} doesn't match "
                f"module.num_gemms {num_experts}"
            )

        # Collective-safe missing-key check: all_reduce(MAX) over a local 0/1 flag
        # so any rank's missing key surfaces everywhere. Flag lives on the current
        # CUDA device -- NCCL has no CPU backend on the EP group.
        local_missing = [
            k for k in (f"weight{i}" for i in range(num_experts)) if k not in state_dict
        ]
        if ep_size > 1:
            missing_flag = torch.tensor(
                [1 if local_missing else 0],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(
                missing_flag,
                op=torch.distributed.ReduceOp.MAX,
                group=get_expert_model_parallel_group(),
            )
            if missing_flag.item() != 0:
                raise ValueError(
                    f"TEGroupedMLP missing expert weights on at least one EP rank "
                    f"(local missing on rank {ep_rank}: {local_missing})"
                )
        elif local_missing:
            raise ValueError(f"TEGroupedMLP missing expert weights: {local_missing}")

        # Per expert, temporarily assign weight = weight{i} and, for the per-expert
        # quantizer layout (GroupedQuantizer), swap in that expert's own TensorQuantizer,
        # so _get_quantized_state extracts each expert's own qformat/scales instead of
        # applying weight0's scales to every expert.
        local_expert_state: dict[str, torch.Tensor] = {}
        seen_qformat = None
        seen_block_size = None
        # Dynamic quantizers we populate a temporary export-only amax on; reset in finally so
        # export leaves module state unchanged (else a dynamic-NVFP4 quantizer keeps a stale max|W|).
        temp_amax_wqs: list = []
        try:
            for local_id in range(num_experts):
                global_id = local_expert_indices[local_id]
                expert_prefix = prefix.format(global_id) + "."
                weight_key = f"weight{local_id}"

                module.weight = getattr(module, weight_key)
                if grouped_wq is not None:
                    module.weight_quantizer = grouped_wq[min(local_id, len(grouped_wq) - 1)]
                    # Dynamic-NVFP4 per-expert quantizers carry no stored amax, but
                    # weight_scale_2 derivation asserts one. Max-calibration weight amax
                    # is exactly max(|W|), so compute it from this expert's weight.
                    _wq = module.weight_quantizer
                    if getattr(_wq, "_amax", None) is None and getattr(_wq, "is_enabled", False):
                        _wq.amax = module.weight.detach().abs().max().float()
                        temp_amax_wqs.append(_wq)

                name_to_value, qformat, block_size = self._get_quantized_state(
                    module, self.dtype, prefix=prefix
                )
                weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)
                name_to_value.pop("weight", None)
                seen_qformat, seen_block_size = qformat, block_size

                weight = state_dict[weight_key].to(self.dtype).cpu()
                weight_scale_cpu = (
                    weight_scale.detach().cpu().clone() if weight_scale is not None else None
                )
                weight_scale_2_cpu = (
                    weight_scale_2.detach().cpu().clone() if weight_scale_2 is not None else None
                )

                if weight_scale_cpu is None:
                    local_expert_state[expert_prefix + "weight"] = weight
                else:
                    local_expert_state[expert_prefix + "weight"] = (
                        weight
                        if not quantize
                        else to_quantized_weight(
                            weight,
                            weight_scale_cpu,
                            qformat,
                            weight_scale_2_cpu,
                            block_size,
                        )
                    )
                    local_expert_state[expert_prefix + "weight_scale"] = weight_scale_cpu.clone()

                if weight_scale_2_cpu is not None:
                    local_expert_state[expert_prefix + "weight_scale_2"] = (
                        weight_scale_2_cpu.clone()
                    )

                for key, val in name_to_value.items():
                    if key == "output_scale":
                        continue
                    local_expert_state[expert_prefix + key] = val.detach().cpu().clone()
        finally:
            for _wq in temp_amax_wqs:
                _wq.reset_amax()
            if grouped_wq is not None:
                module.weight_quantizer = grouped_wq
            if not has_weight and hasattr(module, "weight"):
                delattr(module, "weight")

        # Record quant config for ALL global experts on every rank; otherwise the writer's
        # hf_quant_config.json would miss (EP-1)/EP of the routed experts. All experts in
        # a TEGroupedMLP layer share qformat/block_size, so local values apply globally.
        if seen_qformat is not None and record_quant_config:
            assert seen_block_size is not None
            num_total_experts = num_experts * ep_size
            for global_id in range(num_total_experts):
                self._record_layer_quant_config(
                    prefix.format(global_id) + ".", seen_qformat, seen_block_size
                )

        if ep_size > 1:
            # all_gather_object pickles trip on quantized uint8 tensors whose
            # UntypedStorage has no dtype attr -- round-trip through torch.save bytes.
            _buf = io.BytesIO()
            torch.save(local_expert_state, _buf)
            local_bytes = _buf.getvalue()
            del _buf
            gathered_bytes: list = [None] * ep_size
            torch.distributed.all_gather_object(
                gathered_bytes, local_bytes, group=get_expert_model_parallel_group()
            )
            del local_bytes
            for b in gathered_bytes:
                # weights_only=False: our own torch.save output from a sibling EP rank
                # in this job's collective, not user-supplied.
                s_loaded = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
                self._state_dict.update(s_loaded)
            del gathered_bytes
        else:
            self._state_dict.update(local_expert_state)
        return seen_qformat, seen_block_size

    def _qkv_slicing(
        self,
        module,
        prefix,
        q_proj_name="q_proj",
        k_proj_name="k_proj",
        v_proj_name="v_proj",
        is_mtp=False,
    ):
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        name_to_value, qformat, block_size = self._get_quantized_state(
            module, self.dtype, prefix=prefix
        )

        q_proj_prefix = prefix + q_proj_name + "."
        k_proj_prefix = prefix + k_proj_name + "."
        v_proj_prefix = prefix + v_proj_name + "."
        self._record_layer_quant_config(q_proj_prefix, qformat, block_size)
        self._record_layer_quant_config(k_proj_prefix, qformat, block_size)
        self._record_layer_quant_config(v_proj_prefix, qformat, block_size)
        if qformat in (None, QUANTIZATION_NONE):
            # Split fused linear_qkv exclude into per-HF-name q/k/v_proj entries.
            fused_prefix = prefix.removesuffix(".")
            self.exclude_modules = [m for m in self.exclude_modules if m != fused_prefix]
            self._record_excluded_module(q_proj_prefix)
            self._record_excluded_module(k_proj_prefix)
            self._record_excluded_module(v_proj_prefix)

        config = module.config
        hidden_size = config.hidden_size
        num_query_groups = config.num_query_groups
        head_num = config.num_attention_heads
        head_size = config.kv_channels
        heads_per_group = head_num // num_query_groups
        # Gated attention (Qwen3.5) packs a gate beside every query head, so a group holds
        # [q, gate, k, v]; HF keeps the gate inside ``q_proj``.
        output_gate = getattr(config, "attention_output_gate", False)
        group_dim = (2 * heads_per_group if output_gate else heads_per_group) + 2
        qkv_total_dim = num_query_groups * group_dim

        weight = name_to_value.pop("weight")

        if weight.shape[-1] == 2 * hidden_size:
            print(
                "Parameter linear_qkv.weight has 2x the hidden_size."
                "Set hidden_size to 2x the hidden_size. EAGLE3 is the only known"
                "use case which has this behavior."
            )
            hidden_size = 2 * hidden_size

        # When TP > 1 the weight tensor is already sharded: shape[0] = per_rank_qkv_dim, not
        # qkv_total_dim.  Derive the per-rank dimensions from the actual tensor shape so that
        # all subsequent reshape/slice operations are correct regardless of TP degree.
        per_rank_qkv_dim = weight.shape[0] // head_size
        num_query_groups_local = num_query_groups * per_rank_qkv_dim // qkv_total_dim
        weight = weight.reshape([per_rank_qkv_dim, head_size, hidden_size])
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        q_slice = torch.cat(
            [
                torch.arange(group_dim * i, group_dim * i + heads_per_group)
                for i in range(num_query_groups_local)
            ]
        )
        gate_slice = (
            torch.cat(
                [
                    torch.arange(
                        group_dim * i + heads_per_group, group_dim * i + 2 * heads_per_group
                    )
                    for i in range(num_query_groups_local)
                ]
            )
            if output_gate
            else None
        )
        k_slice = torch.arange(group_dim - 2, per_rank_qkv_dim, group_dim)
        v_slice = torch.arange(group_dim - 1, per_rank_qkv_dim, group_dim)
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

        def _take(tensor, index, last_dim, with_gate=False):
            """Gather ``index`` heads, appending the gate heads for q under gated attention."""
            taken = tensor[index]
            if with_gate:
                taken = torch.cat([taken, tensor[gate_slice]], dim=1)
            return taken.reshape(-1, last_dim)

        gated = [output_gate, False, False]  # q carries the gate; k and v do not

        proj_weights = [_take(weight, s, hidden_size, g) for s, g in zip(slices, gated)]
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
                    [per_rank_qkv_dim, head_size, weight_scale_hidden_size]
                )
                proj_weight_scales = [
                    _take(weight_scale, s, weight_scale_hidden_size, g).to(dtype=weight_scale_dtype)
                    for s, g in zip(slices, gated)
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
            if key == "bias":
                # Slice bias similar to weight
                bias = val.detach().clone()
                bias = bias.reshape([per_rank_qkv_dim, head_size])
                proj_biases = [_take(bias, s, 1, g).reshape(-1) for s, g in zip(slices, gated)]
                proj_bias_keys = [q_proj_prefix + key, k_proj_prefix + key, v_proj_prefix + key]
                for bias_tensor, bias_key in zip(proj_biases, proj_bias_keys):
                    self._state_dict[bias_key] = bias_tensor
            else:
                self._state_dict[q_proj_key] = val.detach().clone()
                self._state_dict[k_proj_key] = val.detach().clone()
                self._state_dict[v_proj_key] = val.detach().clone()

    def _gated_delta_net_slicing(self, module, prefix, is_mtp=False):
        """Split GatedDeltaNet's fused ``in_proj`` into HF's qkv / z / b / a projections.

        Megatron packs ``[query, key, value, z, beta, alpha]``; sizes come from the module so TP
        sharding needs no re-derivation.
        """
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        in_proj = module.in_proj
        name_to_value, qformat, block_size = self._get_quantized_state(
            in_proj, self.dtype, prefix=prefix
        )

        assert tuple(module.in_proj_split_names) == (
            "query",
            "key",
            "value",
            "z",
            "beta",
            "alpha",
        ), (
            f"Unexpected GatedDeltaNet in_proj layout {tuple(module.in_proj_split_names)}; the "
            "split below assumes [query, key, value, z, beta, alpha]"
        )
        sections = dict(zip(module.in_proj_split_names, module.in_proj_split_sections))
        split_sizes = [
            sections["query"] + sections["key"] + sections["value"],
            sections["z"],
            sections["beta"],
            sections["alpha"],
        ]
        proj_names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        proj_prefixes = [prefix + name + "." for name in proj_names]
        # The recipes keep the alpha / beta gates in BF16, but Megatron fuses all six sections
        # behind one quantizer, so they can only be dropped here rather than by a quantizer_name.
        keep_bf16 = {
            p for p, n in zip(proj_prefixes, proj_names) if n in ("in_proj_a", "in_proj_b")
        }

        for proj_prefix in proj_prefixes:
            if proj_prefix in keep_bf16:
                self._record_excluded_module(proj_prefix)
            else:
                self._record_layer_quant_config(proj_prefix, qformat, block_size)
        if qformat in (None, QUANTIZATION_NONE):
            # Split the fused in_proj exclude entry into the per-HF-name projections.
            self.exclude_modules = [
                m for m in self.exclude_modules if m != prefix.removesuffix(".")
            ]
            for proj_prefix in proj_prefixes:
                self._record_excluded_module(proj_prefix)

        weight = name_to_value.pop("weight")
        proj_weights = list(torch.split(weight, split_sizes, dim=0))
        proj_keys = [p + "weight" for p in proj_prefixes]
        weight_scale, weight_scale_2 = self._get_weight_scales(name_to_value, qformat)

        if weight_scale is None:
            for key, proj_weight in zip(proj_keys, proj_weights):
                self._state_dict[key] = proj_weight
        else:
            if len(weight_scale.shape) > 0:
                # Per-channel / per-block scales are laid out along the same (output) dim.
                proj_scales = list(torch.split(weight_scale, split_sizes, dim=0))
            else:
                proj_scales = [weight_scale.detach().clone() for _ in proj_keys]
            for proj_prefix, proj_weight, scale, key in zip(
                proj_prefixes, proj_weights, proj_scales, proj_keys
            ):
                if proj_prefix in keep_bf16:
                    self._state_dict[key] = proj_weight
                    continue
                self._state_dict[key] = to_quantized_weight(
                    proj_weight, scale, qformat, weight_scale_2, block_size
                )
                self._state_dict[key + "_scale"] = scale

        if weight_scale_2 is not None:
            if len(weight_scale_2.shape) > 0:
                raise ValueError("weight_scale_2 must be a scalar!")
            for proj_prefix, key in zip(proj_prefixes, proj_keys):
                if proj_prefix not in keep_bf16:
                    self._state_dict[key + "_scale_2"] = weight_scale_2.detach().clone()

        # weight and weight_scale have been popped; the rest (bias, input_scale, ...) is
        # either split like the weight or replicated onto every projection.
        for key, val in name_to_value.items():
            if key == "bias":
                for proj_bias, proj_prefix in zip(
                    torch.split(val.detach().clone(), split_sizes, dim=0), proj_prefixes
                ):
                    self._state_dict[proj_prefix + key] = proj_bias
            else:
                for proj_prefix in proj_prefixes:
                    if proj_prefix in keep_bf16:
                        continue
                    self._state_dict[proj_prefix + key] = val.detach().clone()

    def _self_attention_scaling(
        self, module, prefix, k_scale_name="k_scale", v_scale_name="v_scale", is_mtp=False
    ):
        """KV cache scaling for CoreAttention module."""
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        k_scale_key = prefix + k_scale_name
        v_scale_key = prefix + v_scale_name
        if hasattr(module, "k_bmm_quantizer") and hasattr(module, "v_bmm_quantizer"):
            kv_scales = get_kv_cache_scaling_factor(module)
            if all(s is not None for s in kv_scales):
                self._state_dict[k_scale_key] = kv_scales[0]
                self._state_dict[v_scale_key] = kv_scales[1]

            kv_cache_dtype = get_kv_cache_dtype(module)
            if kv_cache_dtype in (KV_CACHE_FP8, KV_CACHE_NVFP4):
                # FP8 KV Cache is supported in VLLM; NVFP4 supported in TRTLLM
                self.kv_cache_dtype = kv_cache_dtype

    @staticmethod
    def _merge_nvfp4_expert_scales(scales: list, scales_2: list):
        """Merge per-expert NVFP4 scales onto one global scale, preserving each expert's FP4 range.

        Each expert's block scales were derived against its own ``scale_2``; rescaling them by
        ``scale_2_i / scale_2_max`` keeps the quieter experts from losing a mantissa bit.
        """
        merged_scale_2 = torch.max(torch.stack(scales_2, dim=0), dim=0)[0]
        stacked_2 = torch.stack(scales_2, dim=0).reshape(-1, *([1] * scales[0].dim()))
        merged_scale = (
            torch.stack(scales, dim=0).to(torch.float32) * (stacked_2 / merged_scale_2)
        ).to(scales[0].dtype)
        return merged_scale, merged_scale_2

    def _pack_name_remapping(self, module, prefix, layer_type=None, is_mtp=False, transpose=True):
        """Pack per-expert weights into one tensor; ``transpose`` for HF [E, in, out] layouts."""
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        weight_list = []
        weight_scale_list = []
        weight_scale_2_list = []
        input_scale_list = []

        for expert in module:
            assert layer_type is not None, "layer_type is required for pack_name_remapping"
            name_to_value, qformat, block_size = self._get_quantized_state(
                getattr(expert, layer_type), self.dtype, prefix=prefix
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
            self._record_layer_quant_config(prefix, qformat, block_size)

        merged_weight = torch.stack(weight_list, dim=0)

        # Megatron is [num_experts, out, in]; most HF layouts want [num_experts, in, out], but
        # Qwen3.5 keeps Megatron's orientation.
        if transpose:
            merged_weight = merged_weight.transpose(-2, -1).contiguous()

        if weight_scale_2_list[0] is None:
            merged_weight_scale_2 = None
            if weight_scale_list[0] is not None:
                merged_weight_scale = torch.max(torch.stack(weight_scale_list, dim=0), dim=0)[0]
            else:
                merged_weight_scale = None
        else:
            # NVFP4
            merged_weight_scale, merged_weight_scale_2 = self._merge_nvfp4_expert_scales(
                weight_scale_list, weight_scale_2_list
            )
            if transpose:
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

    def _pack_name_remapping_gpt_oss(self, module, prefix, layer_type=None, is_mtp=False):
        """Pack name remapping into one tensor."""
        if is_mtp:
            prefix = self._mtp_prefix(prefix)
        weight_list = []
        weight_scale_list = []
        weight_scale_2_list = []
        input_scale_list = []
        bias_list = []

        for expert in module:
            assert layer_type is not None, "layer_type is required for pack_name_remapping"
            name_to_value, qformat, block_size = self._get_quantized_state(
                getattr(expert, layer_type), self.dtype, prefix=prefix
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
            self._record_layer_quant_config(prefix, qformat, block_size)

        merged_weight = torch.stack(weight_list, dim=0)

        # Transpose the last two dimensions to match HuggingFace format (except for GptOssForCausalLM)
        # Megatron format: [num_experts, out_features, in_features]
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

    def _gather_exclude_modules(self):
        """Get exclude_modules from all ranks to ensure hf_quant_config is complete."""
        if not torch.distributed.is_initialized():
            return sorted(self.exclude_modules)

        all_exclude_modules = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(all_exclude_modules, self.exclude_modules)
        combined_exclude_modules = set()
        for modules in all_exclude_modules:
            if modules:
                combined_exclude_modules.update(modules)
        return sorted(combined_exclude_modules)

    def _gather_layer_config_dict(self):
        """Get per-layer quantization metadata from all ranks for hf_quant_config."""
        if not torch.distributed.is_initialized():
            return dict(sorted(self.layer_config_dict.items()))

        all_layer_config_dicts = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(all_layer_config_dicts, self.layer_config_dict)
        combined_layer_config_dict = {}
        for layer_config_dict in all_layer_config_dicts:
            if layer_config_dict:
                combined_layer_config_dict.update(layer_config_dict)
        return dict(sorted(combined_layer_config_dict.items()))

    def _gather_kv_cache_dtype(self):
        """Return first non-None kv_cache_dtype across ranks (only attention ranks set it)."""
        local = getattr(self, "kv_cache_dtype", None)
        if not torch.distributed.is_initialized():
            return local
        all_dtypes = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(all_dtypes, local)
        for dt in all_dtypes:
            if dt is not None:
                return dt
        return None


def _read_checkpoint_keys(checkpoint_dir) -> set[str]:
    """Tensor names in a local HuggingFace checkpoint, from its index or single safetensors file."""
    directory = Path(checkpoint_dir)
    index_file = directory / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            return set(json.load(f)["weight_map"])
    single_file = directory / "model.safetensors"
    if single_file.exists():
        with safe_open(str(single_file), framework="pt", device="cpu") as f:
            return set(f.keys())
    return set()


def export_mcore_gpt_to_hf(
    model: torch.nn.Module,
    pretrained_model_name_or_path: str | os.PathLike,
    export_extra_modules: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    export_dir: Path | str = tempfile.gettempdir(),
    trust_remote_code: bool = False,
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
        trust_remote_code=trust_remote_code,
        moe_router_dtype=moe_router_dtype,
    )
    if exporter.export_extra_modules:
        exporter.save_pretrained_extra_modules(export_dir)
    else:
        exporter.save_pretrained(export_dir, pretrained_model_name_or_path)


def import_mcore_gpt_from_hf(
    model: torch.nn.Module,
    pretrained_model_path: str,
    workspace_dir: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    moe_router_dtype: torch.dtype | None = None,
):
    """Import GPTModel state_dict from supported HuggingFace pretrained model path.

    Args:
        model: The Megatron Core GPTModel instance.
        pretrained_model_path: A path to a *directory* containing model weights saved using
            [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
        workspace_dir: The directory to save the workspace.
        dtype: The weights data type to import.
        trust_remote_code: If True, this allows importing from a wider range of sources.
        moe_router_dtype: The data type to import the moe router weights.
    """
    importer = GPTModelImporter(
        model,
        pretrained_model_path,
        workspace_dir=workspace_dir,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        moe_router_dtype=moe_router_dtype,
    )
    importer._import_state_dict()
