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

# based on https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/llm/gpt/model/llama_nemotron.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Dict, Optional, Union
from modelopt.torch.puzzletron.tools.logger import mprint
import torch
import torch.nn.functional as F
from nemo.collections.llm.gpt.model.base import (
    GPTConfig,
    GPTModel,
    torch_dtype_from_mcore_config,
)
from nemo.collections.llm.gpt.model.llama import (
    Llama3Config,
    Llama31Config,
    Llama31Config70B,
    LlamaConfig,
    apply_rope_scaling,
)
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf, dtype_from_str
from nemo.utils import logging
from nemo.utils.import_utils import safe_import
from torch import nn

# from nemo.collections.llm.gpt.model.llama_nemotron import Llama33NemotronSuper49BConfig


_, HAVE_TE = safe_import("transformer_engine")
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.transformer.spec_utils import ModuleSpec

if TYPE_CHECKING:
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM, GenerationConfig
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
    from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig

from modelopt.torch.puzzletron.export.MCore.llama_nemotron_utils import (
    _build_puzzletron_mappings_and_transforms,
    _config_to_dict,
    convert_attention_config_from_cfg_object,
    convert_mlp_config_from_cfg_object,
    convert_nemo_config_to_hf_decilm_config,
    dtype_from_dict,
    merge_qkv_for_puzzletron,
    split_qkv_for_puzzletron,
)
from modelopt.torch.puzzletron.export.MCore.puzzletron_layer_specs import (
    PuzzletronHeterogeneousTransformerConfig,
    get_gpt_heterogeneous_layer_spec_puzzletron,
)


def heterogeneous_layer_spec_puzzletron(config: PuzzletronHeterogeneousTransformerConfig) -> ModuleSpec:
    return get_gpt_heterogeneous_layer_spec_puzzletron(config, use_transformer_engine=HAVE_TE)


# Refactored to inherit directly from GPTConfig instead of Llama31Config70B
# This makes it easier to understand what attributes are set through the hierarchy
@dataclass
class PuzzletronNemotronModelConfig(GPTConfig, PuzzletronHeterogeneousTransformerConfig):
    """Configuration for Puzzletron Nemotron models.

    DESIGN RATIONALE:
    ================
    Refactored from original inheritance (Llama31Config70B + PuzzletronHeterogeneousTransformerConfig)
    to explicit attribute definition for clarity and maintainability. Maintains identical behavior
    to the original Llama hierarchy while enabling future flexibility.

    ATTRIBUTE ORGANIZATION:
    ======================
    Explicitly defines attributes from the Llama hierarchy:
    Llama31Config70B → Llama31Config → Llama3Config → LlamaConfig → GPTConfig

    FUTURE DEVELOPMENT:
    ==================
    Attributes can be freely modified/removed for future Puzzletron models.
    In this case the tests in test_puzzletron_nemotron_config_inheritance.py will need to be updated.
    Current explicit definition is for clarity during transition period.
    """

    # Override attributes from PuzzletronHeterogeneousTransformerConfig with Llama hierarchy values
    # These ensure we maintain the same behavior as the original Llama31Config70B inheritance

    # ===== LlamaConfig attributes =====
    # Core model architecture
    # NOTE: Default is F.silu, but this is overridden during instantiation to match all blocks
    # See instantiate_nemo_config_from_adapted_dict() which enforces same activation across blocks
    activation_func: Callable = F.silu
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096  # (will be overridden by Llama31Config70B)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    # Fusion settings
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    use_transformer_engine_op_fuser: Optional[bool] = None

    # ===== Llama3Config attributes =====
    num_query_groups: int = 8
    init_method_std: float = 0.01  # (will be overridden by Llama31Config)
    layernorm_epsilon: float = 1.0e-05
    rotary_percent: float = 1.0

    # ===== Llama31Config attributes =====
    scale_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    old_context_len: int = 8192
    init_method_std: float = 0.02  # (overrides Llama3Config)

    # ===== Llama31Config70B attributes =====
    # Core model architecture (70B-specific)
    rotary_base: int = 500_000
    seq_length: int = 131072  # (overrides LlamaConfig)
    num_layers: int = 80  #
    hidden_size: int = 8192  #
    ffn_hidden_size: int = 28672  #
    num_attention_heads: int = 64  #
    kv_channels: int = 128  #  (derived from hidden_size // num_attention_heads)
    make_vocab_size_divisible_by: int = 128  #

    # ===== PuzzletronHeterogeneousTransformerConfig attributes =====
    # Actual new PuzzleNemotronModelConfig attributes
    heterogeneous_layers_config_path: Optional[str] = None
    heterogeneous_layers_config_encoded_json: Optional[str] = None
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = heterogeneous_layer_spec_puzzletron

    # HF-specific metadata for lossless round-trip conversion (HF → NeMo → HF)
    # Stores HF config fields that don't have direct NeMo equivalents
    source_hf_config_metadata: Optional[Dict[str, Any]] = None

    # NOTE: How activation_func is handled for Puzzletron models
    # ==============================================================
    # Puzzletron models can define activation functions per-block, but MCore's validation
    # only checks the global activation_func (not per-block activations).
    # See: https://github.com/NVIDIA/Megatron-LM/blob/268fda08592528b7bc1a21aadaed259980ca8efb/megatron/core/transformer/transformer_config.py#L1043-L1061
    #
    # Current approach (enforced in instantiate_nemo_config_from_adapted_dict):
    # - All blocks must use the SAME activation function (None allowed for no-op blocks)
    # - The global activation_func is set to match the blocks' shared activation
    # - This ensures MCore's global validation passes correctly
    #
    # Rationale:
    # 1. MCore validates global activation_func during __post_init__() (lines 1043-1061)
    # 2. NeMo calls __post_init__() AGAIN during trainer.strategy.connect(model)
    #    See: https://github.com/NVIDIA/NeMo/blob/2e19aebd8c8fa9ff7ce9b5076ce130404713443c/nemo/lightning/_strategy_lib.py#L172-L175
    # 3. At runtime, MCore uses per-block activations from get_config_for_layer()
    #    See: https://github.com/NVIDIA/Megatron-LM/blob/268fda08592528b7bc1a21aadaed259980ca8efb/megatron/core/transformer/transformer_block.py#L308-L319
    #
    # For heterogeneous activations across blocks, MCore would need to update their
    # validation logic to support per-block validation (e.g., in get_config_for_layer() or MLP.__init__)

    # ===== Llama31Config method =====
    def configure_model(self, tokenizer, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Llama 3.1 model.

        NOTE: This method is originally from Llama31Config and is explicitly included here
        for consistency and clarity. It maintains the same behavior as the original
        Llama hierarchy inheritance approach.

        Extends the base configuration with Llama 3.1 specific RoPE scaling.
        This method applies RoPE scaling for extended context length support.
        """
        model = super().configure_model(tokenizer, pre_process, post_process, vp_stage)
        # Apply rope scaling for Llama3.1 model
        model.rotary_pos_emb.inv_freq = apply_rope_scaling(
            model.rotary_pos_emb.inv_freq,
            factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            old_context_len=self.old_context_len,
        )
        return model

    @classmethod
    def from_dict_with_preprocessing(cls, config_dict):
        # Potentially adapt the config_dict before instantiation
        instance = cls(**config_dict)
        # Potentially adapt the config after instantiation
        return instance

    # static method
    @staticmethod
    def create_adapted_config_dict_from_puzzletron_config(cfg):
        # TODO: consider doing do this without conversion to dictionary in the future (instead have an adapted config object)
        # Create an empty config object of the same class as cfg
        adapted_cfg_dict = dict()
        orig_cfg_dict = vars(cfg)

        # Extract first set of values from the original config
        adapted_cfg_dict["head_dim"] = orig_cfg_dict["head_dim"]
        adapted_cfg_dict["num_attention_heads"] = orig_cfg_dict["num_attention_heads"]
        # Handle rope_scaling - can be None, missing, or a dict
        adapted_cfg_dict["rope_scaling"] = orig_cfg_dict.get("rope_scaling") or {}

        block_conf = {
            "block_configs": [
                {
                    "attention": convert_attention_config_from_cfg_object(
                        orig_cfg_dict["block_configs"][i].attention,
                        adapted_cfg_dict["num_attention_heads"],
                        adapted_cfg_dict["head_dim"],
                    ),
                    "mlp": {
                        **convert_mlp_config_from_cfg_object(
                            orig_cfg_dict["block_configs"][i].ffn,
                            (
                                orig_cfg_dict["block_configs"][i].parallel_blocks
                                if hasattr(orig_cfg_dict["block_configs"][i], "parallel_blocks")
                                else None
                            ),
                        ),
                        # Store the per-block activation function as a string (for JSON serialization)
                        "hidden_act": (
                            orig_cfg_dict["block_configs"][i].ffn.hidden_act
                            if not (
                                orig_cfg_dict["block_configs"][i].ffn.no_op
                                or orig_cfg_dict["block_configs"][i].ffn.replace_with_linear
                            )
                            else None
                        ),
                    },
                }
                for i in range(len(orig_cfg_dict["block_configs"]))
            ]
        }
        if orig_cfg_dict["o_proj_bias"] != orig_cfg_dict["attention_bias"]:
            raise NotImplementedError("o_proj_bias is not fully supported")
        if orig_cfg_dict["position_embedding_type"] not in ["rope", "yarn"]:
            # this one is not supported by MCore
            raise ValueError(f"only rope and yarn are supported, got {orig_cfg_dict['position_embedding_type']}")

        # Handle dtype (new format uses 'dtype', old format uses 'torch_dtype')
        # Check 'dtype' first, then fall back to 'torch_dtype'
        if "dtype" in orig_cfg_dict and orig_cfg_dict["dtype"] is not None:
            mprint(f"DEBUG: dtype found in config: {orig_cfg_dict['dtype']}")
            adapted_cfg_dict["torch_dtype"] = orig_cfg_dict["dtype"]
        elif "torch_dtype" in orig_cfg_dict and orig_cfg_dict["torch_dtype"] is not None:
            mprint(f"DEBUG: torch_dtype found in config: {orig_cfg_dict['torch_dtype']}")
            adapted_cfg_dict["torch_dtype"] = orig_cfg_dict["torch_dtype"]
        else:
            mprint(f"WARNING: neither dtype nor torch_dtype found in config (or both are None), setting to bfloat16")
            adapted_cfg_dict["torch_dtype"] = "bfloat16"

        # TODO: check how config keys such as position_embedding_type are handled (since they're not passed to the constructor)
        adapted_cfg_dict["heterogeneous_layers_config_path"] = None
        adapted_cfg_dict["block_configs"] = block_conf["block_configs"]
        adapted_cfg_dict["heterogeneous_layers_config_encoded_json"] = json.dumps(block_conf, ensure_ascii=False)
        adapted_cfg_dict["transformer_layer_spec"] = heterogeneous_layer_spec_puzzletron
        adapted_cfg_dict["vocab_size"] = orig_cfg_dict["vocab_size"]
        adapted_cfg_dict["num_layers"] = len(orig_cfg_dict["block_configs"])
        adapted_cfg_dict["hidden_size"] = orig_cfg_dict["hidden_size"]
        # adapted_cfg_dict['num_attention_heads'] = cfg["num_attention_heads"]
        adapted_cfg_dict["kv_channels"] = adapted_cfg_dict["head_dim"]
        adapted_cfg_dict["scale_factor"] = float(adapted_cfg_dict["rope_scaling"].get("factor", 8.0))
        adapted_cfg_dict["rotary_base"] = int(orig_cfg_dict.get("rope_theta", 500_000))
        adapted_cfg_dict["seq_length"] = int(orig_cfg_dict.get("max_position_embeddings", 131072))
        adapted_cfg_dict["init_method_std"] = float(orig_cfg_dict.get("initializer_range", 0.02))
        adapted_cfg_dict["layernorm_epsilon"] = float(orig_cfg_dict.get("rms_norm_eps", 1e-5))
        adapted_cfg_dict["share_embeddings_and_output_weights"] = bool(orig_cfg_dict.get("tie_word_embeddings", False))
        # adapted_cfg_dict["make_vocab_size_divisible_by"] = 128

        # Preserve HF-specific config fields that don't have NeMo equivalents
        # This enables lossless round-trip conversion HF → NeMo → HF
        source_hf_config_metadata = {}

        # eos_token_id: HF can have multiple EOS tokens [128001, 128008, 128009]
        # but NeMo tokenizer only supports single eos_id (uses the last one)
        if "eos_token_id" in orig_cfg_dict:
            source_hf_config_metadata["eos_token_id"] = orig_cfg_dict["eos_token_id"]

        # auto_map: HF-specific field for custom model class loading via trust_remote_code
        # Not relevant to NeMo but needed for HF model.from_pretrained() to work
        if "auto_map" in orig_cfg_dict:
            source_hf_config_metadata["auto_map"] = orig_cfg_dict["auto_map"]

        # dtype: HF uses 'dtype' field, NeMo uses 'torch_dtype', preserve both
        if "dtype" in orig_cfg_dict:
            source_hf_config_metadata["dtype"] = orig_cfg_dict["dtype"]

        # Store as direct config attribute (will be serialized by NeMo automatically)
        adapted_cfg_dict["source_hf_config_metadata"] = source_hf_config_metadata if source_hf_config_metadata else None

        return adapted_cfg_dict


class PuzzletronLlamaNemotronModel(GPTModel):
    """Llama-Nemotron model implementation based on the GPT model architecture.

    This class provides a high-level interface for Llama-Nemotron models,
    implementing the specific architecture and settings needed for Llama-Nemotron models.
    """

    def __init__(
        self,
        config: Annotated[Optional[PuzzletronNemotronModelConfig], Config[PuzzletronNemotronModelConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(
            config or PuzzletronNemotronModelConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform
        )


def instantiate_nemo_config_from_adapted_dict(
    adapted_cfg_dict: dict,
    generation_config: Optional["GenerationConfig"] = None,
) -> PuzzletronNemotronModelConfig:
    """
    Instantiate PuzzletronNemotronModelConfig from adapted config dict.

    This function is shared by the importer and tests to ensure consistency.

    Args:
        adapted_cfg_dict: Dict created by create_adapted_config_dict_from_puzzletron_config
        generation_config: Optional generation config to attach

    Returns:
        PuzzletronNemotronModelConfig instance
    """

    # Helper function for vocab size divisibility
    def make_vocab_size_divisible_by(vocab_size: int) -> int:
        base = 128
        while vocab_size % base != 0:
            base //= 2
        return base

    # Keys used for PuzzletronNemotronModelConfig instantiation
    INSTANTIATION_KEYS = {
        "heterogeneous_layers_config_encoded_json",
        "transformer_layer_spec",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "kv_channels",
        "scale_factor",
        "init_method_std",
        "layernorm_epsilon",
        "seq_length",
        "rotary_base",
        "vocab_size",
        "share_embeddings_and_output_weights",
        "source_hf_config_metadata",
    }

    # Keys that are metadata or derived (not directly passed to constructor)
    metadata_keys = set(adapted_cfg_dict.keys()) - INSTANTIATION_KEYS

    mprint(f"DEBUG: Keys used for instantiation: {sorted(INSTANTIATION_KEYS)}")
    mprint(f"DEBUG: Metadata keys (not used for direct instantiation): {sorted(metadata_keys)}")
    for key in sorted(metadata_keys):
        value = adapted_cfg_dict[key]
        if isinstance(value, (list, dict)):
            mprint(f"  - {key}: {type(value).__name__} with {len(value)} items")
        elif callable(value):
            mprint(f"  - {key}: {value.__name__ if hasattr(value, '__name__') else 'callable'}")
        else:
            mprint(f"  - {key}: {value}")

    model_dtype = dtype_from_dict(adapted_cfg_dict)

    # Determine the unique activation_func from all blocks
    # MCore validates the global activation_func, so we need to set it to match all blocks
    heterogeneous_config = json.loads(adapted_cfg_dict["heterogeneous_layers_config_encoded_json"])
    block_list = heterogeneous_config.get("block_configs", [])

    # Assert that block_configs exists and is not empty
    assert block_list, (
        "No block_configs found in heterogeneous_layers_config_encoded_json. "
        "The JSON structure must contain a 'block_configs' list with at least one block."
    )

    activation_funcs = []

    for i, block in enumerate(block_list):
        # Extract hidden_act from MLP config (if present)
        if "mlp" in block and "hidden_act" in block["mlp"]:
            hidden_act_str = block["mlp"]["hidden_act"]

            # Track None/null values (used for no-op blocks)
            if hidden_act_str is None:
                activation_funcs.append(None)
                continue

            # For now, only support silu and gelu activations
            # See: https://github.com/NVIDIA/Megatron-LM/blob/268fda08592528b7bc1a21aadaed259980ca8efb/megatron/core/transformer/transformer_config.py#L1043-L1048
            if hidden_act_str == "silu":
                activation_funcs.append(F.silu)
            elif hidden_act_str == "gelu":
                activation_funcs.append(F.gelu)
            else:
                raise NotImplementedError(
                    f"Unsupported activation function: '{hidden_act_str}' in block {i}. "
                    f"Only 'silu', 'gelu', and None/null are currently supported. "
                    f"MCore's bias_activation_fusion only validates these activation functions."
                )
        # If no hidden_act key or no MLP, we treat it as None
        else:
            activation_funcs.append(None)

    # Separate None and not-None activations
    not_none_activations = [f for f in activation_funcs if f is not None]

    # Check that all not-None activation functions are the same
    unique_not_none = {id(f) for f in not_none_activations}

    if len(unique_not_none) == 0:
        # No activation functions found (all blocks are no-op or have None)
        # Default to F.silu to pass MCore validation
        global_activation_func = F.silu
        mprint("WARNING: No not-None activation functions found in blocks, defaulting global activation_func to F.silu")
    elif len(unique_not_none) == 1:
        # All not-None blocks use the same activation function (safe)
        global_activation_func = not_none_activations[0]
        func_name = (
            global_activation_func.__name__
            if hasattr(global_activation_func, "__name__")
            else str(global_activation_func)
        )
        none_count = activation_funcs.count(None)
        total_count = len(activation_funcs)
        mprint(
            f"INFO: All {total_count - none_count} not-None blocks use the same activation function: {func_name} ({none_count} None/no-op blocks)"
        )
    else:
        # Multiple different not-None activation functions found (currently not supported/tested)
        func_names = [f.__name__ if hasattr(f, "__name__") else "None" for f in activation_funcs]
        unique_func_names = set(f.__name__ for f in not_none_activations)
        assert False, (
            f"Puzzletron blocks must all use the same activation function (None allowed for no-op blocks). "
            f"Found {len(unique_not_none)} different not-None activation functions across blocks: {unique_func_names}. "
            f"Block activations: {func_names}. "
            f"MCore's validation only checks the global activation_func, which would not match heterogeneous activations. "
            f"Either make all blocks use the same activation, or update MCore to support per-block validation."
        )

    return PuzzletronNemotronModelConfig(
        heterogeneous_layers_config_encoded_json=adapted_cfg_dict["heterogeneous_layers_config_encoded_json"],
        heterogeneous_layers_config_path=None,  # We directly load the block config as json
        transformer_layer_spec=adapted_cfg_dict["transformer_layer_spec"],
        activation_func=global_activation_func,  # Set to match all blocks
        num_layers=adapted_cfg_dict["num_layers"],
        hidden_size=adapted_cfg_dict["hidden_size"],
        num_attention_heads=adapted_cfg_dict["num_attention_heads"],
        kv_channels=adapted_cfg_dict["kv_channels"],
        scale_factor=adapted_cfg_dict["scale_factor"],
        init_method_std=adapted_cfg_dict["init_method_std"],
        layernorm_epsilon=adapted_cfg_dict["layernorm_epsilon"],
        seq_length=adapted_cfg_dict["seq_length"],
        rotary_base=adapted_cfg_dict["rotary_base"],
        make_vocab_size_divisible_by=make_vocab_size_divisible_by(adapted_cfg_dict["vocab_size"]),
        vocab_size=adapted_cfg_dict["vocab_size"],
        share_embeddings_and_output_weights=adapted_cfg_dict["share_embeddings_and_output_weights"],
        # HF-specific metadata for lossless round-trip conversion
        source_hf_config_metadata=adapted_cfg_dict.get("source_hf_config_metadata"),
        fp16=(model_dtype == torch.float16),  # type: ignore
        bf16=(model_dtype == torch.bfloat16),  # type: ignore
        params_dtype=model_dtype,  # type: ignore
        generation_config=generation_config,  # type: ignore
    )


@io.model_importer(PuzzletronLlamaNemotronModel, "hf")
class PuzzletronHFLlamaNemotronImporter(io.ModelConnector["LlamaForCausalLM", PuzzletronLlamaNemotronModel]):
    """Importer for converting Hugging Face Llama-Nemotron models to NeMo format.

    This class handles the conversion of Hugging Face's LlamaForCausalLM models
    to NeMo's PuzzletronLlamaNemotronModel format, including weight mapping and configuration translation.
    """

    # Base mapping using standard LLaMA weight names
    # Layernorm wildcards are replaced with per-layer mappings in convert_state()
    # TODO: MoE and Mamba layer conversions have not been tested yet
    default_mapping = {
        "model.embed_tokens.weight": "embedding.word_embeddings.weight",
        "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
        "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
        "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
        "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
        "model.norm.weight": "decoder.final_layernorm.weight",
        "lm_head.weight": "output_layer.weight",
    }

    def init(self) -> PuzzletronLlamaNemotronModel:
        """Initialize a NeMo LlamaModel instance.

        Returns:
            LlamaModel: Initialized NeMo Llama model with the appropriate configuration
                        and tokenizer.
        """
        config = self.config
        mprint(f"DEBUG: NeMo config dtype settings:")
        mprint(f"  - config.bf16: {config.bf16}")
        mprint(f"  - config.fp16: {config.fp16}")
        mprint(f"  - config.params_dtype: {config.params_dtype}")
        return PuzzletronLlamaNemotronModel(config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import AutoModelForCausalLM

        logging.info(f"Load Puzzletron HF model {str(self)}")
        source = AutoModelForCausalLM.from_pretrained(str(self), trust_remote_code=True, torch_dtype="auto")
        logging.info("Initialize NeMo Puzzletron Llama Nemotron model")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        mprint(f"Converted Llama-Nemotron model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source: Any, target: Any) -> Any:
        """Convert state dict from HF format to NeMo format.

        Maps the weights from the HF model to the NeMo model according to
        the appropriate mapping scheme.

        Args:
            source: Source HF model
            target: Target NeMo model

        Returns:
            The result of applying the transforms
        """
        mapping = self.default_mapping.copy()

        if target.config.normalization == "LayerNorm":
            mapping["model.norm.bias"] = "decoder.final_layernorm.bias"
        if getattr(source.config, "tie_word_embeddings", False):
            del mapping["lm_head.weight"]

        # Puzzletron models must have block_configs for heterogeneous layer support
        assert hasattr(source.config, "block_configs"), "Puzzletron models must have block_configs"

        # Build per-layer specific mappings for heterogeneous support
        attn_mapping, ffn_mapping, mamba_mapping, moe_mapping, transform_specs = (
            _build_puzzletron_mappings_and_transforms(source.config)
        )

        # Remove layernorm wildcards from default_mapping - these will be replaced with
        # specific per-layer mappings based on each layer's architecture.
        for pattern in ["model.layers.*.input_layernorm.weight", "model.layers.*.post_attention_layernorm.weight"]:
            if pattern in mapping:
                del mapping[pattern]

        # Add all layer-specific mappings
        mapping.update(**attn_mapping)
        mapping.update(**ffn_mapping)
        mapping.update(**mamba_mapping)
        mapping.update(**moe_mapping)

        # Create transforms from specification
        transforms = []

        # Helper to create merge_qkv closure with proper layer index capture
        def make_merge_qkv_fn(layer_idx):
            def merge_qkv_fn(ctx, q, k, v):
                return merge_qkv_for_puzzletron(ctx, q, k, v, idx=layer_idx)

            return merge_qkv_fn

        for spec in transform_specs:
            if spec["transform_function"] == "merge_qkv_for_puzzletron":
                # Fixed: proper closure to avoid variable capture issues
                layer_idx = spec["kwargs"]["idx"]
                transforms.append(
                    io.state_transform(
                        source_key=spec["source_key"],
                        target_key=spec["target_key"],
                        fn=make_merge_qkv_fn(layer_idx),
                    )
                )
            elif spec["transform_function"] == "merge_fc1_for_moe":
                transforms.append(
                    io.state_transform(
                        source_key=spec["source_key"],
                        target_key=spec["target_key"],
                        fn=TransformFns.merge_fc1,
                    )
                )

        # Add standard FC1 merge transform
        transforms.append(
            io.state_transform(
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            )
        )
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Get the tokenizer for the HF model.

        Returns:
            AutoTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import (
            AutoTokenizer,
        )

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), trust_remote_code=True)

    @property
    def config(self) -> PuzzletronNemotronModelConfig:
        """Create a NeMo LlamaNemotronConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            PuzzletronNemotronModelConfig: Puzzletron NeMo configuration for Llama models
        """
        from transformers import AutoConfig, GenerationConfig

        source = AutoConfig.from_pretrained(str(self), trust_remote_code=True)

        # Validate that this is a proper Puzzletron-Nemotron checkpoint
        assert getattr(source, "rope_scaling", None), "Llama-Nemotron model should have rope scaling"
        assert (
            getattr(source, "block_configs", None) is not None
        ), "Puzzletron-Nemotron model should be heterogeneous and have block configs"

        adapted_cfg_dict = PuzzletronNemotronModelConfig.create_adapted_config_dict_from_puzzletron_config(
            source
        )  # type: ignore

        try:
            generation_config = GenerationConfig.from_pretrained(str(self))
        except Exception:
            generation_config = None

        output = instantiate_nemo_config_from_adapted_dict(adapted_cfg_dict, generation_config=generation_config)
        return output


@io.model_exporter(PuzzletronLlamaNemotronModel, "hf")
class PuzzletronHFLlamaNemotronExporter(io.ModelConnector[PuzzletronLlamaNemotronModel, "LlamaForCausalLM"]):
    """Exporter for converting NeMo Puzzletron Llama-Nemotron models to Hugging Face format.

    This class handles the conversion of NeMo's PuzzletronLlamaNemotronModel to Hugging Face's
    LlamaForCausalLM format, including weight mapping and configuration translation.
    It supports heterogeneous model architectures with Puzzletron-specific configurations.

    The exporter performs the following key operations:
    1. Initializes a Hugging Face model with appropriate configuration
    2. Maps weights from NeMo format to Hugging Face format
    3. Handles special cases for heterogeneous architectures with Mamba, MoE, and other custom layers
    4. Saves the converted model and tokenizer to the specified output path

    Attributes:
        tokenizer: The tokenizer associated with the NeMo model
        config: The configuration for the Hugging Face model

    Methods:
        init: Initialize a Hugging Face model instance
        apply: Convert and save the model to Hugging Face format
        convert_state: Convert model weights from NeMo to Hugging Face format
    """

    # Base mapping for NeMo -> HF conversion (reversed from importer)
    # Layernorm wildcards are replaced with per-layer mappings in convert_state()
    default_mapping = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    @property
    def config(self) -> "DeciLMConfig":
        """Create a HF DeciLMConfig from the NeMo model config.

        This method constructs a DeciLMConfig for Puzzletron models by parsing the
        heterogeneous_layers_config_encoded_json from the NeMo config and mapping
        the fields to the HF DeciLM format.

        Returns:
            DeciLMConfig: HF configuration for Puzzletron DeciLM models
        """
        # Load the NeMo config
        source_config = io.load_context(str(self), subpath="model.config")

        # Get preserved HF config metadata (stored as direct attribute)
        # This enables lossless round-trip conversion HF → NeMo → HF
        source_hf_config_metadata = getattr(source_config, "source_hf_config_metadata", None) or {}

        # Get EOS token ID(s) - prefer preserved value from source HF config metadata
        # (HF supports multiple EOS tokens, NeMo tokenizer only has single eos_id)
        eos_token_id = source_hf_config_metadata.get("eos_token_id", self.tokenizer.eos_id)

        # Use the shared conversion function
        return convert_nemo_config_to_hf_decilm_config(
            nemo_config=source_config,
            vocab_size=self.tokenizer.vocab_size,
            eos_token_id=eos_token_id,
            bos_token_id=self.tokenizer.bos_id,
            pad_token_id=getattr(self.tokenizer, "pad_id", None),
        )

    def init(self, dtype=torch.bfloat16, from_config=False, model_name=None) -> "LlamaForCausalLM":
        """Initialize a Hugging Face LlamaForCausalLM model instance.

        This method creates a new Hugging Face model instance with the appropriate configuration
        and data type. Puzzletron models always use from_config=True and create a DeciLMForCausalLM.

        Args:
            dtype (torch.dtype, optional): Data type for model parameters. Defaults to torch.bfloat16.
            from_config (bool, optional): Whether to initialize from config or load from pretrained.
                For Puzzletron models, this should always be True. Defaults to False.
            model_name (str, optional): Name of the pretrained model to load. Not used for Puzzletron
                models since we generate the config dynamically. Defaults to None.

        Returns:
            DeciLMForCausalLM: Initialized Hugging Face DeciLM model instance

        Raises:
            ValueError: If model_name is provided (not supported for Puzzletron models)
        """
        from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import DeciLMForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            if from_config:
                # Puzzletron models: create DeciLMForCausalLM from self.config property
                model = DeciLMForCausalLM(self.config)
                model = model.to(dtype=dtype)
                return model
            else:
                # Puzzletron models don't support loading from pretrained HF model cards
                raise ValueError(
                    "Puzzletron models do not have official HF model cards. "
                    "Use from_config=True to create models from NeMo config."
                )

    def apply(self, output_path: Path, target_model_name=None) -> Path:
        """Convert and save a NeMo Puzzletron Llama-Nemotron model to Hugging Face format.

        This method performs the complete conversion process:
        1. Loads the NeMo model checkpoint
        2. Creates the Hugging Face model from config
        3. Converts and transfers the weights
        4. Saves the converted model and tokenizer

        Args:
            output_path (Path): Directory path where the converted model will be saved
            target_model_name (str, optional): Not used for Puzzletron models. Kept for API compatibility.

        Returns:
            Path: Path to the saved Hugging Face model directory
        """
        logging.info("Loading Puzzletron Llama-Nemotron NeMo checkpoint..")
        source, _ = self.nemo_load(str(self))

        # Puzzletron models always use from_config=True to generate DeciLMConfig dynamically
        target = self.init(
            torch_dtype_from_mcore_config(source.config),
            from_config=True,
            model_name=None,
        )
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.tokenizer.save_pretrained(output_path)

        # Copy custom Python files needed for Puzzletron models
        from modelopt.torch.puzzletron.export.MCore.llama_nemotron_utils import (
            copy_puzzletron_python_files_to_decilm_checkpoint,
            embed_chat_template_in_tokenizer_config,
        )

        copy_puzzletron_python_files_to_decilm_checkpoint(output_path)

        # Fix tokenizer: embed chat_template from .jinja file into tokenizer_config.json
        # NeMo's HF → NeMo import extracts chat_template to .jinja but doesn't preserve
        # it in tokenizer_config.json. We restore it here for accuracy parity.
        embed_chat_template_in_tokenizer_config(str(self), output_path)

        return output_path

    def convert_state(self, source: Any, target: Any) -> Any:
        """Convert state dict from NeMo format to HF format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme for Puzzletron models.

        This method follows the same pattern as the importer but with reversed mappings:
        1. Start with default mapping
        2. Remove layernorm wildcards (will be replaced with per-layer mappings)
        3. Build per-layer specific mappings using helper function and reverse them
        4. Create transforms for weight conversions

        Args:
            source: Source NeMo model
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        mapping = self.default_mapping.copy()

        # Handle LayerNorm bias if present
        if source.config.normalization == "LayerNorm":
            mapping["decoder.final_layernorm.bias"] = "model.norm.bias"

        # Handle tied embeddings
        if getattr(source.config, "share_embeddings_and_output_weights", False):
            # Remove output_layer mapping if embeddings are tied
            if "output_layer.weight" in mapping:
                del mapping["output_layer.weight"]

        # Build per-layer specific mappings for heterogeneous support
        attn_mapping, ffn_mapping, mamba_mapping, moe_mapping, transform_specs = (
            _build_puzzletron_mappings_and_transforms(source.config)
        )

        # Remove layernorm wildcards from default_mapping - these will be replaced with
        # specific per-layer mappings based on each layer's architecture.
        for pattern in [
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
        ]:
            if pattern in mapping:
                del mapping[pattern]

        # For exporter: reverse all mappings (HF -> NeMo becomes NeMo -> HF)
        attn_mapping = {v: k for k, v in attn_mapping.items()}
        ffn_mapping = {v: k for k, v in ffn_mapping.items()}
        mamba_mapping = {v: k for k, v in mamba_mapping.items()}
        moe_mapping = {v: k for k, v in moe_mapping.items()}

        # Add all layer-specific mappings
        mapping.update(**attn_mapping)
        mapping.update(**ffn_mapping)
        mapping.update(**mamba_mapping)
        mapping.update(**moe_mapping)

        # Create transforms from specifications (reversed for exporter)
        transforms = []

        # Helper to create split_qkv closure with proper layer index capture
        def make_split_qkv_fn(layer_idx):
            def split_qkv_fn(ctx, qkv):
                return split_qkv_for_puzzletron(ctx, qkv, idx=layer_idx)

            return split_qkv_fn

        for spec in transform_specs:
            if spec["transform_function"] == "merge_qkv_for_puzzletron":
                # For exporter: split QKV (NeMo -> HF)
                layer_idx = spec["kwargs"]["idx"]
                transforms.append(
                    io.state_transform(
                        source_key=spec["target_key"],  # NeMo key
                        target_key=spec["source_key"],  # HF key
                        fn=make_split_qkv_fn(layer_idx),
                    )
                )
            elif spec["transform_function"] == "merge_fc1_for_moe":
                # For exporter: split FC1 for MoE (NeMo -> HF)
                transforms.append(
                    io.state_transform(
                        source_key=spec["target_key"],  # NeMo key
                        target_key=spec["source_key"],  # HF key
                        fn=TransformFns.split_fc1,
                    )
                )

        # Add standard transforms for FC1 splitting and padding pruning
        transforms.extend(
            [
                io.state_transform(
                    source_key="decoder.layers.*.mlp.linear_fc1.weight",
                    target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                    fn=TransformFns.split_fc1,
                ),
                io.state_transform(
                    source_key="embedding.word_embeddings.weight",
                    target_key="model.embed_tokens.weight",
                    fn=TransformFns.prune_padding,
                ),
                io.state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                ),
            ]
        )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return io.load_context(str(self), subpath="model").tokenizer


__all__ = [
    "PuzzletronLlamaNemotronModel",
]
