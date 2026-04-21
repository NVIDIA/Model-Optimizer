# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import abc
import contextlib
import fnmatch
import itertools
import logging
import re
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_pg_size,
    unwrap_model,
)
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from transformers.modeling_utils import PreTrainedModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    MegatronParamMapping,
)
from megatron.bridge.models.conversion.peft_bridge import AdapterWeightConversionTask, MegatronPeftBridge
from megatron.bridge.models.conversion.transformers_compat import (
    rope_theta_from_hf,
)
from megatron.bridge.models.conversion.utils import (
    extract_sort_key,
    get_module_and_param_from_name,
    persistent_buffers,
)
from megatron.bridge.models.decorators.dispatch import dispatch
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)

MappingT = TypeVar("MappingT", bound=MegatronParamMapping)
HFPreTrained = TypeVar("HFPreTrained")
ModelProviderTarget = TypeVar("ModelProviderTarget", bound=ModelProviderMixin)
MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
_BridgeImplClass = TypeVar("_BridgeImplClass", bound="MegatronModelBridge")


class MegatronWeightTuple(NamedTuple):
    """Tuple representing a Megatron model weight with its metadata."""

    param_name: str
    weight: torch.Tensor
    vp_stage: int


class HFWeightTuple(NamedTuple):
    """Tuple representing a HuggingFace model weight with its metadata."""

    param_name: str
    weight: torch.Tensor


@dataclass(frozen=True)
class WeightConversionTask(Generic[MappingT]):
    """A unified task for converting weights between HuggingFace and Megatron formats.

    This class combines both HF->Megatron and Megatron->HF conversion tasks since they
    have different method names (hf_to_megatron vs megatron_to_hf) and can coexist safely.

    The task encapsulates all information needed for weight conversion in either direction,
    with different fields being relevant depending on the conversion type.

    Attributes:
        param_name (str): *unwrapped, local* parameter name (no ``module.`` prefixes).
        global_param_name (str): *unwrapped, global* parameter name (no ``module.`` prefixes).
        mapping (MappingT): Concrete :pyclass:`MegatronParamMapping` instance responsible
            for weight transformation and distribution.

        pp_rank (Optional[int]): Pipeline-parallel rank that owns the parameter (required for saves).
        vp_stage (Optional[int]): Virtual-pipeline stage index (required for loads).
        megatron_module (Optional[torch.nn.Module]): Reference to the Megatron model or
            sub-module that owns the parameter (required for loads).
        param_weight (Optional[torch.Tensor]): The actual parameter tensor that will
            receive the converted weight (required for loads).

    """

    param_name: str
    global_param_name: str
    mapping: MappingT
    pp_rank: Optional[int] = None
    vp_stage: Optional[int] = None
    megatron_module: Optional[torch.nn.Module] = None
    param_weight: Optional[torch.Tensor] = None


def _megatron_local_name_to_global(
    models: MegatronModule | List[MegatronModule],
    config: TransformerConfig,
    param_name: str,
    vp_stage: Optional[int] = None,
) -> str:
    """Adjust layer number and expert number from local to global numbering."""
    layer_module = None
    # PP
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    if "layers." in param_name :
        match = re.match(r"^(.+?\.layers\.\d+)", param_name)
        assert match is not None
        layer_prefix = match.group(1)
        _, layer_module = get_module_and_param_from_name(models=models, param_name=layer_prefix, vp_stage=vp_stage)

        if get_pg_size(pp_group) > 1:
            local_layer_number = int(param_name.split("layers.")[1].split(".")[0])
            if isinstance(layer_module, MegatronModule):
                global_layer_number = layer_module.layer_number - 1
                param_name = param_name.replace(
                    f"layers.{local_layer_number}.",
                    f"layers.{global_layer_number}.",
                )

    # EP
    ep_group = parallel_state.get_expert_model_parallel_group()
    # For now adapters are not sharded across EP ranks
    if ".mlp.experts.linear_fc" in param_name and get_pg_size(ep_group) > 1 and not ".adapter." in param_name:
        # num_experts = config.num_moe_experts
        # num_experts_per_rank = num_experts // ep_group.size()
        assert layer_module is not None, "layer_module is not found. MoE should be attached to layer."
        num_experts_per_rank = layer_module.mlp.num_local_experts

        def _update_expert_number(param_name: str, param_type: str) -> str:
            """Update expert number from local to global for weight or bias parameters."""
            local_expert_number = int(param_name.split(f".{param_type}")[-1])
            global_expert_number = num_experts_per_rank * ep_group.rank() + local_expert_number
            return param_name.replace(
                f".{param_type}{local_expert_number}",
                f".{param_type}{global_expert_number}",
            )

        # Handle weight and bias parameters
        if ".weight" in param_name:
            param_name = _update_expert_number(param_name, "weight")
        elif ".bias" in param_name:
            param_name = _update_expert_number(param_name, "bias")
    return param_name


class MegatronModelBridge(MegatronPeftBridge, Generic[HFPreTrained, ModelProviderTarget, MegatronModel]):
    """
    High-level orchestrator for HuggingFace ↔ Megatron model conversions.

    This abstract base class provides the framework for converting models between
    HuggingFace and Megatron formats. It acts as an orchestrator that coordinates
    the conversion process without directly handling the complex details of
    tensor parallelism or weight transformations.

    The bridge pattern separates concerns:
    - MegatronModelBridge: Orchestrates the overall conversion process
    - MegatronMappingRegistry: Manages parameter name mappings
    - MegatronParamMapping: Handles actual weight transformations and distribution

    Key responsibilities:
    1. Build conversion tasks that map each parameter to its appropriate bridge
    2. Execute tasks with proper error handling and progress tracking
    3. Provide utilities for configuration translation
    4. Handle virtual pipeline parallelism (VP) complexities

    To implement a bridge for a new model architecture:

    1. Create a subclass decorated with @MegatronModelBridge.register_bridge:

        .. code-block:: python

            @MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
            class MegatronCausalLlamaBridge(MegatronModelBridge):
                pass

    2. Implement provider_bridge to create Megatron configurations:

        .. code-block:: python

            def provider_bridge(self, hf_pretrained) -> LlamaModelProvider:
                return LlamaModelProvider(
                    num_layers=hf_pretrained.config.num_hidden_layers,
                    hidden_size=hf_pretrained.config.hidden_size,
                    ...
                )

    3. Implement mapping_registry to define weight mappings:

        .. code-block:: python

            def mapping_registry(self) -> MegatronMappingRegistry:
                return MegatronMappingRegistry(
                    AutoMapping(
                        megatron_param="embedding.word_embeddings.weight",
                        hf_param="model.embed_tokens.weight"
                    ),
                    ...
                )

    Example:
        .. code-block:: python

            # The bridge is typically not instantiated directly
            # Instead, use AutoBridge or AutoBridge which handle this
            bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
            provider = bridge.to_megatron_provider()

    Note:
        This class uses generic type parameters to ensure type safety:
        - HFPreTrained: The HuggingFace model type
        - ModelProviderTarget: The Megatron model provider type
        - MegatronModel: The Megatron model type
    """

    # Provider class to instantiate in provider_bridge (set via @register_bridge decorator)
    # For MLA models, use DeepSeekModelProvider or similar; for standard GPT, use GPTModelProvider
    PROVIDER_CLASS = None  # Set by @register_bridge(provider=...) or defaults to GPTModelProvider

    # Additional file patterns to automatically copy during HF export (e.g., ["*reasoning_parser.py"])
    # Set this in bridge subclasses to include model-specific files beyond standard artifacts
    ADDITIONAL_FILE_PATTERNS = None

    # Common bidirectional config field name mapping: (hf_name, megatron_name)
    # Some mappings may not be used by all models - that's fine, unused fields are skipped
    CONFIG_MAPPING = [
        # Core architecture
        ("num_hidden_layers", "num_layers"),
        ("hidden_size", "hidden_size"),
        ("intermediate_size", "ffn_hidden_size"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_query_groups"),
        ("head_dim", "kv_channels"),
        ("vocab_size", "vocab_size"),
        ("max_position_embeddings", "seq_length"),
        ("rms_norm_eps", "layernorm_epsilon"),
        ("initializer_range", "init_method_std"),
        # Attention and dropout
        ("attention_dropout", "attention_dropout"),
        ("hidden_dropout", "hidden_dropout"),
        ("tie_word_embeddings", "share_embeddings_and_output_weights"),
        ("attention_bias", "add_qkv_bias"),
        ("mlp_bias", "add_bias_linear"),
        ("use_qk_norm", "qk_layernorm"),
        # RoPE
        ("rope_theta", "rotary_base"),
        ("partial_rotary_factor", "rotary_percent"),
        # MoE
        ("num_experts", "num_moe_experts"),
        ("num_local_experts", "num_moe_experts"),
        ("num_experts_per_tok", "moe_router_topk"),
        ("moe_intermediate_size", "moe_ffn_hidden_size"),
        ("aux_loss_alpha", "moe_aux_loss_coeff"),
        ("scoring_func", "moe_router_score_function"),
        ("n_routed_experts", "num_moe_experts"),
        ("n_group", "moe_router_num_groups"),
        ("topk_group", "moe_router_group_topk"),
        ("routed_scaling_factor", "moe_router_topk_scaling_factor"),
        # MLA
        ("q_lora_rank", "q_lora_rank"),
        ("kv_lora_rank", "kv_lora_rank"),
        ("qk_nope_head_dim", "qk_head_dim"),
        ("qk_rope_head_dim", "qk_pos_emb_head_dim"),
        ("v_head_dim", "v_head_dim"),
        # MTP
        ("num_nextn_predict_layers", "mtp_num_layers"),
    ]

    # YARN rope scaling field mapping for GPT models: (hf_rope_scaling_key, megatron_yarn_param)
    # These are only applied when rope_scaling.type == "yarn" and provider is GPTModelProvider
    # Uses yarn_ prefix (e.g., yarn_mscale, yarn_rotary_scaling_factor)
    YARN_ROPE_SCALING_MAPPING = [
        ("factor", "yarn_rotary_scaling_factor"),
        ("original_max_position_embeddings", "yarn_original_max_position_embeddings"),
        ("beta_fast", "yarn_beta_fast"),
        ("beta_slow", "yarn_beta_slow"),
        ("mscale", "yarn_mscale"),
        ("mscale_all_dim", "yarn_mscale_all_dim"),
    ]

    # MLA rope scaling field mapping: (hf_rope_scaling_key, megatron_mla_param)
    # These are applied for MLA models (DeepSeek, Kimi, etc.) which use MLATransformerConfig
    # Uses direct field names without yarn_ prefix (e.g., mscale, rotary_scaling_factor)
    MLA_ROPE_SCALING_MAPPING = [
        ("factor", "rotary_scaling_factor"),
        ("original_max_position_embeddings", "original_max_position_embeddings"),
        ("beta_fast", "beta_fast"),
        ("beta_slow", "beta_slow"),
        ("mscale", "mscale"),
        ("mscale_all_dim", "mscale_all_dim"),
    ]

    # Common bidirectional activation function mapping: hf_name <-> megatron_func
    ACTIVATION_MAPPING = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
        "relu2": squared_relu,
        "tanh": torch.tanh,
        "gelu_pytorch_tanh": fast_gelu,
    }

    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        """Convert HF activation name string to Megatron activation function."""
        if hidden_act not in cls.ACTIVATION_MAPPING:
            raise ValueError(
                f"Unsupported activation function: {hidden_act}. Supported: {list(cls.ACTIVATION_MAPPING.keys())}"
            )
        return cls.ACTIVATION_MAPPING[hidden_act]

    @classmethod
    def megatron_to_hf_activation(cls, activation_func) -> str:
        """Convert Megatron activation function to HF activation name string."""
        for hf_name, megatron_func in cls.ACTIVATION_MAPPING.items():
            if activation_func is megatron_func:
                return hf_name
        raise ValueError(
            f"Unsupported activation function: {activation_func}. Supported: {list(cls.ACTIVATION_MAPPING.values())}"
        )

    def hf_config_to_provider_kwargs(self, hf_config) -> dict:
        """Convert HF config to Megatron provider kwargs using CONFIG_MAPPING.

        Args:
            hf_config: HuggingFace model configuration object

        Returns:
            dict: Provider kwargs ready for GPTModelProvider or similar
        """
        provider_kwargs = {}

        # Map config fields using CONFIG_MAPPING
        # Supports dot notation for nested dict access (e.g., "rope_scaling.factor")
        for hf_name, megatron_name in self.CONFIG_MAPPING:
            has_value = False
            value = None
            if "." in hf_name:
                # Nested dict access: "parent.child" -> getattr(config, parent).get(child)
                parts = hf_name.split(".", 1)
                parent = getattr(hf_config, parts[0], None)
                if parent is not None and isinstance(parent, dict):
                    if parts[1] in parent:
                        value = parent[parts[1]]
                        has_value = True
            else:
                value = getattr(hf_config, hf_name, None)
                has_value = hasattr(hf_config, hf_name)
            if has_value:
                provider_kwargs[megatron_name] = value

        # Extract rotary_base via compat function (handles both legacy rope_theta
        # attribute and transformers 5.0+ rope_parameters dict)
        if "rotary_base" not in provider_kwargs:
            try:
                provider_kwargs["rotary_base"] = rope_theta_from_hf(hf_config)
            except ValueError:
                pass

        # Handle rope scaling: extract params from rope_scaling dict
        # HF configs use either "type" or "rope_type" key for the scaling type
        from megatron.bridge.models.mla_provider import MLAModelProvider

        is_mla_provider = self.PROVIDER_CLASS is not None and issubclass(self.PROVIDER_CLASS, MLAModelProvider)
        rope_scaling = getattr(hf_config, "rope_scaling", None)

        if rope_scaling is not None and isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
            if rope_type == "yarn":
                # Check if this is an MLA provider (uses direct field names)
                # or a GPT provider (uses yarn_ prefixed field names)
                if is_mla_provider:
                    # MLA models: use direct field names (mscale, rotary_scaling_factor, etc.)
                    mla_params = {}
                    for hf_key, megatron_key in self.MLA_ROPE_SCALING_MAPPING:
                        value = rope_scaling.get(hf_key)
                        if value is not None:
                            mla_params[megatron_key] = value
                    if mla_params:
                        provider_kwargs["_mla_rope_params"] = mla_params
                else:
                    # GPT models: use yarn_ prefixed field names
                    yarn_params = {"position_embedding_type": "yarn"}
                    for hf_key, megatron_key in self.YARN_ROPE_SCALING_MAPPING:
                        yarn_params[megatron_key] = rope_scaling.get(hf_key)
                    if "truncate" in rope_scaling:
                        yarn_params["yarn_correction_range_round_to_int"] = rope_scaling["truncate"]
                    if yarn_params:
                        provider_kwargs["_yarn_params"] = yarn_params
        elif is_mla_provider:
            # MLA provider without rope_scaling in HF config:
            # Override rotary_scaling_factor to 1.0 (no scaling) instead of
            # using MLATransformerConfig default of 40
            provider_kwargs["_mla_rope_params"] = {"rotary_scaling_factor": 1.0, "mscale_all_dim": 1.0}

        # Handle vocab_size_divisible_by
        vocab_size = provider_kwargs.get("vocab_size")
        if vocab_size is not None:
            provider_kwargs["make_vocab_size_divisible_by"] = self.make_vocab_size_divisible_by(vocab_size)

        # Determine dtype
        params_dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        provider_kwargs["fp16"] = params_dtype == torch.float16
        provider_kwargs["bf16"] = params_dtype == torch.bfloat16
        provider_kwargs["params_dtype"] = params_dtype

        # Convert activation function (some models use hidden_act, others use hidden_activation)
        hidden_act = getattr(hf_config, "hidden_act", None) or getattr(hf_config, "hidden_activation", "silu")
        provider_kwargs["activation_func"] = self.hf_to_megatron_activation(hidden_act)

        return provider_kwargs

    # Set by @register_bridge decorator
    SOURCE_NAME: str | None = None
    MODEL_TYPE: str | None = None

    def provider_bridge(self, hf_pretrained: HFPreTrained) -> ModelProviderTarget:
        """Create a Megatron model provider from HuggingFace configuration.

        Default implementation that:
        1. Converts HF config to provider kwargs using CONFIG_MAPPING
        2. Creates and returns a GPTModelProvider

        Subclasses should override this to add model-specific configuration
        by calling super().provider_bridge() then setting properties directly
        on the returned provider (e.g., provider.normalization = "RMSNorm").

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or configuration
                containing the source model's architecture details.

        Returns:
            ModelProviderTarget: A configured model provider instance
        """
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        hf_config = hf_pretrained.config

        # Build base provider kwargs using CONFIG_MAPPING
        provider_kwargs = self.hf_config_to_provider_kwargs(hf_config)

        yarn_params = provider_kwargs.pop("_yarn_params", None)
        mla_rope_params = provider_kwargs.pop("_mla_rope_params", None)

        # Use specified provider class, defaulting to GPTModelProvider
        provider_class = self.PROVIDER_CLASS if self.PROVIDER_CLASS is not None else GPTModelProvider
        provider = provider_class(**provider_kwargs)

        # Apply YARN params via setattr (not all providers accept these in __init__)
        if yarn_params:
            for key, value in yarn_params.items():
                setattr(provider, key, value)

        # Apply MLA rope params via setattr (for MLA models like DeepSeek, Kimi)
        if mla_rope_params:
            for key, value in mla_rope_params.items():
                setattr(provider, key, value)

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Convert Megatron provider config to HuggingFace config dict.

        Default implementation that:
        1. Converts provider to HF config using CONFIG_MAPPING
        2. Handles YARN rope scaling parameters
        3. Converts activation function and dtype
        4. Adds architectures and model_type from decorator

        Subclasses should override this to add model-specific configuration
        by calling super().megatron_to_hf_config() then setting values directly
        on the returned dict (e.g., hf_config["rope_scaling"] = {...}).

        Args:
            provider: Megatron model provider instance

        Returns:
            dict: HuggingFace config dictionary
        """
        hf_config = {}

        # Map config fields using CONFIG_MAPPING (reverse direction)
        # Supports dot notation for nested dict building (e.g., "rope_scaling.factor")
        for hf_name, megatron_name in cls.CONFIG_MAPPING:
            has_value = hasattr(provider, megatron_name)
            value = getattr(provider, megatron_name, None)
            if has_value:
                if "." in hf_name:
                    # Nested dict: "parent.child" -> hf_config["parent"]["child"] = value
                    parts = hf_name.split(".", 1)
                    if parts[0] not in hf_config:
                        hf_config[parts[0]] = {}
                    hf_config[parts[0]][parts[1]] = value
                else:
                    hf_config[hf_name] = value

        # Handle YARN rope scaling: check if provider has yarn_* params and build rope_scaling dict
        yarn_rotary_scaling_factor = getattr(provider, "yarn_rotary_scaling_factor", None)
        if yarn_rotary_scaling_factor is not None:
            if "rope_scaling" not in hf_config:
                hf_config["rope_scaling"] = {}
            hf_config["rope_scaling"]["rope_type"] = "yarn"

            for hf_key, megatron_key in cls.YARN_ROPE_SCALING_MAPPING:
                has_value = hasattr(provider, megatron_key)
                value = getattr(provider, megatron_key, None)
                if has_value:
                    hf_config["rope_scaling"][hf_key] = value

            yarn_correction_range_round_to_int = getattr(provider, "yarn_correction_range_round_to_int", None)
            if yarn_correction_range_round_to_int is not None:
                hf_config["rope_scaling"]["truncate"] = yarn_correction_range_round_to_int

        # Convert activation function back to HF format
        activation_func = getattr(provider, "activation_func", None)
        if activation_func is not None:
            hf_config["hidden_act"] = cls.megatron_to_hf_activation(activation_func)

        # Determine torch_dtype
        if getattr(provider, "bf16", False):
            hf_config["torch_dtype"] = "bfloat16"
        elif getattr(provider, "fp16", False):
            hf_config["torch_dtype"] = "float16"
        else:
            hf_config["torch_dtype"] = "float32"

        # Add architectures and model_type from decorator
        if cls.SOURCE_NAME is not None:
            hf_config["architectures"] = [cls.SOURCE_NAME]
        if cls.MODEL_TYPE is not None:
            hf_config["model_type"] = cls.MODEL_TYPE

        return hf_config

    @abc.abstractmethod
    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define weight mappings between HuggingFace and Megatron formats.

        This abstract method must be implemented by subclasses to specify how
        parameters map between the two formats. The returned MegatronMappingRegistry
        contains all param mappings needed for the model architecture.

        Returns:
            MegatronMappingRegistry: MegatronMappingRegistry containing all weight
                mapping definitions.

        Example:
            .. code-block:: python

                def mapping_registry(self):
                    return MegatronMappingRegistry(
                        AutoMapping(
                            megatron_param="embedding.word_embeddings.weight",
                            hf_param="model.embed_tokens.weight"
                        ),
                        QKVMapping(
                            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                            q="model.layers.*.self_attn.q_proj.weight",
                            k="model.layers.*.self_attn.k_proj.weight",
                            v="model.layers.*.self_attn.v_proj.weight"
                        ),
                        # ... more param mappings
                    )
        """
        raise NotImplementedError("Subclass must implement mapping_registry method")

    def _megatron_global_param_names_all_pp_ranks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[str]:
        """Get all parameter names across all pipeline parallel ranks."""
        # Cache the result after first call
        if hasattr(self, "_cached_param_names"):
            return self._cached_param_names

        # Compute the result
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        model_config = unwrap_model(megatron_model)[0].config
        global_param_names = []

        # Ensure megatron_model is a list for consistent handling
        models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]

        for vp_stage, model in enumerate(models_list):
            # persistent buffers are part of the model's state_dict, but not the named_parameters, so we must include them here separately
            for local_param_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_param_name:
                    continue
                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = _megatron_local_name_to_global(
                    models_list, model_config, local_param_name, vp_stage
                )
                if self._is_adapter_param_name(global_param_name):
                    continue
                global_param_names.append(global_param_name)

        gathered_global_param_names = [None] * pp_group.size()
        torch.distributed.all_gather_object(gathered_global_param_names, global_param_names, group=pp_group)

        # flatten the list, sort it and remove duplicates
        # the order matters here, casually re-order will cause a hang.
        # e.g. decoder.layers.0.mlp.experts.linear_fc1.weight100
        flattened_names = list(set(sum(gathered_global_param_names, [])))

        # the order cannot be changed, this sync for all ranks for conversion
        # change this might cause a hang
        gathered_global_param_names = sorted(flattened_names, key=extract_sort_key)

        # Cache the result
        self._cached_param_names = gathered_global_param_names

        return self._cached_param_names

    def _with_progress_tracking(self, tasks, description: str, show_progress: bool = True):
        """Helper method to wrap an iterable with progress tracking.

        Args:
            tasks: Iterable of tasks to process
            description: Description for the progress bar
            show_progress: Whether to show progress (defaults to True)

        Yields:
            Items from the tasks iterable while updating progress
        """
        is_main_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        bridge_name = self.__class__.__name__

        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("({task.completed}/{task.total})"),
                TextColumn("{task.fields[bridge]}"),
                disable=not is_main_rank,
            ) as progress:
                task_id = progress.add_task(description, total=len(tasks), bridge=bridge_name)

                for task in tasks:
                    yield task
                    progress.update(task_id, advance=1)
        else:
            # not using disable above because we notice it will dump some empty progress bar,
            # even when disable is set to True
            for task in tasks:
                yield task

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load weights from HuggingFace state dict.
        This function can be overridden by subclasses to preprocess the HF weights before conversion, such as renaming
        certain parameters to avoid mapping conflicts, or dequantize the weights.

        Note that loading is done lazily before this function is called, so the weights are actually loaded in
        this function when hf_state_dict.__getitem__ is called.

        Args:
            hf_param: The parameter name or dictionary of parameter names to load.
            hf_state_dict: The HuggingFace state dictionary.

        Returns:
            The loaded weights.
        """
        if isinstance(hf_param, str):
            hf_weights = hf_state_dict[hf_param]
        else:
            hf_weights = {k: hf_state_dict[v] for k, v in hf_param.items()}
        return hf_weights

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Modify the converted weights after conversion. By default, no modification is done.
        This function can be overridden by subclasses to postprocess the converted weights, such as merging the
        weights of multiple experts or quantizing the weights.

        Args:
            task: The WeightConversionTask object.
            converted_weights_dict: The converted weights dictionary.
            hf_state_dict: The HuggingFace state dict accessor for expected-key checks.

        Returns:
            The modified weights dictionary.
        """
        return converted_weights_dict

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        allowed_mismatched_params: Optional[List[str]] = None,
    ) -> List[MegatronModel]:
        """Load HuggingFace weights into Megatron models.

        This method orchestrates the complete weight loading process from HuggingFace
        format to Megatron's distributed format. It builds a conversion task and
        executes it with proper progress tracking and error handling.

        The actual weight transformations and distribution are delegated to the
        appropriate MegatronParamMapping instances based on the state mappings.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or state source containing the
                weights to load.
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances (one per virtual pipeline stage).
            allowed_mismatched_params (Optional[List[str]]): List of parameter names or patterns
                to allow mismatch (skip instead of raise error).

        Returns:
            List[MegatronModel]: The input megatron_model as a list with loaded weights.

        Process:
        1. Build a task mapping each Megatron parameter to its source
        2. For each parameter in the task:
            - Fetch source weights from HuggingFace state
            - Apply format transformation via the param mapping
            - Distribute to appropriate TP/PP ranks
            - Copy into the Megatron parameter

        Example:
            .. code-block:: python

                hf_model = PreTrainedCausalLM.from_pretrained("gpt2")
                megatron_model = create_megatron_model()  # Single model or list
                bridge.load_weights_hf_to_megatron(hf_model, megatron_model)

        Note:
            Progress is shown only on rank 0 to avoid cluttered output in
            distributed environments.

        Raises:
            ValueError: If hf_pretrained doesn't have state attribute or if weight shapes don't match.
            AttributeError: If required HF weights are missing.
        """
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # [ModelOpt]: Hide extra parameters registered in Distillation mode
        with contextlib.ExitStack() as stack:
            if hasattr(megatron_model[0], "hide_teacher_model"):
                stack.enter_context(megatron_model[0].hide_teacher_model())
            if hasattr(megatron_model[0], "hide_loss_modules"):
                stack.enter_context(megatron_model[0].hide_loss_modules())

            hf_to_megatron_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

        description = f"Loading from {hf_pretrained.model_name_or_path}"
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            # None means megatron module not on current rank, skip if this task is not going to happen
            if task.megatron_module is None:
                continue
            # 1) Fetch source tensor(s) from HF state dict
            hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)

            # 2) Delegate conversion & distribution to the bridge
            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)

            # 3) Copy into Megatron param if this rank received a shard
            if converted_weights is not None:
                # Assert that param_weight is not None for HF->Megatron tasks
                assert task.param_weight is not None, "param_weight is required for HF->Megatron conversion"

                # Check shape compatibility before copying
                if converted_weights.shape != task.param_weight.shape:
                    # Check whitelist
                    is_whitelisted = False
                    if allowed_mismatched_params:
                        for pattern in allowed_mismatched_params:
                            if fnmatch.fnmatch(task.mapping.megatron_param, pattern) or fnmatch.fnmatch(
                                task.param_name, pattern
                            ):
                                is_whitelisted = True
                                break

                    if is_whitelisted:
                        print_rank_0(
                            f"WARNING: Shape mismatch for megatron param {task.mapping.megatron_param} allowed by whitelist. Skipping."
                        )
                        continue

                    raise ValueError(
                        f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                        f"  Expected shape: {task.param_weight.shape}\n"
                        f"  Got shape: {converted_weights.shape}\n"
                        f"  Bridge type: {type(task.mapping).__name__}\n"
                        f"  HF mapping: {task.mapping.hf_param}"
                    )
                task.param_weight.data.copy_(converted_weights)

        self._broadcast_shared_embeddings(megatron_model)
        return megatron_model

    def stream_weights_hf_to_megatron(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[MegatronWeightTuple]:
        """Generator variant of load_weights_hf_to_megatron for streaming weight conversion.

        This method provides a memory-efficient way to convert weights by yielding
        them one at a time instead of loading all at once. Useful for processing
        very large models or when implementing custom weight handling logic.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or state source containing
                the weights.
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances to extract configuration from.
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.

        Yields:
            MegatronWeightTuple: Named tuples containing:
                - vp_stage: Index of the model in megatron_model list
                - param_name: Name of the parameter
                - weight: Transformed weight tensor for this rank

        Example:
            .. code-block:: python

                # Process weights one by one
                for weight_tuple in bridge.stream_weights_hf_to_megatron(hf_model, megatron_model):
                    print(f"Processing {weight_tuple.param_name}: {weight_tuple.weight.shape}")
                    # Custom processing logic here

                # Or use pre-built conversion tasks
                tasks = bridge.build_conversion_tasks(hf_model, megatron_model)
                for weight_tuple in bridge.stream_weights_hf_to_megatron(hf_model, megatron_model, tasks):
                    print(f"Processing {weight_tuple.param_name}: {weight_tuple.weight.shape}")

        Note:
            Only yields weights that belong to the current rank after TP/PP distribution.

        Raises:
            ValueError: If input parameters are invalid.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        for task in conversion_tasks:
            # None means megatron module not on current rank, skip if this task is not going to happen
            if task.megatron_module is None:
                continue
            hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state
            if isinstance(task.mapping.hf_param, str):
                hf_weights = hf_state_dict[task.mapping.hf_param]
            else:
                hf_weights = {k: hf_state_dict[v] for k, v in task.mapping.hf_param.items()}

            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)
            if converted_weights is not None:
                # Assert that vp_stage is not None for HF->Megatron tasks
                yield MegatronWeightTuple(task.param_name, converted_weights, task.vp_stage)

    def stream_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        merge_adapter_weights: bool = True,
    ) -> Iterable[HFWeightTuple]:
        """Export Megatron weights to HuggingFace format.

        This method orchestrates the conversion of weights from Megatron's distributed
        format back to HuggingFace format. It handles gathering from tensor parallel
        ranks, broadcasting across pipeline parallel ranks, and format conversions.
        All ranks receive the full tensors.

        The export order is determined automatically:
        - First tries safetensors order (if key_to_filename_map is available)
        - Falls back to HuggingFace state dict order

        Args:
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances (one per virtual pipeline stage).
            hf_pretrained (HFPreTrained): HuggingFace model/config for metadata
                and mapping info.
            cpu (bool, optional): Whether to move tensors to CPU before yielding.
                Defaults to True.
            show_progress (bool, optional): Display progress bar during export.
                Defaults to True.
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.
            merge_adapter_weights (bool, optional): When True, materialize and merge LoRA adapter
                weights back into their base tensors so the resulting HF checkpoint contains merged
                weights. Set to False to skip adapter gathering/merge and emit only the base tensors.
                Defaults to True.

        Yields:
            HFWeightTuple: Named tuples of (param_name, weight_tensor) in HF format.

        Example:
            .. code-block:: python

                # Export weights
                for name, weight in bridge.stream_weights_megatron_to_hf(megatron_model, hf_config):
                    print(f"Exported {name}: {weight.shape}")

                # Or use pre-built conversion tasks
                tasks = bridge.build_conversion_tasks(hf_config, megatron_model)
                for name, weight in bridge.stream_weights_megatron_to_hf(
                    megatron_model, hf_config, conversion_tasks=tasks
                ):
                    print(f"Exported {name}: {weight.shape}")

        Raises:
            ValueError: If input parameters are invalid.

        Note:
            All ranks yield the full tensors after gathering from distributed format.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        # Collect adapter conversion tasks when merge is requested
        adapter_tasks_by_base: Dict[str, List[AdapterWeightConversionTask]] = {}
        if merge_adapter_weights:
            adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)

        megatron_to_hf_tasks = conversion_tasks
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(model_config)

        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

        for task in self._with_progress_tracking(megatron_to_hf_tasks, "Converting to HuggingFace", show_progress):
            converted_weights_dict = task.mapping.megatron_to_hf(task.param_weight, task.megatron_module)
            converted_weights_dict = self.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )  # dict will be none except for one expert;
            # All ranks get the full tensor

            adapter_tasks = None
            if merge_adapter_weights and "to_wrap.weight" in task.global_param_name:
                task_global_base_prefix, _, _ = task.global_param_name.partition(".to_wrap.weight")
                adapter_tasks = adapter_tasks_by_base.get(task_global_base_prefix)
            if merge_adapter_weights and adapter_tasks:
                adapter_weights = self.materialize_adapter_weights(adapter_tasks)
                # Merge LoRA adapter weights back into the base tensor for HF export
                converted_weights_dict = self._merge_lora_adapter_weights(
                    megatron_model,
                    converted_weights_dict,
                    adapter_weights,
                )

            for hf_name, tensor in converted_weights_dict.items():
                final_tensor = tensor.cpu() if cpu else tensor

                if not merge_adapter_weights and "to_wrap.weight" in task.global_param_name:
                    hf_name = hf_name[: -len("weight")] + "base_layer.weight"

                # Handle tied embeddings case
                # TODO(yuya): fix this hard coded naming
                if embeddings_are_tied and hf_name == "model.embed_tokens.weight":
                    # Yield the embedding weight
                    yield HFWeightTuple(hf_name, final_tensor)

                    # Also yield as lm_head.weight if it's expected
                    if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
                        expected_keys = hf_pretrained.state.source.get_all_keys()
                        if "lm_head.weight" in expected_keys:
                            yield HFWeightTuple("lm_head.weight", final_tensor.clone().detach())
                elif embeddings_are_tied and hf_name == "lm_head.weight":
                    # This should not happen when embeddings are tied - assert error
                    raise ValueError(
                        "Encountered lm_head.weight when embeddings are tied. This indicates a mapping error."
                    )
                else:
                    # Regular case - yield the tensor normally
                    yield HFWeightTuple(hf_name, final_tensor)

    def dtype_from_hf(self, config, default=None):
        """Extract torch dtype from a HuggingFace config.

        This utility method handles the conversion of dtype specifications in
        HuggingFace configs to PyTorch dtype objects. Supports both direct
        torch.dtype objects and string representations.

        Args:
            config: HuggingFace configuration object with a torch_dtype attribute.
            default (Any, optional): Default value to return if torch_dtype is
                not str or torch.dtype. Defaults to None.

        Returns:
            torch.dtype: The corresponding PyTorch dtype.

        Raises:
            AssertionError: If config doesn't have torch_dtype attribute.
            ValueError: If torch_dtype is neither a string nor torch.dtype.

        Example:
            .. code-block:: python

                dtype = bridge.dtype_from_hf(hf_config)
                print(dtype)  # torch.float16
        """
        assert hasattr(config, "torch_dtype"), "Expected config to have attr `torch_dtype`"
        torch_dtype = config.torch_dtype
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        elif isinstance(torch_dtype, str):
            return self.dtype_from_str(torch_dtype)
        elif default is not None:
            return default

        raise ValueError("torch_dtype is not of type str/torch.dtype")

    def dtype_from_str(self, dtype: str) -> torch.dtype:
        """Convert a string precision identifier to equivalent torch dtype.

        This utility method handles various string representations of PyTorch
        data types, including common abbreviations and mixed precision formats.

        Args:
            dtype (str): String representation of dtype (e.g., "float16", "fp16",
                "bf16-mixed").

        Returns:
            torch.dtype: Corresponding PyTorch dtype (defaults to float32 if unknown).

        Supported formats:
            - float16/fp16/16/16-mixed → torch.float16
            - bfloat16/bf16-mixed → torch.bfloat16
            - Others → torch.float32 (default)

        Example:
            .. code-block:: python

                dtype = bridge.dtype_from_str("fp16")
                print(dtype)  # torch.float16

                dtype = bridge.dtype_from_str("bf16-mixed")
                print(dtype)  # torch.bfloat16
        """
        assert isinstance(dtype, str)
        if dtype in ["float16", "fp16", "16", "16-mixed"]:
            return torch.float16
        elif dtype in ["bfloat16", "bf16-mixed"]:
            return torch.bfloat16
        else:
            return torch.float32

    def make_vocab_size_divisible_by(self, vocab_size: int) -> int:
        """Calculate an appropriate divisor for vocabulary size padding.

        Megatron requires vocabulary sizes to be divisible by certain values for
        efficient tensor parallelism. This method finds the largest power of 2
        (up to 128) that evenly divides the vocabulary size.

        Args:
            vocab_size (int): Original vocabulary size from the model.

        Returns:
            int: Largest power of 2 (≤ 128) that divides vocab_size.

        Example:
            .. code-block:: python

                # For vocab_size=50257 (GPT-2)
                divisor = bridge.make_vocab_size_divisible_by(50257)
                print(divisor)  # 1 (50257 is prime)

                # For vocab_size=32000 (Llama)
                divisor = bridge.make_vocab_size_divisible_by(32000)
                print(divisor)  # 128

        Note:
            The returned value is used by Megatron to potentially pad the
            vocabulary to ensure efficient parallelization.
        """
        base = 128
        while vocab_size % base != 0:
            base //= 2
        return base

    def _get_provider_from_model(self, model: MegatronModule) -> ModelProviderTarget:
        """Extract provider/config from model."""
        model = unwrap_model(model)
        return model.config

    def _share_embeddings_and_output_weights(
        self,
        model_config: TransformerConfig,
    ) -> bool:
        """Shared embedding setting."""
        return getattr(model_config, "share_embeddings_and_output_weights")

    def _unwrap_name(self, name: str) -> str:
        """Unwrap name from DDP or other wrappers.

        Args:
            name: Parameter name that may have 'module.' prefixes

        Returns:
            Unwrapped parameter name with 'module.' prefixes removed

        Example:
            'module.module.decoder.weight' -> 'decoder.weight'
        """
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")

        while name.startswith("module."):
            name = name[len("module.") :]
        return name

    def _broadcast_shared_embeddings(self, megatron_model: Union[MegatronModel, List[MegatronModel]]) -> None:
        """Broadcast shared embeddings and output weights across embedding group.

        When embeddings and output weights are shared and pipeline parallelism is enabled,
        this method ensures all ranks in the embedding group have the same weights by
        broadcasting from rank 0.

        Args:
            megatron_model: Megatron model instance or list of model instances.
        """
        unwrapped_model = unwrap_model(megatron_model)[0]
        # hack for vlm to work properly
        if hasattr(unwrapped_model, "language_model") and unwrapped_model.language_model is not None:
            unwrapped_model = unwrapped_model.language_model
        model_config = unwrapped_model.config
        share_embeddings = self._share_embeddings_and_output_weights(model_config)

        # TODO(yuya): Fix for VPP, the vp stage needs to be passed in for stage checks
        if (share_embeddings and model_config.pipeline_model_parallel_size > 1) and (
            parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
        ):
            # Broadcast embeddings and output weights from rank 0 to embedding group
            embd_group = parallel_state.get_embedding_group()
            embd_group_ranks = torch.distributed.get_process_group_ranks(embd_group)
            if embd_group is not None and torch.distributed.get_rank() in embd_group_ranks:
                # Get embeddings and output weights from rank 0
                if hasattr(unwrapped_model, "embedding") and hasattr(unwrapped_model.embedding, "word_embeddings"):
                    embd_weights = unwrapped_model.embedding.word_embeddings.weight.data
                else:
                    assert hasattr(unwrapped_model, "output_layer"), "Output layer not found"
                    embd_weights = torch.empty_like(unwrapped_model.output_layer.weight.data)
                torch.distributed.broadcast(embd_weights, src=embd_group_ranks[0], group=embd_group)
                if hasattr(unwrapped_model, "output_layer"):
                    unwrapped_model.output_layer.weight.data.copy_(embd_weights)

    def build_conversion_tasks(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: List[MegatronModel],
    ) -> List[None | WeightConversionTask]:
        """Construct the conversion tasks between HF and megatron.

        The algorithm walks over every parameter of every destination model,
        asks the :class:`MegatronMappingRegistry` whether it has a mapping for that
        parameter, and – if the corresponding HF weights actually exist – yields
        an :class:`_HFLoadTask` describing exactly how that parameter will be
        populated.
        """

        # Ensure hf_pretrained has the required state structure
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        hf_keys: Iterable[str] = hf_pretrained.state.source.get_all_keys()

        mapping_registry = self.mapping_registry()
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(model_config)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(megatron_model)

        # Filter out output_layer related parameters if embeddings are tied
        if embeddings_are_tied:
            sorted_global_param_names_all_pp_ranks = [
                name for name in sorted_global_param_names_all_pp_ranks if "output_layer" not in name
            ]

        global_names_index_dict = {name: idx for idx, name in enumerate(sorted_global_param_names_all_pp_ranks)}

        tasks = [None] * len(sorted_global_param_names_all_pp_ranks)
        for vp_stage, model in enumerate(megatron_model):
            # persistent buffers are part of the model's state_dict, but not the named_parameters, so we must include them here separately
            for local_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_name or self._is_adapter_param_name(local_name):
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(megatron_model, model_config, local_name, vp_stage)
                # if name removed due to some reason, continue. e.g. embeddings_are_tied
                if global_name not in global_names_index_dict:
                    print_rank_0(f"WARNING: {global_name} not in global_names_index_dict")
                    continue
                global_name_idx = global_names_index_dict[global_name]
                mapping = mapping_registry.megatron_to_hf_lookup(self._get_lora_unwrapped_name(global_name))

                if not mapping:
                    logger.warning(f"WARNING: No mapping found for megatron_param: {global_name}")
                    continue

                # ensure hf weights exist
                if not mapping.allow_hf_name_mismatch:
                    if isinstance(mapping.hf_param, str):
                        if mapping.hf_param not in hf_keys:
                            logger.warning(f"WARNING: Can't find {mapping.hf_param} in hf_keys")
                            continue
                    else:
                        missing_params = [
                            hf_param for hf_param in mapping.hf_param.values() if hf_param not in hf_keys
                        ]
                        if missing_params:
                            logger.warning(
                                f"WARNING: Can't find the following HF parameters in hf_keys: {missing_params}"
                            )
                            continue

                local_module, local_weights = get_module_and_param_from_name(megatron_model, local_name, vp_stage)
                if local_module is not None and not hasattr(local_module, "config"):
                    # If module is not a MegatronModule (e.g. torch.nn.Conv1d or a module list) we need
                    # to get the config from the model
                    setattr(local_module, "config", model_config)

                tasks[global_name_idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )

        # Fill the remaining ones for pp communications
        for idx, global_name in enumerate(sorted_global_param_names_all_pp_ranks):
            if tasks[idx] is None:
                mapping = mapping_registry.megatron_to_hf_lookup(self._get_lora_unwrapped_name(global_name))
                # Skip tasks with no mapping found
                if mapping is None:
                    continue
                # This is an exception here we pass in global name
                # we are not using global_name to extract module and weights
                # only use it for param mapping auto dispatch checks
                tasks[idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    global_param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mapping,
                )

        return tasks

    @classmethod
    def register_bridge(
        cls,
        *,
        source: Type[PreTrainedModel] | str,
        target: Type[MegatronModel],
        provider: Type[ModelProviderTarget] | None = None,
        model_type: str | None = None,
    ) -> Callable[[_BridgeImplClass], _BridgeImplClass]:
        """Class decorator for registering bridge implementations.

        This decorator registers a MegatronModelBridge subclass with the dispatch
        system, enabling automatic routing of conversions based on the source
        HuggingFace model type and target Megatron model type.

        Args:
            source (Type[PreTrainedModel] | str): HuggingFace PreTrainedModel class
                (e.g., LlamaForCausalLM) or the class name as a string. Using a
                string allows registering bridges for architectures that are only
                available via auto_map.
            target (Type[MegatronModel]): Megatron model class (e.g., GPTModel).
            provider (Type[ModelProviderTarget], optional): Provider class to use
                for this model (e.g., DeepSeekModelProvider for MLA models).
                Defaults to GPTModelProvider if not specified.
            model_type (str, optional): HuggingFace model_type string (e.g., "llama").
                Used for megatron_to_hf_config conversion.

        Returns:
            Callable[[_BridgeImplClass], _BridgeImplClass]: Decorator function
                that registers the bridge implementation.

        Example:
            .. code-block:: python

                @MegatronModelBridge.register_bridge(
                    source=LlamaForCausalLM, target=GPTModel, model_type="llama"
                )
                class MegatronCausalLlamaBridge(MegatronModelBridge):
                    def provider_bridge(self, hf_pretrained):
                        # Implementation
                        pass

                    def mapping_registry(self):
                        # Implementation
                        pass

            String-based registration with custom provider:

            .. code-block:: python

                @MegatronModelBridge.register_bridge(
                    source="DeepseekV3ForCausalLM",
                    target=GPTModel,
                    provider=DeepSeekModelProvider,
                    model_type="deepseek_v3",
                )
                class MegatronDeepseekV3Bridge(MegatronModelBridge):
                    ...

        Note:
            The decorated class is registered with multiple dispatchers to handle
            different conversion scenarios. The registration is automatic when the
            class is defined.
        """

        return create_bridge_decorator(source=source, target=target, provider=provider, model_type=model_type)


def is_tensor_parallel(param) -> bool:
    """Check if a parameter is tensor parallel distributed."""
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


# Core dispatch functions
@dispatch
def get_model_bridge(hf_architecture, hf_config=None) -> "MegatronModelBridge":
    """Get the appropriate model bridge for a given HuggingFace architecture."""
    ...


@dispatch
def stream_weights_megatron_to_hf(
    dispatch_instance: MegatronModel,
    megatron_model: Union[MegatronModel, List[MegatronModel]],
    hf_pretrained: HFPreTrained,
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks: Optional[List[WeightConversionTask]] = None,
    merge_adapter_weights: bool = True,
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron model state to HuggingFace format."""
    ...


@dispatch
def stream_adapter_weights_megatron_to_hf(
    dispatch_instance: MegatronModel,
    megatron_model: Union[MegatronModel, List[MegatronModel]],
    cpu: bool = True,
    show_progress: bool = True,
) -> Iterable[HFWeightTuple]:
    """Bridge only adapter weights from Megatron to HuggingFace format."""
    ...


def register_bridge_implementation(
    *,
    source: Type["PreTrainedModel"] | str,
    target: Type["MegatronModule"],
    bridge_class: Type["MegatronModelBridge"],
) -> None:
    """Register a bridge implementation with the dispatch system.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string.
            Using a string allows registering bridges for architectures that are
            available only via auto_map.
        target: Megatron model class (e.g., GPTModel)
        bridge_class: MegatronModelBridge implementation class
    """
    bridge_class_name = bridge_class.__name__

    @get_model_bridge.impl(source)
    def _get_model_bridge_impl(_, hf_config=None) -> "MegatronModelBridge":
        bridge = bridge_class()
        if hf_config is not None:
            bridge.hf_config = hf_config
        return bridge

    @stream_weights_megatron_to_hf.impl((source, target))
    def _megatron_to_hf_registered_impl(
        _,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
        merge_adapter_weights: bool = True,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()

        # allow bridge to access model config (config-only shims or raw configs lack .config)
        bridge.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained

        return bridge.stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
        )

    @stream_adapter_weights_megatron_to_hf.impl((source, target))
    def _adapter_stream_registered_impl(
        _,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        cpu: bool = True,
        show_progress: bool = True,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()
        return bridge.stream_adapter_weights_megatron_to_hf(
            megatron_model,
            cpu=cpu,
            show_progress=show_progress,
        )

    # Set meaningful names for debugging
    _get_model_bridge_impl.__name__ = f"_bridge_with_{bridge_class_name}"
    _megatron_to_hf_registered_impl.__name__ = f"_megatron_to_hf_with_{bridge_class_name}"
    _adapter_stream_registered_impl.__name__ = f"_adapter_stream_with_{bridge_class_name}"


def create_bridge_decorator(
    *,
    source: Type["PreTrainedModel"] | str,
    target: Type["MegatronModule"],
    provider: Type["ModelProviderMixin"] | None = None,
    model_type: str | None = None,
) -> Callable[[Type["MegatronModelBridge"]], Type["MegatronModelBridge"]]:
    """Create a decorator for registering bridge implementations.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string
            (useful for auto_map architectures)
        target: Megatron model class
        provider: Provider class to use for this model (e.g., DeepSeekModelProvider)
        model_type: HuggingFace model_type string (e.g., "llama", "deepseek_v3")

    Returns:
        Decorator function that registers the bridge implementation
    """

    def decorator(bridge_class: Type["MegatronModelBridge"]) -> Type["MegatronModelBridge"]:
        # Store source name for HF config generation
        bridge_class.SOURCE_NAME = source if isinstance(source, str) else source.__name__
        # Store model_type for HF config generation
        if model_type is not None:
            bridge_class.MODEL_TYPE = model_type
        # Set the provider class on the bridge
        if provider is not None:
            bridge_class.PROVIDER_CLASS = provider
        register_bridge_implementation(source=source, target=target, bridge_class=bridge_class)
        return bridge_class

    return decorator
