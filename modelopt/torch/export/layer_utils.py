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

"""Utils for model_config export.

Some of the logics in this file are empirical and needs constant update if exceptions occur.
"""

from warnings import warn

import torch
import torch.nn as nn

try:
    from transformers.activations import ACT2FN
except Exception:
    warn("Cannot find transformers package. Hugginface modules cannot be exported.")

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import import_plugin

from ..quantization.nn import SequentialQuantizer, TensorQuantizer
from .hf_config_map import HF_CONFIG_MAP
from .mcore_config_map import MCORE_CONFIG_MAP
from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    LINEAR_COLUMN,
    LINEAR_GROUP,
    LINEAR_ROW,
    QUANTIZATION_FP8,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    AttentionConfig,
    ConvConfig,
    DecoderLayerConfig,
    EmbeddingConfig,
    ExpertConfig,
    LayernormConfig,
    LinearActConfig,
    LinearConfig,
    MedusaHeadConfig,
    MLPConfig,
    MOEConfig,
    QKVConfig,
    RecurrentConfig,
    RelativeAttentionTableConfig,
    RgLruConfig,
)
from .model_config_utils import pad_weights
from .postprocess import view_as_float8_e4m3fn_if_needed, view_as_uint8_if_needed
from .quant_utils import (
    get_activation_scaling_factor,
    get_kv_cache_bias,
    get_kv_cache_dtype,
    get_kv_cache_scaling_factor,
    get_prequant_scaling_factor,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    preprocess_linear_fusion,
)

has_mcore = False
with import_plugin("megatron", verbose=False):
    from megatron.core.transformer.module import MegatronModule

    has_mcore = True


def get_experts_list(module: torch.nn.Module, model_type: str):
    """Returns list of grouped experts by linear name for given module."""
    experts_list = []

    # Define linear layer names for different model types
    if "mixtralforcausallm" in model_type:
        linear_names = ["w1", "w2", "w3"]
    elif any(
        qwen_variant in model_type
        for qwen_variant in [
            "qwenmoeforcausallm",
            "qwen2moeforcausallm",
            "qwen3moeforcausallm",
            "qwen3nextforcausallm",
        ]
    ):
        linear_names = ["gate_proj", "down_proj", "up_proj"]
    else:
        raise NotImplementedError(f" {model_type} not supported")

    # Common logic for all supported model types
    experts_list.extend(
        [
            [_get_expert_attr(module.experts, i, linear_name) for i in range(len(module.experts))]
            for linear_name in linear_names
        ]
    )

    return experts_list


def get_dtype(model):
    """Returns the default dtype of the model."""
    for weight in model.parameters():
        if torch.is_floating_point(weight):
            return weight.dtype
    return None


def check_model_compatibility(module_list: list[nn.Module]) -> tuple[bool, bool, bool]:
    """Returns whether the list of modules is compatible with the export logic.

    And if positional embedding and embedding layernorm exists.

    We assumes the model to be assembled with one or two embedding layers,
    a ModuleList of transformer decoders,
    and a final layernorm with optional embedding layernorm.
    Otherwise it will not be supported.
    """
    num_embeddings = 0
    num_module_list = 0
    num_layer_norm = 0
    for module in module_list:
        if is_embedding(module):
            num_embeddings += 1
        elif is_decoder_list(module):
            num_module_list += 1
        elif is_layernorm(module):
            num_layer_norm += 1

    return (
        1 <= num_embeddings <= 2 and num_module_list == 1 and 1 <= num_layer_norm <= 2,
        num_embeddings > 1,
        num_layer_norm > 1,
    )


def get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Returns the root module of the transformer model."""
    if "Megatron" in type(model).__name__:
        if hasattr(model, "model") and "GPTModel" in type(model.model).__name__:
            # NEMO mcore models can be handled with the following branch.
            model = model.model

        # NEMO non mcore models, we need to find the language_model module first.
        children = [model]
        language_model = None
        while children and not language_model:
            next_children = []
            for child in children:
                if type(child).__name__ == "TransformerLanguageModel":
                    language_model = child
                    break
                next_children.extend(list(child.children()))
            children = next_children
        if language_model:
            warn("Warning: this is an old NEMO checkpoint format and will be deprecated soon.")
            layers = list(language_model.embedding.children()) + list(
                language_model.encoder.children()
            )

            if hasattr(language_model, "output_layer"):
                layers.append(language_model.output_layer)

            return layers

    if "GPTModel" in type(model).__name__:
        # mcore models
        layers = []
        if hasattr(model, "embedding"):
            layers = layers + list(model.embedding.children())
        layers = layers + list(model.decoder.children())
        if hasattr(model, "output_layer"):
            layers.append(model.output_layer)
        return layers

    if hasattr(model, "glm"):
        model = model.glm

    if hasattr(model, "transformer"):
        # This is a LMHead model
        # Add lm_head to be processed along with transformer layers
        modules = []
        for m in model.transformer.children():
            # QwenVL's visual encoder name as 'VisionTransformer' has no `layers`.
            if (
                "Transformer" in type(m).__name__
                and hasattr(m, "layers")
                and is_decoder_list(m.layers)
            ):
                modules.append(m.layers)
                modules.append(m.final_layernorm)
            else:
                modules.append(m)
        if hasattr(model, "lm_head"):
            modules += [model.lm_head]
        return modules

    if hasattr(model, "model"):
        # LLAMA, InternLM2
        modules = list(model.model.children())
        # LLAMA
        if hasattr(model, "lm_head"):
            modules += [model.lm_head]
        # InternLM2
        elif hasattr(model, "output"):
            modules += [model.output]

        return modules

    return list(model.children())


def is_linear(module: nn.Module) -> bool:
    """Returns whether the module is a linear layer."""
    return any(k in type(module).__name__ for k in ["Linear", "Conv1D", "NormHead"])


def is_conv(module: nn.Module) -> bool:
    """Returns whether the module is a convolutional layer."""
    return "Conv" in type(module).__name__


def is_embedding(module: nn.Module) -> bool:
    """Returns whether the module is an embedding layer."""
    module_type_name = type(module).__name__
    return (
        "Embedding" in module_type_name
        and "Rotary" not in module_type_name
        and "PhiImage" not in module_type_name
        and "Phi3Image" not in module_type_name
    )


def build_embedding_config(module: nn.Module, normalization_constant: float = 1) -> EmbeddingConfig:
    """Builds the embedding config from the module."""
    assert is_embedding(module)

    world_size = dist.size()
    rank = dist.rank()

    # Special case for chatglm
    if hasattr(module, "word_embeddings"):
        module = module.word_embeddings

    weight = module.weight
    normalized_weight = weight * normalization_constant
    if "Parallel" in type(module).__name__:
        local_weight = normalized_weight
    else:
        padded_weight = pad_weights(normalized_weight, dist.size())
        local_weight = torch.chunk(padded_weight, world_size, dim=0)[rank]
    return EmbeddingConfig(weight=local_weight)


def is_layernorm(module: nn.Module) -> bool:
    """Returns whether the module is a layernorm layer."""
    module_name = type(module).__name__
    return any(norm in module_name for norm in ["LayerNorm", "RMSNorm"])


def build_layernorm_config(module: nn.Module) -> LayernormConfig:
    """Builds the layernorm config from the module."""
    assert is_layernorm(module)

    layernorm_type = LAYERNORM_DEFAULT
    if "RMS" in type(module).__name__:
        layernorm_type = LAYERNORM_RMS

    weight = module.weight

    def _weights_plus_one(module):
        if any(
            name in type(module).__name__
            for name in ["LayerNorm1P", "GemmaRMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm"]
        ):
            return True

        return bool(hasattr(module, "zero_centered_gamma") and module.zero_centered_gamma)

    if _weights_plus_one(module):
        # megatron layernorm's weight needs to be updated.
        weight = weight.float() + 1.0

    config = LayernormConfig(
        weight=weight,
        bias=(module.bias if hasattr(module, "bias") and module.bias is not None else None),
        layernorm_type=layernorm_type,
    )

    # TODO: handle the nemo llama eps config.
    for eps_key in ["eps", "variance_epsilon"]:
        if hasattr(module, eps_key):
            config.eps = getattr(module, eps_key)
            break

    return config


def is_decoder_list(module: nn.Module) -> bool:
    """Returns whether the module is a decoder list."""
    return type(module) is nn.ModuleList


def is_attention(module: nn.Module) -> bool:
    """Returns whether the module is an attention layer."""
    return "Attention" in type(module).__name__


def is_mlp(module: nn.Module) -> bool:
    """Returns whether the module is an MLP layer."""
    return any(key in type(module).__name__.upper() for key in ("MLP", "T5DENSE"))


def is_moe(module: nn.Module) -> bool:
    """Returns whether the module is an MOE layer."""
    return any(
        key in type(module).__name__.lower()
        for key in [
            "MixtralSparseMoeBlock".lower(),
            "ArcticMoE".lower(),
            "DbrxFFN".lower(),
            "MoELayer".lower(),
            "PhimoeSparseMoeBlock".lower(),
            "DeepseekMoE".lower(),
            "Qwen2MoeSparseMoeBlock".lower(),
            "Qwen3MoeSparseMoeBlock".lower(),
            "Qwen3NextSparseMoeBlock".lower(),
        ]
    )


def is_quantlinear(module: nn.Module) -> bool:
    """Returns whether the module is a quantized linear layer."""
    return "QuantLinear" in type(module).__name__ and "lora" not in type(module).__name__.lower()


def dup_kv_weight(v: torch.Tensor, head_size: int, num_head: int, tp_size: int) -> torch.Tensor:
    """Repeat kv heads if tp_size > num_kv_heads."""
    assert tp_size % num_head == 0
    reps = tp_size // num_head

    is_1d = len(v.shape) == 1
    v_1 = v.size(-1) if not is_1d else 1

    v = v.view(-1, head_size, v_1)
    v = v.repeat_interleave(reps, dim=0)
    v = v.view(-1, v_1) if not is_1d else v.reshape(-1)
    return v.contiguous()


def build_qkv(
    qkv_modules: list[nn.Module],
    model_metadata_config,
    ext_config: DecoderLayerConfig = None,
    tp_size: int = 1,
) -> QKVConfig:
    """Converts the qkv modules to the config."""
    config = QKVConfig()
    q_bias = None
    k_bias = None
    v_bias = None

    block_size = get_weight_block_size(qkv_modules[0])

    num_heads = ext_config.num_attention_heads
    training_tp = model_metadata_config["training_tensor_parallel"]
    if len(qkv_modules) == 1:
        # QKV layers combined as a single module, e.g. gpt
        qkv_module = qkv_modules[0]
        assert ext_config is not None, "ext_config is None"
        num_kv_heads = ext_config.num_kv_heads

        if "ColumnParallelLinear" in type(qkv_module).__name__:
            # For NEMO model, num_kv_heads/num_attention_heads is the first dimension of QKV
            model_metadata_config["head_is_first_dim"] = True

        qkv_weight = qkv_module.weight
        if (
            type(qkv_module).__name__ == "Conv1D"
            and not hasattr(qkv_module, "input_quantizer")
            and not hasattr(qkv_module, "output_quantizer")
        ):
            # For unquantized nn.Conv1D, the weights are transposed compared with the nn.Linear
            qkv_weight = qkv_weight.T

        # Handle the case that num_kv_heads/num_attention_heads is the first dimension of QKV.
        # This logic covers MQA and GQA as well.
        keep_channel_order = not model_metadata_config.get("head_is_first_dim", False)

        q_weight, k_weight, v_weight = _split_fused_qkv_weight_and_scaling(
            qkv_weight,
            num_heads,
            num_kv_heads,
            training_tp,
            keep_channel_order,
        )
        qkv_activation_scaling_factor = get_activation_scaling_factor(qkv_module)
        q_activation_scaling_factor = qkv_activation_scaling_factor
        k_activation_scaling_factor = qkv_activation_scaling_factor
        v_activation_scaling_factor = qkv_activation_scaling_factor

        qkv_weight_scaling_factor = get_weight_scaling_factor(qkv_module)

        if qkv_weight_scaling_factor is not None and qkv_weight_scaling_factor.numel() != 1:
            # SQ and AWQ case
            (
                q_weight_scaling_factor,
                k_weight_scaling_factor,
                v_weight_scaling_factor,
            ) = _split_fused_qkv_weight_and_scaling(
                qkv_weight_scaling_factor,
                num_heads,
                num_kv_heads,
                training_tp,
                keep_channel_order,
            )
        else:
            q_weight_scaling_factor = qkv_weight_scaling_factor
            k_weight_scaling_factor = qkv_weight_scaling_factor
            v_weight_scaling_factor = qkv_weight_scaling_factor

        # bias
        if qkv_module.bias is not None:
            q_bias, k_bias, v_bias = _split_fused_qkv_weight_and_scaling(
                qkv_module.bias,
                num_heads,
                num_kv_heads,
                training_tp,
                keep_channel_order,
            )

        q_weight_scaling_factor_2 = k_weight_scaling_factor_2 = v_weight_scaling_factor_2 = (
            get_weight_scaling_factor_2(qkv_module)
        )

        q_prequant_scaling_factor = k_prequant_scaling_factor = v_prequant_scaling_factor = (
            get_prequant_scaling_factor(qkv_module)
        )

        q_quantization = k_quantization = v_quantization = get_quantization_format(qkv_module)

    elif len(qkv_modules) == 3:
        preprocess_linear_fusion(qkv_modules)
        # Separate QKV layers
        q_weight = qkv_modules[0].weight
        q_activation_scaling_factor = get_activation_scaling_factor(qkv_modules[0])
        q_weight_scaling_factor = get_weight_scaling_factor(qkv_modules[0])
        q_quantization = get_quantization_format(qkv_modules[0])
        k_weight = qkv_modules[1].weight
        k_activation_scaling_factor = get_activation_scaling_factor(qkv_modules[1])
        k_weight_scaling_factor = get_weight_scaling_factor(qkv_modules[1])
        k_quantization = get_quantization_format(qkv_modules[1])
        v_weight = qkv_modules[2].weight
        v_activation_scaling_factor = get_activation_scaling_factor(qkv_modules[2])
        v_weight_scaling_factor = get_weight_scaling_factor(qkv_modules[2])
        v_quantization = get_quantization_format(qkv_modules[2])

        q_weight_scaling_factor_2 = get_weight_scaling_factor_2(qkv_modules[0])
        k_weight_scaling_factor_2 = get_weight_scaling_factor_2(qkv_modules[1])
        v_weight_scaling_factor_2 = get_weight_scaling_factor_2(qkv_modules[2])

        q_prequant_scaling_factor = get_prequant_scaling_factor(qkv_modules[0])
        k_prequant_scaling_factor = get_prequant_scaling_factor(qkv_modules[1])
        v_prequant_scaling_factor = get_prequant_scaling_factor(qkv_modules[2])

        if hasattr(qkv_modules[0], "bias"):
            q_bias = qkv_modules[0].bias

        if hasattr(qkv_modules[1], "bias"):
            k_bias = qkv_modules[1].bias

        if hasattr(qkv_modules[2], "bias"):
            v_bias = qkv_modules[2].bias

    else:
        raise NotImplementedError(f"QKV modules format {qkv_modules} not supported")

    # derive num_kv_heads
    head_size = q_weight.size(0) // num_heads
    num_kv_heads = ext_config.num_kv_heads or k_weight.size(0) // head_size
    if tp_size > num_kv_heads:
        if any(
            t is not None and t.numel() > 1
            for t in [
                k_activation_scaling_factor,
                k_weight_scaling_factor_2,
                v_activation_scaling_factor,
                v_weight_scaling_factor_2,
            ]
        ):
            # TODO(oargov): handle cases with biases / scales
            raise NotImplementedError(
                "Duplicating KV heads for KV with non-scalar scales and/or biases is not supported"
            )

        # duplicate kv heads as needed
        k_weight = dup_kv_weight(k_weight, head_size, num_kv_heads, tp_size)
        v_weight = dup_kv_weight(v_weight, head_size, num_kv_heads, tp_size)
        if k_weight_scaling_factor is not None and k_weight_scaling_factor.numel() > 1:
            if len(k_weight_scaling_factor.shape) == 1:
                raise NotImplementedError(
                    "Duplicating KV heads per-channel scales is not supported"
                )
            k_weight_scaling_factor = dup_kv_weight(
                k_weight_scaling_factor, head_size, num_kv_heads, tp_size
            )
        if v_weight_scaling_factor is not None and v_weight_scaling_factor.numel() > 1:
            if len(v_weight_scaling_factor.shape) == 1:
                raise NotImplementedError(
                    "Duplicating KV heads per-channel scales is not supported"
                )
            v_weight_scaling_factor = dup_kv_weight(
                v_weight_scaling_factor, head_size, num_kv_heads, tp_size
            )

        if k_bias is not None:
            k_bias = dup_kv_weight(k_bias, head_size, num_kv_heads, tp_size)
        if v_bias is not None:
            v_bias = dup_kv_weight(v_bias, head_size, num_kv_heads, tp_size)

    config.q = LinearConfig(linear_type=LINEAR_COLUMN)
    config.q.weight = q_weight
    config.q.bias = q_bias if q_bias is not None else None
    config.q.activation_scaling_factor = q_activation_scaling_factor
    config.q.weights_scaling_factor = q_weight_scaling_factor
    config.q.weights_scaling_factor_2 = q_weight_scaling_factor_2
    config.q.prequant_scaling_factor = q_prequant_scaling_factor
    config.q.awq_block_size = block_size
    config.q.quantization = q_quantization

    config.k = LinearConfig(linear_type=LINEAR_COLUMN)
    config.k.weight = k_weight
    config.k.bias = k_bias if k_bias is not None else None
    config.k.activation_scaling_factor = k_activation_scaling_factor
    config.k.weights_scaling_factor = k_weight_scaling_factor
    config.k.weights_scaling_factor_2 = k_weight_scaling_factor_2
    config.k.prequant_scaling_factor = k_prequant_scaling_factor
    config.k.awq_block_size = block_size
    config.k.quantization = k_quantization

    config.v = LinearConfig(linear_type=LINEAR_COLUMN)
    config.v.weight = v_weight
    config.v.bias = v_bias if v_bias is not None else None
    config.v.activation_scaling_factor = v_activation_scaling_factor
    config.v.weights_scaling_factor = v_weight_scaling_factor
    config.v.weights_scaling_factor_2 = v_weight_scaling_factor_2
    config.v.prequant_scaling_factor = v_prequant_scaling_factor
    config.v.awq_block_size = block_size
    config.v.quantization = v_quantization

    if not ext_config.attention_head_size:
        ext_config.attention_head_size = config.q.weight.shape[0] * training_tp // num_heads

    return config


def build_linear_config(module: nn.Module, linear_type: str) -> LinearConfig:
    """Builds the linear config for the module."""
    if has_mcore and not isinstance(module, MegatronModule):
        # Check only for HF model, not Mcore model
        assert is_linear(module)

    torch_weight = module.weight

    if "NormHead" in type(module).__name__:
        torch_weight = torch.nn.functional.normalize(torch_weight)
    elif "Conv1D" in type(module).__name__ and not (
        hasattr(module, "input_quantizer") or hasattr(module, "output_quantizer")
    ):
        # Transpose Conv1D weights to linear unless it has been transposed by the quantization.
        torch_weight = torch_weight.T

    weight = torch_weight

    config = LinearConfig(linear_type=linear_type)
    config.weight = weight

    if hasattr(module, "bias") and module.bias is not None:
        config.bias = module.bias

    config.activation_scaling_factor = get_activation_scaling_factor(module)
    config.weights_scaling_factor = get_weight_scaling_factor(module)
    config.weights_scaling_factor_2 = get_weight_scaling_factor_2(module)
    config.prequant_scaling_factor = get_prequant_scaling_factor(module)
    config.awq_block_size = get_weight_block_size(module)
    config.quantization = get_quantization_format(module)
    return config


def build_fused_linear_config(modules: list[nn.Module], linear_type: str) -> LinearConfig:
    """Returns a fused linear config from a list of modules."""
    assert linear_type == LINEAR_COLUMN, "Only support column fuse"

    preprocess_linear_fusion(modules)

    config = build_linear_config(modules[0], linear_type=linear_type)
    config.weight = torch.cat([module.weight for module in modules], dim=0)

    if config.weights_scaling_factor is not None and config.weights_scaling_factor.numel() != 1:
        config.weights_scaling_factor = torch.cat(
            [get_weight_scaling_factor(module) for module in modules], dim=0
        )

    return config


def build_attention_config(
    module: nn.Module,
    model_metadata_config,
    ext_config: DecoderLayerConfig = None,
    tp_size: int = 1,
) -> AttentionConfig:
    """Builds the attention config from the module."""
    assert is_attention(module)

    config = AttentionConfig()
    if hasattr(module, "rotary_dim"):
        config.rotary_dim = module.rotary_dim
    if hasattr(module, "clip_qkv"):
        config.clip_qkv = module.clip_qkv

    qkv_modules = []
    q = None
    k = None
    v = None
    for name, layer in module.named_children():
        if is_linear(layer):
            if _is_qkv(name):
                qkv_modules.append(layer)
            elif "q" in name:
                q = layer
            elif "k" in name:
                k = layer
            elif "v" in name:
                v = layer
            else:
                # The dense layer
                config.dense = build_linear_config(layer, LINEAR_ROW)
        elif is_layernorm(layer):
            if "q" in name.lower():
                config.q_layernorm = build_layernorm_config(layer)
            elif "k" in name.lower():
                config.k_layernorm = build_layernorm_config(layer)
            else:
                raise NotImplementedError(f"{name}: {layer} not recognized")
        elif (
            "model_type" in model_metadata_config
            and model_metadata_config["model_type"] == "t5"
            and not isinstance(layer, TensorQuantizer)
        ):
            config.rel_attn_table = RelativeAttentionTableConfig(weight=layer.weight.T)

    if not qkv_modules:
        assert q
        assert k
        assert v
        qkv_modules = [q, k, v]
        for layer in qkv_modules:
            # Add the missing zero bias for Whisper model for export purpose
            if layer.bias is None and q.bias is not None:
                layer.bias = torch.nn.Parameter(
                    torch.zeros(layer.weight.size(1), device=layer.weight.device),
                    requires_grad=True,
                )
                print("Add missing zero bias for qkv modules for export purpose")

    config.qkv = build_qkv(qkv_modules, model_metadata_config, ext_config, tp_size=tp_size)

    config.k_cache_scaling_factor, config.v_cache_scaling_factor = get_kv_cache_scaling_factor(
        module
    )
    config.k_cache_bias, config.v_cache_bias = get_kv_cache_bias(module)

    if config.k_cache_scaling_factor is not None:
        assert config.v_cache_scaling_factor is not None
        config.kv_cache_dtype = get_kv_cache_dtype(module)

    return config


def _is_qkv(name) -> bool:
    return all(k in name for k in ["q", "k", "v"]) or "W_pack" in name or "c_attn" in name


def _get_hidden_act(act_func) -> str:
    """Returns the name of the hidden activation function based on ACT2FN."""
    if isinstance(act_func, str):
        return act_func

    # Falcon activation, "nn.GELU" is equivalent to "gelu" in ACT2FN
    if isinstance(act_func, nn.GELU):
        return "gelu"

    if hasattr(act_func, "func") and act_func.func == nn.functional.gelu:
        return "gelu"

    for name, func in ACT2FN.items():
        # TRT LLM uses "squared-relu" activation keyword.
        if name == "relu2":
            name = "squared-relu"
        if isinstance(func, tuple):
            if isinstance(act_func, func[0]):
                return name
        elif isinstance(act_func, func):
            return name

    return act_func.__name__


def build_mlp_config(
    module: nn.Module,
    decoder_type,
    hidden_act: str | None = None,
    merge_gate_fc: bool = False,
) -> MLPConfig:
    """Builds the MLP config for the module."""
    assert is_mlp(module)

    config = MLPConfig(merge_gate_fc=merge_gate_fc)

    def _split_gate_from_fc(decoder_type, module, fc_name, fc_layer):
        if (
            "ColumnParallelLinear" in type(fc_layer).__name__
            and hasattr(module.config, "gated_linear_unit")
            and module.config.gated_linear_unit
        ):
            return True

        if decoder_type != "gpt":
            return False

        return bool("dense_h_to_4h" in fc_name and "dense_h_to_4h_2" not in fc_name)

    # TODO: We may want to refactor these keywords/model mapping
    fc_keywords = {
        "c_fc",  # gpt2
        "fc_in",  # gptj
        "gate_proj",  # llama, baichuan, recurrentgemma, deepseek
        "dense_h_to_4h",  # falcon, chatglm, bloom
        "linear_fc1",
        "w2",  # qwen
        "fc1",  # phi, gemma, whisper
        "gate_up_proj",  # phi
        "wi_0",  # t5
        "wi",  # t5
        "c_fc_0",  # exaone
    }
    proj_keywords = {
        "c_proj",  # gpt2, qwen, exaone
        "fc_out",  # gptj
        "dense_4h_to_h",  # falcon, chatglm, bloom
        "4h_to_h",
        "down_proj",  # llama, baichuan, mpt, phi, recurrentgemma, nemotron, deepseek
        "linear_fc2",
        "proj",
        "fc2",  # phi, gemma, whisper
        "wo",  # t5
    }
    gate_keywords = {
        "up_proj",  # llama, baichuan, recurrentgemma, deepseek
        "dense_h_to_4h_2",
        "w1",  # qwen
        "wi_1",  # t5
        "c_fc_1",  # exaone
    }

    # Arctic (llama-based MoE, decoder_type is "llama") has MLP keyword conflicts with Qwen
    # Arctic's residual MLP use w1 for fc, w2 for proj, w3 for gate
    if type(module).__name__ in ["ArcticMLP", "InternLM2MLP"]:
        fc_keywords.discard("w2")
        gate_keywords.discard("w1")
        fc_keywords.add("w1")
        proj_keywords.add("w2")
        gate_keywords.add("w3")

    if decoder_type == "mpt":
        fc_keywords.add("up_proj")
        gate_keywords.discard("up_proj")

    if type(module).__name__ in [
        "TLGv4MLP",
        "Phi3SmallMLP",
        "NemotronMLP",
    ]:  # for TLGv4ForCausalLM
        fc_keywords.add("up_proj")
        gate_keywords.discard("up_proj")

    fc_linear: nn.Module = None
    gate_linear: nn.Module = None
    proj_linear: nn.Module = None
    for name, layer in module.named_children():
        if is_linear(layer):
            if any(keyword == name for keyword in fc_keywords):
                fc_linear = layer
            elif any(keyword == name for keyword in gate_keywords):
                gate_linear = layer
            elif any(keyword == name for keyword in proj_keywords):
                proj_linear = layer

    # TensorRT-LLM may choose to merge gate and fc during engine building.
    if (
        gate_linear is not None
        and fc_linear is not None
        and (
            merge_gate_fc
            or get_quantization_format(module) in [QUANTIZATION_FP8, QUANTIZATION_NVFP4]
        )
    ):
        preprocess_linear_fusion([fc_linear, gate_linear])

    if fc_linear is not None:
        weight_quantizer = None
        if hasattr(fc_linear, "weight_quantizer"):
            weight_quantizer = fc_linear.weight_quantizer
            if isinstance(weight_quantizer, SequentialQuantizer):
                weight_quantizer = weight_quantizer[0]

        # swap fused fc and gate
        if decoder_type in ["chatglm", "phi3"]:
            weights = torch.chunk(fc_linear.weight, 2, dim=0)
            weights = (weights[1], weights[0])
            fc_linear.weight.data = torch.cat(weights, dim=0)

            if (
                weight_quantizer is not None
                and weight_quantizer.is_enabled
                and weight_quantizer.amax.numel() != 1
            ):
                amax_chunks = torch.chunk(weight_quantizer.amax, 2, dim=0)
                weight_quantizer.amax = torch.cat([amax_chunks[1], amax_chunks[0]], dim=0)

        split_gate = _split_gate_from_fc(decoder_type, module, name, fc_linear)
        if split_gate:
            # We have to split the gate from the fc
            weights = torch.chunk(fc_linear.weight, 2, dim=0)
            weight_scaling_factor = get_weight_scaling_factor(fc_linear)

            weight_scaling_factors = None

            if weight_scaling_factor is not None and weight_scaling_factor.numel() != 1:
                # for Int8 SQ case, we split the weight scaling factor into two parts.
                weight_scaling_factors = torch.chunk(weight_scaling_factor, 2, dim=0)

            config.fc = build_linear_config(fc_linear, LINEAR_COLUMN)
            config.gate = build_linear_config(fc_linear, LINEAR_COLUMN)
            config.fc.weight = weights[0]
            config.gate.weight = weights[1]
            if weight_scaling_factors is not None:
                config.fc.weights_scaling_factor = weight_scaling_factors[0]
                config.gate.weights_scaling_factor = weight_scaling_factors[1]
        else:
            config.fc = build_linear_config(fc_linear, LINEAR_COLUMN)

    if proj_linear is not None:
        config.proj = build_linear_config(proj_linear, LINEAR_ROW)

    if gate_linear is not None:
        config.gate = build_linear_config(gate_linear, LINEAR_COLUMN)

    assert config.proj is not None and config.fc is not None, "proj or fc can not be found"

    # Override hidden_act based on decoder_type
    if decoder_type in ["bloom", "glm"]:
        hidden_act = "gelu"
    if decoder_type == "phi3":
        hidden_act = "swiglu"

    if hidden_act is None:
        if hasattr(module, "activation"):
            hidden_act = module.activation
        elif hasattr(module, "activation_func"):
            # MCore activation_func can be swiglu (gated silu) or squared_relu.
            hidden_act = module.activation_func.__name__.replace("_", "-")
            if hidden_act in ["glu", "silu"]:
                hidden_act = "swiglu" if decoder_type == "gpt" else "silu"
        else:
            for act in ["act", "act_fn", "activation_fn"]:
                if hasattr(module, act):
                    hidden_act = _get_hidden_act(getattr(module, act)).split("_")[0]
                    break

    if hidden_act is None and decoder_type == "qwen":
        # for v1 qwen versions, activation is not explicitly defined as part of the layer's implementation
        hidden_act = "silu"

    if hidden_act is None:
        raise NotImplementedError(f"{module} not supported.")

    config.hidden_act = hidden_act
    return config


def _get_expert_attr(experts: nn.Module, export_id: int, linear_name: str):
    # Generic expert attribute accessor.
    # Works for most MoE models that store experts as a list/ModuleList where
    # each expert has linear layers as direct attributes:
    # experts[0].w1, experts[0].w2, experts[0].w3  (Mixtral)
    # experts[0].gate_proj, experts[0].down_proj, experts[0].up_proj  (Qwen)
    # experts[0].linear_fc1, experts[0].linear_fc2  (Llama MCore)
    return getattr(experts[export_id], linear_name)


def _get_dbrx_expert(experts: nn.Module, export_id: int, linear_name: str):
    # DBRX experts layout is:
    # experts:
    #   w1[0]
    #   w1[1]
    #   ...
    #   w2[0]
    #   w2[1]
    #   ...
    #   v1[0]
    #   v1[1]
    #   ...
    return getattr(experts, linear_name)[export_id]


def _build_stacked_linear(experts: nn.Module, module_name, linear_type, num_experts, expert_getter):
    config = LinearConfig(linear_type=linear_type)

    # weights
    config.weight = torch.stack(
        [expert_getter(experts, i, module_name).weight for i in range(num_experts)]
    )

    # bias
    first_module = expert_getter(experts, 0, module_name)
    if hasattr(first_module, "bias") and first_module.bias is not None:
        raise ValueError("Unexpected bias tensors inside MOE modules.")

    # scaling factors
    def get_stacked_scaling_factors(experts, get_function, module_name):
        expert_0_scaling_factor = get_function(expert_getter(experts, 0, module_name))

        if expert_0_scaling_factor is None:
            return None

        dtype = expert_0_scaling_factor.dtype
        scaling_factors = [
            get_function(expert_getter(experts, i, module_name)) for i in range(num_experts)
        ]

        if dtype == torch.float8_e4m3fn:
            scaling_factors = [sf.view(torch.uint8) for sf in scaling_factors]
            return torch.stack(scaling_factors).view(torch.float8_e4m3fn)

        return torch.stack(scaling_factors)

    config.activation_scaling_factor = get_stacked_scaling_factors(
        experts, get_activation_scaling_factor, module_name
    )
    # The moe plugin only supports a single activation scaling factor for all experts
    if config.activation_scaling_factor is not None:
        config.activation_scaling_factor = config.activation_scaling_factor.max().unsqueeze(0)
    config.weights_scaling_factor = get_stacked_scaling_factors(
        experts, get_weight_scaling_factor, module_name
    )
    config.weights_scaling_factor_2 = get_stacked_scaling_factors(
        experts, get_weight_scaling_factor_2, module_name
    )
    config.prequant_scaling_factor = get_stacked_scaling_factors(
        experts, get_prequant_scaling_factor, module_name
    )
    config.awq_block_size = get_weight_block_size(expert_getter(experts, 0, module_name))
    config.quantization = get_quantization_format(experts)

    return config


def get_expert_linear_names(module: nn.Module) -> list[str]:
    """Get the list of linear names for the experts."""

    def module_match_name_list(module, name_list):
        """Check if the module name matches any of the names in the list.

        e.g. module_match_name_list(QuantQwen3MoeSparseMoeBlock, ['Qwen3MoeSparseMoeBlock']) -> True

        """
        return any(name.lower() in type(module).__name__.lower() for name in name_list)

    if module_match_name_list(
        module,
        [
            "Qwen2MoeSparseMoeBlock",
            "Qwen3MoeSparseMoeBlock",
            "Qwen3NextSparseMoeBlock",
            "DeepseekMoE",
        ],
    ):
        return ["gate_proj", "down_proj", "up_proj"]
    elif module_match_name_list(module, ["MixtralMoeSparseMoeBlock"]):
        return ["linear_fc1", "linear_fc2"]
    elif module_match_name_list(module, ["DBRXMoeSparseMoeBlock"]):
        return ["w1_linear", "w2_linear", "v1_linear"]
    elif module_match_name_list(module, ["GptOssMoE"]):
        # GPT-OSS MoE modules use gate_up_proj and down_proj
        return ["gate_up_proj", "down_proj"]
    else:
        # assuming w1, w2, w3 by default
        return ["w1", "w2", "w3"]


def set_expert_quantizer_amax(
    modules: nn.Module | list[nn.Module],
    quantizer_attrs: str | list[str] | None = None,
    fallback_value: float = 0.5,
    device: torch.device | None = None,
) -> list[nn.Module]:
    """Set amax values for expert quantizers using smart fallback logic.

    Uses smart fallback logic:

    1. Use max from existing quantizers in current batch (best - direct from calibration)
    2. If no existing values found, then:
       - For weight quantizers: calculate from weight statistics
       - For input quantizers: use max from other experts, fallback if none found
    3. Use fallback value as last resort

    This ensures we always have semantically appropriate amax values for export.

    Args:
        modules: Single module or list of modules containing quantizers
        quantizer_attrs: Specific quantizer attributes to handle.
            If None, defaults to ["input_quantizer"] for backward compatibility.
        fallback_value: Final fallback value when other methods fail (default: 0.5)
        device: Target device for tensors (auto-detected if None)

    Returns:
        uncalibrated_modules: a list of uncalibrated experts
    """
    import warnings

    # Normalize inputs
    if not isinstance(modules, list):
        modules = [modules]

    if quantizer_attrs is None:
        quantizer_attrs = ["input_quantizer"]
    elif isinstance(quantizer_attrs, str):
        quantizer_attrs = [quantizer_attrs]

    uncalibrated_modules = []

    # Determine target device if not provided
    if device is None:
        first_module = next(iter(modules))
        if hasattr(first_module, "weight"):
            target_device = first_module.weight.device
        else:
            target_device = torch.device("cpu")
    else:
        target_device = device

    # Collect all valid quantizers
    all_quantizers = []

    for module in modules:
        for attr_name in quantizer_attrs:
            if hasattr(module, attr_name):
                quantizer = getattr(module, attr_name)
                if (
                    quantizer is not None
                    and hasattr(quantizer, "is_enabled")
                    and quantizer.is_enabled
                ):
                    all_quantizers.append((module, attr_name, quantizer))

    target_amax = None

    # Collect ANY existing amax values from current batch (most direct source)
    valid_amax_values = []
    for _, attr_name, quantizer in all_quantizers:
        existing_amax = getattr(quantizer, "amax", None)
        if existing_amax is not None:
            # Convert to tensor and add to collection
            if isinstance(existing_amax, torch.Tensor):
                valid_amax_values.append(existing_amax.to(target_device))
            else:
                valid_amax_values.append(
                    torch.tensor(existing_amax, dtype=torch.float32, device=target_device)
                )

    # Use existing values from current batch if any found
    if len(valid_amax_values) > 0:
        target_amax = torch.max(torch.stack(valid_amax_values))

    # If no existing values in current batch, apply type-specific fallback logic
    elif target_amax is None:
        has_input_quantizers = any("input_quantizer" in attr for _, attr, _ in all_quantizers)
        has_weight_quantizers = any("weight_quantizer" in attr for _, attr, _ in all_quantizers)

        if has_weight_quantizers and not has_input_quantizers:
            # For weight quantizers: calculate from weight statistics
            weight_amax_values = []
            for module, _, _ in all_quantizers:
                # Try to find a weight tensor in the module
                weight_tensor = None
                for weight_attr in ["weight", "gate_up_proj", "down_proj"]:
                    if hasattr(module, weight_attr):
                        weight_tensor = getattr(module, weight_attr)
                        break

                if weight_tensor is not None:
                    weight_amax_values.append(torch.max(torch.abs(weight_tensor)))

            if weight_amax_values:
                target_amax = torch.max(torch.stack(weight_amax_values)).item()
        elif has_input_quantizers:
            # For input quantizers: ideally search other experts for existing input amax values
            # TODO: Implement broader expert search - currently function only has access to current batch
            # For now, this will fall through to fallback value
            pass

    # Final fallback
    if target_amax is None:
        target_amax = fallback_value
        has_input_quantizers = any("input_quantizer" in attr for _, attr, _ in all_quantizers)

    # Apply target amax to quantizers that need it
    for module, attr_name, quantizer in all_quantizers:
        # Check if quantizer needs amax (use property for consistency)
        needs_amax = getattr(quantizer, "amax", None) is None

        # Skip dynamic quantizers for input quantizers
        if "input_quantizer" in attr_name and getattr(quantizer, "_dynamic", False):
            needs_amax = False

        if needs_amax:
            # Create tensor with appropriate value (using function-wide target_device)
            if isinstance(target_amax, torch.Tensor):
                amax_tensor = target_amax.clone().to(dtype=torch.float32, device=target_device)
            else:
                amax_tensor = torch.tensor(target_amax, dtype=torch.float32, device=target_device)

            # Set amax value using property for proper validation and tensor handling
            quantizer.amax = amax_tensor

            uncalibrated_modules.append(module)
            amax_val = amax_tensor.item() if isinstance(amax_tensor, torch.Tensor) else amax_tensor

            if len(valid_amax_values) > 0:
                warnings.warn(
                    f"Missing amax value for {attr_name} in {type(module).__name__}. "
                    f"Setting it to {amax_val:.6f} (max from existing quantizers in current batch). "
                    f"This typically occurs when certain experts are not activated during calibration."
                )
            elif amax_val != fallback_value and "input_quantizer" not in attr_name:
                warnings.warn(
                    f"Missing amax value for {attr_name} in {type(module).__name__}. "
                    f"Setting it to {amax_val:.6f} (computed from weights)."
                )

    return uncalibrated_modules


def build_stacked_experts(
    experts: nn.Module,
    linear_names: list[str],
    num_experts,
    expert_getter,
):
    """Builds the experts_weight_1 and experts_weight_2 configs for the experts."""
    # Resmooth all experts
    preprocess_linear_fusion(
        [expert_getter(experts, i, linear_names[1]) for i in range(num_experts)],
        resmooth_only=True,
    )

    w1w3_names = [linear_names[0]]
    if len(linear_names) == 3:
        w1w3_names.append(linear_names[2])

    preprocess_linear_fusion(
        [
            expert_getter(experts, i, module_name)
            for i in range(num_experts)
            for module_name in w1w3_names
        ],
        resmooth_only=True,
    )

    # Set amax to 0 for uncalibrated experts (calculate max across all experts later)
    expert_modules = [
        expert_getter(experts, i, module_name)
        for module_name in linear_names
        for i in range(num_experts)
    ]
    uncalibrated_experts = set_expert_quantizer_amax(
        modules=expert_modules,
        quantizer_attrs=["input_quantizer"],
        fallback_value=0,
    )

    try:
        # Pre-fuse W1 and W3
        if len(linear_names) == 3:
            for i in range(num_experts):
                preprocess_linear_fusion(
                    [
                        expert_getter(experts, i, module_name)
                        for module_name in [linear_names[0], linear_names[2]]
                    ]
                )

        experts_weight_1 = _build_stacked_linear(
            experts, linear_names[0], LINEAR_COLUMN, num_experts, expert_getter
        )
        experts_weight_2 = _build_stacked_linear(
            experts, linear_names[1], LINEAR_ROW, num_experts, expert_getter
        )

        if len(linear_names) > 2:
            # Only for HF model, as Mcore model only has two fc layers in MoE
            experts_weight_3 = _build_stacked_linear(
                experts, linear_names[2], LINEAR_COLUMN, num_experts, expert_getter
            )

            # Concat w1 and w3 into w1
            experts_weight_1.weight = torch.concat(
                [experts_weight_3.weight, experts_weight_1.weight], dim=-2
            )

            if experts_weight_1.weights_scaling_factor is not None:
                if experts_weight_1.weights_scaling_factor.numel() != num_experts:
                    experts_weight_1.weights_scaling_factor = view_as_float8_e4m3fn_if_needed(
                        torch.concat(
                            [
                                view_as_uint8_if_needed(experts_weight_3.weights_scaling_factor),
                                view_as_uint8_if_needed(experts_weight_1.weights_scaling_factor),
                            ],
                            dim=-2,
                        )
                    )
                else:
                    assert torch.equal(
                        experts_weight_1.weights_scaling_factor,
                        experts_weight_3.weights_scaling_factor,
                    )

            if experts_weight_1.activation_scaling_factor is not None:
                assert torch.equal(
                    experts_weight_1.activation_scaling_factor,
                    experts_weight_3.activation_scaling_factor,
                )

            if experts_weight_1.weights_scaling_factor_2 is not None:
                assert torch.equal(
                    experts_weight_1.weights_scaling_factor_2,
                    experts_weight_3.weights_scaling_factor_2,
                )

        # Explicitly move weight to CPU to reduce GPU memory requirement.
        experts_weight_1.weight = experts_weight_1.weight.cpu()
        experts_weight_2.weight = experts_weight_2.weight.cpu()
        return experts_weight_1, experts_weight_2

    finally:
        # Cleanup: restore original amax values (same logic as old context manager)
        if uncalibrated_experts:
            for module in uncalibrated_experts:
                if hasattr(module.input_quantizer, "_amax"):
                    delattr(module.input_quantizer, "_amax")


def build_moe_config(module: nn.Module, decoder_type) -> MOEConfig:
    """Builds the MOE config for the module."""
    assert is_moe(module)
    assert decoder_type in ["llama", "dbrx", "phi3", "deepseek", "qwen"]

    config = MOEConfig()

    # Router: TRT-LLM uses fp32 for router to keep precision
    if decoder_type in ["llama", "phi3"]:
        if has_mcore and isinstance(module, MegatronModule):
            # For Mcore model
            config.router = build_linear_config(module.router, LINEAR_ROW)
        else:
            # For huggingface model
            config.router = build_linear_config(module.gate, LINEAR_ROW)
    elif decoder_type == "dbrx":
        config.router = build_linear_config(module.router.layer, LINEAR_ROW)
    elif decoder_type == "deepseek":
        # Build a linear on the fly.
        router_linear = nn.Linear(module.gate.gating_dim, module.gate.n_routed_experts, bias=None)
        router_linear.weight = module.gate.weight
        config.router = build_linear_config(router_linear, LINEAR_ROW)
        # Add shared config
        preprocess_linear_fusion([module.shared_experts.gate_proj, module.shared_experts.up_proj])
        config.shared_expert = build_mlp_config(
            module.shared_experts, decoder_type, merge_gate_fc=True
        )
    elif decoder_type == "qwen":
        config.router = build_linear_config(module.gate, LINEAR_ROW)
        # Qwen3 doesn't have shared expert
        if hasattr(module, "shared_expert"):
            preprocess_linear_fusion([module.shared_expert.gate_proj, module.shared_expert.up_proj])
            config.shared_expert = build_mlp_config(
                module.shared_expert, decoder_type, merge_gate_fc=True
            )
            config.shared_expert_gate = build_linear_config(module.shared_expert_gate, LINEAR_ROW)
            config.shared_expert_gate.tp = False
    else:
        raise NotImplementedError(f"{decoder_type} not supported")

    config.router.weight = config.router.weight.type(torch.float)
    config.router.tp = False

    # Experts
    experts = ExpertConfig()
    if decoder_type in ["llama", "phi3"]:
        if has_mcore and isinstance(module, MegatronModule):
            # For Mcore model
            experts.fc, experts.proj = build_stacked_experts(
                module.experts.local_experts,
                ["linear_fc1", "linear_fc2"],
                len(module.experts.local_experts),
                _get_expert_attr,
            )
            # For Mcore model, experts.fc.weight needs to be flipped along axis = 1
            mid_point = experts.fc.weight.shape[1] // 2
            experts.fc.weight = torch.cat(
                [
                    experts.fc.weight[:, mid_point:, :],
                    experts.fc.weight[:, :mid_point, :],
                ],
                dim=1,
            )
        else:
            # For Huggingface model
            experts.fc, experts.proj = build_stacked_experts(
                module.experts,
                ["w1", "w2", "w3"],
                len(module.experts),
                _get_expert_attr,
            )
    elif decoder_type == "dbrx":
        experts.fc, experts.proj = build_stacked_experts(
            module.experts.mlp,
            ["w1_linear", "w2_linear", "v1_linear"],
            len(module.experts.mlp.w1_linear),
            _get_dbrx_expert,
        )
    elif decoder_type in ["deepseek", "qwen"]:
        experts.fc, experts.proj = build_stacked_experts(
            module.experts,
            ["gate_proj", "down_proj", "up_proj"],
            len(module.experts),
            _get_expert_attr,
        )
    else:
        raise NotImplementedError(f"{decoder_type} not supported")

    config.experts = experts

    # activation for mixtral and dbrx
    config.hidden_act = "swiglu"

    return config


def build_conv_config(module: nn.Module) -> ConvConfig:
    """Builds the conv config for this module."""
    return ConvConfig(
        weight=module.weight.unsqueeze(dim=-1),
        bias=module.bias,
    )


def is_recurrent(module: nn.Module) -> bool:
    """Returns whether the module is a recurrent layer."""
    module_name = type(module).__name__
    return "RecurrentBlock" in module_name


def build_recurrent_config(module: nn.Module):
    """Builds the recurrent config for this module."""
    assert is_recurrent(module)

    config = RecurrentConfig()
    config.linear_y = build_linear_config(module.linear_y, linear_type=LINEAR_COLUMN)
    # Separate bias from linear_y to y_bias as guided by TRT LLM.
    config.y_bias = config.linear_y.bias
    config.linear_y.bias = None

    config.linear_x = build_linear_config(module.linear_x, linear_type=LINEAR_COLUMN)
    config.linear_out = build_linear_config(module.linear_out, linear_type=LINEAR_ROW)
    config.conv1d = build_conv_config(module.conv_1d)
    config.rg_lru = RgLruConfig()

    config.rg_lru.input_gate = LinearConfig(
        linear_type=LINEAR_GROUP,
        weight=module.rg_lru.input_gate_weight,
        bias=module.rg_lru.input_gate_bias,
        quantization=get_quantization_format(module),
    )

    config.rg_lru.recurrent_gate = LinearConfig(
        linear_type=LINEAR_GROUP,
        weight=module.rg_lru.recurrent_gate_weight,
        bias=module.rg_lru.recurrent_gate_bias,
        quantization=get_quantization_format(module),
    )

    config.rg_lru.recurrent_param = module.rg_lru.recurrent_param
    return config


def _set_layer_config_from_metaconfig(layer_config, metaconfig):
    for keys, name in HF_CONFIG_MAP:
        for k in keys:
            if k in metaconfig:
                setattr(layer_config, name, metaconfig[k])
    for keys, name in MCORE_CONFIG_MAP:
        for k in keys:
            if k in metaconfig:
                setattr(layer_config, name, metaconfig[k])

    # MCore / NeMo use "rope" as an alias for "rope_gpt_neox"
    if layer_config.position_embedding_type == "rope":
        layer_config.position_embedding_type = "rope_gpt_neox"

    # 2048 is the default TRT LLM max_position_embeddings
    if layer_config.max_position_embeddings == 0:
        layer_config.max_position_embeddings = 2048

    for k in ["_name_or_path"]:
        if k in metaconfig:
            model_path_name = metaconfig[k].replace("-", "_")
            chatglm_version = ""

            for n in ["glm_4", "chatglm3", "chatglm2", "chatglm"]:
                if n in model_path_name:
                    chatglm_version = n.replace("_", "")  # glm_4 -> glm
                    break

            layer_config.chatglm_version = chatglm_version

    # For Falcon variants do not provide new_decoder_architecture, but we can infer it from the model_type
    if "model_type" in metaconfig:
        if metaconfig["model_type"] == "RefinedWeb":
            # Case 1. Falcon-40B / Falcon-40B-instruct
            # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
            layer_config.new_decoder_architecture = True
        elif metaconfig["model_type"] == "RefinedWebModel":
            # Case 2. Falcon-7B / Falcon-7B-instruct
            # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
            layer_config.new_decoder_architecture = False

    # For Falcon variants, they might not specify the number of kv heads with MQA models, e.g., 7b
    if (
        not layer_config.new_decoder_architecture
        and "multi_query" in metaconfig
        and metaconfig["multi_query"]
    ):
        layer_config.num_kv_heads = 1

    # For Phi3
    for k in ["rope_scaling"]:
        if k in metaconfig and metaconfig[k] is not None:
            for k2 in ["short_factor"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_scaling_short_factors = metaconfig[k][k2]
            for k2 in ["long_factor"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_scaling_long_factors = metaconfig[k][k2]
            # For Phi3 small
            for k2 in ["short_mscale"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_short_mscale = metaconfig[k][k2]
            for k2 in ["long_mscale"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_long_mscale = metaconfig[k][k2]


def build_decoder_config(
    module: nn.Module,
    model_metadata_config,
    decoder_type: str,
    tp_size: int = 1,
) -> DecoderLayerConfig:
    """Builds the full decoder config from the module."""
    quantization = get_quantization_format(module)
    config = DecoderLayerConfig(decoder_type=decoder_type)
    # Supporting different attention layer config in MCoreGPTModel. If per layer config
    # exists, override the global config.
    if hasattr(module, "self_attention") and hasattr(module.self_attention, "config"):
        if hasattr(module.self_attention.config, "num_attention_heads"):
            config.num_attention_heads = module.self_attention.config.num_attention_heads
        if hasattr(module.self_attention.config, "kv_channels"):
            config.attention_head_size = module.self_attention.config.kv_channels
        if hasattr(module.self_attention.config, "num_query_groups"):
            config.num_kv_heads = module.self_attention.config.num_query_groups

    # Set all config fields in modelopt from HF config
    _set_layer_config_from_metaconfig(config, model_metadata_config)

    if type(module).__name__ == "DbrxBlock":
        # Flatten DBRX attention and ffn
        module_layers = {}
        module_layers.update(dict(getattr(module, "norm_attn_norm").named_children()))
        module_layers.update({"ffn": module.ffn})
    elif decoder_type in ["t5"]:
        # Combine two modules (T5LayerSelfAttention, T5LayerFF) / three modules
        # ((T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF)) of T5 model
        # (depending on whether it's encoder / decoder) into one decoder layer
        combined_module = nn.ModuleList()
        for sub_module in module:
            for layer in sub_module.children():
                combined_module.append(layer)
        module_layers = dict(combined_module.named_children())
    elif decoder_type in ["bart", "whisper"]:
        if decoder_type == "whisper":
            # Add max_position_embeddings for Whisper model
            if model_metadata_config.get("enc_dec") == "enc":
                config.max_position_embeddings = module.self_attn.config.max_source_positions
            else:
                config.max_position_embeddings = module.self_attn.config.max_target_positions
        # BartEncoderLayer, BartDecoderLayer, WhisperEncoderLayer, WhisperDecoderLayer
        # have MLP component with no Module wrapper.
        # Create a dummy module so that is_mlp may catch it.
        encdec_mlp_submodule_names = ["fc1", "fc2", "activation_fn", "activation_fn"]
        module_layers = dict(module.named_children())

        class EncDecMLP(nn.Module):
            def __init__(self):
                super().__init__()

        encdec_mlp_module = EncDecMLP()
        for submodule_name in encdec_mlp_submodule_names:
            if submodule_name in module_layers:
                setattr(encdec_mlp_module, submodule_name, getattr(module, submodule_name))
                module_layers.pop(submodule_name)
        module_layers.update({"MLP": encdec_mlp_module})
    else:
        module_layers = dict(module.named_children())
        if decoder_type in ["exaone"]:
            module_layers.update({"attn": module_layers["attn"].attention})

    for name, layer in module_layers.items():
        # We assume input_layernorm should be before the post_layernorm in decoder block layout,
        # and residual_layernorm could be after post_layernorm
        if is_layernorm(layer):
            layernorm_config = build_layernorm_config(layer)
            # Special attributes naming for Encoder-Decoder models
            if model_type_is_enc_dec(decoder_type):
                _update_encoder_decoder_layernorm_config(
                    model_metadata_config, config, layernorm_config
                )
            # For all decoder only models
            elif name in ["ln_mlp"]:
                config.mlp_layernorm = layernorm_config
            elif (
                config.decoder_type in ["gemma2", "gemma3"]
            ) and "post_attention_layernorm" in name:
                config.post_layernorm = layernorm_config
            elif (
                config.decoder_type in ["gemma2", "gemma3"]
            ) and "pre_feedforward_layernorm" in name:
                config.pre_feedforward_layernorm = layernorm_config
            elif (
                config.decoder_type in ["gemma2", "gemma3"]
            ) and "post_feedforward_layernorm" in name:
                config.post_feedforward_layernorm = layernorm_config
            elif config.input_layernorm is None:
                config.input_layernorm = layernorm_config
            elif config.post_layernorm is None:
                config.post_layernorm = layernorm_config
            else:
                assert model_metadata_config["parallel_attn_mlp_res"], (
                    "Unexpected layernorm in a layer"
                )
                config.residual_layernorm = layernorm_config

        elif is_attention(layer):
            # For models where a linear may replace the attention/MLP module (e.g. Deci models)
            if is_linear(layer):
                config.attn_replacing_linear = build_linear_config(layer.linear_attn, LINEAR_COLUMN)
            else:
                if decoder_type in ["bloom", "falcon", "phi3small", "internlm"]:
                    model_metadata_config["head_is_first_dim"] = True
                attention_config = build_attention_config(
                    layer, model_metadata_config, config, tp_size=tp_size
                )
                # For decoder of Encoder-Decoder model with self, cross attention
                if (
                    model_type_is_enc_dec(decoder_type)
                    and model_metadata_config["enc_dec"] == "dec"
                ):
                    # We assume self_attention should be before the cross_attention in decoder block layout
                    if config.self_attention is None:
                        config.self_attention = attention_config
                    else:
                        config.cross_attention = attention_config
                elif decoder_type == "mllama":
                    if "cross" in type(layer).__name__.lower():
                        config.cross_attention = attention_config
                    else:
                        config.self_attention = attention_config
                else:
                    config.attention = attention_config
                    if decoder_type == "gemma3":
                        # Gemma3 has q_norm and k_norm within self_attn
                        q_layernorm_config = build_layernorm_config(layer.q_norm)
                        k_layernorm_config = build_layernorm_config(layer.k_norm)
                        config.attention.q_layernorm = q_layernorm_config
                        config.attention.k_layernorm = k_layernorm_config

        elif is_moe(layer):
            if quantization not in [QUANTIZATION_NONE, QUANTIZATION_FP8]:
                warn(f"TensorRT-LLM may not support MOE quantization {quantization}")
            config.mlp = build_moe_config(layer, decoder_type)

        elif is_recurrent(layer):
            config.recurrent = build_recurrent_config(layer)

        # We assume MoE layer should be before the residual MLP layer in decoder block layout,
        # so MLP layer after a MoE layer is treated as a residual MLP layer
        elif is_mlp(layer):
            if config.mlp is None:
                # For models where a linear may replace the attention/MLP module (e.g. Deci models)
                if is_linear(layer):
                    config.mlp_replacing_linear = build_linear_config(
                        layer.linear_mlp, LINEAR_COLUMN
                    )
                else:
                    config.mlp = build_mlp_config(
                        layer,
                        decoder_type,
                        hidden_act=model_metadata_config.get("hidden_act", None),
                    )
            else:
                assert model_metadata_config["parallel_attn_mlp_res"], "Unexpected mlp in a layer"
                config.residual_mlp = build_mlp_config(
                    layer,
                    decoder_type,
                    hidden_act=model_metadata_config.get("hidden_act", None),
                )

    config.gate_attn = getattr(module, "cross_attn_attn_gate", None)
    config.gate_ffwd = getattr(module, "cross_attn_mlp_gate", None)

    config = _move_input_layernorm_for_noop_attention(config)
    return config


def build_medusa_heads_config(
    model: nn.Module | None,
) -> list[MedusaHeadConfig] | None:
    """Build a list of MedusaHeadConfig if exists.

    Following TensorRT-LLM's Medusa implementation, all Medusa heads (num_medusa_heads) should be
    placed inside a 'torch.nn.ModuleList' with attribute name 'medsua_heads'. A Medusa head composes
    an additional 'lm_head' (vocab_size, hidden_size) and a list (num_medusa_layers) of Medusa layer
    (LinearActConfig). The only supported hidden_act for the layer is 'silu'. All Linear layers are
    column-parallel.
    """
    # medusa_heads: nn.Module | None = None

    def get_medusa_heads(model: nn.Module) -> nn.Module | None:
        """Return the MedusaHead is exists."""
        # MCore GPTModel impl
        if hasattr(model, "medusa_heads"):
            return model.medusa_heads
        return None

    medusa_heads = get_medusa_heads(model)

    if medusa_heads is None:
        return None

    configs = []
    for medusa_head in medusa_heads:
        config = MedusaHeadConfig()
        layer_configs = []
        if isinstance(medusa_head, torch.nn.Sequential):
            # In the HF MedusaModel, nn.Sequencetial is used.
            config.lm_head = build_linear_config(medusa_head[-1], LINEAR_COLUMN)
            for layer in medusa_head[0:-1]:
                layer_config = LinearActConfig()
                layer_config.linear = build_linear_config(layer.linear, LINEAR_COLUMN)
                # NOTE: only silu is supported now.
                layer_config.hidden_act = "silu"
                layer_configs.append(layer_config)

        else:
            # In the Megatron, we have MedusaHead define and lm_head is a submodule.
            config.lm_head = build_linear_config(medusa_head.lm_head, LINEAR_COLUMN)
            for layer in medusa_head.medusa_layers:
                layer_config = LinearActConfig()
                layer_config.linear = build_linear_config(layer.linear, LINEAR_COLUMN)
                # NOTE: only silu is supported now.
                layer_config.hidden_act = "silu"
                layer_configs.append(layer_config)
        config.medusa_layers = layer_configs
        configs.append(config)
    return configs


def _split_fused_qkv_weight_and_scaling(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int | None = None,
    training_tp: int = 1,
    keep_channel_order: bool = False,
):
    """Reorder the qkv weight for splitting QKV weights.

    The shape of the fused QKV weights in HF is different from the shape that
    TRT-LLM requires. In particular, the weight of HF consists of interleaved
    q, k, v head weights, while that of TRT-LLM is contiguous.
        HF     : [q1, k1, v1, ..., qh, kh, vh]
        TRT-LLM: [q1, ..., qh, k1, ..., kh, v1, vh]
    where qi, vi, ki are weight vectors corresponding to attention head i.
    It's similar to multi/grouped query attention cases.
    if keep_channel_order
        HF     : [q1, ..., qh, k1, ..., kh, v1, ..., vh]
        TRT-LLM: [q1, ..., qh, k1, ..., kh, v1, ..., vh]
    """
    # Query types and expected kv heads.
    #  - Conventional MHA: num_heads = num_kv_heads
    #  - Multi-Query Attention: num_kv_heads = 1
    #  - Grouped-Query Attention: num_heads % num_kv_heads = 0
    weight_dim = weight.dim()
    assert 1 <= weight_dim <= 2

    qkv_in = weight.shape[-1] if weight_dim > 1 else 1

    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    assert num_heads % num_kv_heads == 0, (
        f"num_heads({num_heads}) must be divisible by num_kv_heads({num_kv_heads}))."
    )

    # The number of attention heads per group: N q head + 1 k head + 1 v head.
    num_group_heads = num_heads // num_kv_heads + 2
    num_kv_heads_single_tp = max(num_kv_heads // training_tp, 1)
    size_per_head = weight.shape[0] // num_kv_heads_single_tp // num_group_heads

    if keep_channel_order:
        kv_size = num_kv_heads * size_per_head
        q_w = weight[: -2 * kv_size, ...]
        k_w = weight[-2 * kv_size : -1 * kv_size, ...]
        v_w = weight[-1 * kv_size :, ...]
    else:
        # Split Q/K/V weights
        weight = weight.reshape(num_kv_heads_single_tp, num_group_heads, size_per_head, qkv_in)
        q_w = weight[:, :-2, ...]  # (nKV, num_heads // nKV, size_per_head, qkv_in)
        k_w = weight[:, -2:-1, ...]  # (nKV, 1, size_per_head, qkv_in)
        v_w = weight[:, -1:, ...]  # (nKV, 1, size_per_head, qkv_in)

        q_w = q_w.reshape(-1, qkv_in)
        k_w = k_w.reshape(-1, qkv_in)
        v_w = v_w.reshape(-1, qkv_in)

    if weight_dim == 1:
        q_w = q_w.reshape(-1)
        k_w = k_w.reshape(-1)
        v_w = v_w.reshape(-1)

    return q_w, k_w, v_w


def _update_encoder_decoder_layernorm_config(model_metadata_config, config, layernorm_config):
    # For encoder of Encoder-Decoder model
    if model_metadata_config["enc_dec"] == "enc":
        if config.attention_layernorm is None:
            config.attention_layernorm = layernorm_config
        else:
            config.mlp_layernorm = layernorm_config
    # For decoder of Encoder-Decoder model
    elif config.self_attention_layernorm is None:
        config.self_attention_layernorm = layernorm_config
    elif config.cross_attention_layernorm is None:
        config.cross_attention_layernorm = layernorm_config
    else:
        config.mlp_layernorm = layernorm_config


def _move_input_layernorm_for_noop_attention(
    complete_decoder_config: DecoderLayerConfig,
) -> DecoderLayerConfig:
    """For models with decoder blocks that skip the attention or recurrent, the only layer norm is AFTER the MLP.

    We build DecoderLayerConfigs with the assumption that there are both norms. This function fixes
    this assumption and moved the input_layernorm to be a post_layernorm if necessary.
    """
    if all(
        config is None
        for config in [
            complete_decoder_config.attention,
            complete_decoder_config.self_attention,
            complete_decoder_config.cross_attention,
            complete_decoder_config.attn_replacing_linear,
            complete_decoder_config.recurrent,
        ]
    ):
        assert complete_decoder_config.post_layernorm is None, (
            "Should not have 2 layer norms with no attention"
        )
        complete_decoder_config.post_layernorm = complete_decoder_config.input_layernorm
        complete_decoder_config.input_layernorm = None

    return complete_decoder_config


def update_experts_avg_prequant_scale(experts: nn.Module):
    """Registers experts_pre_quant_scale attribute of each expert with average pre_quant_scale amongst experts."""
    """In NVFP4_AWQ and INT4_AWQ all the experts share prequant_scaling_factor. """
    experts_linear_names = get_experts_linear_names(experts)
    if "mixtral" in type(experts).__name__.lower():
        get_func = _get_expert_attr
        num_experts = len(experts.experts)
        experts = experts.experts
    elif "dbrx" in type(experts).__name__.lower():
        get_func = _get_dbrx_expert
        num_experts = len(getattr(experts.experts.mlp, experts_linear_names[0]))
        experts = experts.experts.mlp
    else:
        raise NotImplementedError("MoE model not supported")

    for linear_name in experts_linear_names:
        # Check if pre_quant_scale exists
        assert hasattr(get_func(experts, 0, linear_name).input_quantizer, "pre_quant_scale"), (
            "Layer does not have attribute pre_quant_scale"
        )
        experts_avg_pre_quant_scale = torch.mean(
            torch.stack(
                [
                    get_func(experts, i, linear_name).input_quantizer.pre_quant_scale
                    for i in range(num_experts)
                ]
            ),
            dim=0,
        )
        # Register a new experts_pre_quant_scale attribute of each expert with average pre_quant_scale amongst experts
        for i in range(num_experts):
            get_func(
                experts, i, linear_name
            ).input_quantizer.experts_avg_pre_quant_scale = experts_avg_pre_quant_scale.clone()


def get_experts_linear_names(model: torch.nn.Module):
    """Returns linear layer names based on decoder type for MoE models."""
    if "mixtral" in type(model).__name__.lower():
        return ["w1", "w2", "w3"]
    elif "dbrx" in type(model).__name__.lower():
        return ["w1_linear", "w2_linear", "v1_linear"]
    else:
        raise NotImplementedError("MoE model not supported")


def model_type_is_enc_dec(model_type):
    """Check if model_type is a enc-dec model."""
    return model_type in ["t5", "bart", "whisper"]


def get_enc_dec_models(hf_model, model_type):
    """Get the correct encoder, decoder from hf model."""
    assert model_type_is_enc_dec(model_type), "This encoder decoder model is not supported"
    if model_type in ["bart", "whisper"]:
        return [("enc", hf_model.model.encoder), ("dec", hf_model.model.decoder)]
    else:
        return [("enc", hf_model.encoder), ("dec", hf_model.decoder)]


def get_encoder_config(encoder_config):
    """Get the encoder information for decoder model in enc-dec model."""
    encoder_hidden_size = encoder_config.d_model
    encoder_num_heads = (
        encoder_config.encoder_attention_heads
        if hasattr(encoder_config, "encoder_attention_heads")
        else encoder_config.num_heads
    )
    encoder_head_size = (
        encoder_config.d_kv
        if hasattr(encoder_config, "d_kv")
        else encoder_hidden_size // encoder_num_heads
    )
    return encoder_hidden_size, encoder_num_heads, encoder_head_size


def get_enc_dec_token_ids(decoder_config):
    """Parse decoder model token info."""
    decoder_start_token_id = (
        decoder_config.decoder_start_token_id
        if hasattr(decoder_config, "decoder_start_token_id")
        else None
    )
    eos_token_id = decoder_config.eos_token_id if hasattr(decoder_config, "eos_token_id") else None
    bos_token_id = decoder_config.bos_token_id if hasattr(decoder_config, "bos_token_id") else None
    pad_token_id = decoder_config.pad_token_id if hasattr(decoder_config, "pad_token_id") else None
    return decoder_start_token_id, eos_token_id, bos_token_id, pad_token_id
