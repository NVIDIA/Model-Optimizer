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

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Type

import torch.nn as nn

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import BlockConfig
from modelopt.torch.puzzletron.utils.dummy_modules import DummyBlock

__all__ = ["ModelDescriptor"]


class ModelDescriptor(ABC):
    @staticmethod
    @abstractmethod
    def decoder_layer_cls() -> Type[nn.Module] | List[Type[nn.Module]]:
        """Decoder layer class types to patch for heterogeneous config support.

        In most cases this class will hold as attributes both FFN & attention layers.

        Returns:
            nn.Module class type or a list if several class types should be patched.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def block_config_to_layer_overrides(block_config: BlockConfig) -> Dict[str, Any]:
        """Map between BlockConfig and layer config overrides.

        These overrides are consumed by a specific decoder layer and by the whole model.
        Usage can be seen in `deci_x_patcher` under the method `_patched_decoder_layer_init`.

        Example implementation to override the FFN intermediate size of a block:
            >>> def block_config_to_layer_overrides(block_config: BlockConfig) -> Dict[str, Any]:
            >>>     return {"intermediate_size": block_config.ffn.intermediate_size}
        """
        raise NotImplementedError

    @staticmethod
    def mlp_no_op_post_init(decoder_layer: nn.Module):
        """Post-init callback to alter a decoder layer so that FFN/mlp subblock performs as no-op.

        It is recommended to use the utils modules from `no_op.py` to replace layers to dummy
        counterparts.

        Example for replacing a layernorm layer with identity:
            >>> decoder_layer.post_attention_layernorm = Same()

        Example for replacing an MLP layer with zeroes (zeroes since hidden_states are added to
        the residuals hidden_states so a no-op implementation will leave residual the same):
            >>> decoder_layer.mlp = MatchingZeros()

        In case the MLP layer to replace returns multiple outputs i.e `hidden_states, _ = self.mlp()`,
        use the util method `return_tuple_of_size` to return trailing None values:
            >>> decoder_layer.mlp = return_tuple_of_size(MatchingZeros, size=2)()
        """
        raise NotImplementedError

    @staticmethod
    def attn_no_op_post_init(decoder_layer: nn.Module):
        """Post-init callback to alter a decoder layer so that Attention subblock performs as no-op.

        It is recommended to use the utils modules from `no_op.py` to replace layers to dummy
        counterparts.

        Example for replacing a layernorm layer with identity:
            >>> decoder_layer.post_attention_layernorm = Same()

        Example for replacing an attention layer with zeroes:
            >>> decoder_layer.self_attn = MatchingZeros()

        In case the attention layer returns multiple outputs i.e `hidden_states, _ = self.self_attn()`,
        use the util method `return_tuple_of_size` to return trailing None values:
            >>> decoder_layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def init_rotary_embedding(model, runtime):
        """Re-initiate the rotary embeddings based on an existing model.

        In puzzletron we initiate a sharded model by first creating a meta model then replacing
        to the actual device by loading the state_dict with the real weights.

        Rotary embeddings frequencies are tensor buffers that are created dynamically during init
        and are not part of the model state_dict, so cannot be restored after a meta device
        initialization.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def input_embedding_name():
        """Return the name of the input embedding layer."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def output_embedding_name():
        """Return the name of the output embedding layer."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def final_norm_name():
        """Return the name of the final normalization layer."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def layer_block_name(index: int):
        """Return the name of the decoder layer at the given index."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def layer_name_predicates(num_layers: int) -> Dict[str, re.Pattern]:
        """Return predicates for grouping model weights to support subblock checkpointing.

        For every group name return a regex predicate whether a layer name is part of the group.

        Returns:
            Dictionary of group name to regex pattern predicate.
        """
        raise NotImplementedError

    @staticmethod
    def uses_autocast() -> bool:
        """Whether this model supports torch.autocast.

        Some models (e.g., Qwen3-VL MoE) have dtype bugs under autocast.
        Override and return False for models that do not support autocast.
        """
        return True

    @staticmethod
    def get_language_model_config(config):
        """Get the language model config from a PretrainedConfig.

        For regular LM models, returns the config itself.
        For VL/multimodal models with nested configs, override to return the
        language model portion (e.g., config.text_config for Qwen-VL).
        """
        return config

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        """Create a dummy block to replace a layer for sharded model initialization."""
        return DummyBlock(block_index=block_index)

    @classmethod
    def mlp_no_op_supported(cls) -> bool:
        """Check whether `mlp_no_op_post_init` is overridden for mlp no-op support."""
        method_name = ModelDescriptor.mlp_no_op_post_init.__name__
        return getattr(cls, method_name) is not getattr(ModelDescriptor, method_name)

    @classmethod
    def attn_no_op_supported(cls):
        """Check whether `attn_no_op_post_init` is overridden for attention no-op support."""
        method_name = ModelDescriptor.attn_no_op_post_init.__name__
        return getattr(cls, method_name) is not getattr(ModelDescriptor, method_name)

    @classmethod
    def get_weight_groups(
        cls, layer_names: Iterable[str], num_hidden_layers: int
    ) -> Dict[str, List[str]]:
        """Group model weights to support the puzzle subblock checkpointing format.

        This method uses the abstract method `layer_name_predicates` by default.

        Args:
            layer_names: state_dict layer names of the model.
            num_hidden_layers: number of decoder layers in the model.

        Returns:
            Dictionary of group names to list of layer names per group, e.g.:
            >>> {
            ...     "embedding": ["model.embed_tokens.weight"],
            ...     "lm_head": ["lm_head.weight", "model.norm.weight"],
            ...     "block_0_ffn": ["model.layers.0.mlp.down_proj", ...],
            ...     "block_0_attention": ["model.layers.0.self_attn.q_proj", ...],
            ... }
        """
        weight_groups = defaultdict(list)
        for name in layer_names:
            for group, pattern in cls.layer_name_predicates(num_hidden_layers).items():
                if pattern.match(name):
                    weight_groups[group].append(name)
                    break
            else:
                raise ValueError(f"Couldn't find a match for {name}")
        return weight_groups
