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

"""Plugin to add EAGLE support for Megatron-Core GPT model."""

import copy
import warnings

import megatron.core
import torch
import torch.nn.functional as F
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_expert_tensor_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from packaging.version import Version

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..utils import AcceptanceRateValidation, get_default_attention_mask_and_position_ids
from .megatron_medusa import MedusaLayer

try:
    from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
    from megatron.core.post_training.modelopt.layers import Linear
except ImportError:
    warnings.warn("Fail to import megatron.core.post_training! EAGLE feature will be disable!")


def dict_to_config(
    architecture_config,
    use_cpu_initialization=None,
    fp16=False,
    bf16=True,
    sequence_parallel=False,
):
    """Helper function to convert a dictionary to TransformerConfig."""
    config = TransformerConfig(
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=False,
        tensor_model_parallel_size=get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=get_pipeline_model_parallel_world_size(),
        expert_tensor_parallel_size=get_expert_tensor_parallel_world_size(),
        sequence_parallel=sequence_parallel,
        use_cpu_initialization=use_cpu_initialization,
        fp16=fp16,
        bf16=bf16,
        params_dtype=getattr(torch, architecture_config["torch_dtype"]),
        pipeline_dtype=getattr(torch, architecture_config["torch_dtype"]),
        num_layers=architecture_config.get("num_hidden_layers"),
        ffn_hidden_size=architecture_config.get("intermediate_size"),
        num_attention_heads=architecture_config.get("num_attention_heads"),
        kv_channels=architecture_config.get("head_dim"),
        num_query_groups=architecture_config.get("num_key_value_heads"),
        init_method_std=architecture_config.get("initializer_range"),
        layernorm_epsilon=architecture_config.get("rms_norm_eps"),
        add_bias_linear=architecture_config.get("mlp_bias"),
        attention_dropout=architecture_config.get("attention_dropout"),
    )

    config.transformer_layer_spec = None
    config.seq_length = 8192
    config.gradient_accumulation_fusion = False
    config.position_embedding_type = architecture_config.get("position_embedding_type")
    config.rotary_percent = 1.0
    config.rotary_base = architecture_config.get("rope_theta")
    config.rope_scaling = "rope_scaling" in architecture_config
    config.rope_scaling_factor = (
        architecture_config.get("rope_scaling").get("factor")
        if "rope_scaling" in architecture_config
        else None
    )

    config.draft_vocab_size = architecture_config.get("draft_vocab_size")
    config.use_input_layernorm_in_first_layer = architecture_config.get(
        "use_input_layernorm_in_first_layer"
    )
    config.use_last_layernorm = architecture_config.get("use_last_layernorm")
    config.use_aux_hidden_state = architecture_config.get("use_aux_hidden_state")
    config.eagle_aux_hidden_state_layer_ids = architecture_config.get(
        "eagle_aux_hidden_state_layer_ids"
    )
    config.use_mtp_layernorm = architecture_config.get("use_mtp_layernorm")
    config.parallel_draft_step = architecture_config.get("parallel_draft_step")
    config.parallel_draft_heads_num_layers = architecture_config.get(
        "parallel_draft_heads_num_layers"
    )
    config.has_lm_head = architecture_config.get("has_lm_head")

    return config


def mcore_version_higher_than(target_version: str):
    """Check if megatron-core is greater than this version."""
    return Version(megatron.core.__version__) > Version(target_version)


def logits_kld_loss(logits, gt_logits, mapping=None):
    """KL Divergence loss using ground truth logits."""
    gathered_logits = gather_from_tensor_model_parallel_region(logits)
    gathered_gt_logits = gather_from_tensor_model_parallel_region(gt_logits)
    if mapping is not None:
        reverse_mapping = torch.arange(len(mapping)).to(mapping.device) + mapping
        gathered_gt_logits = gathered_gt_logits[:, :, reverse_mapping]
    loss = torch.nn.Softmax(dim=2)(gathered_gt_logits) * torch.nn.LogSoftmax(dim=2)(gathered_logits)
    loss = -torch.sum(loss, 2)
    return loss.transpose(0, 1)


def right_padding(input_ids: torch.Tensor, hidden_states: torch.Tensor = None):
    """Pad zeros to the right so that the padded_input_ids is a multiple of tp."""
    tp = get_tensor_model_parallel_world_size()
    seq_len = input_ids.shape[-1]
    right_padding_len = 0 if seq_len % tp == 0 else (tp - seq_len % tp)

    if right_padding_len > 0:
        right_token_pad = torch.zeros(
            (input_ids.shape[0], right_padding_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        padded_input_ids = torch.cat((input_ids, right_token_pad), dim=-1)
        if hidden_states is not None:
            # If sequence_parallel is used, the input hidden_states is already gathered
            padding_zeros = torch.zeros(
                (right_padding_len, hidden_states.shape[1], hidden_states.shape[2]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            padded_hidden_states = torch.cat((hidden_states, padding_zeros), dim=0)
    else:
        padded_input_ids = input_ids
        padded_hidden_states = hidden_states

    if hidden_states is not None:
        return padded_input_ids, seq_len, padded_hidden_states
    else:
        return padded_input_ids, seq_len


def set_multi_step_attention_mask(attn_mask, step):
    """Given an original attention_mask, construct a multi-step attention_mask.

    i0 i1 i2 i3 i4 i5 i6 i7  (base input_ids)
    =======================
    h0 h1 h2 h3 h4 h5 h6 h7  (base hidden_states)
    l0 l1 l2 l3 l4 l5 l6 l7  (base labels)




    ttt_step=0
                  | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- |
                  | h0 h1 h2 h3 h4 h5 h6 h7 | -- F1 F2 F3 F4 F5 F6 F7 | -- -- G2 G3 G4 G5 G6 G7 |
    =============================================================================================
    F1 l1 | i1 h0 |  x                      |                         |                         |
    F2 l2 | i2 h1 |  x  x                   |                         |                         |
    F3 l3 | i3 h2 |  x  x  x                |                         |                         |
    F4 l4 | i4 h3 |  x  x  x  x             |                         |                         |
    F5 l5 | i5 h4 |  x  x  x  x  x          |                         |                         |
    F6 l6 | i6 h5 |  x  x  x  x  x  x       |                         |                         |
    F7 l7 | i7 h6 |  x  x  x  x  x  x  x    |                         |                         |
    -- -- | -- h7 |  o  o  o  o  o  o  o  o |                         |                         |
    =============================================================================================
    -- -- | i1 -- |                         |                         |                         |
    G2 l2 | i2 F1 |  x  o                   |     x                   |                         |
    G3 l3 | i3 F2 |  x  x  o                |        x                |                         |
    G4 l4 | i4 F3 |  x  x  x  o             |           x             |                         |
    G5 l5 | i5 F4 |  x  x  x  x  o          |              x          |                         |
    G6 l6 | i6 F5 |  x  x  x  x  x  o       |                 x       |                         |
    G7 l7 | i7 F6 |  x  x  x  x  x  x  o    |                    x    |                         |
    -- -- | -- F7 |                         |                         |                         |
    =============================================================================================
    -- -- | i1 -- |                         |                         |                         |
    -- -- | i2 -- |                         |                         |                         |
    H3 l3 | i3 G2 |  x  o  o                |     x  o                |        x                |
    H4 l4 | i4 G3 |  x  x  o  o             |        x  o             |           x             |
    H5 l5 | i5 G4 |  x  x  x  o  o          |           x  o          |              x          |
    H6 l6 | i6 G5 |  x  x  x  x  o  o       |              x  o       |                 x       |
    H7 l7 | i7 G6 |  x  x  x  x  x  o  o    |                 x  o    |                    x    |
    -- -- | -- G7 |                         |                         |                         |
    =============================================================================================
    """
    s = attn_mask.shape[-1]
    for step_idx in range(step):
        mask_0 = attn_mask.clone().detach()
        mask_0[:, :, step_idx, :] = True
        mask_0[:, :, :, :-1] = mask_0[:, :, :, 1:]
        mask_1 = attn_mask.new_ones(attn_mask.shape[0], attn_mask.shape[1], s, s).bool()
        for i in range(step_idx + 1, s - 1):
            mask_1[:, :, i, i] = False

        attn_mask = torch.cat((mask_0, mask_1), dim=-1)

    return attn_mask


class ParallelDraft(MegatronModule):
    """ParallelDraft module with multiple Medusa heads and a shared lm head."""

    def __init__(
        self,
        config,
        vocab_size: int,
        num_heads: int = 1,
        num_layers: int = 1,
        parallel_output: bool = True,
    ):
        """Constructor.

        Args:
            config: MCore transformer config
            vocab_size: vocabulary size
            num_heads: number of draft heads
            num_layers: number of Medusa layers
            parallel_output: if False, then all_gather the logits
        """
        super().__init__(config=config)

        self.medusa_heads = torch.nn.ModuleList(
            [
                torch.nn.ModuleList([MedusaLayer(config) for _ in range(num_layers)])
                for _ in range(num_heads)
            ]
        )

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not parallel_output,
            skip_weight_param_allocation=False,
        )

        def load_state_dict_post_hook(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        self.register_load_state_dict_post_hook(load_state_dict_post_hook)

    def forward(self, x):
        """Forward function."""
        output = []
        for head in self.medusa_heads:
            x_head = x
            for layer in head:
                x_head, _ = layer(x_head)
            output.append(self.lm_head(x_head)[0])
        return output

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Return MCore sharded_state_dict."""
        assert not sharded_offsets, "Unexpected sharded offsets"
        sharded_state_dict = {}
        layer_prefix = f"{prefix}medusa_heads."
        for i, head in enumerate(self.medusa_heads):
            for j, layer in enumerate(head):
                state_dict_prefix = f"{layer_prefix}{i}.{j}."
                sharded_pp_offset = []
                layer_sharded_state_dict = layer.sharded_state_dict(
                    state_dict_prefix, sharded_pp_offset, metadata
                )
                sharded_state_dict.update(layer_sharded_state_dict)
        sharded_state_dict.update(
            self.lm_head.sharded_state_dict(f"{prefix}lm_head.", sharded_offsets, metadata)
        )
        return sharded_state_dict


class EagleLanguageModelEmbedding(LanguageModelEmbedding):
    """Allow last pp stage to also load the embedding."""

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int]] = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Different from the default, we change the state_dict to have 1 replica at pp."""
        state_dict = self.state_dict(prefix="", keep_vars=True)

        weight_prefix = f"{prefix}word_embeddings.weight"
        return {
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(
                tensor=state_dict["word_embeddings.weight"],
                key=weight_prefix,
                allow_shape_mismatch=True,
                prepend_offsets=sharded_offsets,
                # (PP, TP, DP)
                replica_id=(1, 0, get_data_parallel_rank(with_context_parallel=True)),
            )
        }


class EagleTransformerBlock(TransformerBlock):
    """Only store the EAGLE decoder in the last pp stage."""

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"

        sharded_state_dict = {}

        layer_prefix = f"{prefix}layers."
        num_layers = self.config.num_layers
        for global_layer_offset, layer in enumerate(self.layers):
            # global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = (
                f"{layer_prefix}{global_layer_offset}."  # module list index in TransformerBlock
            )

            if num_layers > 1:
                # For multi-layer EAGLE, we use heterogenuous distrubted checkpoint
                sharded_prefix = state_dict_prefix
                sharded_pp_offset = []
            else:
                # For single EAGLE, we use homogenous distrubted checkpoint for backward compatibility
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if module is not self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f"{prefix}{name}.", sharded_offsets, metadata
                    )
                )

        return sharded_state_dict


class EagleModule(MegatronModule):
    """EagleModule definition.

    EagleModule consists of an FC projection and additional decoder layers.
    """

    def __init__(
        self,
        config,
        rotary_pos_emb: torch.nn.Module,
        bias: bool = False,
    ):
        """Constructor.

        EagleModule is essentially a GPTModel except that it only exists in
        the last pp stage. As a result, pre_process must be True (otherwise
        the decoder expects the input is from the receive buffer).
        post_process must be True to perform the final_layernorm.

        Args:
            config: MCore transformer config
            rotary_pos_emb: nn.Module.
        """
        # Override transformer_config before superclass initialization
        config.pipeline_model_parallel_size = 1
        config.virtual_pipeline_model_parallel_size = None
        config.num_layers_in_first_pipeline_stage = None
        config.num_layers_in_last_pipeline_stage = None
        super().__init__(config=config)

        eagle_transformer_layer_spec = self._get_eagle_transformer_layer_spec(config)

        self._num_aux_hidden_states = len(self.config.eagle_aux_hidden_state_layer_ids)
        if self._num_aux_hidden_states > 0:
            self.enorm = TENorm(config, config.hidden_size, config.layernorm_epsilon)
            self._embeddings = None
        elif self.config.use_mtp_layernorm:
            self.enorm = TENorm(config, config.hidden_size, config.layernorm_epsilon)
            self.hnorm = TENorm(config, config.hidden_size, config.layernorm_epsilon)

        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()

        # EAGLE-3 uses aux_hidden_states (usually >= 3); otherwise EAGLE-1
        fc_input_size_multiplier = (
            self._num_aux_hidden_states if self._num_aux_hidden_states > 0 else 2
        )

        # This linear was previously a ColumnParallelLinear. We changed it to a normal linear
        # since ColumnParallelLinear will have try to gather the input sequence when sequence
        # parallel is used and does not allow gathering the outputs.
        with torch.device(device):
            self.fc = Linear(
                config.hidden_size * fc_input_size_multiplier,
                config.hidden_size,
                config=config,
                init_method=(lambda w: None),  # not used
                bias=bias,
            )

        self.rotary_pos_emb = rotary_pos_emb

        # Eagle does not use the final_layernorm in decoder.
        with torch.device(device):
            self.decoder = EagleTransformerBlock(
                config=config,
                spec=eagle_transformer_layer_spec,
                post_layer_norm=config.use_last_layernorm,
                pre_process=True,
                post_process=True,
            )

        if self._num_aux_hidden_states > 0:
            # Register forward hook to the last EAGLE3 layer to extract the pre-norm hidden_state
            # for eagle3 auto regression.
            last_layer = self.decoder.layers[-1]
            last_layer.register_forward_hook(self._eagle3_layer_forward_hook)

            # The first EAGLE3 layer needs to be specialized.
            layer = self.decoder.layers[0]
            self_attention = layer.self_attention
            if not isinstance(self_attention, SelfAttention):
                raise ValueError("EAGLE-3 only support SelfAttention (MHA, GQA).")

            # EAGLE-3's first attention require [input_layernorm_output, aux_hidden_states]
            self_attention.register_forward_pre_hook(self._eagle3_attention_forward_pre_hook)

            # EAGLE-3's first layer reduces hidden_states from 2h to h.
            self_attention.linear_qkv = tensor_parallel.ColumnParallelLinear(
                self_attention.config.hidden_size * 2,
                self_attention.query_projection_size + 2 * self_attention.kv_projection_size,
                config=self_attention.config,
                init_method=self_attention.config.init_method,
                gather_output=False,
                bias=self_attention.config.add_bias_linear or self_attention.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="qkv",
            )

        if self.config.draft_vocab_size != self.config.vocab_size:
            # Need an extra lm_head for eagle module since vocab size is reduced.
            assert self.config.draft_vocab_size <= self.config.vocab_size, (
                "EAGLE module's vocab size should be <= base model vocab size!"
            )

            self.register_buffer(
                "d2t", torch.zeros(self.config.draft_vocab_size, dtype=torch.int64)
            )
        if self.config.draft_vocab_size != self.config.vocab_size or self.config.has_lm_head:
            self.eagle_output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.config.draft_vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=False,
                skip_weight_param_allocation=False,
            )

        if self.config.parallel_draft_step > 1:
            self.parallel_draft_heads = ParallelDraft(
                config=self.config,
                vocab_size=self.config.draft_vocab_size,
                num_heads=self.config.parallel_draft_step - 1,
                num_layers=self.config.parallel_draft_heads_num_layers,
            )

    def _get_eagle_transformer_layer_spec(self, config):
        """Get the TransformerLayer implementation spec.

        IMPORTANT: EagleModule must use arbitrary_attention_mask since we need to
                   manipulate the mask to compute the correct loss. The default
                   causal mask will result in leaking.
        """
        transformer_layer_spec = get_gpt_modelopt_spec(
            config,
            remap_te_layernorm=True,
            use_arbitrary_attention_mask=True,
        )
        # If heterogenous layers (e.g. DeepSeek), transformer_layer_spec is a
        # TransformerBlockSubmodules instead. We use the last layer_specs.
        if "TransformerBlockSubmodules" in str(type(transformer_layer_spec)):
            eagle_transformer_layer_spec = copy.deepcopy(transformer_layer_spec.layer_specs[-1])
        else:
            eagle_transformer_layer_spec = copy.deepcopy(transformer_layer_spec)

        # Force TransformerLayer in case RealQuantTransformerLayer was used.
        eagle_transformer_layer_spec.module = TransformerLayer

        if not self.config.use_input_layernorm_in_first_layer:
            eagle_transformer_layer_spec.submodules.input_layernorm = IdentityOp
        return eagle_transformer_layer_spec

    def _eagle3_layer_forward_hook(self, module, input, output) -> None:
        if not isinstance(module, TransformerLayer):
            raise ValueError(
                "_eagle3_layer_forward_hook can only be registered to TransformerLayer"
            )
        hidden_states = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._next_hidden_states_input = hidden_states

    def _eagle3_attention_forward_pre_hook(self, module, input_layernorm_output):
        assert isinstance(input_layernorm_output[0], torch.Tensor)
        assert self._embeddings is not None
        embeddings = self._embeddings
        self._embeddings = None
        return (torch.cat((embeddings, input_layernorm_output[0]), dim=-1),)

    def forward(
        self,
        embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        inference_context: StaticInferenceContext | None = None,
        extra_block_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Forward function."""
        # NOTE: Even if sequence_parallel is used, the rotary_seq_len must be in the original
        #       length. Since we get the seq_len from hidden_states.shape[0], we need to
        #       multiply the the tp back.
        rotary_seq_len = hidden_states.shape[0]
        if self.config.sequence_parallel:
            rotary_seq_len *= self.config.tensor_model_parallel_size

        if self.config.use_mtp_layernorm:
            embeddings = self.enorm(embeddings)
            hidden_states = self.hnorm(hidden_states)

        # EAGLE-1 uses [s, b, h] input but EAGLE-3 uses [s, b, 2h] input
        if self._num_aux_hidden_states == 0:
            # [s, b, 2h]
            decoder_input = torch.cat((embeddings, hidden_states), dim=-1)
            decoder_input = self.fc(decoder_input)[0]
        else:
            # EAGLE-3 forward
            # EAGLE-3 uses self.fc outside eagle_module forward to convert hidden_states from [s, b, 3h]
            self._embeddings = self.enorm(embeddings)
            decoder_input = hidden_states

        if rotary_pos_emb is None:
            rotary_pos_emb = (
                None if self.config.multi_latent_attention else self.rotary_pos_emb(rotary_seq_len)
            )

        self._next_hidden_states_input = None

        decoder_input_list = [decoder_input]
        self.decoder.set_input_tensor(decoder_input_list[0])
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            inference_context=inference_context,
            **(extra_block_kwargs or {}),
        )

        if self._next_hidden_states_input is None:
            next_hidden_states_input = hidden_states
        else:
            next_hidden_states_input = self._next_hidden_states_input
            self._next_hidden_states_input = None

        return hidden_states, next_hidden_states_input


@EagleDMRegistry.register({GPTModel: "megatron.core.models.gpt.GPTModel"})
class _DynamicEagleGPTModel(EagleModel):
    """A ``megatron.core.models.gpt.GPTModel`` model with dynamic hyperparams."""

    def _set_default_aux_hidden_state_layers(self):
        if hasattr(self.config, "original_num_layers"):
            num_layers = self.config.original_num_layers
        else:
            num_layers = self.config.num_layers
        self.eagle_config.eagle_aux_hidden_state_layer_ids = [
            1,
            max(0, num_layers // 2 - 1),
            max(0, num_layers - 4),
        ]
        self.eagle_config.eagle_aux_hidden_state_layer_ids = list(
            set(self.eagle_config.eagle_aux_hidden_state_layer_ids)
        )

    def _transformer_layer_forward_hook(self, module, input, output) -> None:
        if not isinstance(module, TransformerLayer):
            raise ValueError(
                "_transformer_layer_forward_hook can only be registered to TransformerLayer"
            )
        if module.layer_number - 1 not in self.eagle_config.eagle_aux_hidden_state_layer_ids:
            return
        hidden_states = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._aux_hidden_states.append(hidden_states)

    def _setup(self):
        super()._setup()
        self._register_temp_attribute("eagle_freeze_base_model", True)
        self._register_temp_attribute("calibration_mode", False)

    def modify(
        self,
        eagle_offline,
        eagle_hidden_state_distillation,
        eagle_self_logit_distillation,
        eagle_freeze_base_model,
        eagle_report_acc,
        eagle_reuse_base_decoder,
        eagle_loss_decay_factor,
        eagle_architecture_config,
        eagle_decoder_type,
    ):
        if self.config.pipeline_model_parallel_size > 1:
            warnings.warn(
                "Pipeline parallelism detected! _DynamicEagleGPTModel only supports "
                "pipeline parallelism during TensorRT-LLM checkpoint export."
            )

        # Since there is a chance that EAGLE3 can have heterogenous layers (1st layer
        # qkv is 2x large than the rest), we enable MCore hetereogeneous checkpoint.
        if hasattr(self.config, "hetereogenous_dist_checkpoint"):
            self.config.hetereogenous_dist_checkpoint = True

        super().modify(
            eagle_offline=eagle_offline,
            eagle_hidden_state_distillation=eagle_hidden_state_distillation,
            eagle_self_logit_distillation=eagle_self_logit_distillation,
            eagle_freeze_base_model=eagle_freeze_base_model,
            eagle_report_acc=eagle_report_acc,
            eagle_reuse_base_decoder=eagle_reuse_base_decoder,
            eagle_loss_decay_factor=eagle_loss_decay_factor,
            eagle_architecture_config=eagle_architecture_config,
            eagle_decoder_type=eagle_decoder_type,
        )

        # sequence_parallel is not used in offline eagle
        if self.eagle_offline:
            self.config.sequence_parallel = False

        self.eagle_config = dict_to_config(
            eagle_architecture_config,
            self.config.use_cpu_initialization,
            self.config.fp16,
            self.config.bf16,
            self.config.sequence_parallel,
        )
        self.eagle_config.hidden_size = self.config.hidden_size
        self.eagle_config.vocab_size = self.vocab_size
        self.eagle_config.max_sequence_length = self.max_sequence_length
        self.eagle_config.draft_vocab_size = (
            self.vocab_size
            if self.eagle_config.draft_vocab_size is None
            else self.eagle_config.draft_vocab_size
        )

        if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
            assert eagle_self_logit_distillation, (
                "Only logit distillation is supported when draft_vocab_size != vocab_size!"
            )

        # Use default aux_hidden_state layers if use_aux_hidden_state is True
        # but no layer id is given
        # layer ids are not used in offline eagle, but we need to set this to have correct fc_input_size_multiplier
        if (
            self.eagle_config.use_aux_hidden_state
            and len(self.eagle_config.eagle_aux_hidden_state_layer_ids) == 0
        ):
            self._set_default_aux_hidden_state_layers()

        if len(self.eagle_config.eagle_aux_hidden_state_layer_ids) > 0:
            assert not self.eagle_hidden_state_distillation, (
                "EAGLE-3 does not support hidden state distillation!"
            )

        # EAGLE-3 auxiliary hidden_states (only work for TP+EP, does not work for PP)
        self._aux_hidden_states = []

        if self.eagle_config.position_embedding_type not in ["rope", "yarn"]:
            raise ValueError("For EAGLE, only RoPE or YaRN embedding are supported")

        if not self.pre_process and self.post_process:
            self.embedding = EagleLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )

        # Register TransformerLayer forward hook to extract aux hidden_states.
        if len(self.eagle_config.eagle_aux_hidden_state_layer_ids) > 0:
            for layer in self.decoder.layers:
                layer.register_forward_hook(self._transformer_layer_forward_hook)

        # Freeze all parameters
        if self.eagle_freeze_base_model:
            for name, param in self.named_parameters():
                param.requires_grad = False

        # Only the last PP stage has the additional projection and decoder layer.
        # This is to simplify the export.
        if self.post_process:
            if self.eagle_reuse_base_decoder:
                eagle_config = copy.deepcopy(self.config)
                # Overwrite values from the eagle config
                eagle_config.num_layers = self.eagle_config.num_layers
                eagle_config.use_last_layernorm = self.eagle_config.use_last_layernorm
                eagle_config.use_input_layernorm_in_first_layer = (
                    self.eagle_config.use_input_layernorm_in_first_layer
                )
                eagle_config.eagle_aux_hidden_state_layer_ids = (
                    self.eagle_config.eagle_aux_hidden_state_layer_ids
                )
                eagle_config.use_mtp_layernorm = self.eagle_config.use_mtp_layernorm
                eagle_config.draft_vocab_size = self.eagle_config.draft_vocab_size
                eagle_config.has_lm_head = self.eagle_config.has_lm_head
                self.eagle_module = EagleModule(
                    eagle_config,
                    self.rotary_pos_emb,
                    bias=False,
                )
            else:
                rotary_pos_emb = RotaryEmbedding(
                    kv_channels=self.eagle_config.kv_channels,
                    rotary_percent=self.eagle_config.rotary_percent,
                    rotary_interleaved=False,
                    seq_len_interpolation_factor=None,
                    rotary_base=self.eagle_config.rotary_base,
                    rope_scaling=self.eagle_config.rope_scaling,
                    rope_scaling_factor=self.eagle_config.rope_scaling_factor,
                    use_cpu_initialization=self.eagle_config.use_cpu_initialization,
                )

                self.eagle_module = EagleModule(
                    self.eagle_config,
                    rotary_pos_emb,
                    bias=False,
                )

            # Eagle loss functions
            self.kld = logits_kld_loss

    def _get_eagle_input_hidden_states(self, hidden_states: torch.Tensor, apply_fc: bool = True):
        """When _aux_hidden_states is not empty for online, then this is EAGLE-3.

        Args:
            hidden_states: last hidden_states
            apply_fc: whether to apply EAGLE3 fc
        """
        if not self.eagle_offline:
            if len(self._aux_hidden_states) == 0:
                return hidden_states

            # [s / TP, b, len(self._aux_hidden_states) * h]
            hidden_states = torch.cat(self._aux_hidden_states, dim=-1)
            self._aux_hidden_states.clear()

        if apply_fc:
            # [s / TP, b, 3h] -> [s / TP, b, h]
            return self.eagle_module.fc(hidden_states)[0]
        else:
            return hidden_states

    def _get_eagle_module_inputs(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        ttt_step: int = 0,
    ):
        """Getting EAGLE module inputs."""
        # [b, 1]
        id_padding = torch.zeros(
            (input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device
        )
        padded_input_ids = torch.cat((input_ids[:, 1:], id_padding), dim=-1)

        rotary_pos_emb = self.eagle_module.rotary_pos_emb(padded_input_ids.shape[-1])

        attn_mask = attention_mask.clone().detach()
        attn_mask[:, :, :-1, :-1] = attention_mask[:, :, 1:, 1:]
        attn_mask[:, :, -1, :] = True
        attn_mask[:, :, :, -1] = True

        eagle_inputs = {}

        eagle_inputs["input_ids"] = padded_input_ids
        eagle_inputs["position_ids"] = position_ids

        eagle_inputs["embedding"] = self.embedding(
            input_ids=eagle_inputs["input_ids"],
            position_ids=eagle_inputs["position_ids"],
        )
        eagle_inputs["hidden_states"] = hidden_states

        eagle_inputs["attention_mask"] = set_multi_step_attention_mask(attn_mask, ttt_step)

        eagle_inputs["rotary_pos_emb"] = torch.cat(
            [rotary_pos_emb] * (ttt_step + 1),
            dim=0,
        )

        return eagle_inputs

    def _compute_eagle_loss(self, logits, labels, eagle_logits):
        """Compute the total loss for EAGLE.

        logits: [s, b, vocab // TP]
        labels: [b, s]
        eagle_logits: [s, b, vocab // TP]
        """
        # Compute lm loss (classification loss) or KLDivergence
        if self.eagle_self_logit_distillation:
            mapping = self.eagle_module.d2t if hasattr(self.eagle_module, "d2t") else None
            token_loss = self.kld(eagle_logits[:-1, :, :], logits[1:, :, :], mapping)
        else:
            token_loss = self.compute_language_model_loss(labels[:, 1:], eagle_logits[:-1, :, :])

        # [b, s - 1]
        return token_loss

    def _base_model_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
        return_eagle_inputs: bool = False,
    ):
        # Word and rotary positional embeddings
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from decoder.input_tensor
            decoder_input = None

        extra_kwargs = {"packed_seq_params": None} if mcore_version_higher_than("0.9.0") else {}

        rotary_pos_emb = None
        yarn_mscale = 1.0
        if self.config.multi_latent_attention:
            # For MLA, rotary_pos_emb is computed per attention.
            rotary_pos_emb = None
        elif self.position_embedding_type == "rope":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                self.decoder,
                decoder_input,
                self.config,
                **extra_kwargs,
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        elif self.position_embedding_type == "yarn":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                None,
                decoder_input,
                self.config,
                **extra_kwargs,
            )
            rotary_pos_emb, yarn_mscale = self.rotary_pos_emb(rotary_seq_len)
        else:
            raise ValueError(
                f"Only RoPE or YaRN are supported but got {self.position_embedding_type}"
            )

        # [TODO]: yarn_mscale needs to be passed into TransformerBlock forward when supported.
        # Now the default value for yarn_mscale = 1.0
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if return_eagle_inputs:
            return hidden_states, decoder_input
        else:
            return hidden_states, None

    def _eagle_forward(
        self,
        eagle_inputs,
        output_weight,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        inference_context: StaticInferenceContext | None = None,
        extra_block_kwargs: dict | None = None,
    ):
        eagle_hidden_states, eagle_hidden_states_pre_final_layernorm = self.eagle_module(
            eagle_inputs["embedding"],
            eagle_inputs["hidden_states"],
            eagle_inputs["attention_mask"],
            eagle_inputs["rotary_pos_emb"],
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            inference_context=inference_context,
            **(extra_block_kwargs or {}),
        )

        # Update inference_context.sequence_len_offset after each call of eagle_module
        if inference_context is not None:
            inference_context.sequence_len_offset += eagle_inputs["input_ids"].shape[1]

        if hasattr(self.eagle_module, "eagle_output_layer"):
            eagle_logits, _ = self.eagle_module.eagle_output_layer(eagle_hidden_states)
        else:
            eagle_logits, _ = self.output_layer(eagle_hidden_states, weight=output_weight)

        draft_logits_list = [eagle_logits]
        if self.eagle_config.parallel_draft_step > 1:
            # Get additional draft logits from parallel draft heads
            draft_logits = self.eagle_module.parallel_draft_heads(eagle_hidden_states)
            draft_logits_list += draft_logits

        return (
            eagle_hidden_states,
            draft_logits_list,
            eagle_hidden_states_pre_final_layernorm,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
        return_eagle_inputs: bool = False,
        ttt_steps=4,
        **kwargs,
    ) -> torch.Tensor:
        if position_ids is None or attention_mask is None:
            attention_mask, position_ids = get_default_attention_mask_and_position_ids(input_ids)

        if self.eagle_offline:
            # aux_hidden_states and hidden_states are provided for offline eagle
            # _base_model_forward is skipped
            if return_eagle_inputs:
                raise ValueError("return_eagle_inputs is unsupported in EAGLE offline mode.")
            aux_hidden_states = kwargs.get("aux_hidden_states")
            hidden_states = kwargs.get("hidden_states")
            if aux_hidden_states is None or hidden_states is None:
                raise ValueError(
                    "EAGLE offline mode requires kwargs: aux_hidden_states=[s,b,k*h], "
                    "hidden_states=[s,b,h]."
                )
        else:
            # When return_eagle_inputs is True, return decoder_input_for_eagle.
            # For LLM, decoder_input_for_eagle is just the text embeddings. However, for VLM
            # decoder_input_for_eagle will also contain projected image/video embeddings.
            hidden_states, decoder_input_for_eagle = self._base_model_forward(
                input_ids,
                position_ids,
                attention_mask,
                decoder_input,
                inference_params,
                packed_seq_params,
                extra_block_kwargs,
                return_eagle_inputs=return_eagle_inputs,
            )

            # Typically, this is only the case when PP > 1.
            if not self.post_process:
                return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits_sbh, _ = self.output_layer(hidden_states, weight=output_weight)

        # EAGLE kv cache
        eagle_inference_context = StaticInferenceContext(
            input_ids.shape[0],
            input_ids.shape[1] * ttt_steps,
        )

        if self.eagle_offline:
            eagle_module_input_hidden_states = self._get_eagle_input_hidden_states(
                aux_hidden_states, apply_fc=self.eagle_config.use_aux_hidden_state
            )
        # If EAGLE-3, aux_hidden_states are gathered by the forward_hook
        elif return_eagle_inputs:
            eagle_module_input_hidden_states = self._get_eagle_input_hidden_states(
                hidden_states, apply_fc=False
            )

            if self.config.sequence_parallel:
                eagle_module_input_hidden_states = gather_from_sequence_parallel_region(
                    eagle_module_input_hidden_states
                )
                hidden_states = gather_from_sequence_parallel_region(hidden_states)
            logits_sbh = gather_from_tensor_model_parallel_region(logits_sbh)
            # In case of VLM, there will be other fields for pixels.
            return {
                "input_ids": input_ids.squeeze(0).cpu(),
                "aux_hidden_states": eagle_module_input_hidden_states.squeeze(1).cpu(),
                "hidden_states": hidden_states.squeeze(1).cpu(),
            }
        else:
            eagle_module_input_hidden_states = self._get_eagle_input_hidden_states(
                hidden_states, apply_fc=True
            )

        if labels is not None:
            if labels.shape[1] == input_ids.shape[1] - 1:
                # For offline training, labels may be 1 token shorter than input_ids.
                # We will just pad a 0 to the labels to make the seq_len the same as
                # input_ids. This will introduce a small error in training if logit_distillation
                # is False, and testing accuracy is wrong for the last token.
                right_token_pad = torch.zeros(
                    (labels.shape[0], 1),
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat((labels, right_token_pad), dim=-1)

            # If eagle_freeze_base_model is set to True,
            # the base model is frozen .
            loss = self.compute_language_model_loss(labels, logits_sbh)
            if self.eagle_freeze_base_model:
                loss = 0.0 * loss

        acc = []
        for ttt_step in range(ttt_steps):
            eagle_inputs = self._get_eagle_module_inputs(
                input_ids=input_ids,
                hidden_states=eagle_module_input_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                ttt_step=ttt_step,
            )

            _, eagle_logits, eagle_module_input_hidden_states = self._eagle_forward(
                eagle_inputs,
                output_weight,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                inference_context=eagle_inference_context,
                **(extra_block_kwargs or {}),
            )

            if self.config.sequence_parallel:
                eagle_module_input_hidden_states = gather_from_sequence_parallel_region(
                    eagle_module_input_hidden_states
                )
            eagle_module_input_hidden_states = torch.cat(
                (
                    torch.zeros(
                        (
                            1,
                            eagle_module_input_hidden_states.shape[1],
                            eagle_module_input_hidden_states.shape[2],
                        ),
                        dtype=eagle_module_input_hidden_states.dtype,
                        device=eagle_module_input_hidden_states.device,
                    ),
                    eagle_module_input_hidden_states[:-1, :, :],
                )
            )
            if self.config.sequence_parallel:
                eagle_module_input_hidden_states = scatter_to_sequence_parallel_region(
                    eagle_module_input_hidden_states
                )

            # If labels are not provided, return the original logits. We only return after
            # all eagle weights have been exercised for quantization calibration purpose.
            if labels is None:
                return logits_sbh.transpose(0, 1).contiguous()

            for i in range(self.eagle_config.parallel_draft_step):
                eagle_logit = eagle_logits[i]
                if i > 0:
                    loss_ = self._compute_eagle_loss(
                        logits_sbh[i:], labels[:, i:], eagle_logit[:-i]
                    )
                else:
                    loss_ = self._compute_eagle_loss(logits_sbh, labels, eagle_logit)
                loss_ = loss_[:, ttt_step:]
                loss[:, i + ttt_step + 1 :] += (
                    self.eagle_loss_decay_factor ** (ttt_step + i) * loss_
                )

            if self.eagle_report_acc and not self.training:
                with torch.no_grad():
                    gathered_base_logits = gather_from_tensor_model_parallel_region(logits_sbh)
                    base_top1 = gathered_base_logits.transpose(0, 1).argmax(dim=-1)
                    for i in range(self.eagle_config.parallel_draft_step):
                        gathered_logits = gather_from_tensor_model_parallel_region(eagle_logits[i])
                        gathered_logits = gathered_logits[ttt_step : -(1 + i)]
                        eagle_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                        if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                            eagle_top1 += self.eagle_module.d2t[eagle_top1]
                        top1_p = (
                            torch.eq(base_top1[:, i + ttt_step + 1 :], eagle_top1).sum()
                            / eagle_top1.numel()
                        )
                        acc.append(top1_p)

        if self.eagle_report_acc and not self.training and get_tensor_model_parallel_rank() == 0:
            print(
                f"{torch.distributed.get_rank():3}/{torch.distributed.get_world_size():3}"
                f"EAGLE Top-1: {acc}",
                flush=True,
            )

        return loss

    def pseudo_speculative_generate(
        self,
        input_ids: torch.Tensor,
        steps: int = 1,
    ):
        """Pseudo generate of the EAGLE GPTModel.

        This function does not support kv cache as sequence parallel may be enabled.

        Returns:
            base_token (torch.Tensor): token from base model
            draft_tokens (torch.Tensor): draft tokens from eagle module
        """
        padded_input_ids, seq_len = right_padding(input_ids)

        attention_mask, position_ids = get_default_attention_mask_and_position_ids(padded_input_ids)

        hidden_states, _ = self._base_model_forward(
            padded_input_ids,
            position_ids,
            attention_mask,
        )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits_sbh, _ = self.output_layer(hidden_states, weight=output_weight)

        # Remove the padding
        logits_sbh = logits_sbh[:seq_len, :, :]

        base_token = (
            gather_from_tensor_model_parallel_region(logits_sbh)[-1:, :, :]
            .argmax(dim=-1)
            .transpose(0, 1)
        )

        # Early return
        if steps < 1:
            self._aux_hidden_states.clear()
            return base_token, None

        eagle_ids = torch.cat((input_ids[:, 1:], base_token), dim=-1)

        # EAGLE-3
        # Only the first iteration input_hidden_states are from aux_hidden_state layers
        hidden_states = self._get_eagle_input_hidden_states(hidden_states, apply_fc=True)
        # Remove the padding
        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        hidden_states = hidden_states[:seq_len, :, :]

        draft_tokens = []
        for step in range(steps):
            padded_eagle_ids, seq_len, padded_hidden_states = right_padding(
                eagle_ids, hidden_states
            )
            if self.config.sequence_parallel:
                padded_hidden_states = scatter_to_sequence_parallel_region(padded_hidden_states)
            eagle_attention_mask, eagle_position_ids = get_default_attention_mask_and_position_ids(
                padded_eagle_ids
            )

            eagle_inputs = {}
            eagle_inputs["input_ids"] = padded_eagle_ids
            embeddings = self.embedding(
                input_ids=padded_eagle_ids,
                position_ids=eagle_position_ids,
            )

            eagle_inputs["embedding"] = embeddings
            eagle_inputs["hidden_states"] = padded_hidden_states
            eagle_inputs["attention_mask"] = eagle_attention_mask

            # [TODO] (chenhany): let the module compute itself
            eagle_inputs["rotary_pos_emb"] = None

            _, eagle_logits, eagle_next_hidden_states_input = self._eagle_forward(
                eagle_inputs,
                output_weight,
            )

            # parallel_logits are only used after the last step
            if step == steps - 1 and self.eagle_config.parallel_draft_step > 1:
                parallel_logits = [
                    eagle_logits[i][seq_len - 1 : seq_len]
                    for i in range(1, self.eagle_config.parallel_draft_step)
                ]
            eagle_logits = eagle_logits[0][:seq_len, :, :]
            if self.config.sequence_parallel:
                eagle_next_hidden_states_input = gather_from_sequence_parallel_region(
                    eagle_next_hidden_states_input
                )
            eagle_next_hidden_states_input = eagle_next_hidden_states_input[:seq_len, :, :]

            draft_token = (
                gather_from_tensor_model_parallel_region(eagle_logits)[-1:, :, :]
                .argmax(dim=-1)
                .transpose(0, 1)
            )
            if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                draft_token += self.eagle_module.d2t[draft_token]

            draft_tokens.append(draft_token)

            eagle_ids = torch.cat((eagle_ids, draft_token), dim=-1)
            hidden_states = torch.cat(
                (
                    hidden_states,
                    eagle_next_hidden_states_input[-1:, :, :],
                ),
                dim=0,
            )

        draft_tokens = torch.cat(draft_tokens, dim=-1)
        if self.eagle_config.parallel_draft_step > 1:
            parallel_logits = torch.cat(parallel_logits, dim=0)
            parallel_tokens = (
                (gather_from_tensor_model_parallel_region(parallel_logits))
                .argmax(dim=-1)
                .transpose(0, 1)
            )
            if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                parallel_tokens += self.eagle_module.d2t[parallel_tokens]
            draft_tokens = torch.cat((draft_tokens, parallel_tokens), dim=-1)

        return base_token, draft_tokens


class MegatronARValidation(AcceptanceRateValidation):
    """This is the subclass for megatron model AR validation."""

    def get_ground_truth(self, input_ids, osl):
        """This function returns ground truth output tokens from the base model."""
        input_ids = copy.deepcopy(input_ids)
        for _ in range(osl):
            input_id, _ = self.model.pseudo_speculative_generate(input_ids, steps=0)
            input_ids = torch.cat((input_ids, input_id), dim=-1)
            if input_id[0, 0] == self.end_token:
                break
        return input_ids
