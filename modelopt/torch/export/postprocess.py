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

"""Utils to load and process model_config."""

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from warnings import warn

import torch

from modelopt.torch.quantization.model_calib import disable_pre_quant_scale_and_resmooth
from modelopt.torch.quantization.nn import SequentialQuantizer
from modelopt.torch.quantization.nn.modules.quant_linear import QuantLinear
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.utils import distributed as dist

from .distribute import get_configs_parallel, get_tensors_parallel
from .model_config import (
    LINEAR_COLUMN,
    LINEAR_GROUP,
    LINEAR_ROW,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    QUANTIZATION_NVFP4_SVDQUANT,
    ConvConfig,
    EmbeddingConfig,
    ExpertConfig,
    LinearConfig,
    ModelConfig,
    RelativeAttentionTableConfig,
)
from .model_config_utils import pad_weights
from .quant_utils import get_scaling_factor_from_weight, resmooth_and_get_scale


def _same_tensor(tensors: list[torch.Tensor]):
    return all(t is None for t in tensors) or all(torch.equal(tensors[0], t) for t in tensors[1:])


# Define helper functions
def view_as_uint8_if_needed(tensor):
    """Views fp8 tensor as uint8 tensor for processing using torch operations."""
    is_float8_e4m3fn = tensor.dtype == torch.float8_e4m3fn
    return tensor.view(torch.uint8) if is_float8_e4m3fn else tensor


def view_as_float8_e4m3fn_if_needed(tensor):
    """Views uint8 tensor as fp8 tensor after processing using torch operations."""
    is_uint8 = tensor.dtype == torch.uint8
    return tensor.view(torch.float8_e4m3fn) if is_uint8 else tensor


def _shallow_copy_with_field_instantiation(obj):
    """Creates a shallow copy of the given object, recursively instantiating its fields.

    Unlike `copy.copy(obj)`, which only creates a new instance of the object without
    instantiating its fields (fields in the new object still reference the original object),
    this function ensures that fields are also instantiated recursively, breaking shared
    references with the original object.
    """
    cloned_instance = copy.copy(obj)

    # Iterate over all fields of the instance and clone them if they are instances of dataclasses
    for field in fields(obj):
        field_value = getattr(obj, field.name)
        if hasattr(field_value, "__dataclass_fields__"):
            setattr(cloned_instance, field.name, copy.copy(field_value))

    return cloned_instance


def _split_model_config_for_tp(merged_config, split_factor):
    """This method splits the tensor fields for linear config so the config can be used with more GPUs.

    The implementation is recursive.
    """
    configs = [copy.copy(merged_config) for _ in range(split_factor)]

    if isinstance(merged_config, ConvConfig):
        raise NotImplementedError("Do not support ConvConfig TP merge or split")

    elif isinstance(merged_config, EmbeddingConfig):
        weights = torch.chunk(pad_weights(merged_config.weight, split_factor), split_factor, dim=0)
        for i, config in enumerate(configs):
            config.weight = weights[i]

    elif isinstance(merged_config, ExpertConfig):
        assert merged_config.proj.linear_type != LINEAR_COLUMN  # row
        assert merged_config.fc.linear_type == LINEAR_COLUMN  # column
        # Avoid deepcopy and instantiate expert configs manually to avoid OOM
        configs = [
            _shallow_copy_with_field_instantiation(merged_config) for _ in range(split_factor)
        ]
        weights = torch.chunk(
            pad_weights(merged_config.proj.weight, split_factor), split_factor, dim=2
        )
        for i, config in enumerate(configs):
            config.proj.weight = weights[i]

        # Split w3 and w1 weight tensors separately and then concat for each TP
        merged_w3, merged_w1 = torch.chunk(merged_config.fc.weight, 2, dim=1)
        merged_w3_weights = torch.chunk(merged_w3, split_factor, dim=1)
        merged_w1_weights = torch.chunk(merged_w1, split_factor, dim=1)
        for i, config in enumerate(configs):
            config.fc.weight = torch.concat([merged_w3_weights[i], merged_w1_weights[i]], dim=1)

        # TODO: check if this works with int8_sq
        # Split scaling factors for int4_awq
        if (
            config.fc.weights_scaling_factor is not None
            and config.fc.weights_scaling_factor.dim() == 3
        ):  # column linear
            merged_scaling_factor_w3, merged_scaling_factor_w1 = torch.chunk(
                view_as_uint8_if_needed(merged_config.fc.weights_scaling_factor), 2, dim=1
            )
            scaling_factor_w3 = torch.chunk(merged_scaling_factor_w3, split_factor, dim=1)
            scaling_factor_w1 = torch.chunk(merged_scaling_factor_w1, split_factor, dim=1)
            for i, config in enumerate(configs):
                config.fc.weights_scaling_factor = view_as_float8_e4m3fn_if_needed(
                    torch.concat([scaling_factor_w3[i], scaling_factor_w1[i]], dim=1)
                )

        if (
            config.proj.weights_scaling_factor is not None
            and config.proj.weights_scaling_factor.dim() == 3
        ):  # row linear
            scaling_factor_proj = torch.chunk(
                merged_config.proj.weights_scaling_factor, split_factor, dim=2
            )

            for i, config in enumerate(configs):
                config.proj.weights_scaling_factor = scaling_factor_proj[i]

            # For NVFP4_AWQ and INT4_AWQ we split pre_quant_scale across tp
            if merged_config.proj.prequant_scaling_factor is not None:
                prequant_scaling_factors = torch.chunk(
                    merged_config.proj.prequant_scaling_factor, split_factor, dim=-1
                )
                for i, config in enumerate(configs):
                    config.proj.prequant_scaling_factor = prequant_scaling_factors[i]

    elif isinstance(merged_config, LinearConfig):
        # The scaling factors split rule is summarized as below:

        # S: all ranks should have the same scaling factor.
        # P0: Split the scaling factors on dim 0.
        # P1: Split the scaling factors on dim 1.
        # NA: Not valid / present

        # ws: weight scaling factor
        # as: activation scaling factor
        # ps: prequant scaling factor

        # C: Column Linear
        # R: Row Linear

        # F: FP8
        # I8: INT8 SQ
        # I4: INT4 AWQ

        # Split Rules:
        #      ws  as  ps
        # FC   S   S   NA
        # FR   S   S   NA
        # I8C  P0  S   S
        # I8R  S   S   P0
        # I4C  P0  S   S
        # I4R  P1  S   P0

        # For INT4 AWQ reference implementation: please check examples/llama/weight.py in the tekit repo
        # For normal linear layers, we split the column linear on the dim 0 and row on the dim 1
        assert merged_config.linear_type != LINEAR_GROUP, (
            "Do not support group linear TP merge or split"
        )

        # Do not do anything if we don't need to process TP.
        if not merged_config.tp:
            return configs

        split_axis = 0 if merged_config.linear_type == LINEAR_COLUMN else 1
        if merged_config.linear_type == LINEAR_COLUMN:
            merged_config.weight = pad_weights(merged_config.weight, split_factor)
        weights = torch.chunk(merged_config.weight, split_factor, dim=split_axis)
        for i, config in enumerate(configs):
            config.weight = weights[i]

        # Only split the bias for column linear.
        if merged_config.linear_type == LINEAR_COLUMN and merged_config.bias is not None:
            biases = torch.chunk(merged_config.bias, split_factor, dim=0)
            for i, config in enumerate(configs):
                config.bias = biases[i]

        if merged_config.linear_type == LINEAR_COLUMN:
            if (
                merged_config.weights_scaling_factor is not None
                and merged_config.weights_scaling_factor.numel() != 1
            ):
                # INT4 AWQ and INT8 sq all linear cases
                scaling_factors = torch.chunk(
                    view_as_uint8_if_needed(merged_config.weights_scaling_factor),
                    split_factor,
                    dim=0,
                )

                for i, config in enumerate(configs):
                    config.weights_scaling_factor = view_as_float8_e4m3fn_if_needed(
                        scaling_factors[i]
                    )

        else:
            if (
                merged_config.weights_scaling_factor is not None
                and merged_config.awq_block_size != 0
            ):
                scaling_factors = torch.chunk(
                    view_as_uint8_if_needed(merged_config.weights_scaling_factor),
                    split_factor,
                    dim=1,
                )

                for i, config in enumerate(configs):
                    config.weights_scaling_factor = view_as_float8_e4m3fn_if_needed(
                        scaling_factors[i]
                    )

            if merged_config.prequant_scaling_factor is not None:
                prequant_scaling_factors = torch.chunk(
                    merged_config.prequant_scaling_factor, split_factor, dim=0
                )
                for i, config in enumerate(configs):
                    config.prequant_scaling_factor = prequant_scaling_factors[i]

    elif isinstance(merged_config, RelativeAttentionTableConfig):
        weights = torch.chunk(merged_config.weight, split_factor, dim=0)
        for i, config in enumerate(configs):
            config.weight = weights[i]

    elif is_dataclass(merged_config):
        for field in fields(merged_config):
            field_configs = _split_model_config_for_tp(
                getattr(merged_config, field.name),
                split_factor,
            )
            for r in range(split_factor):
                setattr(configs[r], field.name, field_configs[r])
    elif isinstance(merged_config, list):
        for i in range(len(merged_config)):
            field_configs = _split_model_config_for_tp(
                merged_config[i],
                split_factor,
            )
            for r in range(split_factor):
                configs[r][i] = field_configs[r]

    return configs


def _split_model_config_for_pp(merged_config, split_factor):
    """This method splits ModelConfig for inference pipeline parallel."""
    num_layers = len(merged_config.layers)
    assert num_layers % split_factor == 0
    layers_per_pp = num_layers // split_factor

    configs = [copy.copy(merged_config) for _ in range(split_factor)]
    for i, config in enumerate(configs):
        if i > 0:
            config.vocab_embedding = None
            config.position_embedding = None
            config.ln_embed = None
        if i < split_factor - 1:
            config.ln_f = None
            config.lm_head = None
        config.layers = config.layers[i * layers_per_pp : (i + 1) * layers_per_pp]

    return configs


def _merge_model_configs_to_first_tp(config, ranks: list[int], group=None):
    """This method merges the tensor fields for linear config so the config can be used with fewer GPUs.

    The implementation is recursive.
    """
    if isinstance(config, ConvConfig):
        raise NotImplementedError("Do not support ConvConfig TP merge or split")

    elif isinstance(config, EmbeddingConfig):
        assert config.weight is not None
        with get_tensors_parallel(config.weight, ranks, group) as weights:
            if weights:
                config.weight = torch.cat(weights, dim=0)

    elif isinstance(config, LinearConfig):
        # The scaling factors merge rule is summarized as below:

        # S: all ranks should have the same scaling factor.
        # M: Pick elementwise max among the ranks. Merged shape same as single rank.
        # C0: Concat the scaling factors on dim 0. Merged shape == tensor_parallel * original shape.
        # C1: Concat the scaling factors on dim 1. Merged shape == original shape * tensor_parallel.
        # NA: Not valid / present

        # ws: weight scaling factor
        # as: activation scaling factor
        # ps: prequant scaling factor

        # C: Column Linear
        # R: Row Linear

        # F: FP8
        # I8: INT8 SQ
        # I4: INT4 AWQ

        # Merge Rules:
        #      ws  as  ps
        # FC   M   M   NA
        # FR   M   M   NA
        # I8C  C0  M   S
        # I8R  M   M   C0
        # I4C  C0  M   S
        # I4R  C1  M   C0

        assert config.linear_type != LINEAR_GROUP, "Do not support group linear TP merge or split"

        # No merge is needed if tp is disabled.
        if not config.tp:
            return

        # Handling constants
        for field_name in [
            "activation_scaling_factor",
            "weights_scaling_factor",
            "weights_scaling_factor_2",
        ]:
            field_value = getattr(config, field_name)
            if field_value is not None and field_value.numel() == 1:
                with get_tensors_parallel(field_value, ranks, group) as scaling_factors:
                    if scaling_factors:
                        # Scaling factor is a scalar.
                        setattr(
                            config,
                            field_name,
                            torch.stack(scaling_factors).max(dim=0).values,
                        )

        # We merge column linear on the dim 0 and row on the dim 1
        merge_axis = 0 if config.linear_type == LINEAR_COLUMN else 1

        assert config.weight is not None
        with get_tensors_parallel(config.weight, ranks, group) as weights:
            if weights:
                config.weight = torch.cat(weights, dim=merge_axis)

        # Only cat the bias for column linear.
        if config.linear_type == LINEAR_COLUMN and config.bias is not None:
            with get_tensors_parallel(config.bias, ranks, group) as biases:
                if biases:
                    config.bias = torch.cat(biases, dim=0)

        if config.linear_type == LINEAR_COLUMN:
            if (
                config.weights_scaling_factor is not None
                and config.weights_scaling_factor.numel() != 1
            ):
                # INT8 sq
                with get_tensors_parallel(
                    config.weights_scaling_factor, ranks, group
                ) as w_scaling_factors:
                    if w_scaling_factors:
                        config.weights_scaling_factor = torch.cat(w_scaling_factors, dim=0)

            if config.prequant_scaling_factor is not None:
                with get_tensors_parallel(
                    config.prequant_scaling_factor, ranks, group
                ) as p_scaling_factors:
                    if p_scaling_factors:
                        # INT4 AWQ, desmooth and de-smooth and re-smooth across all ranks
                        if config.awq_block_size != 0:
                            (
                                config.weight,
                                config.weights_scaling_factor,
                                config.prequant_scaling_factor,
                            ) = resmooth_and_get_scale(
                                merged_weights=config.weight,
                                pre_quant_scales=p_scaling_factors,
                                ranks=len(ranks),
                                group_size=config.awq_block_size,
                                quantization=config.quantization,
                            )
                            if config.quantization in [
                                QUANTIZATION_NVFP4_AWQ,
                                QUANTIZATION_NVFP4_SVDQUANT,
                            ]:
                                # We have to update weight_scaling_factor and weight_scaling_factor_2
                                config.weights_scaling_factor, config.weights_scaling_factor_2 = (
                                    NVFP4QTensor.get_weights_scaling_factor(
                                        config.weight, config.awq_block_size
                                    )
                                )
                        else:
                            assert _same_tensor(p_scaling_factors), (
                                f"Failed to merge config {config} with others"
                            )
        else:
            if config.weights_scaling_factor is not None:
                with get_tensors_parallel(
                    config.weights_scaling_factor, ranks, group
                ) as w_scaling_factors:
                    if w_scaling_factors:
                        if config.awq_block_size != 0:
                            # INT4 AWQ
                            if w_scaling_factors[0].ndim == 2:
                                scaling_factors_total_size = 0
                                for _, w_scaling_factor in enumerate(w_scaling_factors):
                                    scaling_factors_total_size += w_scaling_factor.numel()
                                if scaling_factors_total_size != config.weight.numel():
                                    # The weights from each rank are padded to a multiple of group_size in this case.
                                    # We need to merge the weights and recalculate the scaling factors.
                                    config.weights_scaling_factor = get_scaling_factor_from_weight(
                                        config.weight, config.awq_block_size
                                    )
                                    if config.quantization in [
                                        QUANTIZATION_NVFP4,
                                        QUANTIZATION_NVFP4_AWQ,
                                        QUANTIZATION_NVFP4_SVDQUANT,
                                    ]:
                                        (
                                            config.weights_scaling_factor,
                                            config.weights_scaling_factor_2,
                                        ) = NVFP4QTensor.get_weights_scaling_factor(
                                            config.weight, config.awq_block_size
                                        )
                                else:
                                    config.weights_scaling_factor = torch.cat(
                                        w_scaling_factors, dim=1
                                    )
                            else:
                                raise NotImplementedError(
                                    "Unexpected dimensions for scaling factors."
                                )
                        else:
                            # INT8 SQ
                            config.weights_scaling_factor = (
                                torch.stack(w_scaling_factors).max(dim=0).values
                            )

            if config.prequant_scaling_factor is not None:
                with get_tensors_parallel(
                    config.prequant_scaling_factor, ranks, group
                ) as p_scaling_factors:
                    if p_scaling_factors:
                        config.prequant_scaling_factor = torch.cat(p_scaling_factors, dim=0)

    elif is_dataclass(config):
        for field in fields(config):
            _merge_model_configs_to_first_tp(getattr(config, field.name), ranks, group)
    elif isinstance(config, list):
        for i in range(len(config)):
            _merge_model_configs_to_first_tp(config[i], ranks, group)


def _model_model_configs_to_first_pp(
    model_config: ModelConfig,
    ranks: list[int],
    workspace_path: Path | str | None = None,
):
    """Merges the mode_config from each rank to the first pp rank.

    Args:
        model_config: model or module config
        ranks: a list of ranks in the same pipeline parallel group
        workspace_path: the path to the NFS directory for postprocess cross rank communication.
    """
    # TODO: There is an NCCL error if we try group sync based on the pp_ranks.
    # So we just ask all groups to sync together for now.
    group = None

    # Merge decoder layers.
    decoder_layers = []
    for layer in model_config.layers:
        with get_configs_parallel(
            layer, ranks, group, workspace_path=workspace_path
        ) as layer_configs:
            layer_configs_copy = []
            if layer_configs:
                layer_configs_copy.append(layer_configs[0])
                # Have to copy the config from the other pps as the shm will be releases after
                layer_configs_copy.extend(config for config in layer_configs[1:] if config)
                decoder_layers.append(layer_configs_copy)

    # If we take a 80-layer TP8/PP4 export, then locally each pp rank has 20 layers.
    # If we list the global layer id (0-base) in decoder.layers, then
    #
    # decoder_layers = [
    #   [decoder.layers.0  (owned by rank 0-7), 20 (rank 8-15), 40 (rank 16-23), 60 (rank 24-31)],
    #   [decoder.layers.1  (owned by rank 0-7), 21 (rank 8-15), 41 (rank 16-23), 61 (rank 24-31)],
    #   ...
    #   [decoder.layers.19 (owned by rank 0-7), 39 (rank 8-15), 59 (rank 16-23), 79 (rank 24-31)],
    # ]
    #
    # To merge into PP1, we need to gather and shuffle back to the order of
    #
    # shuffled_config_layers = [
    #   decoder.layers.0  (owned by rank 0-7),
    #   decoder.layers.1  (owned by rank 0-7),
    #   ...
    #   decoder.layers.19 (owned by rank 0-7),
    #   decoder.layers.20 (owned by rank 0-7),
    #   decoder.layers.21 (owned by rank 0-7),
    #   ...
    #   decoder.layers.78 (owned by rank 0-7),
    #   decoder.layers.79 (owned by rank 0-7),
    # ]
    model_config.layers = [
        layer for shuffled_layers in zip(*decoder_layers) for layer in shuffled_layers
    ]

    # Get the ln_f from the last PP rank
    with get_configs_parallel(
        model_config.ln_f, ranks, group, workspace_path=workspace_path
    ) as configs:
        if configs and configs[-1] is not None:
            model_config.ln_f = configs[-1]

    # Get the lm_head from the last PP rank
    with get_configs_parallel(
        model_config.lm_head, ranks, group, workspace_path=workspace_path
    ) as configs:
        if configs and configs[-1] is not None:
            model_config.lm_head = configs[-1]


def postprocess_model_config(
    model_config,
    inference_tensor_parallel: int = 1,
    inference_pipeline_parallel: int = 1,
    training_pipeline_parallel: int = 1,
    workspace_path: Path | str | None = None,
) -> list[ModelConfig]:
    """Postprocesses the model configs with trained tensor parallel to target inference tensor parallel.

    If the training_pipeline_parallel > 1, the model configs across PP will be merged to one.

    Returns:
        The processed model config as a list.
            For the merging case:
                The merged rank will return the merged model_config as an single item list.
                The other ranks will return an empty list as we no longer export them.
            For the split case:
                The splitted model config list is returned.
    """
    rank = dist.rank()

    # We assume the ranks ardistributed in training as [PP size, TP size].
    training_tensor_parallel = dist.size() // training_pipeline_parallel
    tp_rank = rank % training_tensor_parallel
    pp_rank = rank // training_tensor_parallel

    print(f"current rank: {rank}, tp rank: {tp_rank}, pp rank: {pp_rank}")
    dist.barrier()

    # Merge PP ranks to the first
    if training_pipeline_parallel > 1:
        # The pp_ranks for the same tp is [tp_rank, tp_rank + tp, tp_rank + tp * 2, ...]
        pp_ranks = torch.arange(tp_rank, dist.size(), training_tensor_parallel, dtype=int).tolist()

        print(f"PP: Current rank {rank}, merge to {pp_ranks[0]}. Merge group {pp_ranks}")
        dist.barrier()
        _model_model_configs_to_first_pp(
            model_config,
            pp_ranks,
            workspace_path=workspace_path,
        )

    # Returns the empty model_config on other PP ranks.
    if pp_rank != 0:
        model_config.rank = -1
        return []

    # Now we are with PP == 1
    model_config.pipeline_parallel = 1
    model_config.rank = tp_rank

    tp_world_size = dist.size() // training_pipeline_parallel
    if inference_tensor_parallel < tp_world_size:
        # Merge the model_configs to target inference tensor parallel.
        assert tp_world_size % inference_tensor_parallel == 0, (
            f"Cannot merge {tp_world_size} configs to {inference_tensor_parallel}"
        )

        num_configs_per_group = tp_world_size // inference_tensor_parallel
        local_tp_group_id = tp_rank // num_configs_per_group
        tp_ranks = list(
            range(
                local_tp_group_id * num_configs_per_group,
                (local_tp_group_id + 1) * num_configs_per_group,
            )
        )

        print(f"TP: Current rank {rank}, merge to {tp_ranks[0]}. Merge group {tp_ranks}.")
        # We sync on all TP ranks (and pp_rank = 0)
        group = dist.get_group(list(range(training_tensor_parallel)))
        _merge_model_configs_to_first_tp(model_config, tp_ranks, group)
        model_config.tensor_parallel = inference_tensor_parallel
        if rank == tp_ranks[0]:
            model_config.rank = local_tp_group_id
            splitted_model_configs = [model_config]
        else:
            # Mark this config to be invalid and return it as invalid.
            model_config.rank = -1
            return []

    elif inference_tensor_parallel > tp_world_size:
        assert tp_world_size == 1, (
            "We only support splitting a single model config to multiple GPUs"
        )
        split_factor = inference_tensor_parallel // tp_world_size
        splitted_model_configs = _split_model_config_for_tp(
            model_config,
            split_factor,
        )
        for i, config in enumerate(splitted_model_configs):
            config.rank = i
            config.tensor_parallel = inference_tensor_parallel

    else:
        splitted_model_configs = [model_config]

    if inference_pipeline_parallel > 1:
        splitted_model_configs_tp_pp = []
        for i, model_config_tp in enumerate(splitted_model_configs):
            splitted_model_configs_pp = _split_model_config_for_pp(
                model_config_tp, inference_pipeline_parallel
            )
            for j, config in enumerate(splitted_model_configs_pp):
                config.rank = i + j * inference_tensor_parallel
                config.tensor_parallel = inference_tensor_parallel
                config.pipeline_parallel = inference_pipeline_parallel
            splitted_model_configs_tp_pp.extend(splitted_model_configs_pp)
        return splitted_model_configs_tp_pp
    else:
        return splitted_model_configs


def pad_embedding_lm_head(model_config: ModelConfig, padding_factor: int = 64):
    """Pad lm_head and embedding as multiples of 64 for AWQ quantization."""
    vocab_size = model_config.vocab_size
    # Pad the lm_head and vocab_embedding only if the lm_head is quantized with AWQ.
    if vocab_size % padding_factor == 0:
        return

    pad_vocab_size = int((vocab_size + padding_factor - 1) / padding_factor) * padding_factor
    model_config.vocab_size = pad_vocab_size

    if hasattr(model_config, "vocab_embedding"):
        embedding_config = model_config.vocab_embedding
        original_weight = embedding_config.weight
        pad_size = (0, 0, 0, pad_vocab_size - original_weight.shape[0])
        embedding_config.weight = torch.nn.functional.pad(
            original_weight, pad_size, mode="constant", value=0
        )

    if hasattr(model_config, "lm_head"):
        lm_head_config = model_config.lm_head
        original_weight = lm_head_config.weight
        original_bias = lm_head_config.bias
        original_wsf = lm_head_config.weights_scaling_factor

        lm_head_config.weight = torch.nn.functional.pad(
            original_weight,
            (0, 0, 0, pad_vocab_size - original_weight.shape[0]),
            mode="constant",
            value=0,
        )
        if original_bias is not None:
            lm_head_config.bias = torch.nn.functional.pad(
                original_bias,
                (0, pad_vocab_size - original_bias.shape[0]),
                mode="constant",
                value=0,
            )

        if original_wsf is not None:
            assert len(original_wsf.shape) == 2, "AWQ weight scaling factor should be 2D."
            pad_weights_scaling_factor = (
                torch.ones(
                    (pad_vocab_size, original_wsf.shape[1]),
                    dtype=original_wsf.dtype,
                )
                / 7.0  # int4: maxbound = 7.0
            )

            pad_weights_scaling_factor[:vocab_size, :] = original_wsf
            lm_head_config.weights_scaling_factor = pad_weights_scaling_factor


def update_lm_head_quantization(
    config: ModelConfig, lm_head: QuantLinear, inference_tensor_parallel: int = 1
):
    """Update lm_head quantization config for TRT-LLM export."""
    if not hasattr(lm_head, "weight_quantizer") or not hasattr(lm_head, "input_quantizer"):
        return
    weight_quantizer = lm_head.weight_quantizer
    input_quantizer = lm_head.input_quantizer

    disable_lm_head_quantization = False

    assert config.layers is not None, "Expect layers in the model config."

    # check if weight shape is not multiples of block size
    awq_block_size = 0
    weight_quantizer = (
        weight_quantizer[0]
        if isinstance(weight_quantizer, SequentialQuantizer)
        else weight_quantizer
    )
    if hasattr(weight_quantizer, "block_sizes") and weight_quantizer.block_sizes is not None:
        awq_block_size = weight_quantizer.block_sizes[-1]

    if (
        awq_block_size > 0
        and weight_quantizer.is_enabled
        and lm_head.weight.shape[0] % (awq_block_size * inference_tensor_parallel) != 0
    ):
        disable_lm_head_quantization = True

    # disable quantizer
    if disable_lm_head_quantization:
        if hasattr(input_quantizer, "_pre_quant_scale"):
            disable_pre_quant_scale_and_resmooth(lm_head, delete_pre_quant_scale=True)

        lm_head.weight_quantizer.disable()

        input_quantizer.disable()
        print("Disable lm_head quantization for TRT-LLM export due to deployment limitations.")

    else:
        warn(
            "Enable lm_head quantization. lm_head quantization may lead to additional accuracy loss."
        )


def check_weight_shape_valid(config, inference_tensor_parallel=1, training_tensor_parallel=1):
    """Check if weight shape are valid with inference TP.

    This function is recursive.
    """

    def _check_merged_weight(merged_k, tp):
        assert merged_k % tp == 0, f"Weights with shape {merged_k} cannot be split into {tp} ranks."

    def _check_merged_weight_scaling_factor(merged_k, tp, awq_block_size):
        if awq_block_size > 0 and (merged_k // tp) % awq_block_size != 0:
            raise NotImplementedError(
                f"Weight shape {merged_k} of each TP tp={tp} "
                f"is not divisible for block size {awq_block_size} for block quantization."
            )

    def _check_merged_channel_is_valid(merged_k, tp, awq_block_size):
        _check_merged_weight(merged_k=merged_k, tp=tp)
        _check_merged_weight_scaling_factor(merged_k=merged_k, tp=tp, awq_block_size=awq_block_size)

    if isinstance(config, LinearConfig):
        # check weight shape
        if not config.tp:
            inference_tensor_parallel = 1
        if config.linear_type == LINEAR_COLUMN:
            _, k = config.weight.shape
            merged_k = k * training_tensor_parallel
            _check_merged_channel_is_valid(
                merged_k, tp=inference_tensor_parallel, awq_block_size=config.awq_block_size
            )
        elif config.linear_type == LINEAR_ROW:
            k, m = config.weight.shape
            merged_k = k * training_tensor_parallel
            merged_m = m * training_tensor_parallel
            # For int4_awq, weight scaling factors will be split as (k, (merged_m // TP) // block_size)
            _check_merged_weight(merged_k=merged_k, tp=inference_tensor_parallel)
            _check_merged_weight_scaling_factor(
                merged_m, tp=inference_tensor_parallel, awq_block_size=config.awq_block_size
            )

        return

    if isinstance(config, ExpertConfig):
        _, _, k = config.fc.weight.shape
        merged_k = k * training_tensor_parallel
        _check_merged_channel_is_valid(
            merged_k, tp=inference_tensor_parallel, awq_block_size=config.fc.awq_block_size
        )
        return

    if is_dataclass(config):
        for field in fields(config):
            check_weight_shape_valid(
                getattr(config, field.name),
                inference_tensor_parallel,
                training_tensor_parallel,
            )
    elif isinstance(config, list):
        for config_i in config:
            check_weight_shape_valid(config_i, inference_tensor_parallel, training_tensor_parallel)


def postprocess_tensors(
    weights: dict[str, torch.tensor],
    dtype: torch.dtype,
    force_cpu: bool = True,
    force_contiguous: bool = True,
    force_non_view: bool = True,
):
    """Make all tensors in the weights to be with the target dtype, on CPU, contiguous and own the memory."""
    for k, v in weights.items():
        if force_cpu:
            v = v.cpu()
        if (
            any(
                k.endswith(suffix)
                for suffix in [
                    "weight",
                    "bias",
                    "prequant_scaling_factor",
                    "recurrent_param",
                    "rel_attn_table",
                ]
            )
            and "router" not in k
            and v.dtype
            in [
                torch.float16,
                torch.bfloat16,
                torch.float,
            ]
        ):
            v = v.type(dtype)
        if force_contiguous:
            v = v.contiguous()
        if force_non_view and v._is_view():
            v = v.clone()
        weights[k] = v
