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

"""A simple generate Megatron (V)LM models."""

import torch
from megatron.core import mpu
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.timers import Timer
from megatron.core.transformer import MegatronModule
from tqdm import tqdm

__all__ = ["megatron_generate", "megatron_prefill"]


def get_current_memory_info():
    """Get current memory usage."""
    remaining_mem, total_mem = torch.cuda.mem_get_info()
    info = "rank {:3}/{:3}  memory remaining {:03}% ({:d}/{:d} MB) ".format(
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
        int(remaining_mem * 100 / total_mem),
        remaining_mem // 1048576,
        total_mem // 1048576,
    )
    return info


def megatron_prefill(
    model: MegatronModule,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    image_sizes: torch.LongTensor | None = None,
    skip_return_logits: bool = False,
) -> torch.Tensor:
    """A simple prefill function for Megatron Core V(LM) models."""
    if not isinstance(model, MegatronModule):
        raise ValueError("megatron_prefill only supports Megatron Core models.")

    model.eval()

    # Create a static inference context if KV-cache is enabled.
    max_batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[-1]

    def _dummy_loss_func(output_tensor, non_loss_data=True):
        """Need a dummy loss function."""
        return output_tensor

    def _forward_step_func(data, model):
        """Forward step function."""
        batch_size = data["tokens"].shape[0]
        seq_len = data["tokens"].shape[-1]
        device = data["tokens"].device

        # ModelOpt transoformer_spec by default use arbitrary attention mask type; hence we need to
        # compute the attention_mask for prefilling. Alternatively, if "causal" attention mask type
        # is used, the attention_mask is not needed. During generation, the attn_mask_type is overridden
        # to "no_mask" by SelfAttention.forward() if inference_context is provided.
        attention_mask = (
            torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
            .bool()
            .view(batch_size, 1, seq_len, seq_len)
        )

        # NOTE: we don't support traditional positional embedding. Only RoPE or YaRN are supported.
        position_ids = None

        output_tensor = model(
            data["tokens"],
            position_ids,
            attention_mask,
            runtime_gather_output=True,
        )
        return output_tensor, _dummy_loss_func

    if model.config.sequence_parallel:
        tp = model.config.tensor_model_parallel_size
        num_pad_tokens = (tp - input_ids.shape[-1] % tp) % tp
    else:
        num_pad_tokens = 0

    if num_pad_tokens > 0:
        padding_shape = (input_ids.shape[0], num_pad_tokens)
        padded_tokens = torch.full(padding_shape, 0, dtype=input_ids.dtype, device=input_ids.device)
        tokens = torch.cat((input_ids, padded_tokens), dim=-1)
    else:
        tokens = input_ids

    list_of_logits = get_forward_backward_func()(
        forward_step_func=_forward_step_func,
        data_iterator=[{"tokens": tokens}],
        model=model,
        num_microbatches=1,
        seq_length=tokens.shape[-1],
        micro_batch_size=max_batch_size,
        decoder_seq_length=tokens.shape[-1],
        forward_only=True,
        collect_non_loss_data=True,
    )
    if skip_return_logits:
        return None

    if mpu.is_pipeline_last_stage():
        logits = list_of_logits[0][:, :seq_length, :].detach()
    else:
        logits = None

    if model.config.bf16:
        logits_dtype = torch.bfloat16
    elif model.config.fp16:
        logits_dtype = torch.float16
    else:
        logits_dtype = torch.float32
    logits = broadcast_from_last_pipeline_stage(
        [max_batch_size, seq_length, model.vocab_size], logits_dtype, logits
    )

    return logits


def megatron_generate(
    model: MegatronModule,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    image_sizes: torch.LongTensor | None = None,
    osl: int = 32,
    eos_token_id: list[int] = [],
    enable_kv_cache: bool = True,
    disable_tqdm: bool = False,
    return_dict: bool = False,
) -> torch.Tensor | dict:
    """A simple generate function for Megatron Core V(LM) models.

    This function supports TP, PP, EP, and ETP. Sequence parallelism is only supported without KV-cache
    decoding (automatically turned off if KV-cache is enabled). Context parallelism is not tested.
    For MHA and GQA, both native DotProductAttention and TEDotProductAttention are supported. For MLA,
    only TEDotProductAttention is supported.

    When PP>1, all input args must be provided by all PP ranks. Similarly, outputs are broadcasted to
    all PP ranks (from the last pipeline stage).

    Args:
        model: The model to generate from.
        input_ids: The sequence used as a prompt to generate.
        pixel_values: (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
        image_grid_thw: (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        image_sizes: The image sizes.
        osl: The maximum sequence length to generate.
        eos_token_id: The end of sequence token id.
        enable_kv_cache: Whether to enable KV-cache decoding.
        disable_tqdm: Whether to disable the tqdm progress bar.
        return_dict: Whether to return a dictionary that includes other metrics.
    """
    if not isinstance(model, MegatronModule):
        raise ValueError("megatron_generate only supports Megatron Core models.")

    if model.config.sequence_parallel and enable_kv_cache:
        enable_kv_cache = False
        print("Turing off kv-cache decoding since is not implemented for sequence parallelism!")

    model.eval()

    # Create a static inference context if KV-cache is enabled.
    max_batch_size = input_ids.shape[0]
    max_seq_len = input_ids.shape[-1] + osl
    inference_context = (
        StaticInferenceContext(max_batch_size, max_seq_len) if enable_kv_cache else None
    )

    def _dummy_loss_func(output_tensor, non_loss_data=True):
        """Need a dummy loss function."""
        return output_tensor

    def _forward_step_func(data, model):
        """Forward step function."""
        batch_size = data["tokens"].shape[0]
        seq_len = data["tokens"].shape[-1]
        device = data["tokens"].device

        # ModelOpt transoformer_spec by default use arbitrary attention mask type; hence we need to
        # compute the attention_mask for prefilling. Alternatively, if "causal" attention mask type
        # is used, the attention_mask is not needed. During generation, the attn_mask_type is overridden
        # to "no_mask" by SelfAttention.forward() if inference_context is provided.
        if seq_len > 1:
            attention_mask = (
                torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
                .bool()
                .view(batch_size, 1, seq_len, seq_len)
            )
        else:
            attention_mask = None

        # NOTE: we don't support traditional positional embedding. Only RoPE or YaRN are supported.
        position_ids = None

        # Check if this is a VLM model (has vision inputs)
        _has_pixel_values = data.get("pixel_values") is not None
        _has_image_grid_thw = data.get("image_grid_thw") is not None
        _has_image_sizes = data.get("image_sizes") is not None
        has_vision_inputs = _has_pixel_values or _has_image_grid_thw or _has_image_sizes

        if has_vision_inputs:
            # For VLM models:
            # - position_ids: [batch, seq_len] (required for RoPE with multi-modal positions)
            # - attention_mask: [batch, seq_len] (simple 1D boolean mask, not 4D causal)
            vlm_position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            vlm_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

            forward_args = {
                "input_ids": data["tokens"],
                "position_ids": vlm_position_ids,
                "attention_mask": vlm_attention_mask,
                "inference_context": inference_context,
                "runtime_gather_output": True,
            }
            # Add vision inputs
            if _has_pixel_values:
                forward_args["pixel_values"] = data["pixel_values"]
            if _has_image_grid_thw:
                forward_args["image_grid_thw"] = data["image_grid_thw"]
            if _has_image_sizes:
                forward_args["image_sizes"] = data["image_sizes"]

            output_tensor = model(**forward_args)
        else:
            # For text-only LLM models
            output_tensor = model(
                data["tokens"],
                position_ids,
                attention_mask,
                inference_context=inference_context,
                runtime_gather_output=True,
            )
        return output_tensor, _dummy_loss_func

    disable_tqdm = disable_tqdm or torch.distributed.get_rank() > 0

    output_ids = torch.tensor([])
    step_pbar = tqdm(range(osl), disable=disable_tqdm, leave=False)

    time_ttft = 0
    time_remaining_outputs = 0
    timer = Timer("generate")
    timer.start(barrier=True)

    for step in step_pbar:
        step_pbar.set_description(get_current_memory_info())

        if model.config.sequence_parallel:
            tp = model.config.tensor_model_parallel_size
            num_pad_tokens = (tp - input_ids.shape[-1] % tp) % tp
        else:
            num_pad_tokens = 0

        if inference_context is not None and step > 0:
            tokens = input_ids[:, -1:]
            inference_context.enable_decode_mode()
        elif num_pad_tokens > 0:
            padding_shape = (input_ids.shape[0], num_pad_tokens)
            padded_tokens = torch.full(
                padding_shape, 0, dtype=input_ids.dtype, device=input_ids.device
            )
            tokens = torch.cat((input_ids, padded_tokens), dim=-1)
        else:
            tokens = input_ids

        data_dict = {"tokens": tokens}
        # Vision inputs should only be passed during prefill (step 0), not during decode steps
        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            data_dict["image_grid_thw"] = image_grid_thw
        if image_sizes is not None:
            data_dict["image_sizes"] = image_sizes

        list_of_logits = get_forward_backward_func()(
            forward_step_func=_forward_step_func,
            data_iterator=[data_dict],
            model=model,
            num_microbatches=1,
            seq_length=tokens.shape[-1],
            micro_batch_size=max_batch_size,
            decoder_seq_length=tokens.shape[-1],
            forward_only=True,
            collect_non_loss_data=True,
        )

        if inference_context is not None:
            inference_context.sequence_len_offset += tokens.shape[-1]

        if mpu.is_pipeline_last_stage():
            eager_ids = (
                list_of_logits[0][:, -(num_pad_tokens + 1), :].argmax(dim=-1, keepdim=True).detach()
            )
        else:
            eager_ids = None

        eager_ids = broadcast_from_last_pipeline_stage(
            [max_batch_size, 1], input_ids.dtype, eager_ids
        )

        if step > 0:
            output_ids = torch.cat([output_ids, eager_ids], dim=-1)
        else:
            time_ttft = timer.elapsed(barrier=True)
            output_ids = eager_ids

        input_ids = torch.cat([input_ids, eager_ids], dim=-1)

        if eager_ids.item() in eos_token_id:
            break

    time_remaining_outputs = timer.elapsed(barrier=True)

    # print(f"time_ttft: {time_ttft}, time_remaining_outputs: {time_remaining_outputs}")

    if return_dict:
        return {
            "output_ids": output_ids,
            "ttft": time_ttft,
            "tps": time_remaining_outputs / (output_ids.shape[-1] - 1),
        }
    else:
        return output_ids
