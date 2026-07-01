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

import inspect

import torch
from megatron.core import mpu
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
    recv_from_prev_pipeline_rank_,
    send_to_next_pipeline_rank,
)
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.timers import Timer
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_attr_wrapped_model, get_batch_on_this_cp_rank
from tqdm import tqdm

__all__ = ["megatron_generate", "megatron_prefill"]

# ``is_hybrid_cp`` exists only in newer Megatron-Core; pass it only if the signature accepts it.
_CP_BATCH_KWARGS = (
    {"is_hybrid_cp": False}
    if "is_hybrid_cp" in inspect.signature(get_batch_on_this_cp_rank).parameters
    else {}
)


def cp_split_sequence(input_ids: torch.Tensor, cp_group) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition ``input_ids`` + global ``position_ids`` across the CP group (zigzag layout).

    Returns ``(local_input_ids, local_position_ids)`` for this CP rank; inverse is
    :func:`cp_gather_logits`. Sequence length must be divisible by ``2 * cp_size``.
    """
    cp_size = torch.distributed.get_world_size(cp_group)
    if input_ids.shape[-1] % (2 * cp_size) != 0:
        raise ValueError(
            f"Context parallelism requires sequence length divisible by 2 * cp_size "
            f"(= {2 * cp_size}), got {input_ids.shape[-1]}."
        )
    # .contiguous(): get_batch_on_this_cp_rank reshapes via .view(), which rejects expand()'s stride-0 view.
    position_ids = (
        torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand(input_ids.shape[0], -1)
        .contiguous()
    )
    batch = get_batch_on_this_cp_rank(
        {"input_ids": input_ids, "position_ids": position_ids},
        cp_group=cp_group,
        **_CP_BATCH_KWARGS,
    )
    return batch["input_ids"], batch["position_ids"]


def cp_gather_logits(local_logits: torch.Tensor, cp_group, global_seq_len: int) -> torch.Tensor:
    """Reconstruct full-sequence logits from per-CP-rank logits (inverse of :func:`cp_split_sequence`).

    Rank ``r`` holds the zigzag pair ``[chunk_r, chunk_{2*cp_size-1-r}]``; all-gather and re-order
    the chunks back into ``[B, global_seq_len, vocab]``.
    """
    cp_size = torch.distributed.get_world_size(cp_group)
    if global_seq_len % (2 * cp_size) != 0:
        raise ValueError(
            f"global_seq_len (= {global_seq_len}) must be divisible by 2 * cp_size (= {2 * cp_size})."
        )
    chunk = global_seq_len // (2 * cp_size)
    gathered = [torch.empty_like(local_logits) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, local_logits.contiguous(), group=cp_group)
    chunks: list[torch.Tensor | None] = [None] * (2 * cp_size)
    for r in range(cp_size):
        chunks[r] = gathered[r][:, :chunk, :]
        chunks[2 * cp_size - 1 - r] = gathered[r][:, chunk:, :]
    return torch.cat(chunks, dim=1)


def get_current_memory_info():
    """Get current memory usage."""
    remaining_mem, total_mem = torch.cuda.mem_get_info()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    remaining_pct = int(remaining_mem * 100 / total_mem)
    info = (
        f"rank {rank:3}/{world_size:3}  memory remaining "
        f"{remaining_pct:03}% ({remaining_mem // 1048576:d}/{total_mem // 1048576:d} MB) "
    )
    return info


@torch.no_grad()
def megatron_prefill(
    model: MegatronModule,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    image_sizes: torch.LongTensor | None = None,
    position_ids: torch.LongTensor | None = None,
    skip_return_logits: bool = False,
) -> torch.Tensor:
    """A simple prefill function for Megatron Core V(LM) models.

    Supports TP, PP, SP, CP, EP, and combinations thereof. For PP, activations are communicated
    explicitly between pipeline stages (rather than through get_forward_backward_func)
    so that the training pipeline scheduler does not interfere with inference.

    Under CP (>1) pass each rank its local ``input_ids`` slice (via :func:`cp_split_sequence`); the
    CP-aware attention handles causal masking internally (requires ``AttnMaskType.causal``). RoPE is
    applied by the model per CP rank, so ``position_ids`` are only needed for absolute/learned
    position embeddings.
    """
    if not isinstance(model, MegatronModule):
        raise ValueError("megatron_prefill only supports Megatron Core models.")

    model.eval()

    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[-1]
    device = input_ids.device

    pp_first = mpu.is_pipeline_first_stage()
    pp_last = mpu.is_pipeline_last_stage()
    is_pp = not (pp_first and pp_last)
    pp_dtype = model.config.pipeline_dtype or (
        torch.bfloat16 if model.config.bf16 else torch.float32
    )

    if model.config.sequence_parallel:
        tp = model.config.tensor_model_parallel_size
        num_pad_tokens = (tp - seq_length % tp) % tp
    else:
        num_pad_tokens = 0

    if num_pad_tokens > 0:
        tokens = torch.cat(
            [
                input_ids,
                torch.zeros(batch_size, num_pad_tokens, dtype=input_ids.dtype, device=device),
            ],
            dim=-1,
        )
    else:
        tokens = input_ids

    padded_seq_len = tokens.shape[-1]

    cp_size = mpu.get_context_parallel_world_size()

    # Under CP a local triu mask would be wrong for the per-rank zigzag chunks; pass None and let
    # the CP-aware causal attention build it. Without CP the causal mask must be supplied explicitly.
    if cp_size > 1:
        attention_mask = None
    else:
        attention_mask = (
            torch.triu(
                torch.ones((batch_size, padded_seq_len, padded_seq_len), device=device), diagonal=1
            )
            .bool()
            .view(batch_size, 1, padded_seq_len, padded_seq_len)
        )

    # position_ids feed only absolute/learned embeddings (RoPE is internal). Default to a contiguous
    # local range; CP callers pass the partitioned global positions.
    if position_ids is None:
        position_ids = (
            torch.arange(padded_seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
    elif num_pad_tokens > 0:
        # Match the SP token padding so position_ids lines up with `tokens`.
        position_ids = torch.cat(
            [
                position_ids,
                torch.zeros(batch_size, num_pad_tokens, dtype=position_ids.dtype, device=device),
            ],
            dim=-1,
        )

    # For PP, receive activations from the previous stage before calling forward.
    if is_pp and not pp_first:
        pp_dtype = model.config.pipeline_dtype or (
            torch.bfloat16 if model.config.bf16 else torch.float32
        )
        recv_buffer = torch.empty(
            (padded_seq_len, batch_size, model.config.hidden_size),
            dtype=pp_dtype,
            device=device,
        )
        recv_from_prev_pipeline_rank_(recv_buffer)
        get_attr_wrapped_model(model, "set_input_tensor")(recv_buffer)

    has_vision_inputs = (
        pixel_values is not None or image_grid_thw is not None or image_sizes is not None
    )
    if has_vision_inputs:
        forward_kwargs: dict = {
            "input_ids": tokens,
            "position_ids": position_ids,
            "attention_mask": torch.ones(
                (batch_size, padded_seq_len), dtype=torch.bool, device=device
            ),
            "runtime_gather_output": True,
        }
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw
        if image_sizes is not None:
            forward_kwargs["image_sizes"] = image_sizes
        output = model(**forward_kwargs)
    else:
        output = model(tokens, position_ids, attention_mask, runtime_gather_output=True)

    # Some VLM wrappers (e.g. Gemma3VLModel) return ``(logits, loss_mask)`` rather than a bare tensor.
    if isinstance(output, (tuple, list)):
        output = output[0]

    # For PP non-last stages, forward activations to the next stage and return early.
    if is_pp and not pp_last:
        pp_dtype = model.config.pipeline_dtype or (
            torch.bfloat16 if model.config.bf16 else torch.float32
        )
        send_to_next_pipeline_rank(output.to(dtype=pp_dtype))

    # .contiguous() is required because the slice is a view with the padded stride; the broadcast
    # below asserts contiguity when SP pads seq_length up to a multiple of TP.
    logits = output[:, :seq_length, :].detach().contiguous() if pp_last else None

    if model.config.bf16:
        logits_dtype = torch.bfloat16
    elif model.config.fp16:
        logits_dtype = torch.float16
    else:
        logits_dtype = torch.float32

    # All PP ranks must participate in the broadcast to stay in sync.
    # VLM wrappers (e.g. Qwen3VLModel) hold the output layer on the inner language_model, so the
    # vocab size lives there rather than on the wrapper itself.
    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(model, "language_model", model).vocab_size
    result = broadcast_from_last_pipeline_stage(
        [batch_size, seq_length, vocab_size], logits_dtype, logits
    )
    return None if skip_return_logits else result


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

    pp_first = mpu.is_pipeline_first_stage()
    pp_last = mpu.is_pipeline_last_stage()
    is_pp = not (pp_first and pp_last)
    pp_dtype = model.config.pipeline_dtype or (
        torch.bfloat16 if model.config.bf16 else torch.float32
    )

    # Create a static inference context if KV-cache is enabled.
    max_batch_size = input_ids.shape[0]
    max_seq_len = input_ids.shape[-1] + osl
    inference_context = (
        StaticInferenceContext(max_batch_size, max_seq_len) if enable_kv_cache else None
    )

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
            num_pad_tokens = 0
        elif num_pad_tokens > 0:
            padding_shape = (input_ids.shape[0], num_pad_tokens)
            padded_tokens = torch.full(
                padding_shape, 0, dtype=input_ids.dtype, device=input_ids.device
            )
            tokens = torch.cat((input_ids, padded_tokens), dim=-1)
        else:
            tokens = input_ids

        batch_size = tokens.shape[0]
        seq_len = tokens.shape[-1]
        device = tokens.device

        # ModelOpt transformer_spec uses arbitrary attention mask type by default; compute causal
        # mask for prefill. During decode, attn_mask_type is overridden to "no_mask" by
        # SelfAttention.forward() when inference_context is provided.
        if seq_len > 1:
            attention_mask = (
                torch.triu(torch.ones((batch_size, seq_len, seq_len), device=device), diagonal=1)
                .bool()
                .view(batch_size, 1, seq_len, seq_len)
            )
        else:
            attention_mask = None

        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Check if this is a VLM model (vision inputs only passed at step 0 / prefill)
        _has_pixel_values = step == 0 and pixel_values is not None
        _has_image_grid_thw = step == 0 and image_grid_thw is not None
        _has_image_sizes = step == 0 and image_sizes is not None
        has_vision_inputs = _has_pixel_values or _has_image_grid_thw or _has_image_sizes

        # For PP, receive activations from the previous stage before calling forward.
        if is_pp and not pp_first:
            recv_buffer = torch.empty(
                (seq_len, batch_size, model.config.hidden_size),
                dtype=pp_dtype,
                device=device,
            )
            recv_from_prev_pipeline_rank_(recv_buffer)
            get_attr_wrapped_model(model, "set_input_tensor")(recv_buffer)

        if has_vision_inputs:
            forward_args = {
                "input_ids": tokens,
                "position_ids": position_ids,
                "attention_mask": torch.ones(
                    (batch_size, seq_len), dtype=torch.bool, device=device
                ),
                "inference_context": inference_context,
                "runtime_gather_output": True,
            }
            if _has_pixel_values:
                forward_args["pixel_values"] = pixel_values
            if _has_image_grid_thw:
                forward_args["image_grid_thw"] = image_grid_thw
            if _has_image_sizes:
                forward_args["image_sizes"] = image_sizes
            output = model(**forward_args)
        else:
            output = model(
                tokens,
                position_ids,
                attention_mask,
                inference_context=inference_context,
                runtime_gather_output=True,
            )

        if inference_context is not None:
            inference_context.sequence_len_offset += seq_len

        # For PP non-last stages, forward activations to the next stage.
        if is_pp and not pp_last:
            send_to_next_pipeline_rank(output.to(dtype=pp_dtype))

        if pp_last:
            eager_ids = output[:, -(num_pad_tokens + 1), :].argmax(dim=-1, keepdim=True).detach()
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
