# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""C1 memory consumer: feed the previous verify's rejected rows to the next DFlash draft step.

Enable it in the server's environment and load the rejmem worker extension::

    MODELOPT_REJMEM_CONSUMER=1 PYTHONPATH=<modelopt tree> vllm serve <target> \\
        --speculative-config '{"method": "dflash", "model": "<drafter>", "num_speculative_tokens": 15}' \\
        --worker-extension-cls modelopt.torch.speculative.plugins.rejmem.vllm_worker.RejMemWorkerExtension \\
        --max-model-len <max prompt_len + max_tokens + 3 * num_speculative_tokens>

The ``3 * NS`` margin covers the memory rows of the step that async scheduling runs past ``max_tokens``.
It needs vLLM 0.33.0 with Model Runner V2's DFlash speculator, non-causal draft layers, the FLASH_ATTN
draft backend and draft sliding windows of at least ``2 * NS + 1``, and refuses anything else at startup.

The speculator's context precompute already projects and RoPEs the rejected rows of a verify, then
drops them (position 0, PAD slot). Here rejected row ``i`` of a request whose new anchor is ``a`` keeps
its real position ``a + i`` and takes the KV slot of logical index ``a + 1 + NS + i``, right after the
draft block, and the draft seq_len grows by ``r``: every query of the block attends to the ``r`` memory
rows, and FlashAttention's bottom-right aligned window moves right by ``r`` (CONTRACT.md sections 1-2).
The draft seq_len equals the target's seq_len plus ``1 + NS``, so vLLM's host-side bounds still hold.
A memory row left without a KV slot (``max_model_len`` too small) stops the worker with an error one or
two draft steps later; ``dropped_memory_rows()`` counts them synchronously, e.g. after a rollout batch.
"""

import hashlib
import inspect

import torch
import vllm
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.cp_utils import cp_local_slot
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash

from modelopt import env

__all__ = ["dropped_memory_rows", "install"]

_VLLM_VERSION = "0.33.0"
# sha256 of the vLLM 0.33.0 sources this module replaces, copies or wraps.
_SOURCE_SHA256 = {
    "prepare_dflash_inputs": "85265096f19142fe24a1bddf6a27bfa4775cbc9ccf16f7fc0e356f35ff0840c4",
    "_prepare_dflash_inputs_kernel": "98c1fc588215ae00e6c0979d91e6fc07f707961354dd09d2d46d003954f925c3",
    "DFlashSpeculator.propose": "b63f4ad371bec2be33179a522569eb226d1140d6428a71581c4bc829543f2b42",
    "DFlashSpeculator.set_attn": "968678300f8342154cbab8ee1f1d663a55972ae86a8dd205ed1ce901e2c3b799",
}

_installed = False
_active = False  # set once this worker's DFlash speculator passed _check_speculator
# Memory rows left without a KV slot: (device counter, pinned host copy).
_dropped_rows: tuple[torch.Tensor, torch.Tensor] | None = None


def install() -> None:
    """Patch vLLM's DFlash speculator when ``MODELOPT_REJMEM_CONSUMER`` is on; no-op otherwise."""
    global _installed
    if _installed or not env.rejmem_consumer():
        return
    _check_vllm()
    original_set_attn = dflash.DFlashSpeculator.set_attn
    original_initialize_from_config = gpu_worker.Worker.initialize_from_config

    def set_attn(self, *args, **kwargs):
        global _active
        original_set_attn(self, *args, **kwargs)
        _check_speculator(self)
        _active = True

    def initialize_from_config(self, *args, **kwargs):
        original_initialize_from_config(self, *args, **kwargs)
        # The KV cache setup has built the draft attention by now. Without the MRV2 DFlash
        # speculator (V1 runner, another method, no drafter) the patch would never run.
        if not _active:
            raise NotImplementedError(
                "MODELOPT_REJMEM_CONSUMER needs Model Runner V2's DFlash speculator, which this "
                "server did not build."
            )

    dflash.DFlashSpeculator.set_attn = set_attn
    gpu_worker.Worker.initialize_from_config = initialize_from_config
    # propose() looks the module attribute up at call time.
    dflash.prepare_dflash_inputs = prepare_dflash_inputs
    _installed = True


def dropped_memory_rows() -> int:
    """Memory rows that found no KV slot since startup; 0 unless ``max_model_len`` is too small.

    Synchronizes with the GPU, so unlike the check in each draft step it also covers the last steps
    before the server goes idle: call it before using a finished rollout batch.
    """
    if _dropped_rows is None:
        return 0
    torch.cuda.synchronize(_dropped_rows[0].device)
    return int(_dropped_rows[0].item())


def _check_vllm() -> None:
    if vllm.__version__ != _VLLM_VERSION:
        raise RuntimeError(
            f"MODELOPT_REJMEM_CONSUMER supports vLLM {_VLLM_VERSION} only, found {vllm.__version__}."
        )
    sources = {
        "prepare_dflash_inputs": dflash.prepare_dflash_inputs,
        "_prepare_dflash_inputs_kernel": dflash._prepare_dflash_inputs_kernel.fn,
        "DFlashSpeculator.propose": dflash.DFlashSpeculator.propose,
        "DFlashSpeculator.set_attn": dflash.DFlashSpeculator.set_attn,
    }
    changed = [
        name
        for name, obj in sources.items()
        if hashlib.sha256(inspect.getsource(obj).encode()).hexdigest() != _SOURCE_SHA256[name]
    ]
    if changed:
        raise RuntimeError(
            f"MODELOPT_REJMEM_CONSUMER: {', '.join(changed)} in {dflash.__name__} differ from "
            f"vLLM {_VLLM_VERSION}; the memory consumer patches that exact code."
        )


def _check_speculator(spec: dflash.DFlashSpeculator) -> None:
    """Refuse the drafters for which slot redirect does not give the memory semantics."""
    if type(spec) is not dflash.DFlashSpeculator:
        raise NotImplementedError(
            f"MODELOPT_REJMEM_CONSUMER supports the DFlash speculator only, not {type(spec).__name__}."
        )
    # A causal query would see memory rows by offset, not all of them.
    if any(spec.model.get_draft_attn_causal()):
        raise NotImplementedError("MODELOPT_REJMEM_CONSUMER needs non-causal draft layers.")
    backends = {group.backend.get_name() for groups in spec.attn_groups for group in groups}
    if backends != {"FLASH_ATTN"}:
        # The sliding-window shift by r is FlashAttention's bottom-right aligned window.
        raise NotImplementedError(
            f"MODELOPT_REJMEM_CONSUMER needs the FLASH_ATTN draft backend, got {sorted(backends)}."
        )
    # The symmetric (W - 1, W - 1) window ends after the memory rows: with r = NS of them, the
    # last query still sees the anchor only if W >= 2 * NS + 1.
    min_window = 2 * spec.num_speculative_steps + 1
    windows = [layer.self_attn.attn.impl.sliding_window[0] + 1 for layer in spec.model.model.layers]
    if any(0 < w < min_window for w in windows):
        raise NotImplementedError(
            f"MODELOPT_REJMEM_CONSUMER needs draft sliding windows of at least "
            f"2 * num_speculative_tokens + 1 = {min_window}, got {sorted(set(windows))}."
        )


# Copy of vllm 0.33.0 dflash/speculator.py::_prepare_dflash_inputs_kernel. Changed: the context rows
# (the rejected suffix gets real positions and the slots after the block) and seq_lens (+ num_rejected).
@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_is_padding_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    out_dropped_rows_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    SAMPLE_FROM_ANCHOR: tl.constexpr,  # noqa: N803
    PAD_SLOT_ID: tl.constexpr,  # noqa: N803
    CP_SIZE: tl.constexpr,  # noqa: N803
    CP_INTERLEAVE: tl.constexpr,  # noqa: N803
    BLOCK_SIZE: tl.constexpr,  # noqa: N803
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    # --- Context positions / slots ---
    # The rejected suffix rows are this round's memory: row i keeps its real position
    # anchor + i and is stored at logical KV index anchor + i + num_query_per_req.
    is_memory = is_ctx & ~is_valid_ctx
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_kv_pos = tl.where(is_memory, ctx_pos + num_query_per_req, ctx_pos)
    ctx_block_num = ctx_kv_pos // (block_size * CP_SIZE)
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block. Old sliding-window context positions can map
    # to it after eviction. Neither they nor a memory row past max_model_len
    # may write draft KV.
    ctx_resident = is_ctx & (ctx_block_id != 0) & (ctx_kv_pos < max_model_len)
    local_ctx_slot = cp_local_slot(
        ctx_kv_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID
    )
    ctx_slot = tl.where(
        ctx_resident,
        local_ctx_slot,
        PAD_SLOT_ID,
    )
    tl.atomic_add(out_dropped_rows_ptr, tl.sum((is_memory & ~ctx_resident).to(tl.int32)))
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // (block_size * CP_SIZE)
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    # A null block is never a writable cache slot. This can occur when a
    # sliding-window block table contains evicted/global padding entries.
    q_resident = is_query & (q_block_id != 0)
    local_q_slot = cp_local_slot(
        query_pos,
        q_block_id,
        block_size,
        cp_rank,
        CP_SIZE,
        CP_INTERLEAVE,
        PAD_SLOT_ID,
    )
    q_slot = tl.where(
        q_resident,
        local_q_slot,
        PAD_SLOT_ID,
    )

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    tl.store(out_is_padding_ptr + query_idx, False, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # When SAMPLE_FROM_ANCHOR (DSpark), so we sample at EVERY query position
    # and each position k predicts the NEXT token (sampled position = query_pos + 1).
    # Otherwise (DFlash default) the anchor is the bonus token and only the mask tokens
    # at offsets > 0 are sampled from, each AT its own position.
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens is the absolute sequence length the draft attention
        # reads up to (context + query + memory), not just the count of
        # accepted tokens this step.
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req + num_rejected, max_model_len),
        )
        # Copy sampling state.
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # Padded sample slots point at query index 0 (a valid row in
            # last_hidden_states) so CG replay never reads OOB. Padded
            # sample idx mappings point to -1, which is ignored during
            # sampling to prevent writing stale values to draft logits.
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            # Pad query slot mappings past num_query_tokens with PAD so the
            # captured CG sees PAD slots (no K/V write) for replay sizes
            # larger than the current request count.
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_input_ids_ptr + block, 0, mask=mask)
                tl.store(out_query_positions_ptr + block, 0, mask=mask)
                tl.store(out_is_padding_ptr + block, True, mask=mask)
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
    num_reqs_override: int | None = None,
) -> None:
    """VLLM 0.33.0's ``prepare_dflash_inputs`` launching the memory kernel above."""
    global _dropped_rows
    if _dropped_rows is None:
        _dropped_rows = (
            torch.zeros(1, dtype=torch.int32, device=query_slot_mapping.device),
            torch.zeros(1, dtype=torch.int32).pin_memory(),
        )
    dropped, dropped_host = _dropped_rows
    # Read without a sync: the copy below lands a step or two later.
    if dropped_host.item():
        raise RuntimeError(
            f"MODELOPT_REJMEM_CONSUMER: {dropped_host.item()} memory rows found no KV slot; set "
            "--max-model-len >= prompt_len + max_tokens + 3 * num_speculative_tokens."
        )

    num_reqs = input_batch.num_reqs if num_reqs_override is None else num_reqs_override
    assert num_reqs <= input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens[:num_reqs].max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))  # noqa: N806
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dflash_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.is_padding,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        dropped,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    dropped_host.copy_(dropped, non_blocking=True)
