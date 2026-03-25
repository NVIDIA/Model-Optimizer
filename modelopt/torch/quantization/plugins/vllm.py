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

"""Support quantization for VLLM layers."""

import importlib
from contextlib import contextmanager

import torch

from ...utils.distributed import ParallelState
from ..nn import QuantLinearConvBase, QuantModule, QuantModuleRegistry, TensorQuantizer

# Try multiple import paths for vLLM compatibility across versions
if importlib.util.find_spec("vllm.attention"):
    import vllm.attention as vllm_attention  # vllm < 0.16.0
else:
    import vllm.model_executor.layers.attention as vllm_attention  # vllm >= 0.16.0

import vllm.model_executor.layers.fused_moe.layer as vllm_fused_moe_layer
import vllm.model_executor.layers.linear as vllm_linear
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

try:
    from vllm.forward_context import get_forward_context as _get_forward_context
except ImportError:
    _get_forward_context = None

vllm_shared_fused_moe_layer = None
for module_path in [
    "vllm.model_executor.layers.fused_moe.shared_fused_moe",  # 0.11.0+
    "vllm.model_executor.layers.shared_fused_moe.shared_fused_moe",  # 0.10.2
]:
    try:
        vllm_shared_fused_moe_layer = importlib.import_module(module_path)
        break
    except ImportError:
        continue


if importlib.util.find_spec("vllm.attention.layers"):  # vllm < 0.15.0
    from vllm.attention.layers.cross_attention import CrossAttention
    from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
else:
    try:
        from vllm.model_executor.layers.attention.cross_attention import CrossAttention
    except ImportError:
        CrossAttention = None
    try:
        from vllm.model_executor.layers.attention.encoder_only_attention import EncoderOnlyAttention
    except ImportError:
        EncoderOnlyAttention = None

if importlib.util.find_spec("vllm.attention.layer"):
    import vllm.attention.layer as vllm_attention

try:
    VllmMLAAttention = vllm_attention.MLAAttention
except ImportError:
    VllmMLAAttention = None

vllm_fused_moe_package = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")


class FakeQuantMethod:
    """A class that implements fake quantization methods for vLLM models.

    This class provides functionality to apply quantization methods to model layers
    in a way that's compatible with vLLM's architecture.
    """

    def __init__(self, quant_method):
        """Initialize the FakeQuantMethod.

        Args:
            quant_method: The quantization method to be applied to the model layers.
        """
        self.quant_method = quant_method

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the quantization method to a given layer.

        Args:
            layer (torch.nn.Module): The neural network layer to be quantized.
            x (torch.Tensor): The input tensor to the layer.
            bias (torch.Tensor | None, optional): The bias tensor to the layer. Defaults to None.

        Returns:
            torch.Tensor: The quantized output tensor.
        """
        x = layer.input_quantizer(x)
        if layer.weight_quantizer.is_enabled:
            original_weight = layer.weight
            quantized_tensor = layer.weight_quantizer(layer.weight)
            # parameterize the quantized weight
            if isinstance(original_weight, torch.nn.Parameter) and not isinstance(
                quantized_tensor, torch.nn.Parameter
            ):
                quantized_tensor = torch.nn.Parameter(
                    quantized_tensor, requires_grad=original_weight.requires_grad
                )
            layer.weight = quantized_tensor
            output = self.quant_method.apply(layer, x, bias)
            layer.weight = original_weight
        else:
            output = self.quant_method.apply(layer, x, bias)
        output = layer.output_quantizer(output)
        return output


def create_parallel_state():
    """Create a parallel state for vLLM."""
    dp_group = get_dp_group().device_group
    tp_group = get_tp_group().device_group
    try:
        ep_group = get_ep_group().device_group
    except AssertionError:
        ep_group = None
    return ParallelState(dp_group, tp_group, ep_group)


class _VLLMParallelLinear(QuantModule):
    def _setup(self):
        self.input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.output_quantizer.disable()
        assert type(self.quant_method) is vllm_linear.UnquantizedLinearMethod, (
            f"quant_method is {type(self.quant_method)}"
        )
        self.fake_quant_method = FakeQuantMethod(self.quant_method)
        self.parallel_state = create_parallel_state()

    def forward(self, input_):
        # This context manager will conflict with torch.compile
        # with replace_function(self, "quant_method", self.fake_quant_method):
        # Manually replace quant_method instead
        self._quant_method = self.quant_method
        self.quant_method = self.fake_quant_method
        output = super().forward(input_)
        self.quant_method = self._quant_method
        return output


@QuantModuleRegistry.register({vllm_linear.RowParallelLinear: "vllm_RowParallelLinear"})
class _QuantVLLMRowParallelLinear(_VLLMParallelLinear):
    pass


@QuantModuleRegistry.register({vllm_linear.ColumnParallelLinear: "vllm_ColumnParallelLinear"})
class _QuantVLLMColumnParallelLinear(_VLLMParallelLinear):
    pass


@QuantModuleRegistry.register(
    {vllm_linear.MergedColumnParallelLinear: "vllm_MergedColumnParallelLinear"}
)
class _QuantVLLMMergedColumnParallelLinear(_VLLMParallelLinear):
    pass


@QuantModuleRegistry.register({vllm_linear.QKVParallelLinear: "vllm_QKVParallelLinear"})
class _QuantVLLMQKVParallelLinear(_VLLMParallelLinear):
    pass


# ReplicatedLinear is for MoE router and should not be quantized


class _QuantFusedMoEBase(QuantModule):
    def _setup(self):
        self.w13_input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.w2_input_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_input)
        self.w13_weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.w2_weight_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_weight)
        self.w13_output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.w2_output_quantizer = TensorQuantizer(QuantLinearConvBase.default_quant_desc_output)
        self.w13_output_quantizer.disable()
        self.w2_output_quantizer.disable()
        assert type(self.quant_method) is vllm_fused_moe_layer.UnquantizedFusedMoEMethod, (
            f"quant_method is {type(self.quant_method)}"
        )
        self.parallel_state = create_parallel_state()

    def invoke_fused_moe_quantized(
        self,
        A: torch.Tensor,  # noqa: N803
        B: torch.Tensor,  # noqa: N803
        C: torch.Tensor,  # noqa: N803
        *args,
        **kwargs,
    ):
        if B is self.w13_weight:
            # First layer of expert
            A = self.w13_input_quantizer(A)  # noqa: N806
            if self.w13_weight_quantizer.is_enabled:
                orig, self.w13_weight = self.w13_weight, self.w13_weight_quantizer(self.w13_weight)
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
                self.w13_weight = orig
            else:
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
            if self.w13_output_quantizer.is_enabled:
                C[:] = self.w13_output_quantizer(C)
        elif B is self.w2_weight:
            A = self.w2_input_quantizer(A)  # noqa: N806
            if self.w2_weight_quantizer.is_enabled:
                orig, self.w2_weight = self.w2_weight, self.w2_weight_quantizer(self.w2_weight)
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
                self.w2_weight = orig
            else:
                vllm_fused_moe_package._invoke_fused_moe_kernel(A, B, C, *args, **kwargs)
            if self.w2_output_quantizer.is_enabled:
                C[:] = self.w2_output_quantizer(C)
        else:
            raise ValueError("Cannot determine first or second layer of expert")

    @contextmanager
    def _patch_moe_kernel(self):
        """Temporarily replace vLLM fused_moe kernel with quantized version."""
        for attr in ["invoke_fused_moe_kernel", "invoke_fused_moe_triton_kernel"]:
            if hasattr(vllm_fused_moe_package, attr):
                orig = getattr(vllm_fused_moe_package, attr)
                setattr(vllm_fused_moe_package, "_invoke_fused_moe_kernel", orig)
                setattr(vllm_fused_moe_package, attr, self.invoke_fused_moe_quantized)
                try:
                    yield
                finally:
                    setattr(vllm_fused_moe_package, attr, orig)
                return
        raise ValueError("fused_moe_kernel is not found")

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        with self._patch_moe_kernel():
            return super().forward(hidden_states, router_logits)

    @torch.no_grad()
    def fold_weight(self, keep_attrs: bool = False):
        # the MoE weights can be super large, it consumes too much memory, so we need to fold the weight one by one
        for i in range(self.w13_weight.shape[0]):
            self.w13_weight[i].copy_(
                self.w13_weight_quantizer(self.w13_weight[i].float().contiguous()).to(
                    self.w13_weight.dtype
                )
            )
        self.w13_weight_quantizer.disable()
        for i in range(self.w2_weight.shape[0]):
            self.w2_weight[i].copy_(
                self.w2_weight_quantizer(self.w2_weight[i].float().contiguous()).to(
                    self.w2_weight.dtype
                )
            )
        self.w2_weight_quantizer.disable()

        torch.cuda.empty_cache()


@QuantModuleRegistry.register({vllm_fused_moe_layer.FusedMoE: "vllm_FusedMoE"})
class _QuantVLLMFusedMoE(_QuantFusedMoEBase):
    pass


if vllm_shared_fused_moe_layer is not None:

    @QuantModuleRegistry.register(
        {vllm_shared_fused_moe_layer.SharedFusedMoE: "vllm_SharedFusedMoE"}
    )
    class _QuantVLLMSharedFusedMoE(_QuantFusedMoEBase):
        pass


def _get_seq_lens_and_block_table(
    attn_metadata: object | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Get (seq_lens, block_table) from attention metadata.

    Supports v0-style metadata (top-level seq_lens, block_table) and
    FlashInferMetadata (TRTLLM prefill/decode with nested seq_lens, block_tables).
    Returns (None, None) when not available.
    """
    if attn_metadata is None:
        return None, None
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    if seq_lens is not None and block_table is not None:
        return seq_lens, block_table
    # FlashInferMetadata: decode at front, prefill at back; TRTLLM uses block_tables/seq_lens
    decode = getattr(attn_metadata, "decode", None)
    prefill = getattr(attn_metadata, "prefill", None)
    decode_sl = getattr(decode, "seq_lens", None) if decode is not None else None
    decode_bt = getattr(decode, "block_tables", None) if decode is not None else None
    prefill_sl = getattr(prefill, "seq_lens", None) if prefill is not None else None
    prefill_bt = getattr(prefill, "block_tables", None) if prefill is not None else None
    if (
        decode_sl is not None
        and decode_bt is not None
        and prefill_sl is None
        and prefill_bt is None
    ):
        return decode_sl, decode_bt
    if (
        prefill_sl is not None
        and prefill_bt is not None
        and decode_sl is None
        and decode_bt is None
    ):
        return prefill_sl, prefill_bt
    if (
        decode_sl is not None
        and decode_bt is not None
        and prefill_sl is not None
        and prefill_bt is not None
    ):
        seq_lens = torch.cat([decode_sl, prefill_sl], dim=0)
        block_table = torch.cat([decode_bt, prefill_bt], dim=0)
        return seq_lens, block_table
    return None, None


@QuantModuleRegistry.register({vllm_attention.Attention: "vllm_Attention"})
class _QuantVLLMAttention(QuantModule):
    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()
        # FP8 quantizers for the skip zone (first_m and last_n positions).
        # When skip zones are configured and these quantizers are enabled (via mtq.quantize),
        # skip-zone tokens are fake-quantized with FP8 rather than left as BF16.
        # The BF16 buffer stores the original BF16 values of last_n tokens so that when
        # they graduate into the quant zone they are quantized cleanly with k_bmm_quantizer
        # (NVFP4) rather than compounding FP8→NVFP4 error.
        self.k_bmm_fp8_quantizer = TensorQuantizer()
        self.v_bmm_fp8_quantizer = TensorQuantizer()
        self.parallel_state = create_parallel_state()
        # K/V skip config set by set_kv_quant_skip_tokens() after mtq.quantize().
        # skip_first_m positions use FP8, skip_last_n positions use FP8 + BF16 buffer.
        self.kv_quant_skip_first_m: int = 0
        self.kv_quant_skip_last_n: int = 0

    def _quantize_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: object,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-token skip-zone KV quantization.

        Quant zone [skip_first_m, seq_len - skip_last_n): NVFP4 via k_bmm_quantizer.
        Skip zone (first_m + last_n): FP8 via k_bmm_fp8_quantizer if enabled, else BF16.
        Graduation: last_n tokens stored in a BF16 buffer; NVFP4 applied at graduation
        to avoid compounding FP8 → NVFP4 error.
        """
        # key/value arrive as 2D [num_tokens, num_kv_heads * head_dim] at this call site;
        # Attention.forward reshapes them to 3D only inside super().forward().
        # Reshape here for per-head indexing; reshape back before returning.
        orig_key_shape = key.shape
        orig_val_shape = value.shape
        if key.ndim == 2:
            key = key.view(-1, self.num_kv_heads, self.head_size)
            head_size_v = getattr(self, "head_size_v", self.head_size)
            value = value.view(-1, self.num_kv_heads, head_size_v)

        k_quantizer = self.k_bmm_quantizer
        v_quantizer = self.v_bmm_quantizer
        k_skip_quantizer = getattr(self, "k_bmm_fp8_quantizer", lambda x: x)
        v_skip_quantizer = getattr(self, "v_bmm_fp8_quantizer", lambda x: x)
        skip_quant_enabled = (
            k_skip_quantizer is not None
            and isinstance(k_skip_quantizer, TensorQuantizer)
            and k_skip_quantizer.is_enabled
        )

        skip_first_m = self.kv_quant_skip_first_m
        skip_last_n = self.kv_quant_skip_last_n
        device = kv_cache.device
        # kv_cache layout:
        #   FlashInfer (blocks_first=True):  (num_blocks, 2, block_size, num_heads, head_dim)
        #   FA2 / others (blocks_first=False): (2, num_blocks, block_size, num_heads, head_dim)
        block_size = kv_cache.shape[2]
        blocks_first = kv_cache.shape[1] == 2

        if not isinstance(seq_lens, torch.Tensor):
            seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)
        else:
            seq_lens_t = seq_lens.to(device=device, dtype=torch.long)
        num_seqs = seq_lens_t.shape[0]

        # Detect pure decode via max_query_len (Python int, no GPU sync).
        max_query_len = getattr(attn_metadata, "max_query_len", None)
        if max_query_len is not None:
            is_pure_decode = max_query_len == 1
        else:
            query_start_loc_fb = getattr(attn_metadata, "query_start_loc", None)
            assert query_start_loc_fb is not None, (
                "Neither attn_metadata.max_query_len nor attn_metadata.query_start_loc "
                "is available for KV skip-zone fake-quantization."
            )
            ql = query_start_loc_fb[1 : num_seqs + 1] - query_start_loc_fb[:num_seqs]
            is_pure_decode = bool(ql.max().item() == 1)

        if is_pure_decode:
            # --- DECODE PATH ---
            if skip_last_n == 0:
                # Quant zone: [skip_first_m, seq_len). Skip zone: [0, skip_first_m).
                # Pre-quantize key/value; impl.forward writes quantized values to kv_cache.
                # No kv_cache read-modify-write; no write-write conflict with impl.forward.
                curr_pos = seq_lens_t - 1  # (num_seqs,)
                in_quant = (curr_pos >= skip_first_m).view(-1, 1, 1)
                if skip_quant_enabled:
                    key = torch.where(in_quant, k_quantizer(key), k_skip_quantizer(key))
                    value = torch.where(in_quant, v_quantizer(value), v_skip_quantizer(value))
                else:
                    key = torch.where(in_quant, k_quantizer(key), key)
                    value = torch.where(in_quant, v_quantizer(value), value)
            else:
                # Current token is always in the last-N skip zone.
                # The token graduating into the quant zone: pos_grad = seq_lens_t - 1 - skip_last_n.
                # Read its original BF16 value from the buffer and apply NVFP4 — avoids
                # compounding FP8 → NVFP4 error.
                num_blocks = kv_cache.shape[0] if blocks_first else kv_cache.shape[1]
                seq_arange = torch.arange(num_seqs, device=device, dtype=torch.long)
                pos_grad = seq_lens_t - 1 - skip_last_n  # (num_seqs,)
                valid = (pos_grad >= skip_first_m) & (pos_grad >= 0)
                pos_clamped = pos_grad.clamp(min=0)
                block_idx = (pos_clamped // block_size).clamp(max=block_table.shape[1] - 1).long()
                slot_in_block = (pos_clamped % block_size).long()
                block_ids = block_table[seq_arange, block_idx].long().clamp(0, num_blocks - 1)
                bm = valid.view(-1, 1, 1)

                # Lazy-init BF16 buffer for the skip zone window.
                # Shape: (num_seqs, skip_last_n, num_kv_heads, head_dim).
                # Allocated on first call (e.g. during CUDA graph warm-up); stable thereafter.
                bf16_buf_k = getattr(self, "bf16_buffer_k", None)
                if (
                    bf16_buf_k is None
                    or bf16_buf_k.shape[0] < num_seqs
                    or bf16_buf_k.shape[1] != skip_last_n
                ):
                    n_heads = key.shape[1]
                    h_dim = key.shape[2]
                    self.bf16_buffer_k = torch.zeros(
                        num_seqs,
                        skip_last_n,
                        n_heads,
                        h_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    self.bf16_buffer_v = torch.zeros_like(self.bf16_buffer_k)
                bf16_buf_k = self.bf16_buffer_k
                bf16_buf_v = self.bf16_buffer_v

                # Graduate: read BF16 from buffer slot 0, apply NVFP4, write to kv_cache.
                k_bf16_grad = bf16_buf_k[:num_seqs, 0]  # (num_seqs, n_heads, h_dim)
                v_bf16_grad = bf16_buf_v[:num_seqs, 0]
                if blocks_first:
                    kv_cache[block_ids, 0, slot_in_block] = torch.where(
                        bm, k_quantizer(k_bf16_grad), kv_cache[block_ids, 0, slot_in_block]
                    )
                    kv_cache[block_ids, 1, slot_in_block] = torch.where(
                        bm, v_quantizer(v_bf16_grad), kv_cache[block_ids, 1, slot_in_block]
                    )
                else:
                    kv_cache[0, block_ids, slot_in_block] = torch.where(
                        bm, k_quantizer(k_bf16_grad), kv_cache[0, block_ids, slot_in_block]
                    )
                    kv_cache[1, block_ids, slot_in_block] = torch.where(
                        bm, v_quantizer(v_bf16_grad), kv_cache[1, block_ids, slot_in_block]
                    )

                # Roll buffer left: evict graduated slot, shift remaining forward.
                if skip_last_n > 1:
                    bf16_buf_k[:num_seqs, :-1] = bf16_buf_k[:num_seqs, 1:].clone()
                    bf16_buf_v[:num_seqs, :-1] = bf16_buf_v[:num_seqs, 1:].clone()

                # Push current BF16 key/value into newest buffer slot (before FP8 is applied).
                bf16_buf_k[:num_seqs, -1] = key.to(torch.bfloat16)
                bf16_buf_v[:num_seqs, -1] = value.to(torch.bfloat16)

                # FP8-quantize the current token entering the skip zone (if configured).
                if skip_quant_enabled:
                    key = k_skip_quantizer(key)
                    value = v_skip_quantizer(value)
        else:
            # --- PREFILL PATH ---
            # Compute each token's absolute position in its sequence:
            #   abs_pos[i] = (seq_lens_t[s] - query_lens[s]) + (i - query_start_loc[s])
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            assert query_start_loc is not None, (
                "attn_metadata.query_start_loc is required for prefill KV quantization."
            )
            query_lens_per_seq = (
                query_start_loc[1 : num_seqs + 1] - query_start_loc[:num_seqs]
            )  # (num_seqs,)
            total_tokens = key.shape[0]

            seq_ids = torch.repeat_interleave(
                torch.arange(num_seqs, device=device, dtype=torch.long), query_lens_per_seq
            )
            seq_start_in_full = seq_lens_t - query_lens_per_seq  # (num_seqs,)
            local_offset = (
                torch.arange(total_tokens, device=device, dtype=torch.long)
                - query_start_loc[seq_ids]
            )
            abs_pos = seq_start_in_full[seq_ids] + local_offset  # (total_tokens,)

            seq_len_per_token = seq_lens_t[seq_ids]  # (total_tokens,)
            last_n_start_per_token = (seq_len_per_token - skip_last_n).clamp(min=0)

            # Populate BF16 buffer with last skip_last_n tokens of each sequence
            # (original BF16, before any quantization) for clean decode-time graduation.
            if skip_last_n > 0:
                bf16_buf_k = getattr(self, "bf16_buffer_k", None)
                if (
                    bf16_buf_k is None
                    or bf16_buf_k.shape[0] < num_seqs
                    or bf16_buf_k.shape[1] != skip_last_n
                ):
                    n_heads = key.shape[1]
                    h_dim = key.shape[2]
                    self.bf16_buffer_k = torch.zeros(
                        num_seqs,
                        skip_last_n,
                        n_heads,
                        h_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    self.bf16_buffer_v = torch.zeros_like(self.bf16_buffer_k)
                bf16_buf_k = self.bf16_buffer_k
                bf16_buf_v = self.bf16_buffer_v
                bf16_buf_k[:num_seqs].zero_()
                bf16_buf_v[:num_seqs].zero_()
                for s in range(num_seqs):
                    tok_s = int(query_start_loc[s].item())
                    tok_e = int(query_start_loc[s + 1].item())
                    if tok_s >= tok_e:
                        continue
                    ln_start_s = int(last_n_start_per_token[tok_s].item())
                    abs_pos_s = abs_pos[tok_s:tok_e]
                    in_last_n_mask = abs_pos_s >= ln_start_s
                    k_last_n = key[tok_s:tok_e][in_last_n_mask].to(torch.bfloat16)
                    v_last_n = value[tok_s:tok_e][in_last_n_mask].to(torch.bfloat16)
                    n_fill = min(k_last_n.shape[0], skip_last_n)
                    if n_fill > 0:
                        # Right-align: most recent tokens at end of buffer
                        bf16_buf_k[s, skip_last_n - n_fill :] = k_last_n[-n_fill:]
                        bf16_buf_v[s, skip_last_n - n_fill :] = v_last_n[-n_fill:]

            # Apply quantization: NVFP4 for quant zone, FP8 for skip zones.
            quant_mask = (abs_pos >= skip_first_m) & (abs_pos < last_n_start_per_token)
            if skip_quant_enabled:
                skip_mask = (abs_pos < skip_first_m) | (abs_pos >= last_n_start_per_token)
                bm_quant = quant_mask.view(-1, 1, 1)
                bm_skip = skip_mask.view(-1, 1, 1)
                key = torch.where(
                    bm_quant,
                    k_quantizer(key),
                    torch.where(bm_skip, k_skip_quantizer(key), key),
                )
                value = torch.where(
                    bm_quant,
                    v_quantizer(value),
                    torch.where(bm_skip, v_skip_quantizer(value), value),
                )
            else:
                bm = quant_mask.view(-1, 1, 1)
                key = torch.where(bm, k_quantizer(key), key)
                value = torch.where(bm, v_quantizer(value), value)

        # Restore original 2D shape so super().forward() receives what it expects.
        key = key.view(orig_key_shape)
        value = value.view(orig_val_shape)
        return key, value

    def forward(self, query, key, value, *args, **kwargs):
        # Q is not cached so quantize it here.
        query = self.q_bmm_quantizer(query)

        if self.kv_quant_skip_first_m > 0 or self.kv_quant_skip_last_n > 0:
            # kv_cache and attn_metadata are not passed as arguments in vLLM 0.15+;
            # they live in the forward context. QuantModule does an in-place class
            # swap so Attention attributes (kv_cache, layer_name) are on self.
            kv_cache = None
            attn_metadata = None
            if _get_forward_context is not None:
                try:
                    fwd_ctx = _get_forward_context()
                    kv_cache = self.kv_cache[fwd_ctx.virtual_engine]
                    attn_metadata = fwd_ctx.attn_metadata
                    if isinstance(attn_metadata, dict):
                        attn_metadata = attn_metadata[self.layer_name]
                except Exception:
                    pass

            if kv_cache is not None and attn_metadata is not None:
                seq_lens, block_table = _get_seq_lens_and_block_table(attn_metadata)
                if seq_lens is not None and block_table is not None:
                    # Inference path: apply per-token skip-zone quantization.
                    key, value = self._quantize_kv(
                        key, value, kv_cache, attn_metadata, seq_lens, block_table
                    )
                    return super().forward(query, key, value, *args, **kwargs)

        # skip zones == 0: either no-skip inference (quantize normally) or calibration.
        # Skip zones are always 0 during mtq.quantize() — they are set afterward via
        # set_kv_quant_skip_tokens(). _if_calib is True only during the calibration
        # forward loop, so it is safe to use as the calibration signal.
        key = self.k_bmm_quantizer(key)
        value = self.v_bmm_quantizer(value)
        if self.k_bmm_fp8_quantizer._if_calib:
            self.k_bmm_fp8_quantizer(key)
            self.v_bmm_fp8_quantizer(value)

        return super().forward(query, key, value, *args, **kwargs)


def set_kv_quant_skip_tokens(model, skip_first_m: int = 0, skip_last_n: int = 0):
    """Configure first-M / last-N KV cache quantization skipping for all attention layers.

    After mtq.quantize(), call this to skip fake-quantization of the first
    ``skip_first_m`` and last ``skip_last_n`` KV tokens at attention time.
    Setting both to 0 disables skipping (all tokens are quantized as normal).
    """
    for module in model.modules():
        if isinstance(module, _QuantVLLMAttention):
            module.kv_quant_skip_first_m = skip_first_m
            module.kv_quant_skip_last_n = skip_last_n


if CrossAttention is not None:

    @QuantModuleRegistry.register({CrossAttention: "vllm_CrossAttention"})
    class _QuantVLLMCrossAttention(_QuantVLLMAttention):
        pass


if EncoderOnlyAttention is not None:

    @QuantModuleRegistry.register({EncoderOnlyAttention: "vllm_EncoderOnlyAttention"})
    class _QuantVLLMEncoderOnlyAttention(_QuantVLLMAttention):
        pass


if VllmMLAAttention is not None:

    @QuantModuleRegistry.register({VllmMLAAttention: "vllm_MLAAttention"})
    class _QuantVLLMMLAAttention(QuantModule):
        def _setup(self):
            self.q_bmm_quantizer = TensorQuantizer()
            self.kv_c_bmm_quantizer = TensorQuantizer()
            self.k_pe_bmm_quantizer = TensorQuantizer()
            self.parallel_state = create_parallel_state()

        def forward(self, query, kv_c, k_pe, *args, **kwargs):
            query = self.q_bmm_quantizer(query)
            kv_c = self.kv_c_bmm_quantizer(kv_c)
            k_pe = self.k_pe_bmm_quantizer(k_pe)
            return super().forward(query, kv_c, k_pe, *args, **kwargs)
