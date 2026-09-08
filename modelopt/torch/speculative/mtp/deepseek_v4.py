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

"""DeepSeek-V4 checkpoint-native MTP adapter.

The released DSV4 inference implementation is intentionally inference-only:
its MTP forward is decorated with ``torch.inference_mode`` and calls TileLang
quantized kernels.  This adapter instead uses the differentiable DSV4
Transformers decoder primitive and materializes the checkpoint's FP8/MXFP4
weights as BF16 MTP master parameters.  The target embedding and LM head are
loaded separately and frozen.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.speculative.eagle.utils import IGNORE_TOKEN_ID, masked_soft_target_cross_entropy

from .adapter import NativeMTPAdapter, NativeMTPCheckpointError, register_native_mtp_adapter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = [
    "DeepSeekV4MTPAdapter",
    "NativeMTPBoostModel",
    "decode_dsv4_fp8_weight",
    "decode_dsv4_mxfp4_weight",
    "dsv4_target_features",
    "encode_dsv4_fp8_weight",
    "encode_dsv4_mxfp4_weight",
    "load_dsv4_target_model",
]


_MTP_PREFIX = "mtp.0."
_FP8_BLOCK_SIZE = 128
_MXFP4_BLOCK_SIZE = 32
_FP8_MAX = 448.0
_MXFP4_MAX = 6.0
_MXFP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set the default dtype while constructing a meta module."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def _default_device(device: torch.device) -> Iterator[None]:
    """Temporarily route implicit vendor allocations to the target device."""
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _is_fsdp_efficient_nonzero_rank() -> bool:
    """Return whether this process should leave model state on ``meta`` for FSDP2."""
    efficient_loading = os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "false").lower()
    if efficient_loading not in ("1", "true", "yes", "on"):
        return False
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() != 0
    return int(os.environ.get("RANK", "0")) != 0


def _dsv4_hadamard_fallback(x: torch.Tensor) -> torch.Tensor:
    """Apply the vendor's normalized Hadamard rotation using native Torch ops."""
    if x.dtype is not torch.bfloat16:
        raise ValueError(f"DSV4 Hadamard input must be BF16, got {x.dtype}")
    width = x.shape[-1]
    if width < 1:
        raise ValueError("DSV4 Hadamard input must have a nonempty final dimension")
    padded_width = 1 << (width - 1).bit_length()
    output = F.pad(x.reshape(-1, width), (0, padded_width - width))
    stride = 1
    while stride < padded_width:
        blocks = output.reshape(-1, padded_width // (2 * stride), 2, stride)
        left = blocks[:, :, 0]
        right = blocks[:, :, 1]
        output = torch.cat((left + right, left - right), dim=-1).reshape(-1, padded_width)
        stride *= 2
    return (output[:, :width] * (width**-0.5)).reshape(x.shape)


def _require_safetensors():
    """Import safetensors only for checkpoint I/O."""
    try:
        from safetensors.torch import safe_open, save_file
    except ImportError as exc:
        raise ImportError(
            "Loading a checkpoint-native MTP requires the optional safetensors dependency."
        ) from exc
    return safe_open, save_file


def _require_dsv4_transformers():
    """Import DSV4 primitives lazily so codec tests need no Transformers install."""
    try:
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4DecoderLayer,
            DeepseekV4HyperHead,
            DeepseekV4RMSNorm,
            DeepseekV4RotaryEmbedding,
        )
        from transformers.utils import ModelOutput
    except ImportError as exc:
        raise ImportError(
            "DeepSeek-V4 native MTP boosting requires a Transformers version with "
            "transformers.models.deepseek_v4 (ModelOpt's supported Transformers 5 build)."
        ) from exc
    return (
        DeepseekV4Config,
        DeepseekV4DecoderLayer,
        DeepseekV4HyperHead,
        DeepseekV4RMSNorm,
        DeepseekV4RotaryEmbedding,
        ModelOutput,
    )


def load_dsv4_target_model(
    model_path: str | Path,
    converted_checkpoint: str | Path,
    device: torch.device | str,
    *,
    max_batch_size: int,
    max_seq_len: int,
) -> nn.Module:
    """Load the frozen vendor target used by both online training and feature dumps."""
    model_path = Path(model_path)
    inference_dir = model_path / "inference"
    config_path = inference_dir / "config.json"
    if not inference_dir.is_dir() or not config_path.is_file():
        raise NativeMTPCheckpointError(
            f"{model_path} does not contain the DeepSeek-V4 inference release files"
        )

    checkpoint = Path(converted_checkpoint)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if checkpoint.is_dir():
        checkpoint = checkpoint / f"model{rank}-mp{world_size}.safetensors"
    if not checkpoint.is_file():
        raise NativeMTPCheckpointError(f"Converted target checkpoint does not exist: {checkpoint}")

    sys.path.insert(0, str(inference_dir))
    try:
        from model import ModelArgs, Transformer
    except ImportError as error:
        raise ImportError(
            "Could not import the DeepSeek-V4 inference implementation. Install the "
            "dependencies listed in inference/requirements.txt."
        ) from error
    finally:
        sys.path.pop(0)

    # The upstream release lists fast_hadamard_transform, but its PyPI source
    # archive omits CUDA sources and it has no ARM64 wheel.  Its only uses in
    # this target path rotate width-128 tensors, for which this native-Torch
    # fallback is mathematically identical and small relative to target cost.
    try:
        from fast_hadamard_transform import hadamard_transform as _hadamard_transform  # noqa: F401
    except (ImportError, OSError):
        setattr(
            sys.modules[Transformer.__module__],
            "rotate_activation",
            _dsv4_hadamard_fallback,
        )

    with config_path.open() as config_file:
        model_args = ModelArgs(**json.load(config_file))
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.n_mtp_layers = 0

    target_device = torch.device(device)
    previous_device = torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_device(target_device)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Transformer(model_args)
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)

    try:
        from safetensors.torch import load_model
    except ImportError as exc:
        raise ImportError("Loading the DSV4 target requires safetensors.") from exc
    load_model(model, str(checkpoint), strict=False)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.inference_mode()
def dsv4_target_features(
    model: nn.Module, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the target's raw HC state and normalized LM-head input."""
    # The vendor FP8 kernels allocate outputs with torch.get_default_dtype().
    # The release implementation intentionally omits ``device=`` from helper
    # allocations such as its cached sparse-attention indices.  Keep those
    # allocations colocated with the MP rank's inputs during the entire
    # forward, rather than falling back to CPU or cuda:0.
    with _default_device(input_ids.device), _default_dtype(torch.bfloat16):
        hidden_states = model.embed(input_ids)
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for layer in model.layers:
            hidden_states = layer(hidden_states, 0, input_ids)
        lm_head_input = model.head.hc_head(
            hidden_states,
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
        )
        return hidden_states, model.norm(lm_head_input)


def _native_mtp_output_type(model_output_type: type) -> type:
    """Create the dataclass output required by Transformers' ``ModelOutput`` base."""

    @dataclass
    class NativeMTPBoostOutput(model_output_type):
        loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        hidden_states: torch.Tensor | None = None
        raw_hidden_states: torch.Tensor | None = None
        train_acc: list[list[float]] | None = None
        eagle_loss: torch.Tensor | None = None

    return NativeMTPBoostOutput


def _e8m0_exponents(scale: torch.Tensor) -> torch.Tensor:
    """Return signed powers-of-two exponents from a raw UE8M0 tensor."""
    return scale.contiguous().view(torch.uint8).to(torch.int32) - 127


def _as_e8m0(raw: torch.Tensor) -> torch.Tensor:
    """Bit-cast raw E8M0 bytes to the DSV4 safetensors storage dtype."""
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:
        raise RuntimeError("This PyTorch build does not provide torch.float8_e8m0fnu")
    return raw.contiguous().view(dtype)


def decode_dsv4_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize DSV4's E4M3 × UE8M0 128x128-block FP8 weight format."""
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("DSV4 block-FP8 weights and scales must both be rank 2")
    rows, columns = weight.shape
    scale_rows, scale_columns = scale.shape
    padded_rows = scale_rows * _FP8_BLOCK_SIZE
    padded_columns = scale_columns * _FP8_BLOCK_SIZE
    if rows > padded_rows or columns > padded_columns:
        raise ValueError(
            f"FP8 scale shape {tuple(scale.shape)} cannot cover weight shape {tuple(weight.shape)}"
        )

    padded = F.pad(weight.float(), (0, padded_columns - columns, 0, padded_rows - rows))
    exponents = _e8m0_exponents(scale)
    expanded = exponents.repeat_interleave(_FP8_BLOCK_SIZE, 0).repeat_interleave(_FP8_BLOCK_SIZE, 1)
    return torch.ldexp(padded, expanded)[:rows, :columns].to(dtype)


def encode_dsv4_fp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Statically quantize a BF16 master weight to DSV4's block-FP8 layout."""
    if weight.ndim != 2:
        raise ValueError(f"DSV4 block-FP8 weight must be rank 2, got rank {weight.ndim}")
    rows, columns = weight.shape
    padded_rows = math.ceil(rows / _FP8_BLOCK_SIZE) * _FP8_BLOCK_SIZE
    padded_columns = math.ceil(columns / _FP8_BLOCK_SIZE) * _FP8_BLOCK_SIZE
    padded = F.pad(weight.float(), (0, padded_columns - columns, 0, padded_rows - rows))
    blocks = padded.view(
        padded_rows // _FP8_BLOCK_SIZE,
        _FP8_BLOCK_SIZE,
        padded_columns // _FP8_BLOCK_SIZE,
        _FP8_BLOCK_SIZE,
    )
    amax = blocks.abs().amax(dim=(1, 3))
    min_exp = torch.full_like(amax, -127.0)
    exponent = torch.ceil(torch.where(amax > 0, torch.log2(amax / _FP8_MAX), min_exp))
    exponent = exponent.clamp(min=-127, max=127).to(torch.int32)
    expanded = exponent.repeat_interleave(_FP8_BLOCK_SIZE, 0).repeat_interleave(_FP8_BLOCK_SIZE, 1)
    quantized = torch.ldexp(padded, -expanded).clamp(-_FP8_MAX, _FP8_MAX)
    raw_scale = (exponent + 127).to(torch.uint8)
    return quantized[:rows, :columns].to(torch.float8_e4m3fn), _as_e8m0(raw_scale)


def decode_dsv4_mxfp4_weight(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize DSV4's packed E2M1 × UE8M0-per-32-element expert format."""
    if packed_weight.ndim < 1 or scale.ndim < 1:
        raise ValueError("DSV4 MXFP4 weights and scales must have at least one dimension")
    packed = packed_weight.contiguous().view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    logical_shape = nibbles.shape
    if logical_shape[-1] % _MXFP4_BLOCK_SIZE:
        raise ValueError(f"MXFP4 logical width must be divisible by {_MXFP4_BLOCK_SIZE}")
    expected_scale_shape = (*logical_shape[:-1], logical_shape[-1] // _MXFP4_BLOCK_SIZE)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale.shape)} does not match expected {expected_scale_shape}"
        )

    values = nibbles.new_tensor(_MXFP4_VALUES, dtype=torch.float32)
    sign = torch.where((nibbles & 0x08) == 0, 1.0, -1.0)
    magnitudes = values[(nibbles & 0x07).to(torch.long)]
    decoded = sign * magnitudes
    blocks = decoded.view(*logical_shape[:-1], -1, _MXFP4_BLOCK_SIZE)
    return (
        torch.ldexp(blocks, _e8m0_exponents(scale).unsqueeze(-1)).reshape(logical_shape).to(dtype)
    )


def encode_dsv4_mxfp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Statically quantize a BF16 master expert weight to DSV4's MXFP4 layout."""
    if weight.ndim < 1 or weight.shape[-1] % _MXFP4_BLOCK_SIZE:
        raise ValueError(
            "DSV4 MXFP4 weight width must be divisible by "
            f"{_MXFP4_BLOCK_SIZE}, got {tuple(weight.shape)}"
        )
    shape = weight.shape
    blocks = weight.float().view(*shape[:-1], -1, _MXFP4_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    min_exp = torch.full_like(amax, -127.0)
    exponent = torch.ceil(torch.where(amax > 0, torch.log2(amax / _MXFP4_MAX), min_exp))
    exponent = exponent.clamp(min=-127, max=127).to(torch.int32)
    normalized = torch.ldexp(blocks, -exponent.unsqueeze(-1))

    bounds = normalized.new_tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    # The native MXFP4 encoder selects the next code only when the magnitude
    # is strictly above a midpoint.  ``right=False`` preserves exact midpoint
    # ties (for example .25 and .75) in the lower representable bucket.
    magnitude_code = torch.bucketize(normalized.abs(), bounds, right=False).to(torch.uint8)
    sign_code = (normalized < 0).to(torch.uint8) << 3
    codes = (magnitude_code | sign_code).reshape(shape)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    raw_scale = (exponent + 127).to(torch.uint8)
    return packed.view(torch.int8), _as_e8m0(raw_scale)


@dataclass(frozen=True)
class _CheckpointLayout:
    """The indexed safetensors layout needed for strict load/export."""

    path: Path
    config: dict[str, Any]
    weight_map: dict[str, str]
    embedding_key: str
    lm_head_key: str


class _TensorReader:
    """Open each source shard at most once while reading selected tensors."""

    def __init__(self, layout: _CheckpointLayout):
        self.layout = layout
        self._stack: contextlib.ExitStack | None = None
        self._handles: dict[str, Any] = {}

    def __enter__(self) -> _TensorReader:
        self._stack = contextlib.ExitStack()
        return self

    def __exit__(self, *args) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None
        self._handles.clear()

    def get(self, key: str) -> torch.Tensor:
        try:
            shard = self.layout.weight_map[key]
        except KeyError as exc:
            raise NativeMTPCheckpointError(f"Missing checkpoint tensor {key!r}") from exc
        if shard not in self._handles:
            assert self._stack is not None, "_TensorReader must be used as a context manager"
            safe_open, _ = _require_safetensors()
            self._handles[shard] = self._stack.enter_context(
                safe_open(str(self.layout.path / shard), framework="pt", device="cpu")
            )
        return self._handles[shard].get_tensor(key)


def _read_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    try:
        with config_path.open() as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise NativeMTPCheckpointError(f"Missing {config_path}") from exc
    if config.get("model_type") != "deepseek_v4":
        raise NativeMTPCheckpointError(
            "DeepSeek-V4 adapter requires model_type='deepseek_v4', "
            f"got {config.get('model_type')!r}"
        )
    return config


def _read_weight_map(path: Path) -> dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            weight_map = json.load(handle).get("weight_map")
        if not isinstance(weight_map, dict):
            raise NativeMTPCheckpointError(f"Invalid safetensors index at {index_path}")
        return weight_map

    single = path / "model.safetensors"
    if not single.is_file():
        raise NativeMTPCheckpointError(
            f"Expected model.safetensors.index.json or model.safetensors in {path}"
        )
    safe_open, _ = _require_safetensors()
    with safe_open(str(single), framework="pt", device="cpu") as handle:
        return dict.fromkeys(handle.keys(), single.name)


def _expected_mtp_keys(config: Mapping[str, Any]) -> set[str]:
    try:
        num_experts = int(config["n_routed_experts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeMTPCheckpointError("DSV4 config is missing a valid n_routed_experts") from exc
    if int(config.get("num_nextn_predict_layers", 0) or 0) != 1:
        raise NativeMTPCheckpointError(
            "The initial native DSV4 MTP adapter supports exactly one checkpoint MTP layer"
        )

    keys = {
        "attn.attn_sink",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "attn_norm.weight",
        "e_proj.weight",
        "e_proj.scale",
        "enorm.weight",
        "ffn.gate.bias",
        "ffn.gate.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w1.scale",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w2.scale",
        "ffn.shared_experts.w3.weight",
        "ffn.shared_experts.w3.scale",
        "ffn_norm.weight",
        "h_proj.weight",
        "h_proj.scale",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "hnorm.weight",
        "norm.weight",
    }
    for projection in ("wkv", "wo_a", "wo_b", "wq_a", "wq_b"):
        keys.update((f"attn.{projection}.weight", f"attn.{projection}.scale"))
    for expert_index in range(num_experts):
        for projection in ("w1", "w2", "w3"):
            keys.update(
                (
                    f"ffn.experts.{expert_index}.{projection}.weight",
                    f"ffn.experts.{expert_index}.{projection}.scale",
                )
            )
    return {_MTP_PREFIX + key for key in keys}


def _checkpoint_layout(model_path: str | Path, *, strict: bool) -> _CheckpointLayout:
    path = Path(model_path)
    if not path.is_dir():
        raise NativeMTPCheckpointError(f"Native MTP checkpoint must be a directory, got {path}")
    config = _read_config(path)
    weight_map = _read_weight_map(path)
    actual_mtp = {key for key in weight_map if key.startswith(_MTP_PREFIX)}
    if not actual_mtp:
        raise NativeMTPCheckpointError("DSV4 checkpoint has no mtp.0.* tensors")
    if strict:
        expected = _expected_mtp_keys(config)
        missing = sorted(expected - actual_mtp)
        unexpected = sorted(actual_mtp - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing[:8]}" + ("..." if len(missing) > 8 else ""))
            if unexpected:
                details.append(
                    f"unexpected={unexpected[:8]}" + ("..." if len(unexpected) > 8 else "")
                )
            raise NativeMTPCheckpointError(
                "DSV4 native MTP checkpoint coverage is not exact: " + ", ".join(details)
            )

    embedding_key = "embed.weight"
    if embedding_key not in weight_map:
        raise NativeMTPCheckpointError("DSV4 checkpoint is missing its shared embed.weight")
    if "head.weight" in weight_map:
        lm_head_key = "head.weight"
    elif config.get("tie_word_embeddings"):
        lm_head_key = embedding_key
    else:
        raise NativeMTPCheckpointError(
            "DSV4 checkpoint is missing head.weight and does not tie embeddings"
        )
    return _CheckpointLayout(path, config, weight_map, embedding_key, lm_head_key)


def _decode_weight(
    weight: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        if scale is None:
            raise NativeMTPCheckpointError("DSV4 FP8 weight has no UE8M0 scale tensor")
        return decode_dsv4_fp8_weight(weight, scale, dtype=dtype)
    if weight.dtype in (torch.int8, torch.uint8):
        if scale is None:
            raise NativeMTPCheckpointError("DSV4 MXFP4 weight has no UE8M0 scale tensor")
        return decode_dsv4_mxfp4_weight(weight, scale, dtype=dtype)
    return weight.to(dtype)


def _state_tensor(state_dict: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    if name in state_dict:
        return state_dict[name]
    module_name = "module." + name
    if module_name in state_dict:
        return state_dict[module_name]
    raise KeyError(f"Missing native MTP state tensor {name!r}")


class NativeMTPBoostModel(nn.Module):
    """Standalone differentiable DSV4 MTP with frozen shared target endpoints."""

    is_native_mtp_boost = True

    def __init__(
        self,
        config: Any,
        *,
        mtp_layer_index: int,
        rollout_steps: int = 1,
        hsm_mode: str = "off",
    ) -> None:
        """Construct one trainable native MTP layer and its frozen target endpoints."""
        super().__init__()
        (
            _,
            decoder_layer_cls,
            hyper_head_cls,
            rms_norm_cls,
            rotary_embedding_cls,
            model_output_cls,
        ) = _require_dsv4_transformers()
        self.config = config
        self._model_output_type = _native_mtp_output_type(model_output_cls)
        self.mtp_layer_index = mtp_layer_index
        self.rollout_steps = rollout_steps
        self.hsm_mode = hsm_mode
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = nn.Module()
        self.mtp.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mtp.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mtp.enorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp.hnorm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp.decoder = decoder_layer_cls(config, mtp_layer_index)
        self.mtp.hc_head = hyper_head_cls(config)
        self.mtp.norm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp.rotary_emb = rotary_embedding_cls(config)

        # HF represents routing correction state as persistent buffers.  The
        # FSDP2 CPU-efficient full-state broadcast expects every persistent
        # state entry to have a DTensor placement, and floating MTP state must
        # also participate in optimization. Promote such state without
        # changing its checkpoint key; integer routing tables remain frozen.
        for module in self.mtp.modules():
            for name, buffer in list(module._buffers.items()):
                if (
                    buffer is None
                    or name in module._non_persistent_buffers_set
                    or isinstance(buffer, nn.Parameter)
                ):
                    continue
                delattr(module, name)
                module.register_parameter(
                    name,
                    nn.Parameter(buffer, requires_grad=buffer.is_floating_point()),
                )

    @property
    def device(self) -> torch.device:
        """Return the trainable MTP device."""
        return self.mtp.e_proj.weight.device

    def freeze_target_endpoints(self) -> None:
        """Ensure shared target embedding/head never become optimizer parameters."""
        self.embedding.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def set_online_target_feature_provider(
        self,
        provider: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        *,
        model_parallel: bool,
    ) -> None:
        """Attach a frozen target callable without registering its weights on this module."""
        self._target_feature_provider = provider
        self._target_model_parallel = model_parallel

    def _online_target_features(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        provider = getattr(self, "_target_feature_provider", None)
        if provider is None:
            raise ValueError(
                "Online native MTP training requires an attached target feature provider"
            )

        target_input_ids = input_ids.detach()
        batch_start = 0
        if getattr(self, "_target_model_parallel", False):
            gathered = [
                torch.empty_like(target_input_ids)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered, target_input_ids)
            target_input_ids = torch.cat(gathered)
            batch_start = torch.distributed.get_rank() * input_ids.shape[0]

        raw_hidden_states, lm_head_hidden_states = provider(target_input_ids)
        batch_slice = slice(batch_start, batch_start + input_ids.shape[0])
        return (
            raw_hidden_states[batch_slice].detach().clone(),
            lm_head_hidden_states[batch_slice].detach().clone(),
        )

    def _attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        valid = positions[:, None] >= positions[None, :]
        window = int(self.config.sliding_window)
        valid &= positions[:, None] - positions[None, :] < window
        min_value = torch.finfo(dtype).min
        causal = torch.where(
            valid,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), min_value, dtype=dtype, device=device),
        )
        batch_size = 1 if attention_mask is None else attention_mask.shape[0]
        causal = causal.view(1, 1, sequence_length, sequence_length).expand(batch_size, -1, -1, -1)
        if attention_mask is None:
            return causal
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"attention_mask must have shape {(batch_size, sequence_length)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        padding = ~attention_mask.to(torch.bool)
        return causal.masked_fill(padding[:, None, None, :], min_value)

    @staticmethod
    def _shift_tokens(input_ids: torch.Tensor, steps: int) -> torch.Tensor:
        """Shift teacher-forced tokens left without wrapping the trailing positions."""
        shifted = torch.zeros_like(input_ids)
        shifted[:, :-steps] = input_ids[:, steps:]
        return shifted

    @staticmethod
    def _sample_uniform_hsm(hidden_state_history: list[torch.Tensor]) -> torch.Tensor:
        """Sample one prior raw MTP state per token, matching EAGLE uniform HSM."""
        stacked = torch.stack(hidden_state_history, dim=0)
        num_candidates, batch_size, sequence_length, hc_mult, hidden_size = stacked.shape
        candidate_indices = torch.randint(
            num_candidates,
            (1, batch_size, sequence_length, 1, 1),
            device=stacked.device,
        )
        return stacked.gather(
            0,
            candidate_indices.expand(-1, -1, -1, hc_mult, hidden_size),
        ).squeeze(0)

    def _run_mtp_round(
        self,
        input_ids: torch.Tensor,
        raw_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        token_shift: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one normal-causal MTP round without a KV cache."""
        dtype = self.mtp.e_proj.weight.dtype
        shifted_input_ids = self._shift_tokens(input_ids, token_shift)
        tokens = self.mtp.enorm(self.embedding(shifted_input_ids).to(dtype))
        hidden_states = self.mtp.hnorm(raw_hidden_states.to(dtype))
        hidden_states = self.mtp.e_proj(tokens).unsqueeze(2) + self.mtp.h_proj(hidden_states)

        batch_size, sequence_length = input_ids.shape
        positions = (
            torch.arange(sequence_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        position_embeddings = {
            layer_type: self.mtp.rotary_emb(tokens, position_ids=positions, layer_type=layer_type)
            for layer_type in ("main", "compress")
        }
        causal_mask = self._attention_mask(
            attention_mask,
            sequence_length=sequence_length,
            dtype=dtype,
            device=input_ids.device,
        )
        raw_hidden_states = self.mtp.decoder(
            hidden_states,
            input_ids=shifted_input_ids,
            position_embeddings=position_embeddings,
            position_ids=positions,
            attention_mask=causal_mask,
            past_key_values=None,
        )
        hidden_states = self.mtp.norm(self.mtp.hc_head(raw_hidden_states))
        return raw_hidden_states, hidden_states, self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        mtp_boost_inputs: Mapping[str, torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
        *,
        target_mtp_hidden_states: torch.Tensor | None = None,
        target_lm_head_hidden_states: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        """Run cache-free MTP rollouts and optionally compute EAGLE's logit loss.

        ``mtp_boost_inputs`` is the offline dataset contract.  Direct target
        tensor kwargs are retained for small standalone callers.
        """
        if (
            mtp_boost_inputs is None
            and target_mtp_hidden_states is None
            and target_lm_head_hidden_states is None
        ):
            target_mtp_hidden_states, target_lm_head_hidden_states = self._online_target_features(
                input_ids
            )
        if mtp_boost_inputs is not None:
            if target_mtp_hidden_states is not None or target_lm_head_hidden_states is not None:
                raise ValueError(
                    "Pass native target features either nested or as direct kwargs, not both"
                )
            try:
                target_mtp_hidden_states = mtp_boost_inputs["target_mtp_hidden_states"]
                target_lm_head_hidden_states = mtp_boost_inputs["target_lm_head_hidden_states"]
            except KeyError as exc:
                raise KeyError(
                    "mtp_boost_inputs must contain target_mtp_hidden_states and "
                    "target_lm_head_hidden_states"
                ) from exc
        if target_mtp_hidden_states is None:
            raise ValueError("Native MTP boost requires target_mtp_hidden_states")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, sequence], got {tuple(input_ids.shape)}")
        expected_hidden_shape = (
            *input_ids.shape,
            int(self.config.hc_mult),
            self.config.hidden_size,
        )
        if tuple(target_mtp_hidden_states.shape) != expected_hidden_shape:
            raise ValueError(
                "target_mtp_hidden_states must have shape "
                f"{expected_hidden_shape}, got {tuple(target_mtp_hidden_states.shape)}"
            )

        _, sequence_length = input_ids.shape
        rollout_steps = getattr(self, "rollout_steps", 1)
        hsm_mode = getattr(self, "hsm_mode", "off")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if hsm_mode not in {"off", "uniform_layer_sample"}:
            raise ValueError(f"Unsupported native MTP HSM mode {hsm_mode!r}")
        if hsm_mode == "uniform_layer_sample" and rollout_steps < 2:
            raise ValueError("uniform_layer_sample HSM requires at least two MTP rollouts")
        if target_lm_head_hidden_states is not None and rollout_steps >= sequence_length:
            raise ValueError(
                "rollout_steps must be smaller than the sequence length when computing MTP loss"
            )

        teacher_probabilities = None
        teacher_logits = None
        if target_lm_head_hidden_states is not None:
            expected_teacher_shape = (*input_ids.shape, self.config.hidden_size)
            if tuple(target_lm_head_hidden_states.shape) != expected_teacher_shape:
                raise ValueError(
                    "target_lm_head_hidden_states must have shape "
                    f"{expected_teacher_shape}, got {tuple(target_lm_head_hidden_states.shape)}"
                )
            # Cached target features are teacher-only inputs.  Keep their
            # reconstruction outside autograd even if callers pass tensors
            # with ``requires_grad=True``.
            with torch.no_grad():
                teacher_logits = self.lm_head(
                    target_lm_head_hidden_states.to(
                        device=self.lm_head.weight.device,
                        dtype=self.lm_head.weight.dtype,
                    )
                )
                teacher_probabilities = torch.softmax(teacher_logits, dim=-1)
        if loss_mask is None:
            loss_mask = (
                attention_mask.to(torch.bool)
                if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.bool)
            )
            if labels is not None:
                loss_mask &= labels.ne(IGNORE_TOKEN_ID)

        raw_hidden_states = target_mtp_hidden_states.detach()
        raw_hidden_state_history = [raw_hidden_states]
        losses = []
        accuracies = []
        hidden = None
        logits = None
        for rollout_index in range(rollout_steps):
            raw_hidden_states, hidden, logits = self._run_mtp_round(
                input_ids,
                raw_hidden_states,
                attention_mask,
                token_shift=rollout_index + 1,
            )
            if teacher_probabilities is not None and teacher_logits is not None:
                # At round n, MTP token t predicts target token t + n + 1.
                target_start = rollout_index + 1
                student_logits = logits[:, :-target_start]
                aligned_mask = loss_mask[:, target_start:].to(student_logits.dtype)
                losses.append(
                    masked_soft_target_cross_entropy(
                        teacher_probabilities[:, target_start:], student_logits, aligned_mask
                    )
                )
                with torch.no_grad():
                    valid = aligned_mask.bool()
                    correct = (
                        teacher_logits[:, target_start:].argmax(-1) == student_logits.argmax(-1)
                    ) & valid
                    accuracies.append((correct.sum().float() / valid.sum().clamp_min(1)).item())
            if rollout_index + 1 < rollout_steps:
                raw_hidden_state_history.append(raw_hidden_states)
                raw_hidden_states = (
                    self._sample_uniform_hsm(raw_hidden_state_history)
                    if hsm_mode == "uniform_layer_sample"
                    else raw_hidden_states
                )

        loss = sum(losses) if losses else None

        return self._model_output_type(
            loss=loss,
            logits=logits,
            hidden_states=hidden,
            raw_hidden_states=raw_hidden_states,
            train_acc=[accuracies or [0.0]],
            eagle_loss=loss,
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        **_: Any,
    ) -> None:
        """Save a resumable BF16 training checkpoint for ``Trainer`` integration."""
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        source_state = self.state_dict() if state_dict is None else state_dict
        state = {
            key.removeprefix("module."): value.detach().to(
                device="cpu",
                dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
            )
            for key, value in source_state.items()
            if key.removeprefix("module.").startswith("mtp.")
        }
        if not state:
            raise RuntimeError("Native MTP save received no mtp.* state tensors")
        checkpoint = output / "mtp_boost.pt"
        torch.save(state, checkpoint)
        # Trainer resume discovery recognizes this conventional filename.  A
        # hard link avoids duplicating the large BF16 MTP master checkpoint.
        resume_checkpoint = output / "pytorch_model.bin"
        try:
            os.link(checkpoint, resume_checkpoint)
        except OSError:
            shutil.copy2(checkpoint, resume_checkpoint)
        with (output / "mtp_boost_config.json").open("w") as handle:
            json.dump(
                {
                    "adapter": "deepseek_v4",
                    "mtp_layer_index": self.mtp_layer_index,
                    "rollout_steps": self.rollout_steps,
                    "hsm_mode": self.hsm_mode,
                },
                handle,
                indent=2,
                sort_keys=True,
            )


_DIRECT_STATE_KEYS = {
    "attn.attn_sink": "mtp.decoder.self_attn.sinks",
    "attn.kv_norm.weight": "mtp.decoder.self_attn.kv_norm.weight",
    "attn.q_norm.weight": "mtp.decoder.self_attn.q_a_norm.weight",
    "attn.wkv.weight": "mtp.decoder.self_attn.kv_proj.weight",
    "attn.wo_a.weight": "mtp.decoder.self_attn.o_a_proj.weight",
    "attn.wo_b.weight": "mtp.decoder.self_attn.o_b_proj.weight",
    "attn.wq_a.weight": "mtp.decoder.self_attn.q_a_proj.weight",
    "attn.wq_b.weight": "mtp.decoder.self_attn.q_b_proj.weight",
    "attn_norm.weight": "mtp.decoder.input_layernorm.weight",
    "e_proj.weight": "mtp.e_proj.weight",
    "enorm.weight": "mtp.enorm.weight",
    "ffn.gate.bias": "mtp.decoder.mlp.gate.e_score_correction_bias",
    "ffn.gate.weight": "mtp.decoder.mlp.gate.weight",
    "ffn.shared_experts.w1.weight": "mtp.decoder.mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w2.weight": "mtp.decoder.mlp.shared_experts.down_proj.weight",
    "ffn.shared_experts.w3.weight": "mtp.decoder.mlp.shared_experts.up_proj.weight",
    "ffn_norm.weight": "mtp.decoder.post_attention_layernorm.weight",
    "h_proj.weight": "mtp.h_proj.weight",
    "hc_attn_base": "mtp.decoder.attn_hc.base",
    "hc_attn_fn": "mtp.decoder.attn_hc.fn",
    "hc_attn_scale": "mtp.decoder.attn_hc.scale",
    "hc_ffn_base": "mtp.decoder.ffn_hc.base",
    "hc_ffn_fn": "mtp.decoder.ffn_hc.fn",
    "hc_ffn_scale": "mtp.decoder.ffn_hc.scale",
    "hc_head_base": "mtp.hc_head.hc_base",
    "hc_head_fn": "mtp.hc_head.hc_fn",
    "hc_head_scale": "mtp.hc_head.hc_scale",
    "hnorm.weight": "mtp.hnorm.weight",
    "norm.weight": "mtp.norm.weight",
}


def _expert_state_keys(index: int) -> dict[str, str]:
    prefix = f"mtp.0.ffn.experts.{index}"
    state_prefix = "mtp.decoder.mlp.experts"
    return {
        f"{prefix}.w1.weight": f"{state_prefix}.gate_up_proj",
        f"{prefix}.w2.weight": f"{state_prefix}.down_proj",
        f"{prefix}.w3.weight": f"{state_prefix}.gate_up_proj",
    }


@register_native_mtp_adapter
class DeepSeekV4MTPAdapter(NativeMTPAdapter):
    """Strict loader/exporter for DSV4 Pro's checkpoint-native ``mtp.0`` block."""

    name = "deepseek_v4"

    @classmethod
    def supports(cls, model_path: str | Path) -> bool:
        """Return whether the checkpoint is a DeepSeek-V4 model with a native MTP."""
        try:
            layout = _checkpoint_layout(model_path, strict=False)
        except (NativeMTPCheckpointError, ImportError, OSError, json.JSONDecodeError):
            return False
        return layout.config.get("model_type") == "deepseek_v4" and any(
            key.startswith(_MTP_PREFIX) for key in layout.weight_map
        )

    @classmethod
    def _transformers_config(cls, layout: _CheckpointLayout) -> tuple[Any, int]:
        config_cls, *_ = _require_dsv4_transformers()
        base_layers = int(layout.config["num_hidden_layers"])
        config_data = dict(layout.config)
        # The upstream HF model intentionally omits MTP layers.  Its decoder
        # primitive still uses the per-layer schedules, so retain the extra
        # checkpoint schedule entry while instantiating the single MTP layer.
        config_data["num_hidden_layers"] = base_layers + 1
        config = config_cls(**config_data)
        config._attn_implementation = "eager"
        return config, base_layers

    @classmethod
    def _build_empty_model(
        cls,
        config: Any,
        mtp_layer_index: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        rollout_steps: int,
        hsm_mode: str,
    ) -> NativeMTPBoostModel:
        if dtype is not torch.bfloat16:
            raise ValueError(
                f"DSV4 native MTP training uses BF16 master weights; got dtype={dtype!r}"
            )
        target_device = torch.device("cpu" if device is None else device)
        with torch.device("meta"), _default_dtype(dtype):
            model = NativeMTPBoostModel(
                config,
                mtp_layer_index=mtp_layer_index,
                rollout_steps=rollout_steps,
                hsm_mode=hsm_mode,
            )
        model.to_empty(device=target_device)
        # Rotary buffers are non-persistent and were constructed on ``meta``.
        # Accelerate's CPU-efficient FSDP2 path snapshots these buffers before
        # moving the module to meta, so meta-only ranks still need real buffer
        # storage here. The launcher moves this small CPU buffer to the local
        # CUDA device before Trainer initialization.
        _, _, _, _, rotary_embedding_cls, _ = _require_dsv4_transformers()
        rotary_device = torch.device("cpu") if target_device.type == "meta" else target_device
        model.mtp.rotary_emb = rotary_embedding_cls(config).to(rotary_device)
        return model

    @classmethod
    def _copy_tensor(
        cls, model_state: Mapping[str, torch.Tensor], name: str, value: torch.Tensor
    ) -> None:
        target = _state_tensor(model_state, name)
        if tuple(target.shape) != tuple(value.shape):
            raise NativeMTPCheckpointError(
                f"Shape mismatch loading {name}: checkpoint {tuple(value.shape)}, "
                f"model {tuple(target.shape)}"
            )
        target.copy_(value.to(device=target.device, dtype=target.dtype))

    @classmethod
    def _load_model(cls, model: NativeMTPBoostModel, layout: _CheckpointLayout) -> None:
        model_state = model.state_dict()
        num_experts = int(layout.config["n_routed_experts"])
        with torch.no_grad(), _TensorReader(layout) as reader:
            cls._copy_tensor(model_state, "embedding.weight", reader.get(layout.embedding_key))
            cls._copy_tensor(model_state, "lm_head.weight", reader.get(layout.lm_head_key))

            for source_suffix, state_name in _DIRECT_STATE_KEYS.items():
                source_name = _MTP_PREFIX + source_suffix
                weight = reader.get(source_name)
                scale_name = source_name.replace(".weight", ".scale")
                scale = (
                    reader.get(scale_name)
                    if source_name.endswith(".weight") and scale_name in layout.weight_map
                    else None
                )
                cls._copy_tensor(
                    model_state,
                    state_name,
                    _decode_weight(weight, scale, dtype=model_state[state_name].dtype),
                )

            gate_up = _state_tensor(model_state, "mtp.decoder.mlp.experts.gate_up_proj")
            down = _state_tensor(model_state, "mtp.decoder.mlp.experts.down_proj")
            intermediate = gate_up.shape[1] // 2
            for expert_index in range(num_experts):
                keys = _expert_state_keys(expert_index)
                for source_name, state_name in keys.items():
                    source_weight = reader.get(source_name)
                    source_scale = reader.get(source_name.replace(".weight", ".scale"))
                    decoded = _decode_weight(source_weight, source_scale, dtype=gate_up.dtype)
                    if source_name.endswith(".w1.weight"):
                        if tuple(decoded.shape) != tuple(
                            gate_up[expert_index, :intermediate].shape
                        ):
                            raise NativeMTPCheckpointError(
                                f"Unexpected source shape for {source_name}"
                            )
                        gate_up[expert_index, :intermediate].copy_(decoded.to(gate_up.device))
                    elif source_name.endswith(".w3.weight"):
                        if tuple(decoded.shape) != tuple(
                            gate_up[expert_index, intermediate:].shape
                        ):
                            raise NativeMTPCheckpointError(
                                f"Unexpected source shape for {source_name}"
                            )
                        gate_up[expert_index, intermediate:].copy_(decoded.to(gate_up.device))
                    else:
                        if tuple(decoded.shape) != tuple(down[expert_index].shape):
                            raise NativeMTPCheckpointError(
                                f"Unexpected source shape for {source_name}"
                            )
                        down[expert_index].copy_(decoded.to(down.device))
        model.freeze_target_endpoints()

    @classmethod
    def attach_online_target(
        cls,
        model: nn.Module,
        model_path: str | Path,
        target_checkpoint: str | Path,
        *,
        device: torch.device | str,
        max_batch_size: int,
        max_seq_len: int,
    ) -> None:
        """Attach the frozen vendor target feature provider for online training."""
        if not isinstance(model, NativeMTPBoostModel):
            raise TypeError("DeepSeek-V4 online target requires a NativeMTPBoostModel")

        target_device = torch.device(device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DeepSeek-V4 online target execution requires CUDA")
        torch.cuda.set_device(target_device)

        expected_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if expected_world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size > 1 and not Path(target_checkpoint).is_dir():
            raise NativeMTPCheckpointError(
                "Distributed DSV4 online training requires a directory containing "
                f"model{{rank}}-mp{world_size}.safetensors target shards"
            )

        target = load_dsv4_target_model(
            model_path,
            target_checkpoint,
            target_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        model.set_online_target_feature_provider(
            lambda input_ids: dsv4_target_features(target, input_ids),
            model_parallel=world_size > 1,
        )

    @classmethod
    def create(
        cls,
        model_path: str | Path,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        rollout_steps: int,
        hsm_mode: str,
    ) -> NativeMTPBoostModel:
        """Load the checkpoint's native MTP, using meta state on nonzero FSDP2 ranks."""
        layout = _checkpoint_layout(model_path, strict=True)
        config, mtp_layer_index = cls._transformers_config(layout)
        leave_on_meta = _is_fsdp_efficient_nonzero_rank()
        model = cls._build_empty_model(
            config,
            mtp_layer_index,
            dtype=dtype,
            device="meta" if leave_on_meta else device,
            rollout_steps=rollout_steps,
            hsm_mode=hsm_mode,
        )
        if leave_on_meta:
            model.freeze_target_endpoints()
        else:
            cls._load_model(model, layout)
        return model

    @classmethod
    def _encoded_mtp_tensors(
        cls,
        layout: _CheckpointLayout,
        state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        encoded: dict[str, torch.Tensor] = {}
        num_experts = int(layout.config["n_routed_experts"])
        with _TensorReader(layout) as reader:
            for source_suffix, state_name in _DIRECT_STATE_KEYS.items():
                source_name = _MTP_PREFIX + source_suffix
                source = reader.get(source_name)
                value = _state_tensor(state_dict, state_name)
                if source_name.endswith(".weight") and source.dtype == torch.float8_e4m3fn:
                    quantized, scale = encode_dsv4_fp8_weight(value)
                    encoded[source_name] = quantized.cpu().contiguous()
                    encoded[source_name.replace(".weight", ".scale")] = scale.cpu().contiguous()
                elif source_name.endswith(".weight") and source.dtype in (torch.int8, torch.uint8):
                    quantized, scale = encode_dsv4_mxfp4_weight(value)
                    encoded[source_name] = quantized.cpu().contiguous()
                    encoded[source_name.replace(".weight", ".scale")] = scale.cpu().contiguous()
                elif not source_name.endswith(".scale"):
                    encoded[source_name] = value.to(dtype=source.dtype, device="cpu").contiguous()

            gate_up = _state_tensor(state_dict, "mtp.decoder.mlp.experts.gate_up_proj")
            down = _state_tensor(state_dict, "mtp.decoder.mlp.experts.down_proj")
            intermediate = gate_up.shape[1] // 2
            for expert_index in range(num_experts):
                for projection, value in (
                    ("w1", gate_up[expert_index, :intermediate]),
                    ("w2", down[expert_index]),
                    ("w3", gate_up[expert_index, intermediate:]),
                ):
                    source_name = f"{_MTP_PREFIX}ffn.experts.{expert_index}.{projection}.weight"
                    source = reader.get(source_name)
                    if source.dtype in (torch.int8, torch.uint8):
                        quantized, scale = encode_dsv4_mxfp4_weight(value)
                    elif source.dtype == torch.float8_e4m3fn:
                        quantized, scale = encode_dsv4_fp8_weight(value)
                    else:
                        encoded[source_name] = value.to(
                            dtype=source.dtype, device="cpu"
                        ).contiguous()
                        continue
                    encoded[source_name] = quantized.cpu().contiguous()
                    encoded[source_name.replace(".weight", ".scale")] = scale.cpu().contiguous()
        expected = _expected_mtp_keys(layout.config)
        if set(encoded) != expected:
            missing = sorted(expected - set(encoded))
            unexpected = sorted(set(encoded) - expected)
            raise RuntimeError(
                "Native MTP export did not cover exactly the checkpoint MTP tensors: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        return encoded

    @classmethod
    def _link_or_copy(cls, source: str | Path, destination: str | Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    @classmethod
    def export(
        cls,
        model: nn.Module,
        base_checkpoint: str | Path,
        output_path: str | Path,
        *,
        state_dict: dict[str, torch.Tensor] | None,
    ) -> Path:
        """Write trained MTP tensors into a linked copy of the native checkpoint."""
        layout = _checkpoint_layout(base_checkpoint, strict=True)
        if not isinstance(model, NativeMTPBoostModel):
            raise TypeError(
                "DeepSeek-V4 native MTP export requires the NativeMTPBoostModel returned by "
                "create_native_mtp_boost_model"
            )
        output = Path(output_path)
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing native MTP export directory: {output}"
            )
        output.mkdir(parents=True)
        full_state = model.state_dict() if state_dict is None else state_dict
        encoded = cls._encoded_mtp_tensors(layout, full_state)
        mtp_shards = {layout.weight_map[key] for key in encoded}

        try:
            for source_item in layout.path.iterdir():
                destination = output / source_item.name
                if source_item.name in mtp_shards:
                    continue
                if source_item.is_file():
                    cls._link_or_copy(source_item, destination)
                elif source_item.is_dir():
                    shutil.copytree(source_item, destination, copy_function=cls._link_or_copy)

            safe_open, save_file = _require_safetensors()
            for shard_name in mtp_shards:
                shard_path = layout.path / shard_name
                shard_tensors: dict[str, torch.Tensor] = {}
                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    metadata = handle.metadata()
                    # ``safe_open`` exposes ``keys()`` but is not itself iterable.
                    for key in handle.keys():  # noqa: SIM118
                        shard_tensors[key] = encoded.get(key, handle.get_tensor(key)).contiguous()
                save_file(shard_tensors, str(output / shard_name), metadata=metadata)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
        return output
