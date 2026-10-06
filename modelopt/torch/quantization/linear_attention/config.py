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

"""Saved execution policy for GDN training-time numerical emulation."""

from typing import Literal

from pydantic import Field, PrivateAttr, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = [
    "LinearAttentionConfig",
    "LinearAttentionDecodeConfig",
    "LinearAttentionPolicyEntry",
    "LinearAttentionReplayConfig",
]


class _StateConfig(ModeloptBaseConfig):
    block_v: Literal[16, 32, 64, 128] = ModeloptField(default=64)

    @model_validator(mode="before")
    @classmethod
    def _load_legacy_defaults(cls, values):
        if isinstance(values, dict):
            values = dict(values)
            for key, default in (("mode", "chunk"), ("quantize_initial", True)):
                if values.get(key) == default:
                    values.pop(key)
        return values


class _SolveConfig(ModeloptBaseConfig):
    # Retained solely to deserialize old pickled policies and validate their fixed setting.
    method: Literal["exact"] = ModeloptField(default="exact")


class LinearAttentionReplayConfig(ModeloptBaseConfig):
    """Anchor refresh and update encoding schedule; quantizers are configured by quant_cfg."""

    window: int = Field(default=8, ge=1, le=64, strict=True)
    encoding: Literal["once", "reencode"] = ModeloptField(default="once")
    _legacy_factor_qdq: bool | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _load_legacy_factor_qdq(cls, values, handler):
        legacy = None
        if isinstance(values, dict) and "factor_qdq" in values:
            values = dict(values)
            legacy = values.pop("factor_qdq")
            if not isinstance(legacy, bool):
                raise ValueError("Legacy factor_qdq must be a boolean")
        result = handler(values)
        if legacy is not None:
            result._legacy_factor_qdq = legacy
        return result


class LinearAttentionDecodeConfig(ModeloptBaseConfig):
    """Explicit suffix recurrence; workload supplies per-sequence prefix lengths.

    ``precision='vllm_0_15'`` uses the pinned serving forward arithmetic for
    the chunked prefix and token suffix, with a differentiable Torch adjoint.
    ``replayssm`` imports INT8/Hadamard checkpoint and ring kernels from the
    quantized-ReplaySSM serving fork. ``full`` retains the FP32/FP64 Torch reference.
    TensorQuantizer configures quantization independently; this profile does not enable QDQ.
    """

    mode: Literal["token", "replay"] = ModeloptField(default="token")
    readout: Literal["working", "stored"] = ModeloptField(default="stored")
    quantize_initial: bool = ModeloptField(default=True)
    prefill_state_qdq: bool = ModeloptField(default=False)
    precision: Literal["full", "vllm_0_15", "replayssm"] = ModeloptField(default="full")
    state_codec: Literal["tile", "int8_hadamard32"] = ModeloptField(default="tile")
    decay_log_step: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    replay: LinearAttentionReplayConfig | None = ModeloptField(default=None)

    @model_validator(mode="before")
    @classmethod
    def _load_legacy_defaults(cls, values):
        if isinstance(values, dict) and values.get("implementation") == "torch":
            values = {key: value for key, value in values.items() if key != "implementation"}
        return values

    @model_validator(mode="after")
    def _validate_replay(self):
        if (self.mode == "replay") != (self.replay is not None):
            raise ValueError("replay settings must be supplied exactly when mode='replay'")
        if self.state_codec == "int8_hadamard32" and self.prefill_state_qdq:
            raise ValueError(
                "Hadamard state QDQ starts at decode handoff; disable prefill_state_qdq"
            )
        if self.precision == "vllm_0_15" and (
            self.mode != "token"
            or self.readout != "working"
            or self.state_codec != "tile"
            or not self.quantize_initial
            or self.prefill_state_qdq
            or self.decay_log_step is not None
        ):
            raise ValueError(
                "vllm_0_15 requires token mode, working readout, tile codec, initial-state QDQ, "
                "and no additional prefill or gate rounding"
            )
        if self.precision == "replayssm" and (
            self.readout != "working"
            or self.state_codec != "int8_hadamard32"
            or not self.quantize_initial
            or self.prefill_state_qdq
            or self.decay_log_step is not None
            or (self.replay is not None and self.replay.encoding != "once")
        ):
            raise ValueError(
                "replayssm requires working readout, int8_hadamard32, initial-state QDQ, "
                "encode-once replay, and no additional prefill or gate rounding"
            )
        return self


class LinearAttentionConfig(ModeloptBaseConfig):
    """Execution policy shared by GDN/KDA state QAT and retained GDN W QAT.

    ``serving`` uses native forward arithmetic with a differentiable adjoint;
    ``reference`` retains explicit mathematical experiments. ``fla`` is the
    disabled/default or W-only path. State formats and grouping come from
    TensorQuantizer; ``state.block_v`` controls legacy tile grouping.
    """

    schema_version: Literal[1, 2] = ModeloptField(default=2)
    backend: Literal["fla", "serving", "reference"] = ModeloptField(default="fla")
    chunk_size: Literal[64] = ModeloptField(default=64)
    state: _StateConfig = ModeloptField(default=_StateConfig())
    decode: LinearAttentionDecodeConfig | None = ModeloptField(default=None)

    @model_validator(mode="before")
    @classmethod
    def _load_legacy_defaults(cls, values):
        if isinstance(values, dict) and values.get("backend") == "matmul":
            decode = values.get("decode") or {}
            precision = (
                decode.get("precision", "full") if isinstance(decode, dict) else decode.precision
            )
            values = {**values, "backend": "reference" if precision == "full" else "serving"}
        if isinstance(values, dict) and values.get("backend") == "serving":
            decode = values.get("decode")
            if isinstance(decode, dict):
                values = {
                    **values,
                    "decode": {"precision": "vllm_0_15", "readout": "working", **decode},
                }
        if isinstance(values, dict) and values.get("schema_version") == 1:
            decode = values.get("decode")
            if isinstance(decode, dict) and isinstance(decode.get("replay"), dict):
                replay = {"factor_qdq": True, **decode["replay"]}
                values = {**values, "decode": {**decode, "replay": replay}}
        if isinstance(values, dict) and "solve" in values:
            _SolveConfig.model_validate(values["solve"])
            values = {key: value for key, value in values.items() if key != "solve"}
        return values

    @model_validator(mode="after")
    def _validate_decode_backend(self):
        if (self.backend != "fla") != (self.decode is not None):
            raise ValueError("State QAT requires an explicit decode policy")
        if self.decode is not None and (self.backend == "serving") != (
            self.decode.precision != "full"
        ):
            raise ValueError(
                "Serving state QAT requires a native precision profile; full uses reference"
            )
        if self.decode is not None and self.decode.state_codec == "int8_hadamard32":
            if self.state.block_v < 32:
                raise ValueError("int8_hadamard32 requires block_v >= 32")
        return self


class LinearAttentionPolicyEntry(ModeloptBaseConfig):
    """Assign a complete policy to supported modules matching ``module_name``.

    Rules apply in order: the last match wins, without merging nested fields.
    A rule must match at least one supported linear-attention module.
    """

    module_name: str = Field(...)
    cfg: LinearAttentionConfig = ModeloptField(default=LinearAttentionConfig())
