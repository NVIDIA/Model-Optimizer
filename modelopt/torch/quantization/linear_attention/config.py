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

from pydantic import Field, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = [
    "LinearAttentionConfig",
    "LinearAttentionDecodeConfig",
    "LinearAttentionMatmulConfig",
    "LinearAttentionPolicyEntry",
    "LinearAttentionReplayConfig",
    "LinearAttentionSolveConfig",
]

_PrefillSite = Literal[
    "key_interaction",
    "wy_value",
    "wy_key",
    "state_read",
    "state_update",
    "output_state",
    "output_score",
    "output_value",
]
_ElementwiseSite = Literal[
    "gate_prefix",
    "gate_exp",
    "value_residual",
    "state_decay",
    "state_add",
    "output_add",
]
_ArithmeticDtype = Literal["float32", "float16", "bfloat16"]


class LinearAttentionMatmulConfig(ModeloptBaseConfig):
    """Round an accumulator after each left-to-right reduction block.

    Partial products use the baseline working dtype. This specifies an emulation
    schedule, not the internal accumulation order of a hardware MMA instruction.
    """

    accumulator_dtype: _ArithmeticDtype | None = ModeloptField(default=None)
    reduction_block: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _require_complete_schedule(self):
        if (self.accumulator_dtype is None) != (self.reduction_block is None):
            raise ValueError("accumulator_dtype and reduction_block must be specified together")
        return self


class _StateConfig(ModeloptBaseConfig):
    mode: Literal["chunk"] = ModeloptField(default="chunk")
    block_v: Literal[16, 32, 64, 128] = ModeloptField(default=64)
    quantize_initial: Literal[True] = ModeloptField(default=True)


class LinearAttentionSolveConfig(ModeloptBaseConfig):
    """Exact solve; inverse approximation is a later delivery."""

    method: Literal["exact"] = ModeloptField(default="exact")


class LinearAttentionReplayConfig(ModeloptBaseConfig):
    """Anchor refresh and encoded rank-one update policy."""

    window: int = Field(default=8, ge=1, le=64, strict=True)
    factor_qdq: bool = ModeloptField(default=True)
    encoding: Literal["once", "reencode"] = ModeloptField(default="once")


class LinearAttentionDecodeConfig(ModeloptBaseConfig):
    """Explicit suffix recurrence; workload supplies per-sequence prefix lengths."""

    mode: Literal["token", "replay"] = ModeloptField(default="token")
    implementation: Literal["torch", "triton"] = ModeloptField(default="torch")
    readout: Literal["working", "stored"] = ModeloptField(default="stored")
    quantize_initial: bool = ModeloptField(default=True)
    prefill_state_qdq: bool = ModeloptField(default=False)
    state_codec: Literal["tile", "int8_hadamard32"] = ModeloptField(default="tile")
    decay_log_step: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    replay: LinearAttentionReplayConfig | None = ModeloptField(default=None)

    @model_validator(mode="after")
    def _validate_replay(self):
        if (self.mode == "replay") != (self.replay is not None):
            raise ValueError("replay settings must be supplied exactly when mode='replay'")
        if self.implementation == "triton" and self.replay and self.replay.encoding != "once":
            raise ValueError("The Triton replay candidate implements encode-once only")
        if self.state_codec == "int8_hadamard32" and self.prefill_state_qdq:
            raise ValueError(
                "Hadamard state QDQ starts at decode handoff; disable prefill_state_qdq"
            )
        return self


class LinearAttentionConfig(ModeloptBaseConfig):
    """GDN/KDA chunk-64 policy; unsupported numerical modes fail config validation.

    ``state.block_v`` defines one dynamic scale per ``[Dk, block_v]`` tile of each
    sequence/head. The initial state and every chunk's final state are rounded when
    the module's state quantizer is enabled. Outputs use the incoming rounded state.
    Decode's ``int8_hadamard32`` codec instead fixes scales to one key channel and
    32 values; ``state.block_v`` remains the execution tile width.
    """

    schema_version: Literal[1] = ModeloptField(default=1)
    backend: Literal["fla", "matmul"] = ModeloptField(default="fla")
    chunk_size: Literal[64] = ModeloptField(default=64)
    state: _StateConfig = ModeloptField(default=_StateConfig())
    solve: LinearAttentionSolveConfig = ModeloptField(default=LinearAttentionSolveConfig())
    decode: LinearAttentionDecodeConfig | None = ModeloptField(default=None)
    matmul: dict[_PrefillSite, LinearAttentionMatmulConfig] = ModeloptField(default={})
    elementwise: dict[_ElementwiseSite, _ArithmeticDtype] = ModeloptField(default={})

    @model_validator(mode="after")
    def _validate_arithmetic_backend(self):
        if self.backend == "fla" and (
            self.matmul
            or self.elementwise
            or self.solve.method != "exact"
            or self.decode is not None
        ):
            raise ValueError("Prefill arithmetic policies require backend='matmul'")
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
