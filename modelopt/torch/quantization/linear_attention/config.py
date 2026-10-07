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

"""Saved execution policy for GDN/KDA state QAT."""

from typing import Literal

from pydantic import Field, model_validator

from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField

__all__ = ["LinearAttentionConfig", "LinearAttentionPolicyEntry"]


# Deserialization-only names for pre-unification pickles. ModeloptBaseConfig
# registers each original class name with torch.serialization's safe globals.
class _StateConfig(ModeloptBaseConfig):
    """Legacy state settings; migrated by LinearAttentionConfig on load."""


class _SolveConfig(ModeloptBaseConfig):
    """Legacy solve settings; migrated by LinearAttentionConfig on load."""


class LinearAttentionReplayConfig(ModeloptBaseConfig):
    """Legacy replay settings; migrated by LinearAttentionConfig on load."""


class LinearAttentionDecodeConfig(ModeloptBaseConfig):
    """Legacy decode settings; migrated by LinearAttentionConfig on load."""


def _legacy_fields(value):
    if isinstance(value, ModeloptBaseConfig):
        # model_dump would omit the fields of these deserialization-only classes.
        fields = dict(vars(value))
        factor_qdq = (value.__pydantic_private__ or {}).get("_legacy_factor_qdq")
        if factor_qdq is not None:
            fields["factor_qdq"] = factor_qdq
        return fields
    if not isinstance(value, dict):
        raise ValueError("Legacy linear-attention settings must be dictionaries")
    return dict(value)


def _consume_fixed_settings(values, **defaults):
    """Remove fixed legacy settings without silently changing their numerics."""
    for key, default in defaults.items():
        if values.pop(key, default) != default:
            raise ValueError(f"Native state QAT requires {key}={default!r}")


class LinearAttentionConfig(ModeloptBaseConfig):
    """One execution policy shared by the chunked prefix and recurrent suffix.

    ``serving`` uses native forward arithmetic with a differentiable adjoint;
    ``fla`` is the disabled/default path. ``vllm_0_15`` uses public vLLM arithmetic;
    ``replayssm`` uses the serving fork's INT8/Hadamard checkpoint and ring kernels.
    ``replay_window=1`` refreshes the state every token; larger windows enable replay.
    Both profiles leave prefill unquantized and quantize the handoff state.

    TensorQuantizer independently enables QDQ and owns its format and grouping.
    ``state_block_v`` controls grouping only for legacy per-tile quantizers.
    """

    schema_version: Literal[3] = ModeloptField(default=3)
    backend: Literal["fla", "serving"] = ModeloptField(default="fla")
    precision: Literal["vllm_0_15", "replayssm"] = ModeloptField(default="vllm_0_15")
    replay_window: int = Field(default=1, ge=1, le=64, strict=True)
    state_block_v: Literal[16, 32, 64, 128] = ModeloptField(default=64)

    @property
    def state_codec(self):
        """The native profile fixes the codec; it is not an independent setting."""
        return "int8_hadamard32" if self.precision == "replayssm" else "tile"

    @model_validator(mode="before")
    @classmethod
    def _load_legacy_settings(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        version = values.get("schema_version", 3)
        if version in (1, 2):
            values["schema_version"] = 3
        if values.get("backend") == "reference":
            raise ValueError("Reference training is retired; select backend='serving'")
        _consume_fixed_settings(values, chunk_size=64)
        if "solve" in values:
            solve = _legacy_fields(values.pop("solve"))
            _consume_fixed_settings(solve, method="exact")
            if solve:
                raise ValueError(f"Unsupported legacy solve settings: {solve}")
        if "state" in values:
            state = _legacy_fields(values.pop("state"))
            _consume_fixed_settings(state, mode="chunk", quantize_initial=True)
            if "state_block_v" in values:
                raise ValueError("Supply state_block_v without legacy state settings")
            values["state_block_v"] = state.pop("block_v", 64)
            if state:
                raise ValueError(f"Unsupported legacy state settings: {state}")
        if "decode" in values:
            decode = values.pop("decode")
            if (values.get("backend", "fla") != "fla") != (decode is not None):
                raise ValueError("Legacy state QAT requires an explicit decode policy")
            if decode is not None:
                if "precision" in values or "replay_window" in values:
                    raise ValueError(
                        "Supply precision/replay_window without legacy decode settings"
                    )
                decode = _legacy_fields(decode)
                precision = decode.pop(
                    "precision", "full" if values.get("backend") == "matmul" else "vllm_0_15"
                )
                if precision == "full":
                    raise ValueError(
                        "Reference training is retired; select a native serving precision"
                    )
                values["precision"] = precision
                replay = decode.pop("replay", None)
                mode = decode.pop("mode", "token")
                if mode not in ("token", "replay") or (mode == "replay") != (replay is not None):
                    raise ValueError("Legacy replay settings require mode='replay'")
                _consume_fixed_settings(
                    decode,
                    readout="working",
                    quantize_initial=True,
                    prefill_state_qdq=False,
                    decay_log_step=None,
                    implementation="torch",
                    state_codec="int8_hadamard32" if precision == "replayssm" else "tile",
                )
                if decode:
                    raise ValueError(f"Unsupported legacy decode settings: {decode}")
                if replay is not None:
                    replay = _legacy_fields(replay)
                    factor_qdq = replay.pop("factor_qdq", version == 1)
                    if factor_qdq is not False:
                        raise ValueError(
                            "Replay factor QDQ is retired; native ReplaySSM stores BF16 factors"
                        )
                    values["replay_window"] = replay.pop("window", 8)
                    _consume_fixed_settings(replay, encoding="once")
                    if replay:
                        raise ValueError(f"Unsupported legacy replay settings: {replay}")
                    if precision != "replayssm":
                        raise ValueError("Replay requires precision='replayssm'")
        if values.get("backend") == "matmul":
            if "precision" not in values:
                raise ValueError("Reference training is retired; select a native serving precision")
            values["backend"] = "serving"
        return values

    @model_validator(mode="after")
    def _validate_profile(self):
        if self.precision == "replayssm":
            if self.backend != "serving":
                raise ValueError("replayssm requires backend='serving'")
            if self.state_block_v < 32:
                raise ValueError("int8_hadamard32 requires state_block_v >= 32")
        elif self.replay_window != 1:
            raise ValueError("Replay requires precision='replayssm'")
        return self

    def __setstate__(self, state):
        # Config objects can be pickled inside ModelOpt checkpoints. Normalize
        # their old nested fields before any runtime consumer sees the policy.
        migrated = type(self).model_validate(state["__dict__"])
        super().__setstate__(migrated.__getstate__())


class LinearAttentionPolicyEntry(ModeloptBaseConfig):
    """Assign a complete policy to supported modules matching ``module_name``.

    Rules apply in order: the last match wins, without merging fields.
    A rule must match at least one supported linear-attention module.
    """

    module_name: str = Field(...)
    cfg: LinearAttentionConfig = ModeloptField(default=LinearAttentionConfig())
