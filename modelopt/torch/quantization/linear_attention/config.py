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


class LinearAttentionConfig(ModeloptBaseConfig):
    """One execution policy shared by the chunked prefix and recurrent suffix.

    ``serving`` uses native forward arithmetic with a differentiable adjoint;
    ``fla`` is the disabled/default path. ``vllm_0_15`` uses public vLLM arithmetic;
    ``replayssm`` uses the serving fork's INT8/Hadamard checkpoint and ring kernels.
    ``replay_window=1`` refreshes the state every token; larger windows enable replay.
    Fresh prefill has no internal state QDQ; incoming continuation state and
    the handoff to a nonempty suffix use the selected checkpoint encoding.

    TensorQuantizer independently enables QDQ and owns its format and grouping.
    ``state_block_v`` controls grouping only for legacy per-tile quantizers.
    """

    backend: Literal["fla", "serving"] = ModeloptField(default="fla")
    precision: Literal["vllm_0_15", "replayssm"] = ModeloptField(default="vllm_0_15")
    replay_window: int = Field(default=1, ge=1, le=64, strict=True)
    state_block_v: Literal[16, 32, 64, 128] = ModeloptField(default=64)

    @property
    def state_codec(self):
        """The native profile fixes the codec; it is not an independent setting."""
        return "int8_hadamard32" if self.precision == "replayssm" else "tile"

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


class LinearAttentionPolicyEntry(ModeloptBaseConfig):
    """Assign a complete policy to supported modules matching ``module_name``.

    Rules apply in order: the last match wins, without merging fields.
    A rule must match at least one supported linear-attention module.
    """

    module_name: str = Field(...)
    cfg: LinearAttentionConfig = ModeloptField(default=LinearAttentionConfig())
