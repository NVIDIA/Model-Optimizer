# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context manager that patches mcore MambaLayer to inject per-layer config from block_configs.

TransformerLayer (used for attention and MLP slots in MambaStack) is already patched by
mbridge_gpt_patcher when the same context is active. This module patches MambaLayer so that
Mamba slots in a hybrid stack also receive per-layer config overrides.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, List, Optional, Union

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import BlockConfig

from block_config_to_mcore import get_overrides_for_layer
from mbridge_gpt_patcher import _get_ctx, _merge_config_overrides

logger = logging.getLogger(__name__)


@contextmanager
def mbridge_mamba_patcher(
    block_configs: Optional[List[Union[BlockConfig, dict]]],
    num_attention_heads: int,
    hidden_size: int,
):
    """Context manager that patches MambaLayer.__init__ for heterogeneous per-layer config.

    When active, each MambaLayer built by MambaStack will receive config overrides from
    block_configs[layer_index]. The same block_configs/context is shared with the GPT
    patcher (TransformerLayer is used for attention/MLP slots in MambaStack).

    Args:
        block_configs: List of block configs (one per slot in the hybrid pattern), or None.
        num_attention_heads: Base model num_attention_heads.
        hidden_size: Base model hidden_size.
    """
    from megatron.core.ssm.mamba_layer import MambaLayer

    ctx = _get_ctx()
    # Share context with GPT patcher so TransformerLayer (attention/MLP in Mamba) also gets overrides
    prev_block_configs = ctx.block_configs
    prev_num_heads = ctx.num_attention_heads
    prev_hidden = ctx.hidden_size

    ctx.block_configs = block_configs
    ctx.num_attention_heads = num_attention_heads
    ctx.hidden_size = hidden_size

    orig_init = MambaLayer.__init__

    def patched_init(
        self: Any,
        config: Any,
        submodules: Any,
        layer_number: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        # In MambaStack, layer_number is already global: i + 1 + pp_layer_offset
        block_cfgs = ctx.block_configs
        if not block_cfgs:
            return orig_init(self, config, submodules, layer_number, *args, **kwargs)

        overrides = get_overrides_for_layer(
            block_cfgs,
            layer_number,
            num_attention_heads=ctx.num_attention_heads or config.num_attention_heads,
            hidden_size=ctx.hidden_size or config.hidden_size,
            strict_mamba_slot=True,
        )
        if overrides and overrides.config_overrides:
            config = _merge_config_overrides(config, overrides.config_overrides)

        orig_init(self, config, submodules, layer_number, *args, **kwargs)

    try:
        MambaLayer.__init__ = patched_init
        yield
    finally:
        MambaLayer.__init__ = orig_init
        ctx.block_configs = prev_block_configs
        ctx.num_attention_heads = prev_num_heads
        ctx.hidden_size = prev_hidden
