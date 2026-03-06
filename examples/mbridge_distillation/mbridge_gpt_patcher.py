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

"""Context manager that patches mcore build_module to inject per-layer config from block_configs."""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional, Union

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import BlockConfig

from block_config_to_mcore import (
    MCoreLayerOverrides,
    get_overrides_for_layer,
)

logger = logging.getLogger(__name__)
# Ensure our INFO messages are visible even if parent loggers are set higher
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("INFO:%(name)s:%(message)s"))
    logger.addHandler(_h)
    logger.propagate = False

_mbridge_ctx = threading.local()


def _get_ctx():
    if not hasattr(_mbridge_ctx, "block_configs"):
        _mbridge_ctx.block_configs = None
        _mbridge_ctx.num_attention_heads = None
        _mbridge_ctx.hidden_size = None
        _mbridge_ctx.apply_no_ops = True
    return _mbridge_ctx


def _merge_config_overrides(
    config: Any,
    overrides: Dict[str, Any],
) -> Any:
    """Return a shallow copy of config with overrides applied, preserving the class."""
    import copy

    # Create a shallow copy to preserve class and avoid deepcopying ProcessGroups
    # This fixes the issue where the config class changes (e.g. from GPTOSSProvider to TransformerConfig)
    new_config = copy.copy(config)

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(new_config, k):
            setattr(new_config, k, v)
        else:
            raise ValueError(f"mbridge_gpt_patcher: {k} is not a valid attribute of {type(new_config)}")

    return new_config


def _apply_gpt_no_op(
    layer: Any,
    overrides: MCoreLayerOverrides,
) -> None:
    """Replace attention/mlp submodules with IdentityOp when no_op is set."""
    from megatron.core.transformer.identity_op import IdentityOp

    if overrides.attention_no_op and hasattr(layer, "self_attention"):
        layer.self_attention = IdentityOp()
        logger.info(
            "mbridge_gpt_patcher: replaced self_attention with IdentityOp for no_op layer"
        )
    if overrides.mlp_no_op and hasattr(layer, "mlp"):
        layer.mlp = IdentityOp()
        logger.info(
            "mbridge_gpt_patcher: replaced mlp with IdentityOp for no_op layer"
        )


def _is_transformer_layer_class(module: type) -> bool:
    """True if module is TransformerLayer or a subclass (e.g. MoETransformerLayer, MLPLayer).

    Uses issubclass so that MambaStack's MoE slot (MoETransformerLayer) and MLP slot
    (MLPLayer) are intercepted in addition to plain TransformerLayer attention slots.
    """
    if not isinstance(module, type):
        return False
    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer
        return issubclass(module, TransformerLayer)
    except ImportError:
        # Fallback: name-based check
        return (
            module.__name__ in ("TransformerLayer", "MoETransformerLayer", "MLPLayer")
            and "megatron" in (getattr(module, "__module__", "") or "")
        )


@contextmanager
def mbridge_gpt_patcher(
    block_configs: Optional[List[Union[BlockConfig, dict]]],
    num_attention_heads: int,
    hidden_size: int,
    apply_no_ops: bool = True,
):
    """Context manager that patches build_module to inject per-layer config from block_configs.

    When active, each TransformerLayer built via spec_utils.build_module will receive
    config overrides (e.g. ffn_hidden_size, num_query_groups) from block_configs[layer_index],
    and attention/mlp will be replaced with IdentityOp when block_config specifies no_op
    (only when apply_no_ops=True).
    Patches both spec_utils.build_module and transformer_block.build_module (the call site)
    so the hook runs regardless of which module reference the layer spec holds.

    Args:
        block_configs: List of block configs (one per decoder layer), or None to disable.
        num_attention_heads: Base model num_attention_heads.
        hidden_size: Base model hidden_size.
        apply_no_ops: If True (default), replace attention/mlp with IdentityOp when
            block_config specifies no_op. Set to False for Mamba/hybrid models where
            MambaStack already encodes each slot as a single dedicated layer type
            (mamba, attention, mlp, or moe) and the spec handles no_ops intrinsically.
    """
    # Use sys.modules so we patch the same modules used at runtime (e.g. Bridge's 3rdparty).
    import megatron.core.transformer.spec_utils  # noqa: F401 - ensure loaded
    import megatron.core.transformer.transformer_block  # noqa: F401 - ensure loaded
    import megatron.core.ssm.mamba_block  # noqa: F401 - ensure loaded

    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
    from megatron.core.utils import get_pg_rank

    spec_utils_mod = sys.modules["megatron.core.transformer.spec_utils"]
    transformer_block_mod = sys.modules["megatron.core.transformer.transformer_block"]
    mamba_block_mod = sys.modules["megatron.core.ssm.mamba_block"]

    print(
        f"[mbridge_gpt_patcher] context entered; spec_utils={getattr(spec_utils_mod, '__file__', None)!r}",
        flush=True,
    )
    logger.info(
        "mbridge_gpt_patcher: entering context; spec_utils=%s, transformer_block=%s",
        getattr(spec_utils_mod, "__file__", None),
        getattr(transformer_block_mod, "__file__", None),
    )

    ctx = _get_ctx()
    prev_block_configs = ctx.block_configs
    prev_num_heads = ctx.num_attention_heads
    prev_hidden = ctx.hidden_size

    ctx.block_configs = block_configs
    ctx.num_attention_heads = num_attention_heads
    ctx.hidden_size = hidden_size
    ctx.apply_no_ops = apply_no_ops

    orig_build_module = spec_utils_mod.build_module

    def patched_build_module(
        spec_or_module: Any, *args: Any, **kwargs: Any
    ) -> Any:
        logger.info(
            "mbridge_gpt_patcher: build_module called (spec type=%s, has module=%s)",
            type(spec_or_module).__name__,
            getattr(spec_or_module, "module", None) is not None,
        )
        # Resolve module class the same way build_module does
        if isinstance(spec_or_module, type):
            module = spec_or_module
        elif hasattr(spec_or_module, "module") and isinstance(
            getattr(spec_or_module, "module", None), type
        ):
            module = spec_or_module.module
        else:
            logger.info(
                "mbridge_gpt_patcher: build_module passthrough (not a type/spec with type module)"
            )
            return orig_build_module(spec_or_module, *args, **kwargs)

        module_name = getattr(module, "__name__", None)
        module_module = getattr(module, "__module__", "")
        is_tlayer = _is_transformer_layer_class(module)
        logger.info(
            "mbridge_gpt_patcher: resolved module %s from %s; is_transformer_layer=%s",
            module_name,
            module_module,
            is_tlayer,
        )

        if not is_tlayer:
            return orig_build_module(spec_or_module, *args, **kwargs)

        layer_number = kwargs.get("layer_number", 1)
        print(
            f"mbridge_gpt_patcher: intercepting TransformerLayer build (layer_number={layer_number})",
            flush=True,
        )
        logger.info(
            "mbridge_gpt_patcher: intercepting TransformerLayer build (layer_number=%s)",
            layer_number,
        )
        block_cfgs = ctx.block_configs
        if not block_cfgs:
            logger.info(
                "mbridge_gpt_patcher: no block_configs in context, passing through"
            )
            return orig_build_module(spec_or_module, *args, **kwargs)

        config = kwargs.get("config")
        pg_collection = kwargs.get("pg_collection")
        vp_stage = kwargs.get("vp_stage")

        if config is None:
            logger.info(
                "mbridge_gpt_patcher: no config in kwargs, passing through"
            )
            return orig_build_module(spec_or_module, *args, **kwargs)

        if pg_collection is None:
            try:
                from megatron.core.process_groups_config import ProcessGroupCollection
                pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            except Exception as e:
                logger.info(
                    "mbridge_gpt_patcher: could not get pg_collection: %s", e
                )
                pg_collection = None
        global_layer_number = layer_number + (
            0
            if pg_collection is None
            else get_transformer_layer_offset(
                config, vp_stage, get_pg_rank(pg_collection.pp)
            )
        )
        logger.info(
            "mbridge_gpt_patcher: layer_number=%s -> global_layer_number=%s",
            layer_number,
            global_layer_number,
        )

        overrides = get_overrides_for_layer(
            block_cfgs,
            global_layer_number,
            num_attention_heads=ctx.num_attention_heads or getattr(
                config, "num_attention_heads", None
            ),
            hidden_size=ctx.hidden_size or getattr(config, "hidden_size", None),
        )
        if overrides and overrides.config_overrides:
            logger.info(
                "mbridge_gpt_patcher: applying config overrides for layer %s: %s",
                global_layer_number,
                overrides.config_overrides,
            )
            kwargs = dict(kwargs)
            kwargs["config"] = _merge_config_overrides(
                config, overrides.config_overrides
            )
        else:
            logger.info(
                "mbridge_gpt_patcher: no config overrides for layer %s",
                global_layer_number,
            )

        result = orig_build_module(spec_or_module, *args, **kwargs)

        if ctx.apply_no_ops and overrides and (overrides.attention_no_op or overrides.mlp_no_op):
            logger.info(
                "mbridge_gpt_patcher: applying no_op for layer %s (attn=%s, mlp=%s)",
                global_layer_number,
                overrides.attention_no_op,
                overrides.mlp_no_op,
            )
            _apply_gpt_no_op(result, overrides)
        elif not ctx.apply_no_ops and overrides and (overrides.attention_no_op or overrides.mlp_no_op):
            logger.info(
                "mbridge_gpt_patcher: skipping no_op for layer %s (apply_no_ops=False; "
                "MambaStack slot specs handle this intrinsically)",
                global_layer_number,
            )

        return result

    try:
        spec_utils_mod.build_module = patched_build_module
        transformer_block_mod.build_module = patched_build_module
        mamba_block_mod.build_module = patched_build_module
        print(
            "[mbridge_gpt_patcher] patched spec_utils.build_module, transformer_block.build_module, and mamba_block.build_module",
            flush=True,
        )
        logger.info(
            "mbridge_gpt_patcher: patched spec_utils.build_module, transformer_block.build_module, and mamba_block.build_module"
        )
        yield
    finally:
        spec_utils_mod.build_module = orig_build_module
        transformer_block_mod.build_module = orig_build_module
        mamba_block_mod.build_module = orig_build_module
        logger.info("mbridge_gpt_patcher: restored original build_module, exiting context")
        ctx.block_configs = prev_block_configs
        ctx.num_attention_heads = prev_num_heads
        ctx.hidden_size = prev_hidden
