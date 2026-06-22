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

"""Patch MCore layer construction for heterogeneous AnyModel/Puzzletron checkpoints.

Megatron Bridge providers hold a single global ``TransformerConfig``. When
``provider.provide()`` builds the model, every decoder layer is constructed from
this one shared config — wrong for heterogeneous (AnyModel / Puzzletron)
checkpoints where each layer can have its own ``num_moe_experts``,
``ffn_hidden_size``, GQA grouping, Mamba state dims, etc.

The :func:`mbridge_patcher` context manager intercepts MCore's per-layer
construction by monkey-patching ``build_module`` in three module namespaces and
``MambaLayer.__init__``, injecting per-layer config overrides from the
provider's ``block_configs`` and (optionally) replacing disabled subblocks with
:class:`NoOpWithBias` / :class:`NoOpRMSNorm`.

See the docstrings of the individual functions / classes for the details:

  * :class:`NoOpWithBias`, :class:`NoOpRMSNorm` — why ``IdentityOp`` is the wrong
    no-op replacement (output-tuple contract, distributed-checkpoint sharding).
  * :func:`_apply_no_ops` and :data:`_NO_OP_RULES` — which subblocks get
    replaced and why we skip layers whose ``*_bda`` is already ``IdentityFuncOp``.
  * :func:`mbridge_patcher` — the three ``build_module`` namespaces that must be
    patched simultaneously (``spec_utils``, ``transformer_block``, ``mamba_block``)
    and why ``MambaLayer`` needs its own ``__init__`` patch.
  * ``provider_patch._has_fully_pruned_slot`` — when ``apply_no_ops=True`` is
    needed for Mamba/hybrid providers (fully-pruned ``M`` / ``E`` slots).
"""

from __future__ import annotations

import copy
import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from block_config_utils import MCoreLayerOverrides, get_overrides_for_layer

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "NoOpRMSNorm",
    "NoOpWithBias",
    "mbridge_patcher",
]


# ---------------------------------------------------------------------------
# Thread-local patcher state
# ---------------------------------------------------------------------------
#
# Using thread-local storage means re-entrant / nested uses of mbridge_patcher
# (e.g., for a teacher and student model built on different threads) are safe.
# Within a single thread, the outermost context's values are restored on exit.

_ctx = threading.local()


def _get_ctx():
    """Return (and lazily initialise) the thread-local patcher context."""
    if not hasattr(_ctx, "block_configs"):
        _ctx.block_configs = None
        _ctx.num_attention_heads = None
        _ctx.hidden_size = None
        _ctx.apply_no_ops = True
    return _ctx


# ---------------------------------------------------------------------------
# No-op replacement modules
# ---------------------------------------------------------------------------


class NoOpWithBias(torch.nn.Module):
    """No-op module that satisfies MCore's ``(output, bias)`` tuple contract.

    MCore's ``bias_dropout_add_func`` expects the output of ``self_attention()``
    and ``mlp()`` to be a 2-tuple ``(output_tensor, optional_bias)``. This
    module returns ``(torch.zeros_like(x), None)`` so that::

        bda((zeros, None), residual, dropout) = dropout(zeros) + residual = residual

    i.e. the block has zero contribution to the hidden state — the correct
    semantics for a disabled subblock.

    ``IdentityOp.forward(x)`` returns the bare tensor and would raise
    ``TypeError: cannot unpack non-sequence Tensor`` in the BDA path, which is
    why we cannot simply reuse it here.

    The ``get_extra_state`` / ``set_extra_state`` hooks mirror the pattern used
    by Megatron's TE submodules so distributed-checkpoint sharding (one
    ``ShardedObject`` replica per TP rank) does not fail validation when
    ``_extra_state`` is missing on some ranks only.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        return torch.zeros_like(x), None

    def get_extra_state(self) -> None:
        return None

    def set_extra_state(self, state: Any) -> None:
        return None


class NoOpRMSNorm(torch.nn.Module):
    """No-op stand-in for ``torch.nn.RMSNorm`` / MCore wrapped RMS norms.

    Returns ``x`` unchanged, has no ``weight`` parameter, and accepts arbitrary
    extra ``*args`` / ``**kwargs`` so calls matching a broader norm API stay
    valid. Contrast with :class:`NoOpWithBias`, which implements the
    ``(output, bias)`` tuple contract for ``self_attention`` / ``mlp``.

    See :class:`NoOpWithBias` for why ``get_extra_state`` / ``set_extra_state``
    are required for ``torch_dist`` checkpoint sharding when replacing TE norms.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x

    def get_extra_state(self) -> None:
        return None

    def set_extra_state(self, state: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Config override application
# ---------------------------------------------------------------------------


def _apply_config_overrides(config: Any, overrides: dict) -> Any:
    """Return a *shallow copy* of ``config`` with ``overrides`` applied.

    Shallow copy avoids deep-copying CUDA tensors or ``ProcessGroup`` objects
    that may live on the config. The original config is never mutated.

    Raises:
        AttributeError: If an override key does not exist on the config object.
            A typo would silently produce the wrong model if we allowed it
            through.
    """
    if not overrides:
        return config
    patched = copy.copy(config)
    for key, value in overrides.items():
        if not hasattr(patched, key):
            raise AttributeError(
                f"mbridge_patcher: cannot override '{key}' — field does not exist on "
                f"{type(patched).__name__}. Check block_config_to_mcore_overrides()."
            )
        setattr(patched, key, value)
    return patched


def _lookup_and_apply_config(
    ctx: Any,
    config: Any,
    idx: int,
    *,
    strict_mamba_slot: bool = False,
) -> tuple[MCoreLayerOverrides | None, Any]:
    """Look up per-layer overrides for ``idx`` and apply ``config_overrides``.

    Reads ``num_attention_heads`` / ``hidden_size`` / ``block_configs`` from the
    thread-local patcher context, falling back to the corresponding fields on
    ``config`` when the context value is unset. Returns the
    :class:`MCoreLayerOverrides` for this layer (or ``None``) plus a (possibly
    new) ``config`` with ``overrides.config_overrides`` applied via
    :func:`_apply_config_overrides`.

    The ``strict_mamba_slot`` flag is forwarded to ``get_overrides_for_layer``
    and should be ``True`` only from the MambaLayer path (each MambaStack slot
    is a single dedicated subblock type).
    """
    overrides = get_overrides_for_layer(
        ctx.block_configs,
        idx,
        num_attention_heads=ctx.num_attention_heads or getattr(config, "num_attention_heads", 1),
        hidden_size=ctx.hidden_size or getattr(config, "hidden_size", 0),
        strict_mamba_slot=strict_mamba_slot,
    )
    if overrides and overrides.config_overrides:
        return overrides, _apply_config_overrides(config, overrides.config_overrides)
    return overrides, config


# ---------------------------------------------------------------------------
# No-op submodule replacement
# ---------------------------------------------------------------------------


def _is_identity_func_op(module: Any) -> bool:
    """Return True iff ``module`` is MCore's ``IdentityFuncOp``.

    When MCore wires a subblock as ``submodule = IdentityOp`` and
    ``bda = IdentityFuncOp`` (the canonical no-op pattern for inactive
    subblocks in MambaStack — e.g. the attention block of an ``E`` slot or the
    MLP block of a ``*`` slot), the call chain
    ``bda(...)( submodule(x), residual, dropout )`` evaluates to
    ``submodule(x) = x`` — a pure tensor pass-through. Replacing
    ``submodule`` with :class:`NoOpWithBias` (which returns
    ``(zeros_like(x), None)``) would turn that into a tuple-returning
    expression that breaks the next layer's ``pre_mlp_layernorm``. Detecting
    this lets :func:`_apply_no_ops` leave the canonical chain alone.
    """
    try:
        from megatron.core.transformer.identity_op import IdentityFuncOp
    except ImportError:
        return False
    return isinstance(module, IdentityFuncOp)


def _reset_is_moe_layer(layer: Any, idx: int) -> None:
    """Clear ``layer.is_moe_layer`` after the mlp subblock is no-op'd.

    If left as ``True``, the CUDA-graph early-return path in the
    transformer-layer forward executes
    ``cudagraph_outputs = self.mlp(pre_mlp_layernorm_output)`` and then treats
    its tuple result as a list — crashing. Resetting here ensures that path
    is never taken for a no-op'd MoE slot.
    """
    if getattr(layer, "is_moe_layer", False):
        layer.is_moe_layer = False
        logger.debug("Layer %d: reset is_moe_layer=False after mlp no_op replacement", idx)


@dataclass(frozen=True)
class _NoOpRule:
    """Declarative rule for replacing one disabled subblock on a built MCore layer.

    Maps a BlockConfig flag (``attention_no_op`` / ``mlp_no_op``) to:

    * ``layer_attr`` — submodule attribute we want to disable
      (``self_attention`` / ``mixer`` / ``mlp``).
    * ``bda_attr`` — accompanying ``*_bda`` attribute. When already
      ``IdentityFuncOp`` the surrounding chain is a pure tensor pass-through,
      so the rule is skipped (see :func:`_is_identity_func_op`).
    * ``norm_attr`` — pre-subblock norm to also replace with
      :class:`NoOpRMSNorm`, or ``None`` if the layer has no separate norm.
    * ``post_hook`` — additional fix-up after replacement (currently used
      only by the mlp rule, see :func:`_reset_is_moe_layer`).
    * ``flag_group`` — used only by the post-loop diagnostic to detect that
      an ``attention_no_op`` flag was set on a layer with neither
      ``self_attention`` nor ``mixer``.
    """

    flag_attr: str
    layer_attr: str
    bda_attr: str
    norm_attr: str | None
    description: str
    post_hook: Callable[[Any, int], None] | None = None


_NO_OP_RULES: tuple[_NoOpRule, ...] = (
    # TransformerLayer attention subblock
    _NoOpRule(
        flag_attr="attention_no_op",
        layer_attr="self_attention",
        bda_attr="self_attn_bda",
        norm_attr=None,
        description="self_attention",
    ),
    # MambaLayer SSM subblock — fully-pruned `M` slot from the MIP solver.
    # Without this rule the SSM mixer keeps its uninitialised weights and
    # bf16 backward NaNs at iter 1.
    _NoOpRule(
        flag_attr="attention_no_op",
        layer_attr="mixer",
        bda_attr="mamba_bda",
        norm_attr="norm",
        description="MambaLayer mixer",
    ),
    # TransformerLayer mlp subblock
    _NoOpRule(
        flag_attr="mlp_no_op",
        layer_attr="mlp",
        bda_attr="mlp_bda",
        norm_attr="pre_mlp_layernorm",
        description="mlp",
        post_hook=_reset_is_moe_layer,
    ),
)


def _apply_no_ops(layer: Any, overrides: MCoreLayerOverrides, idx: int) -> None:
    """Replace disabled subblock submodules with :class:`NoOpWithBias` on a built layer.

    Iterates over :data:`_NO_OP_RULES` and replaces ``layer.<layer_attr>`` with
    :class:`NoOpWithBias` (and the matching norm with :class:`NoOpRMSNorm`) when

      * the BlockConfig flag for that subblock is set, AND
      * the layer has the corresponding attribute, AND
      * the accompanying ``*_bda`` is **not** ``IdentityFuncOp`` — otherwise
        the canonical Mamba/hybrid no-op chain (pure tensor pass-through)
        would break if we substituted a tuple-returning module.

    Diagnostic: warns if ``attention_no_op`` is set on a layer with neither
    ``self_attention`` nor ``mixer``, indicating a new MCore layer class we
    don't know how to lower (the slot will likely NaN at iter 1).
    """
    attention_handled = False

    for rule in _NO_OP_RULES:
        if not getattr(overrides, rule.flag_attr):
            continue
        if not hasattr(layer, rule.layer_attr):
            continue

        if rule.flag_attr == "attention_no_op":
            attention_handled = True

        if _is_identity_func_op(getattr(layer, rule.bda_attr, None)):
            logger.debug(
                "Layer %d: %s no-op left as-is (bda is IdentityFuncOp, canonical "
                "Mamba/hybrid pass-through)",
                idx,
                rule.description,
            )
            continue

        setattr(layer, rule.layer_attr, NoOpWithBias())
        if rule.norm_attr is not None:
            setattr(layer, rule.norm_attr, NoOpRMSNorm())
        logger.debug("Layer %d: replaced %s with NoOpWithBias", idx, rule.description)

        if rule.post_hook is not None:
            rule.post_hook(layer, idx)

    if overrides.attention_no_op and not attention_handled:
        logger.warning(
            "Layer %d (%s): attention_no_op=True but layer has no recognised "
            "attention/mixer subblock. Skipping — fully-pruned slot may NaN at iter 1.",
            idx,
            type(layer).__name__,
        )


# ---------------------------------------------------------------------------
# The unified patcher context manager
# ---------------------------------------------------------------------------


@contextmanager
def mbridge_patcher(
    block_configs: list[Any] | None,
    num_attention_heads: int,
    hidden_size: int,
    *,
    apply_no_ops: bool = True,
):
    """Patch MCore layer construction to inject per-layer config from ``block_configs``.

    This context manager must be active during ``provider.provide()`` (or
    transitively during ``provide_distributed_model()``) for the overrides to
    take effect. ``provider_patch.py`` activates this automatically when
    ``block_configs`` is set on the provider.

    Three ``build_module`` namespaces must be patched simultaneously because
    they each hold an independent module-level reference to the function:

      1. ``megatron.core.transformer.spec_utils`` — the primary definition.
      2. ``megatron.core.transformer.transformer_block`` — local alias.
      3. ``megatron.core.ssm.mamba_block`` — local alias used by ``MambaStack``.

    Without (3), every slot of a hybrid MambaModel would be built with the
    *global* config and per-layer overrides would never apply.

    ``MambaLayer`` is constructed directly (not via ``build_module``), so we
    additionally patch ``MambaLayer.__init__`` to inject Mamba-specific
    overrides (``mamba_state_dim``, ``mamba_head_dim``, ``mamba_num_groups``,
    ``mamba_num_heads``) and to run :func:`_apply_no_ops` for fully-pruned
    ``M`` slots.

    Args:
        block_configs: List of BlockConfig-like objects (one per decoder
            layer/slot), or ``None`` / empty list to make this context a no-op.
        num_attention_heads: Global model ``num_attention_heads`` (used for
            GQA validation and as fallback when ``config.num_attention_heads``
            is unavailable).
        hidden_size: Global model ``hidden_size``.
        apply_no_ops: If ``True`` (default, appropriate for GPT-style models),
            replace disabled attention/MLP submodules with
            :class:`NoOpWithBias` after construction. Set ``False`` for
            canonical (un-pruned) Mamba/hybrid models where the MambaStack
            layer spec already assigns each slot to a single dedicated MCore
            layer type. ``provider_patch._has_fully_pruned_slot`` flips this
            back to ``True`` when the MIP solver fully prunes a slot.
    """
    if not block_configs:
        yield
        return

    import megatron.core.transformer.spec_utils
    import megatron.core.transformer.transformer_block

    has_mamba_block = False
    try:
        import megatron.core.ssm.mamba_block  # noqa: F401  ensure loaded

        has_mamba_block = True
    except ImportError:
        logger.debug("megatron.core.ssm.mamba_block not available; skipping mamba_block patch")

    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        get_transformer_layer_offset,
    )
    from megatron.core.utils import get_pg_rank

    spec_utils_mod = sys.modules["megatron.core.transformer.spec_utils"]
    transformer_block_mod = sys.modules["megatron.core.transformer.transformer_block"]
    mamba_block_mod = sys.modules.get("megatron.core.ssm.mamba_block") if has_mamba_block else None

    orig_build_module = spec_utils_mod.build_module

    ctx = _get_ctx()
    saved_state = (ctx.block_configs, ctx.num_attention_heads, ctx.hidden_size, ctx.apply_no_ops)
    ctx.block_configs = block_configs
    ctx.num_attention_heads = num_attention_heads
    ctx.hidden_size = hidden_size
    ctx.apply_no_ops = apply_no_ops

    # -------------------------------------------------------------------------
    # build_module replacement
    # -------------------------------------------------------------------------

    def patched_build_module(spec_or_module: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept TransformerLayer construction and apply per-layer config overrides."""
        # Resolve the module class from the spec argument.
        if isinstance(spec_or_module, type):
            module_cls = spec_or_module
        elif hasattr(spec_or_module, "module") and isinstance(
            getattr(spec_or_module, "module"), type
        ):
            module_cls = spec_or_module.module
        else:
            return orig_build_module(spec_or_module, *args, **kwargs)

        # Only intercept TransformerLayer and its subclasses. Using issubclass (not
        # name matching) is critical: MambaStack uses MoETransformerLayer and
        # MLPLayer, both of which subclass TransformerLayer under different names.
        if not (isinstance(module_cls, type) and issubclass(module_cls, TransformerLayer)):
            return orig_build_module(spec_or_module, *args, **kwargs)

        # Compute the global 1-based layer number from the local index plus
        # the PP-stage offset.
        layer_number = kwargs.get("layer_number", 1)
        config = kwargs.get("config")
        vp_stage = kwargs.get("vp_stage")
        pg_collection = kwargs.get("pg_collection")

        if config is None or ctx.block_configs is None:
            return orig_build_module(spec_or_module, *args, **kwargs)

        if pg_collection is None:
            # pg_collection is normally passed in kwargs by MCore ≥ 0.9.
            # Best-effort fallback for older MCore builds.
            try:
                from megatron.core.process_groups_config import ProcessGroupCollection

                pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            except Exception:
                pass

        pp_offset = (
            get_transformer_layer_offset(config, vp_stage, get_pg_rank(pg_collection.pp))
            if pg_collection is not None
            else 0
        )
        global_layer_number = layer_number + pp_offset

        overrides, patched_config = _lookup_and_apply_config(ctx, config, global_layer_number)

        if overrides and overrides.config_overrides:
            logger.debug(
                "Layer %d (%s): applying config overrides %s",
                global_layer_number,
                module_cls.__name__,
                overrides.config_overrides,
            )
            kwargs = dict(kwargs)
            kwargs["config"] = patched_config

        layer = orig_build_module(spec_or_module, *args, **kwargs)

        if overrides and (overrides.attention_no_op or overrides.mlp_no_op):
            if ctx.apply_no_ops:
                _apply_no_ops(layer, overrides, global_layer_number)
            else:
                logger.debug(
                    "Layer %d: skipping no_op replacement (apply_no_ops=False; "
                    "spec handles this intrinsically for Mamba/hybrid models)",
                    global_layer_number,
                )

        return layer

    # -------------------------------------------------------------------------
    # MambaLayer.__init__ replacement
    # -------------------------------------------------------------------------

    has_mamba_layer_patch = False
    orig_mamba_init = None

    try:
        from megatron.core.ssm.mamba_layer import MambaLayer

        orig_mamba_init = MambaLayer.__init__

        def patched_mamba_init(
            self_layer: Any,
            config: Any,
            submodules: Any,
            layer_number: int = 1,
            *args: Any,
            **kwargs: Any,
        ):
            """Inject per-layer Mamba config overrides before MambaLayer.__init__ runs.

            In MambaStack, ``layer_number`` is already the *global* 1-based index
            (set in mamba_block.py as ``i + 1 + pp_layer_offset``), so we use it
            directly as the index into ``block_configs``.

            After ``orig_mamba_init`` constructs ``self.mixer`` / ``self.norm`` /
            ``self.mamba_bda``, we run :func:`_apply_no_ops` so fully-pruned ``M``
            slots get their SSM mixer replaced with :class:`NoOpWithBias`.
            ``patched_build_module`` only intercepts ``TransformerLayer`` subclasses,
            so MambaLayer cannot rely on it for no-op replacement.
            """
            overrides = None
            if ctx.block_configs:
                overrides, config = _lookup_and_apply_config(
                    ctx, config, layer_number, strict_mamba_slot=True
                )
                if overrides and overrides.config_overrides:
                    logger.debug(
                        "MambaLayer %d: applying config overrides %s",
                        layer_number,
                        overrides.config_overrides,
                    )

            orig_mamba_init(self_layer, config, submodules, layer_number, *args, **kwargs)

            if (
                ctx.apply_no_ops
                and overrides is not None
                and (overrides.attention_no_op or overrides.mlp_no_op)
            ):
                _apply_no_ops(self_layer, overrides, layer_number)

        MambaLayer.__init__ = patched_mamba_init
        has_mamba_layer_patch = True
        logger.debug("mbridge_patcher: patched MambaLayer.__init__")

    except ImportError:
        logger.debug("megatron.core.ssm.mamba_layer not available; skipping MambaLayer patch")

    # -------------------------------------------------------------------------
    # Apply build_module patches and yield
    # -------------------------------------------------------------------------

    try:
        spec_utils_mod.build_module = patched_build_module
        transformer_block_mod.build_module = patched_build_module
        if mamba_block_mod is not None:
            mamba_block_mod.build_module = patched_build_module

        logger.info(
            "mbridge_patcher: active — block_configs=%d layers, apply_no_ops=%s, "
            "mamba_block_patched=%s, mamba_layer_patched=%s",
            len(block_configs),
            apply_no_ops,
            has_mamba_block,
            has_mamba_layer_patch,
        )
        yield

    finally:
        spec_utils_mod.build_module = orig_build_module
        transformer_block_mod.build_module = orig_build_module
        if mamba_block_mod is not None:
            mamba_block_mod.build_module = orig_build_module

        if has_mamba_layer_patch:
            MambaLayer.__init__ = orig_mamba_init

        ctx.block_configs, ctx.num_attention_heads, ctx.hidden_size, ctx.apply_no_ops = saved_state

        logger.info("mbridge_patcher: exited, all patches restored")
