# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context manager that patches MCore layer construction for heterogeneous AnyModel models.

Problem statement
-----------------
Megatron Bridge providers (``GPTModelProvider``, ``MambaModelProvider``, etc.) are dataclasses
that hold a *single* global ``TransformerConfig``. When ``provider.provide()`` builds the model,
MCore constructs every decoder layer using this one shared config. For heterogeneous
(AnyModel / Puzzletron) checkpoints, each layer has its own architecture: different
``num_moe_experts``, ``ffn_hidden_size``, GQA ``num_query_groups``, Mamba state dims, etc.
Building all layers from the same global config produces the wrong weight shapes and causes
weight-loading failures.

How the patcher works
---------------------
The patcher intercepts MCore's per-layer construction by monkey-patching
``megatron.core.transformer.spec_utils.build_module``.  Every transformer layer is built
through this single function, so intercepting it lets us inject per-layer config overrides
before the layer's ``__init__`` runs.

Three module-level references to ``build_module`` must be patched simultaneously:

    1. ``megatron.core.transformer.spec_utils``        — the primary definition.
    2. ``megatron.core.transformer.transformer_block`` — a local alias imported at module load.
    3. ``megatron.core.ssm.mamba_block``               — another local alias (critical!).

Why mamba_block needs its own patch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``megatron.core.ssm.mamba_block`` (``MambaStack``) does::

    from megatron.core.transformer.spec_utils import build_module   # top-of-file import

This creates a local reference in the ``mamba_block`` module namespace.  Patching only
``spec_utils.build_module`` leaves ``mamba_block.build_module`` pointing at the original,
so all 52 slots of a hybrid MambaModel (Mamba, MoE, attention) are built with the *global*
config and the per-layer overrides are never applied.  Patching the local reference fixes
this.

MambaLayer.__init__ patching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``MambaLayer`` is not built via ``build_module`` — MambaStack calls it directly.  So we also
patch ``MambaLayer.__init__`` to inject Mamba-specific per-layer overrides (``mamba_state_dim``,
``mamba_head_dim``, ``mamba_num_groups``, ``mamba_num_heads``).

No-op submodule replacement (GPT models only)
----------------------------------------------
For GPT / dense-attention models, a BlockConfig may mark the attention or MLP sub-block as
``no_op``.  After the layer is instantiated with the correct config, those submodules are
replaced with ``NoOpWithBias``.

**Why IdentityOp is wrong for this job**

MCore's bias-dropout-add residual path expects the output of ``self_attention()`` and
``mlp()`` to be a *tuple* ``(output_tensor, bias_or_None)``::

    # megatron/core/fusions/fused_bias_dropout.py
    def _bias_dropout_add_func(x_with_bias, residual, prob, training):
        x, bias = x_with_bias   # ← unpacking requires a 2-tuple

``IdentityOp.forward(x, *args, **kwargs)`` returns the plain tensor ``x``.  Passing that to
the BDA function raises ``TypeError: cannot unpack non-sequence Tensor``.

**``NoOpWithBias`` — the correct replacement**

``NoOpWithBias.forward(x, *args, **kwargs)`` returns ``(torch.zeros_like(x), None)``.
Plugging that into the BDA formula::

    bda((zeros, None), residual, dropout) = dropout(zeros) + residual = residual

The block leaves the hidden state unchanged — exactly the semantics of "no_op".

**MoE ``is_moe_layer`` flag**

When a MoE layer's MLP is replaced with ``NoOpWithBias``, the ``is_moe_layer`` flag on the
layer (set at ``__init__`` time based on ``isinstance(self.mlp, MoELayer)``) must be reset to
``False``.  If left as ``True``, the CUDA-graph early-return path in the transformer-layer
forward runs::

    cudagraph_outputs = self.mlp(pre_mlp_layernorm_output)
    return cudagraph_outputs + [residual]   # treats tuple as a list → crash

Setting ``layer.is_moe_layer = False`` ensures this path is never taken.

Mamba/hybrid models — apply_no_ops=False
-----------------------------------------
In ``MambaStack``, each slot is *already* a single dedicated MCore layer type chosen by the
layer spec: ``MambaLayer``, ``TransformerLayer`` (attention slot), ``MLPLayer``, or
``MoETransformerLayer``.  The spec itself encodes the "no_op" concept — there is no combined
attention + MLP layer to partially disable.  Setting ``apply_no_ops=False`` (which is done
automatically by ``_patched_provide`` in ``provider_patch.py``) prevents the patcher from
incorrectly trying to replace non-existent submodules.
"""

from __future__ import annotations

import copy
import logging
import sys
import threading
from contextlib import contextmanager
from typing import Any, List, Optional, Union

import torch

from block_config_utils import MCoreLayerOverrides, get_overrides_for_layer

logger = logging.getLogger(__name__)


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
# NoOpWithBias — the correct no-op replacement for attention and MLP submodules
# ---------------------------------------------------------------------------


class NoOpWithBias(torch.nn.Module):
    """A no-op module that satisfies MCore's (output, bias) contract.

    MCore's ``bias_dropout_add_func`` expects the output of ``self_attention()`` and ``mlp()``
    to be a 2-tuple ``(output_tensor, optional_bias)``.  This module returns
    ``(torch.zeros_like(x), None)`` so that::

        bda((zeros, None), residual, dropout) = dropout(zeros) + residual = residual

    i.e., the block has zero contribution to the hidden state — the correct semantics for
    a disabled ("no_op") attention or MLP slot.

    The input ``x`` is the post-layernorm hidden states fed to the submodule.  Its shape
    always matches the expected output shape (same hidden dimension), so ``zeros_like(x)``
    produces a correctly shaped zero tensor without any extra shape bookkeeping.

    Transformer-Engine / Megatron submodules often expose ``get_extra_state`` /
    ``set_extra_state`` so ``state_dict`` includes ``_extra_state``.  Megatron distributed
    checkpointing wraps each ``_extra_state`` as a ``ShardedObject`` with one replica per
    tensor-parallel rank.  If a replacement module omits ``_extra_state`` on some ranks only,
    validation fails (missing shard).  These hooks mirror Megatron's ``ColumnParallelLinear``
    pattern: empty extra state, but present on every rank.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        return torch.zeros_like(x), None

    def get_extra_state(self) -> None:
        """Return empty TE-compatible extra state for distributed checkpoint sharding."""
        return None

    def set_extra_state(self, state: Any) -> None:
        """Ignore loaded extra state (no TE buffers on this no-op)."""
        return None


class NoOpRMSNorm(torch.nn.Module):
    """No-op stand-in for ``torch.nn.RMSNorm`` / MCore wrapped RMS norms.

    Transformer layers call ``input_layernorm(x)`` and ``pre_mlp_layernorm(x)`` with a
    single tensor and expect a tensor back.  This module returns ``x`` unchanged (identity),
    has **no** ``weight`` parameter, and accepts extra ``*args`` / ``**kwargs`` so calls
    that match a broader norm API remain valid.

    Contrast with :class:`NoOpWithBias`, which implements the ``(output, bias)`` tuple
    contract required by ``self_attention`` and ``mlp``.

    See :class:`NoOpWithBias` docstring for why ``get_extra_state`` / ``set_extra_state``
    are required for Megatron ``torch_dist`` checkpoints when replacing TE norms.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros_like(x) #x

    def get_extra_state(self) -> None:
        """Return empty TE-compatible extra state for distributed checkpoint sharding."""
        return None

    def set_extra_state(self, state: Any) -> None:
        """Ignore loaded extra state (no TE buffers on this no-op)."""
        return None


# ---------------------------------------------------------------------------
# Config override application
# ---------------------------------------------------------------------------


def _apply_config_overrides(config: Any, overrides: dict) -> Any:
    """Return a *shallow copy* of ``config`` with ``overrides`` applied.

    Shallow copy avoids deep-copying CUDA tensors or ``ProcessGroup`` objects that may
    live on the config.  The original config is never mutated.

    Raises:
        AttributeError: If an override key does not exist on the config object.
            This is a hard error — a typo in the override key would silently produce the
            wrong model if we allowed it to pass through.
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


# ---------------------------------------------------------------------------
# No-op submodule replacement (GPT models only)
# ---------------------------------------------------------------------------


def _apply_no_ops(layer: Any, overrides: MCoreLayerOverrides, global_layer_number: int) -> None:
    """Replace disabled attention/MLP submodules with ``NoOpWithBias`` on a built layer.

    Must be called after the layer has been fully constructed so the submodule attributes
    exist.

    Side effects:
        - If attention_no_op: replaces ``layer.self_attention`` with ``NoOpWithBias()``.
        - If mlp_no_op: replaces ``layer.mlp`` with ``NoOpWithBias()`` and resets
          ``layer.is_moe_layer = False`` (see module docstring for why).
    """
    if overrides.attention_no_op:
        if not hasattr(layer, "self_attention"):
            logger.warning(
                "Layer %d: attention_no_op=True but layer has no self_attention attribute "
                "(layer type=%s). Skipping.",
                global_layer_number,
                type(layer).__name__,
            )
        else:
            logger.debug(
                "Layer %d: replacing self_attention with NoOpWithBias", global_layer_number
            )
            #TODO check if input_layernorm is an IdentityOp
            layer.self_attention = NoOpWithBias()

    if overrides.mlp_no_op:
        if not hasattr(layer, "mlp"):
            logger.warning(
                "Layer %d: mlp_no_op=True but layer has no mlp attribute "
                "(layer type=%s). Skipping.",
                global_layer_number,
                type(layer).__name__,
            )
        else:
            logger.debug(
                "Layer %d: replacing mlp with NoOpWithBias", global_layer_number
            )
            layer.pre_mlp_layernorm = NoOpRMSNorm()
            layer.mlp = NoOpWithBias()

            # Critical: reset is_moe_layer so the CUDA graph MoE early-return path is not
            # taken with our NoOpWithBias (see module docstring for the full explanation).
            if getattr(layer, "is_moe_layer", False):
                layer.is_moe_layer = False
                logger.debug(
                    "Layer %d: reset is_moe_layer=False after mlp no_op replacement",
                    global_layer_number,
                )


# ---------------------------------------------------------------------------
# The unified patcher context manager
# ---------------------------------------------------------------------------


@contextmanager
def mbridge_patcher(
    block_configs: Optional[List[Union["BlockConfig", dict]]],
    num_attention_heads: int,
    hidden_size: int,
    *,
    apply_no_ops: bool = True,
):
    """Patch MCore layer construction to inject per-layer config from ``block_configs``.

    This context manager must be active during ``provider.provide()`` (or transitively
    during ``provide_distributed_model()``) for the overrides to take effect.  The
    ``provider_patch.py`` module activates this automatically when ``block_configs`` is set
    on the provider.

    The patcher covers three ``build_module`` namespaces (see module docstring for why all
    three are necessary) and ``MambaLayer.__init__``.

    Args:
        block_configs: List of BlockConfig-like objects (one per decoder layer/slot), or
            ``None`` / empty list to make this context a no-op.
        num_attention_heads: Global model ``num_attention_heads`` (used for GQA validation
            and as fallback when ``config.num_attention_heads`` is unavailable).
        hidden_size: Global model ``hidden_size``.
        apply_no_ops: If ``True`` (default, appropriate for GPT-style models), replace
            disabled attention/MLP submodules with ``NoOpWithBias`` after construction.
            Set ``False`` for Mamba/hybrid models where the MambaStack layer spec already
            assigns each slot to a single dedicated MCore layer type — no post-construction
            submodule replacement is needed or correct.
    """
    if not block_configs:
        # Nothing to do. Yield immediately rather than importing / patching anything.
        yield
        return

    # --- Ensure all module namespaces are loaded before we capture their build_module ---
    import megatron.core.transformer.spec_utils        # noqa: F401  ensure loaded
    import megatron.core.transformer.transformer_block  # noqa: F401  ensure loaded

    has_mamba_block = False
    try:
        import megatron.core.ssm.mamba_block  # noqa: F401  ensure loaded
        has_mamba_block = True
    except ImportError:
        logger.debug("megatron.core.ssm.mamba_block not available; skipping mamba_block patch")

    from megatron.core.transformer.transformer_layer import TransformerLayer, get_transformer_layer_offset
    from megatron.core.utils import get_pg_rank

    spec_utils_mod        = sys.modules["megatron.core.transformer.spec_utils"]
    transformer_block_mod = sys.modules["megatron.core.transformer.transformer_block"]
    mamba_block_mod       = sys.modules.get("megatron.core.ssm.mamba_block") if has_mamba_block else None

    orig_build_module = spec_utils_mod.build_module

    # --- Save and populate thread-local context ---
    ctx = _get_ctx()
    saved_state = (ctx.block_configs, ctx.num_attention_heads, ctx.hidden_size, ctx.apply_no_ops)
    ctx.block_configs        = block_configs
    ctx.num_attention_heads  = num_attention_heads
    ctx.hidden_size          = hidden_size
    ctx.apply_no_ops         = apply_no_ops

    # -------------------------------------------------------------------------
    # build_module replacement
    # -------------------------------------------------------------------------

    def patched_build_module(spec_or_module: Any, *args: Any, **kwargs: Any) -> Any:
        """Intercept TransformerLayer construction and apply per-layer config overrides."""

        # Step 1: Resolve the module class from the spec argument.
        if isinstance(spec_or_module, type):
            module_cls = spec_or_module
        elif hasattr(spec_or_module, "module") and isinstance(
            getattr(spec_or_module, "module"), type
        ):
            module_cls = spec_or_module.module
        else:
            # Not a type or typed spec (e.g. a string or callable spec) — pass through.
            return orig_build_module(spec_or_module, *args, **kwargs)


        # Step 2: Only intercept TransformerLayer and its subclasses.
        #
        # Using issubclass (not name matching) is critical: MambaStack uses
        # MoETransformerLayer and MLPLayer, both of which are subclasses of TransformerLayer
        # but have different names. A name-only check would silently miss them.
        if not (isinstance(module_cls, type) and issubclass(module_cls, TransformerLayer)):
            return orig_build_module(spec_or_module, *args, **kwargs)

        # Step 3: Compute the global 1-based layer number.
        #
        # layer_number in kwargs is the *local* layer index within the current PP stage
        # (1-based, starting from 1 for the first layer on this stage).
        # To index into block_configs correctly, we need the *global* layer number.
        layer_number  = kwargs.get("layer_number", 1)
        config        = kwargs.get("config")
        vp_stage      = kwargs.get("vp_stage")
        pg_collection = kwargs.get("pg_collection")

        if config is None or ctx.block_configs is None:
            return orig_build_module(spec_or_module, *args, **kwargs)

        if pg_collection is None:
            # pg_collection is normally passed in kwargs by MCore ≥ 0.9.
            # Provide a best-effort fallback for older MCore builds.
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

        # Step 4: Look up per-layer overrides.
        effective_num_heads = ctx.num_attention_heads or getattr(config, "num_attention_heads", 1)
        effective_hidden    = ctx.hidden_size or getattr(config, "hidden_size", 0)
        overrides = get_overrides_for_layer(
            ctx.block_configs,
            global_layer_number,
            num_attention_heads=effective_num_heads,
            hidden_size=effective_hidden,
        )

        # Step 5: Apply config overrides before building the layer.
        if overrides and overrides.config_overrides:
            logger.debug(
                "Layer %d (%s): applying config overrides %s",
                global_layer_number,
                module_cls.__name__,
                overrides.config_overrides,
            )
            kwargs = dict(kwargs)
            kwargs["config"] = _apply_config_overrides(config, overrides.config_overrides)

        # Step 6: Build the layer with the (potentially overridden) config.
        layer = orig_build_module(spec_or_module, *args, **kwargs)

        # Step 7: Optionally replace disabled submodules with no-ops.
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

            In MambaStack, layer_number is already the *global* 1-based index:
                layer_number = i + 1 + pp_layer_offset   (set in mamba_block.py)
            So we use it directly as the index into block_configs without any additional
            PP offset calculation.
            """
            if ctx.block_configs:
                effective_num_heads = ctx.num_attention_heads or getattr(config, "num_attention_heads", 1)
                effective_hidden    = ctx.hidden_size or getattr(config, "hidden_size", 0)
                overrides = get_overrides_for_layer(
                    ctx.block_configs,
                    layer_number,
                    num_attention_heads=effective_num_heads,
                    hidden_size=effective_hidden,
                    strict_mamba_slot=True,  # each MambaStack slot must be a single subblock type
                )
                if overrides and overrides.config_overrides:
                    logger.debug(
                        "MambaLayer %d: applying config overrides %s",
                        layer_number,
                        overrides.config_overrides,
                    )
                    config = _apply_config_overrides(config, overrides.config_overrides)

            orig_mamba_init(self_layer, config, submodules, layer_number, *args, **kwargs)

        MambaLayer.__init__ = patched_mamba_init
        has_mamba_layer_patch = True
        logger.debug("mbridge_patcher: patched MambaLayer.__init__")

    except ImportError:
        logger.debug("megatron.core.ssm.mamba_layer not available; skipping MambaLayer patch")

    # -------------------------------------------------------------------------
    # Apply build_module patches and yield
    # -------------------------------------------------------------------------

    try:
        spec_utils_mod.build_module        = patched_build_module
        transformer_block_mod.build_module = patched_build_module
        if mamba_block_mod is not None:
            mamba_block_mod.build_module   = patched_build_module

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
        # Restore all patches unconditionally, even if an exception occurred.
        spec_utils_mod.build_module        = orig_build_module
        transformer_block_mod.build_module = orig_build_module
        if mamba_block_mod is not None:
            mamba_block_mod.build_module   = orig_build_module

        if has_mamba_layer_patch:
            MambaLayer.__init__ = orig_mamba_init  # noqa: F821  (defined above in try block)

        # Restore thread-local context to the values that were active before this call.
        ctx.block_configs, ctx.num_attention_heads, ctx.hidden_size, ctx.apply_no_ops = saved_state

        logger.info("mbridge_patcher: exited, all patches restored")
