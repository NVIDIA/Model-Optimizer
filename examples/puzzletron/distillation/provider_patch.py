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

"""Patch Megatron Bridge ModelProviderMixin.provide() to activate mbridge_patcher automatically.

Overview
--------
After obtaining a Bridge provider via ``bridge.to_megatron_provider()``, call
``set_provider_block_configs(provider, block_configs)`` to attach the heterogeneous layer
configs.  From that point on, any call to ``provider.provide()`` or
``provider.provide_distributed_model()`` will automatically run inside ``mbridge_patcher``,
ensuring every layer is built with the correct per-layer config.

Two-level patching
------------------
Megatron Bridge has a class hierarchy where ``ModelProviderMixin`` defines the default
``provide()`` abstract method.  Many concrete providers (``GPTModelProvider``,
``MambaModelProvider``) subclass it and override ``provide()`` to build the specific MCore
model.

If we only patch ``ModelProviderMixin.provide``, subclasses that override ``provide()`` will
call *their own* implementation, not the patched one (Python MRO).  To handle this,
``set_provider_block_configs()`` checks whether the provider instance's ``provide`` method is
already our patched version, and if not, patches the **instance method** directly.

This design means:
  - ``apply_patch()`` (class-level) is sufficient for providers that do not override provide().
  - ``set_provider_block_configs()`` always guarantees the instance is covered.

Thread safety
-------------
``_patched_provide`` reads ``self.block_configs``, ``self.num_attention_heads``, and
``self.hidden_size`` from the provider instance at call time, so it is safe to call from
multiple threads as long as those attributes are not mutated concurrently.
"""

from __future__ import annotations

import logging
import types
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)

_PATCH_SENTINEL = "_mbridge_provider_patch_applied"
_DISTILLATION_PATCH_SENTINEL = "_mbridge_distillation_patch_applied"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def set_provider_block_configs(
    provider: Any,
    block_configs: Optional[List[Union[dict, Any]]],
) -> None:
    """Attach ``block_configs`` to a provider and ensure ``provide()`` uses mbridge_patcher.

    Call this after ``bridge.to_megatron_provider()`` and before
    ``provider.provide_distributed_model()``.

    Args:
        provider: Any ``ModelProviderMixin`` instance returned by Bridge
            (e.g. ``GPTModelProvider``, ``MambaModelProvider``).
        block_configs: List of block config dicts or BlockConfig objects, one per decoder
            layer.  Pass ``None`` or ``[]`` to disable per-layer overrides entirely.
    """
    provider.block_configs = block_configs

    if not block_configs:
        # Nothing to activate — leave the provider's provide() as-is.
        return

    # Check if the instance's provide() already points to our patched function.
    current_func = getattr(getattr(provider, "provide", None), "__func__", None)
    if current_func is _patched_provide:
        # Class-level patch already covers this provider.
        logger.debug(
            "%s: provide() already patched at class level; no instance patch needed.",
            type(provider).__name__,
        )
        return

    # The provider's subclass overrides provide() directly.  Patch the instance.
    logger.debug(
        "%s: provide() is overridden by subclass; patching instance method.",
        type(provider).__name__,
    )
    provider._mbridge_orig_provide = provider.provide
    provider.provide = types.MethodType(_patched_provide, provider)


def apply_patch() -> None:
    """Patch ``ModelProviderMixin.provide`` at the *class* level.

    Safe to call multiple times — subsequent calls are no-ops.

    Note: ``set_provider_block_configs()`` handles providers whose subclass overrides
    ``provide()`` by patching the instance method instead.  Calling ``apply_patch()`` is
    still recommended as a defence-in-depth measure for providers that rely on the MRO.
    """
    from megatron.bridge.models.model_provider import ModelProviderMixin

    if getattr(ModelProviderMixin, _PATCH_SENTINEL, False):
        logger.debug("apply_patch: ModelProviderMixin.provide already patched; skipping.")
        return

    setattr(ModelProviderMixin, "_mbridge_orig_provide", ModelProviderMixin.provide)
    ModelProviderMixin.provide = _patched_provide
    setattr(ModelProviderMixin, _PATCH_SENTINEL, True)
    logger.info("apply_patch: patched ModelProviderMixin.provide for heterogeneous AnyModel support")


def remove_patch() -> None:
    """Restore ``ModelProviderMixin.provide`` to its original implementation.

    Safe to call even if ``apply_patch()`` was never called.
    """
    from megatron.bridge.models.model_provider import ModelProviderMixin

    if not getattr(ModelProviderMixin, _PATCH_SENTINEL, False):
        return

    orig = getattr(ModelProviderMixin, "_mbridge_orig_provide", None)
    if orig is not None:
        ModelProviderMixin.provide = orig
        delattr(ModelProviderMixin, "_mbridge_orig_provide")
    if hasattr(ModelProviderMixin, _PATCH_SENTINEL):
        delattr(ModelProviderMixin, _PATCH_SENTINEL)
    logger.info("remove_patch: restored original ModelProviderMixin.provide")


# ---------------------------------------------------------------------------
# The patched provide() implementation
# ---------------------------------------------------------------------------


def _patched_provide(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Replacement for ``ModelProviderMixin.provide()`` that activates ``mbridge_patcher``.

    If ``self.block_configs`` is not set (or is empty), the original ``provide()`` is called
    unchanged — no overhead, no patching.

    If ``self.block_configs`` is set:
        1. Reads ``num_attention_heads`` and ``hidden_size`` from the provider.
        2. Detects whether this is a Mamba/hybrid provider (to set ``apply_no_ops``).
        3. Activates ``mbridge_patcher`` and delegates to the original ``provide()``.
    """
    from layer_patchers import mbridge_patcher

    block_configs = getattr(self, "block_configs", None)

    # Retrieve the original provide method saved at patch time.
    orig = getattr(self, "_mbridge_orig_provide", None)
    if orig is None:
        raise RuntimeError(
            f"{type(self).__name__}.provide(): _mbridge_orig_provide not found. "
            "Call apply_patch() or set_provider_block_configs() before using the provider."
        )

    def call_orig():
        # orig may be a bound method (instance patch) or unbound function (class patch).
        return orig(*args, **kwargs) if hasattr(orig, "__self__") else orig(self, *args, **kwargs)

    if not block_configs:
        return call_orig()

    # Extract the global model dimensions needed for GQA validation and Mamba overrides.
    num_attention_heads = (
        getattr(self, "num_attention_heads", None) or getattr(self, "num_heads", None)
    )
    hidden_size = getattr(self, "hidden_size", None)

    if num_attention_heads is None or hidden_size is None:
        logger.warning(
            "%s.provide(): block_configs is set but num_attention_heads=%r or hidden_size=%r "
            "is missing on the provider. Falling back to homogeneous (global config only).",
            type(self).__name__,
            num_attention_heads,
            hidden_size,
        )
        return call_orig()

    # Mamba/hybrid providers: each MambaStack slot is already a single dedicated layer type
    # chosen by the spec (MambaLayer, TransformerLayer for attention, MLPLayer, or
    # MoETransformerLayer).  The spec handles the "no_op" concept intrinsically — there is
    # no combined attention+MLP TransformerLayer to partially disable.  Setting
    # apply_no_ops=False prevents incorrectly replacing non-existent submodules.
    is_mamba_provider = "Mamba" in type(self).__name__
    apply_no_ops = not is_mamba_provider

    logger.info(
        "%s.provide(): activating mbridge_patcher "
        "(block_configs=%d layers, apply_no_ops=%s)",
        type(self).__name__,
        len(block_configs),
        apply_no_ops,
    )

    with mbridge_patcher(
        block_configs=block_configs,
        num_attention_heads=int(num_attention_heads),
        hidden_size=int(hidden_size),
        apply_no_ops=apply_no_ops,
    ):
        return call_orig()


# ---------------------------------------------------------------------------
# Distillation-aware patching
# ---------------------------------------------------------------------------
#
# ``DistillationProvider.provide()`` calls ``self._super_class.provide(self, ...)``
# directly — bypassing both the class-level ``ModelProviderMixin.provide`` patch
# and any instance-level patch placed by ``set_provider_block_configs()``.
#
# ``apply_distillation_patch()`` patches ``DistillationProvider.provide`` at the
# class level to intercept that call and wrap ``_super_class.provide`` with
# ``mbridge_patcher`` when ``self.student_block_configs`` is present.
# ``set_student_block_configs()`` attaches that attribute to the provider.


def apply_distillation_patch() -> None:
    """Patch ``DistillationProvider.provide`` for heterogeneous student support.

    ``DistillationProvider.provide()`` builds the student model by calling
    ``self._super_class.provide(self, ...)`` directly, bypassing both the
    class-level ``ModelProviderMixin.provide`` patch (set by ``apply_patch()``)
    and any instance-level patch set by ``set_provider_block_configs()``.

    This function patches ``DistillationProvider.provide`` at the class level so
    that, when ``self.student_block_configs`` is set, it temporarily replaces
    ``self._super_class.provide`` with a wrapper that activates ``mbridge_patcher``
    around the actual student build.  The replacement is restored in a ``finally``
    block, making the operation safe even if an exception occurs during model
    construction.

    The teacher model is built via ``self.teacher.provide_distributed_model()``
    inside ``DistillationProvider.provide()``, which calls ``self.teacher.provide()``.
    That path goes through the instance-level patch applied by
    ``set_provider_block_configs(teacher, teacher_block_configs)``, so the teacher
    is handled correctly without any additional changes here.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    from megatron.bridge.models.distillation_provider import DistillationProvider

    if getattr(DistillationProvider, _DISTILLATION_PATCH_SENTINEL, False):
        logger.debug(
            "apply_distillation_patch: DistillationProvider.provide already patched; skipping."
        )
        return

    orig_distillation_provide = DistillationProvider.provide
    setattr(DistillationProvider, "_mbridge_orig_distillation_provide", orig_distillation_provide)

    def _patched_distillation_provide(
        self: Any, pre_process: Any = None, post_process: Any = None, vp_stage: Any = None
    ) -> Any:
        """Wrap the student model build in mbridge_patcher when student_block_configs is set."""
        from layer_patchers import mbridge_patcher

        student_bc = getattr(self, "student_block_configs", None)
        num_attn_heads = (
            getattr(self, "num_attention_heads", None) or getattr(self, "num_heads", None)
        )
        hidden_size = getattr(self, "hidden_size", None)

        def _call_orig():
            return orig_distillation_provide(self, pre_process, post_process, vp_stage)

        if not student_bc or num_attn_heads is None or hidden_size is None:
            return _call_orig()

        # Detect student architecture: Mamba providers use the spec as the no_op mechanism,
        # so apply_no_ops=False prevents incorrect submodule replacement.
        is_student_mamba = "Mamba" in self._super_class.__name__
        apply_no_ops = not is_student_mamba

        logger.info(
            "DistillationProvider.provide(): activating mbridge_patcher for student "
            "(student_block_configs=%d layers, apply_no_ops=%s, super_class=%s)",
            len(student_bc),
            apply_no_ops,
            self._super_class.__name__,
        )

        # Temporarily replace _super_class.provide with a wrapper that activates
        # mbridge_patcher around the student build.  The original is restored in
        # the finally block to leave the class unmodified after construction.
        orig_super_provide = self._super_class.provide

        def _wrapped_super_provide(self_inner: Any, pre: Any = None, post: Any = None, vp: Any = None) -> Any:
            with mbridge_patcher(
                block_configs=student_bc,
                num_attention_heads=int(num_attn_heads),
                hidden_size=int(hidden_size),
                apply_no_ops=apply_no_ops,
            ):
                return orig_super_provide(self_inner, pre, post, vp)

        self._super_class.provide = _wrapped_super_provide
        try:
            return _call_orig()
        finally:
            self._super_class.provide = orig_super_provide

    DistillationProvider.provide = _patched_distillation_provide
    setattr(DistillationProvider, _DISTILLATION_PATCH_SENTINEL, True)
    logger.info(
        "apply_distillation_patch: patched DistillationProvider.provide "
        "for heterogeneous student support"
    )


def remove_distillation_patch() -> None:
    """Restore ``DistillationProvider.provide`` to its original implementation.

    Safe to call even if ``apply_distillation_patch()`` was never called.
    """
    try:
        from megatron.bridge.models.distillation_provider import DistillationProvider
    except ImportError:
        return

    if not getattr(DistillationProvider, _DISTILLATION_PATCH_SENTINEL, False):
        return

    orig = getattr(DistillationProvider, "_mbridge_orig_distillation_provide", None)
    if orig is not None:
        DistillationProvider.provide = orig
        delattr(DistillationProvider, "_mbridge_orig_distillation_provide")
    if hasattr(DistillationProvider, _DISTILLATION_PATCH_SENTINEL):
        delattr(DistillationProvider, _DISTILLATION_PATCH_SENTINEL)
    logger.info("remove_distillation_patch: restored original DistillationProvider.provide")


def set_student_block_configs(
    distillation_provider: Any,
    student_block_configs: Optional[List[Union[dict, Any]]],
) -> None:
    """Attach ``student_block_configs`` to a ``DistillationProvider``.

    Call this after ``convert_to_distillation_provider()`` and before
    ``provider.provide_distributed_model()`` to configure per-layer overrides
    for the student model.

    ``apply_distillation_patch()`` must also be called (before or after this function)
    to activate the student mbridge_patcher during model construction.

    Args:
        distillation_provider: A ``DistillationProvider`` instance returned by
            ``convert_to_distillation_provider()``.
        student_block_configs: Per-layer block config dicts/objects for the student
            (one per decoder layer), or ``None`` / ``[]`` to use the global homogeneous
            config (no per-layer overrides).
    """
    distillation_provider.student_block_configs = student_block_configs
    if student_block_configs:
        logger.info(
            "%s: student_block_configs attached (%d layers)",
            type(distillation_provider).__name__,
            len(student_block_configs),
        )
    else:
        logger.info(
            "%s: student_block_configs is None/empty — student will use global homogeneous config.",
            type(distillation_provider).__name__,
        )
