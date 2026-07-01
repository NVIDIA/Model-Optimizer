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

"""Reusable distillation helpers for heterogeneous AnyModel/Puzzletron models.

These helpers are consumed by the distillation example (``PuzzletronHooks``) but
contain no dependency on the example's ``DistillHooks`` base class, so they live
in the library alongside the layer/provider patchers:

- :func:`build_distill_config_container` — build a Bridge ``ConfigContainer`` for
  distillation from ``_pretrain_common()`` defaults.
- :func:`sync_teacher_from_student` / :func:`sync_teacher_config_from_student` —
  copy shared parallelism / checkpoint-layout fields from student to teacher.
- :func:`install_hybrid_moe_aux_loss_size_fix` — align ``track_moe_metrics``'s
  tracker size with what MoE routers actually use, preventing NCCL deadlocks when
  a PP stage has zero surviving MoE layers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from megatron.bridge.utils.common_utils import get_rank_safe

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)

__all__ = [
    "build_distill_config_container",
    "install_hybrid_moe_aux_loss_size_fix",
    "sync_teacher_config_from_student",
    "sync_teacher_from_student",
]


# ---------------------------------------------------------------------------
# ConfigContainer construction helpers
# ---------------------------------------------------------------------------


def build_distill_config_container(
    distill_provider, student_checkpoint_path: str
) -> ConfigContainer:
    """Build a ``ConfigContainer`` for distillation using Bridge ``_pretrain_common()`` defaults.

    Sets ``model`` to the :class:`DistillationProvider`, points the HF tokenizer at the
    student checkpoint, and aligns dataset ``seq_length`` with the provider. YAML / CLI
    overrides are applied afterwards by the caller.
    """
    from megatron.bridge.recipes.common import _pretrain_common

    cfg = _pretrain_common()
    cfg.model = distill_provider

    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = student_checkpoint_path

    provider_seq_len = getattr(distill_provider, "seq_length", None)
    if provider_seq_len:
        cfg.dataset.seq_length = provider_seq_len

    return cfg


def sync_teacher_from_student(student_provider, teacher_provider) -> None:
    """Copy shared parallelism / checkpoint layout fields from student to teacher."""
    shared_attrs = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "pipeline_dtype",
        "seq_length",
        "hetereogenous_dist_checkpoint",
    )
    for attr in shared_attrs:
        if hasattr(teacher_provider, attr):
            setattr(teacher_provider, attr, getattr(student_provider, attr))


def sync_teacher_config_from_student(distill_provider) -> None:
    """Copy shared fields from the student (distill_provider) to its teacher."""
    teacher = getattr(distill_provider, "teacher", None)
    if teacher is not None:
        sync_teacher_from_student(distill_provider, teacher)


# ---------------------------------------------------------------------------
# Hybrid MoE aux-loss tracker fix
# ---------------------------------------------------------------------------
# Bridge's `training_log` passes ``num_layers = hybrid_layer_pattern.count("E")``
# (e.g. 13) to ``track_moe_metrics`` for hybrid models, but Megatron-Core MoE
# routers write into ``tracker[name]["values"]`` of size ``self.config.num_layers``
# (e.g. 56 — the *full* pattern length, see core/transformer/moe/router.py).
#
# When ``_apply_no_ops`` (in layer_patchers.py) replaces the MLP of a fully
# pruned ``E`` slot with ``NoOpWithBias`` and resets ``is_moe_layer=False``, the
# router for that layer never fires.  If a particular PP stage ends up with
# *zero* surviving MoE layers, no router on that stage writes to the tracker,
# so ``track_moe_metrics(..., force_initialize=True)`` creates the entry with
# size 13 on those ranks while ranks on stages with surviving MoE layers still
# carry size-56 tensors from their routers.
#
# The first cross-PP ``all_reduce`` inside
# ``reduce_aux_losses_tracker_across_ranks`` then deadlocks (NCCL silently
# hangs on mismatched buffer sizes).  py-spy stack:
#     all_reduce -> reduce_aux_losses_tracker_across_ranks
#                -> track_moe_metrics -> training_log
#
# Fix: monkey-patch ``track_moe_metrics`` to override the ``num_layers``
# argument with ``config.num_layers`` (matching the router) and synthesise
# ``moe_layer_freq`` from ``hybrid_layer_pattern`` so the per-MoE-layer
# averaging denominator (``num_moe_layers`` at moe_utils.py:1090) stays
# correct (= count of "E" slots).
def install_hybrid_moe_aux_loss_size_fix(config: ConfigContainer) -> None:
    """Align track_moe_metrics's tracker size with what MoE routers actually use.

    No-op for non-hybrid models or when no MoE experts are configured.
    """
    model_cfg = config.model
    rank0 = get_rank_safe() == 0

    def _info(msg: str, *args: Any) -> None:
        # Use print (not logger) so the message survives even if the logger is
        # not yet configured / muted on rank 0 at this point in startup.
        if rank0:
            print("[MoEAuxFix-install] " + (msg % args if args else msg), flush=True)

    is_hybrid = getattr(model_cfg, "is_hybrid_model", False)
    num_experts = getattr(model_cfg, "num_moe_experts", None)
    pattern: str = getattr(model_cfg, "hybrid_layer_pattern", "") or ""
    full_num_layers: int | None = getattr(model_cfg, "num_layers", None)
    block_configs = getattr(model_cfg, "block_configs", None)

    # For Puzzletron heterogeneous students, ``hybrid_layer_pattern`` is empty
    # because layer types are encoded per-layer in ``block_configs`` instead.
    # We therefore derive ``moe_layer_freq`` from ``block_configs`` when
    # available, and otherwise fall back to None (only affects the per-layer
    # logging denominator, not the all_reduce size).
    moe_layer_freq: list[int] | None = None
    if pattern:
        moe_layer_freq = [1 if c == "E" else 0 for c in pattern]
    elif block_configs:
        derived: list[int] = []
        for bc in block_configs:
            ffn = getattr(bc, "ffn", None) if not isinstance(bc, dict) else bc.get("ffn")
            if ffn is None:
                derived.append(0)
                continue
            ffn_no_op = (
                getattr(ffn, "no_op", False)
                if not isinstance(ffn, dict)
                else ffn.get("no_op", False)
            )
            ffn_type = (
                getattr(ffn, "block_type", None)
                if not isinstance(ffn, dict)
                else ffn.get("block_type")
            )
            ffn_type_str = str(ffn_type).lower() if ffn_type is not None else ""
            is_moe = (not ffn_no_op) and ("moe" in ffn_type_str or "expert" in ffn_type_str)
            derived.append(1 if is_moe else 0)
        if any(derived):
            moe_layer_freq = derived

    moe_count = sum(moe_layer_freq) if moe_layer_freq else 0
    _info(
        "called: model_cls=%s is_hybrid_model=%s num_moe_experts=%s "
        "num_layers=%s hybrid_layer_pattern_len=%d count_E=%d "
        "block_configs_len=%s moe_layer_freq_sum=%s",
        type(model_cfg).__name__,
        is_hybrid,
        num_experts,
        full_num_layers,
        len(pattern),
        pattern.count("E"),
        len(block_configs) if block_configs is not None else None,
        moe_count if moe_layer_freq is not None else None,
    )

    if not is_hybrid:
        _info("skip: is_hybrid_model is False")
        return
    if num_experts is None:
        _info("skip: num_moe_experts is None")
        return
    if full_num_layers is None:
        _info("skip: num_layers missing")
        return

    import megatron.core.transformer.moe.moe_utils as _moe_utils

    # Bridge does `from ...moe_utils import track_moe_metrics` (named import),
    # so train_utils binds the original at import time and patching only
    # _moe_utils.track_moe_metrics has NO effect on the binding train_utils
    # actually calls. We must patch BOTH module dicts: the canonical source
    # AND every importer that already grabbed the name.
    targets: list[tuple[Any, str]] = [(_moe_utils, "track_moe_metrics")]
    try:
        import megatron.bridge.training.utils.train_utils as _bridge_train_utils

        if hasattr(_bridge_train_utils, "track_moe_metrics"):
            targets.append((_bridge_train_utils, "track_moe_metrics"))
    except ImportError:
        _info("skip: megatron.bridge.training.utils.train_utils not importable")

    if getattr(_moe_utils.track_moe_metrics, "_hybrid_size_fix_applied", False):
        _info("skip: already applied (idempotent)")
        return

    orig_track_moe_metrics = _moe_utils.track_moe_metrics

    def _describe_moe_layer_freq(value: Any) -> str:
        # Megatron-Core overloads moe_layer_freq as either an int (periodicity),
        # a list[int] (per-layer 0/1 mask), or None. The diagnostic must handle
        # all three or it will explode (e.g. sum(int) -> TypeError).
        if value is None:
            return "None"
        if isinstance(value, (list, tuple)):
            return f"list(len={len(value)}, sum={sum(value)})"
        return f"scalar({value!r})"

    def _patched_track_moe_metrics(*args: Any, **kwargs: Any) -> Any:
        kwargs["num_layers"] = full_num_layers
        kwargs.setdefault("moe_layer_freq", moe_layer_freq)
        if kwargs.get("moe_layer_freq") is None:
            kwargs["moe_layer_freq"] = moe_layer_freq

        # Per-rank diagnostics around the cross-rank all_reduces inside
        # track_moe_metrics → reduce_aux_losses_tracker_across_ranks.  If a
        # rank prints "[MoEAuxFix] enter" but never prints "[MoEAuxFix] exit",
        # that rank is the one stuck in the all_reduce.  Compare the printed
        # tracker buffer sizes across ranks — they MUST all match (= full_num_layers
        # + (mtp_num_layers or 0)) for the PP all_reduce to succeed.
        rank = get_rank_safe()
        iteration = kwargs.get("iteration", "?")
        if logger.isEnabledFor(logging.DEBUG):
            tracker = _moe_utils.get_moe_layer_wise_logging_tracker()
            sizes_before = {k: tuple(v["values"].shape) for k, v in tracker.items()}
            keys_before = sorted(tracker.keys())
            logger.debug(
                "[MoEAuxFix] enter rank=%s iter=%s num_layers_arg=%s "
                "moe_layer_freq=%s tracker_keys=%s tracker_sizes=%s",
                rank,
                iteration,
                kwargs["num_layers"],
                _describe_moe_layer_freq(kwargs["moe_layer_freq"]),
                keys_before,
                sizes_before,
            )
        try:
            ret = orig_track_moe_metrics(*args, **kwargs)
        except Exception as exc:
            logger.error("[MoEAuxFix] EXCEPTION rank=%s iter=%s: %r", rank, iteration, exc)
            raise
        logger.debug("[MoEAuxFix] exit rank=%s iter=%s", rank, iteration)
        return ret

    _patched_track_moe_metrics._hybrid_size_fix_applied = True  # type: ignore[attr-defined]

    patched_locations = []
    for mod, attr in targets:
        try:
            setattr(mod, attr, _patched_track_moe_metrics)
            patched_locations.append(f"{mod.__name__}.{attr}")
        except Exception as exc:
            _info("warn: failed to patch %s.%s: %r", mod.__name__, attr, exc)

    moe_slots = sum(moe_layer_freq) if moe_layer_freq else 0
    _info(
        "INSTALLED: num_layers=%d MoE slots=%s (out of %d) patched=%s",
        full_num_layers,
        moe_slots if moe_layer_freq is not None else "unknown",
        len(pattern),
        patched_locations,
    )
    logger.info(
        "Installed hybrid MoE aux-loss tracker size fix: num_layers=%d, "
        "MoE slots=%s (out of %d), patched=%s",
        full_num_layers,
        moe_slots if moe_layer_freq is not None else "unknown",
        len(pattern),
        patched_locations,
    )
