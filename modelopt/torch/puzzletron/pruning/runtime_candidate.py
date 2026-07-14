# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared reversible application of one complete typed block candidate."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

__all__ = ["RemovableCandidateHandle", "apply_runtime_candidate"]


class RemovableCandidateHandle:
    """Own hooks and entered contexts for one live candidate application."""

    def __init__(self, handles: list[Any], contexts: list[AbstractContextManager]):
        self._handles = list(handles)
        self._contexts: list[AbstractContextManager] = []
        self._removed = False
        try:
            for context in contexts:
                context.__enter__()
                self._contexts.append(context)
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        if self._removed:
            return
        self._removed = True
        error: BaseException | None = None
        for context in reversed(self._contexts):
            try:
                context.__exit__(None, None, None)
            except BaseException as exc:  # preserve cleanup of later resources
                error = error or exc
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as exc:
                error = error or exc
        self._contexts.clear()
        self._handles.clear()
        if error is not None:
            raise error


def apply_runtime_candidate(
    layer,
    parent_block_config,
    child_block_config,
    *,
    expert_keep_ids=None,
) -> RemovableCandidateHandle:
    """Apply Mamba/MoE/no-op typed semantics without resizing parameters.

    The low-level hook construction remains colocated with the AutoModel
    solution recipe for now; this API is the single lifecycle boundary used by
    both replace-one-block scoring and elastic bypass.
    """

    from ..plugins.automodel.solution_recipe import ReplaceBlockScoringRecipe

    handles, contexts = ReplaceBlockScoringRecipe._typed_subblock_runtime_hooks(
        layer,
        teacher_block_config=parent_block_config,
        child_block_config=child_block_config,
        expert_keep_ids=expert_keep_ids,
    )
    return RemovableCandidateHandle(handles, contexts)
