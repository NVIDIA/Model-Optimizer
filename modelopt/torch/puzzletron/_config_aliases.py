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

"""Shared compatibility checks for Puzzletron experiment configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

__all__ = []

_COMPATIBILITY_ALIASES = (
    ("puzzle_dir", "experiment.dir", "puzzle_dir", True),
    ("input_hf_model_path", "model.source", "input_hf_model_path", False),
    ("teacher_dir", "convert.teacher_dir", "teacher_dir", True),
    ("dataset_path", "data.path", "dataset_path", True),
    ("trust_remote_code", "model.trust_remote_code", "model.trust_remote_code", False),
)


def _lookup(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def _compatible_values(left: Any, right: Any, *, path_like: bool) -> bool:
    if path_like:
        return (
            Path(os.path.normpath(str(left))).expanduser()
            == Path(os.path.normpath(str(right))).expanduser()
        )
    return left == right


def _validate_compatibility_aliases(config: Mapping[str, Any]) -> None:
    for legacy_path, canonical_path, preferred_override, path_like in _COMPATIBILITY_ALIASES:
        try:
            legacy_value = _lookup(config, legacy_path)
            canonical_value = _lookup(config, canonical_path)
        except KeyError:
            continue
        if not _compatible_values(legacy_value, canonical_value, path_like=path_like):
            raise ValueError(
                f"Experiment aliases {legacy_path!r} and {canonical_path!r} disagree; "
                f"keep them identical or override {preferred_override!r} so composed "
                "references stay synchronized"
            )
