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

"""Dependency-light access to the canonical artifact-import contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__all__ = ["IMPORT_CAMPAIGN_MANIFEST", "imported_stage_manifest_is_complete"]

_MODULE_NAME = "_puzzletron_artifact_import_contract"
_MODULE_PATH = Path(__file__).resolve().parents[1] / "artifact_import_contract.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load Puzzletron artifact import contract from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, _MODULE)
_SPEC.loader.exec_module(_MODULE)

IMPORT_CAMPAIGN_MANIFEST = _MODULE.IMPORT_CAMPAIGN_MANIFEST
imported_stage_manifest_is_complete = _MODULE.imported_stage_manifest_is_complete
