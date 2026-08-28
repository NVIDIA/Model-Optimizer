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

"""Compatibility loaded only by Puzzletron-owned vLLM server subprocesses."""

import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path


def _vllm_package_has_extension(package_spec, module_name: str) -> bool:
    """Return whether the installed vLLM package contains a native extension."""
    package_paths = package_spec.submodule_search_locations or ()
    return any(
        (Path(package_path) / f"{module_name}{suffix}").is_file()
        for package_path in package_paths
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _install_stable_libtorch_c_stub() -> None:
    """Satisfy vllm._C imports when stable-libtorch owns operator registration."""
    if "vllm._C" in sys.modules:
        return
    try:
        package_spec = importlib.util.find_spec("vllm")
    except (ImportError, AttributeError, ValueError):
        return
    if (
        package_spec is None
        or _vllm_package_has_extension(package_spec, "_C")
        or not _vllm_package_has_extension(package_spec, "_C_stable_libtorch")
    ):
        return

    # Stable-libtorch vLLM builds register their Torch operators through
    # _C_stable_libtorch, but some runtime modules still import vllm._C for its
    # registration side effect.  The stable extension is loaded independently;
    # this empty module only preserves that companion import contract.
    c_stub = types.ModuleType("vllm._C")
    c_stub.__package__ = "vllm"
    sys.modules["vllm._C"] = c_stub


def _load_inherited_sitecustomize() -> None:
    """Run the next sitecustomize module after this compatibility hook."""
    current_directory = Path(__file__).resolve().parent
    search_path = [
        entry for entry in sys.path if Path(entry or os.curdir).resolve() != current_directory
    ]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", search_path)
    if spec is None or spec.loader is None or spec.origin == __file__:
        return
    current_module = sys.modules.get("sitecustomize")
    inherited = importlib.util.module_from_spec(spec)
    sys.modules["sitecustomize"] = inherited
    try:
        spec.loader.exec_module(inherited)
    finally:
        if current_module is None:
            sys.modules.pop("sitecustomize", None)
        else:
            sys.modules["sitecustomize"] = current_module


try:
    import torch
except ImportError:
    torch = None

if torch is not None and importlib.util.find_spec("torch._opaque_base") is None:
    # Some NVIDIA Torch 2.11 alpha builds predate torch._opaque_base.  vLLM's
    # LayerName optimization is optional, so disable it and provide the base
    # class needed to import vLLM.  Newer Torch builds retain their native API.
    os.environ["VLLM_USE_LAYERNAME"] = "0"
    opaque_base = types.ModuleType("torch._opaque_base")
    setattr(opaque_base, "OpaqueBase", object)
    sys.modules["torch._opaque_base"] = opaque_base

_install_stable_libtorch_c_stub()
_load_inherited_sitecustomize()
