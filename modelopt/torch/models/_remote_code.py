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

"""Loading the modeling code a checkpoint ships with itself."""

import contextlib
import filecmp
import glob
import importlib
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable
from types import ModuleType


def stage_full_remote_closure() -> None:
    """Make transformers stage every top-level module of a local checkpoint's remote code.

    transformers' ``get_cached_module_file`` copies a remote module and its *direct* relative imports
    into the module's stage dir under ``transformers_modules``, but does not recurse. Remote code more
    than one import deep -- Kimi-K2.5's ``modeling_kimi_k25 -> modeling_deepseek ->
    configuration_deepseek`` -- is then staged incomplete, and the later recursive walk in
    ``get_relative_import_files`` fails with ``FileNotFoundError``, from
    ``get_class_from_dynamic_module`` and ``from_pretrained`` alike.

    Wrap ``get_cached_module_file`` so that, loading from a local checkpoint dir, it copies every
    sibling ``*.py`` into the stage dir it just populated, as transformers copies its own: again
    when the staged copy differs from the checkpoint's, then invalidating the import system's
    caches. Modules in a subdirectory of the checkpoint are not copied. A copy that fails only
    warns, since only a deeper import needs the file, and the other modules are still staged.
    Idempotent.
    """
    import transformers.dynamic_module_utils as dmu
    from transformers.utils import HF_MODULES_CACHE

    if getattr(dmu.get_cached_module_file, "_modelopt_full_closure", False):
        return
    orig = dmu.get_cached_module_file

    def _staged(pretrained_model_name_or_path, module_file, *args, **kwargs):
        resolved = orig(pretrained_model_name_or_path, module_file, *args, **kwargs)
        src = str(pretrained_model_name_or_path)
        if os.path.isdir(src):
            staged = str(resolved)
            if not os.path.isabs(staged):
                staged = os.path.join(HF_MODULES_CACHE, staged)
            stage_dir = os.path.dirname(staged)
            copied = False
            for py in sorted(glob.glob(os.path.join(src, "*.py"))):
                dst = os.path.join(stage_dir, os.path.basename(py))
                try:
                    if os.path.exists(dst) and filecmp.cmp(py, dst, shallow=False):
                        continue
                    _copy_into_place(py, dst)
                except OSError as e:  # a failed copy only matters if a deeper import needs the file
                    warnings.warn(
                        f"Could not stage {os.path.basename(py)} of the remote code of {src}: {e}"
                    )
                    continue
                copied = True
            if copied:
                # The import system caches what a directory holds: the files just staged must not
                # stay hidden from the imports that need them.
                importlib.invalidate_caches()
        return resolved

    setattr(_staged, "_modelopt_full_closure", True)
    dmu.get_cached_module_file = _staged


def _copy_into_place(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` through a temp file renamed into place, removed if the copy fails.

    Every rank loading the checkpoint stages into the same dir and imports from it: renamed into
    place (atomic), a file appears there whole, never half written.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst))
    os.close(fd)
    try:
        shutil.copy(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# Patches keep_patched applies to remote modules, by module-name suffix.
_KEPT_PATCHES: dict[str, Callable[[ModuleType], bool]] = {}


def _is_remote(name: str, suffix: str) -> bool:
    """Whether the module ``name`` is remote code named ``*<suffix>``.

    transformers' own port of a model can share the remote module's name, as
    transformers.models.kimi_k25.modeling_kimi_k25 does from 5.13.
    """
    import transformers.dynamic_module_utils as dmu

    return name.startswith(f"{dmu.TRANSFORMERS_DYNAMIC_MODULE_NAME}.") and name.endswith(suffix)


def _apply(suffix: str, patch: Callable[[ModuleType], bool]) -> dict[str, bool]:
    """Apply ``patch`` to every loaded remote module named ``*<suffix>``: whether it changed each."""
    return {
        name: patch(module)
        for name, module in list(sys.modules.items())
        if module is not None and _is_remote(name, suffix)
    }


def keep_patched(suffix: str, patch: Callable[[ModuleType], bool]) -> list[str]:
    """Apply ``patch`` to every loaded remote module named ``*<suffix>``, and again whenever
    transformers resolves a class from one.

    transformers re-executes a remote module in place when the files it hashes change, and loads it
    afresh under another name for another path to the same checkpoint; either way the module's
    globals are rebound and a patch applied once is gone. Wrapping ``get_class_in_module`` re-applies
    every kept patch to every loaded remote module it is for, each time transformers resolves a
    class, whichever module defines that class, and hands back the class as the patched module binds
    it, so ``from_pretrained`` builds from the patched module. ``patch`` returns whether it changed
    the module; a re-application that did is reported on the rank it happened on. Returns the names
    of the modules patched now. Installs nothing in the loader unless it patched a module: a patch
    that raises, or no module to patch, leaves transformers' loader as it was, so a later load in the
    process is not refused by it. A patch that raises on one of several matching modules leaves those
    it already patched patched. Idempotent per suffix.
    """
    import transformers.dynamic_module_utils as dmu

    patched = list(_apply(suffix, patch))
    if not patched:
        return patched
    _KEPT_PATCHES[suffix] = patch
    if not getattr(dmu.get_class_in_module, "_modelopt_keep_patched", False):
        orig = dmu.get_class_in_module

        def _resolved(class_name, module_path, *args, **kwargs):
            cls = orig(class_name, module_path, *args, **kwargs)
            # Resolving may have executed a remote module again, and not necessarily the one
            # defining the class (that can be imported, from a sibling or transformers' own port).
            repatched = [
                name
                for kept_suffix, kept_patch in _KEPT_PATCHES.items()
                for name, changed in _apply(kept_suffix, kept_patch).items()
                if changed
            ]
            if not repatched:
                return cls
            from modelopt.torch.utils import distributed as dist

            for name in repatched:
                print(
                    f"[rank {dist.rank()}] re-patched {name}, which transformers loaded again after "
                    "it was patched",
                    flush=True,
                )
            # transformers read the class before the patch ran, which may have rebound it: read it
            # again, from the module as patched (loaded already, so not executed again).
            return orig(class_name, module_path)

        setattr(_resolved, "_modelopt_keep_patched", True)
        dmu.get_class_in_module = _resolved
    return patched
