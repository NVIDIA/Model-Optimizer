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

import glob
import os
import shutil
import warnings


def stage_full_remote_closure() -> None:
    """Make transformers stage every module of a local checkpoint's remote code, not just one hop.

    transformers' ``get_cached_module_file`` copies a remote module and its *direct* relative imports
    into ``transformers_modules/<repo>/<hash>/``, but does not recurse. Remote code more than one import
    deep -- Kimi-K2.5's ``modeling_kimi_k25 -> modeling_deepseek -> configuration_deepseek`` -- is
    then staged incomplete, and the later recursive walk in ``get_relative_import_files`` fails with
    ``FileNotFoundError``, from ``get_class_from_dynamic_module`` and ``from_pretrained`` alike.

    Wrap ``get_cached_module_file`` so that, loading from a local checkpoint dir, it copies every
    sibling ``*.py`` into the stage dir it just populated. Idempotent.
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
            try:
                for py in glob.glob(os.path.join(src, "*.py")):
                    dst = os.path.join(stage_dir, os.path.basename(py))
                    if not os.path.exists(dst):
                        shutil.copy(py, dst)
            except OSError as e:  # a failed copy only matters if a deeper import needs the file
                warnings.warn(f"Could not stage the remote code of {src}: {e}")
        return resolved

    setattr(_staged, "_modelopt_full_closure", True)
    dmu.get_cached_module_file = _staged
