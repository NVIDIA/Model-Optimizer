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

"""Utility functions for loading CPP / CUDA extensions."""

import os
import warnings
from pathlib import Path
from time import time
from types import ModuleType
from typing import Any

import torch
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from torch.utils.cpp_extension import load

__all__ = ["load_cpp_extension"]


def load_cpp_extension(
    name: str,
    sources: list[str | Path],
    cuda_version_specifiers: str | None,
    fail_msg: str = "",
    raise_if_failed: bool = False,
    **load_kwargs: Any,
) -> ModuleType | None:
    """Load a C++ / CUDA extension using torch.utils.cpp_extension.load() if the current CUDA version satisfies it.

    Loading first time may take a few mins because of the compilation, but subsequent loads are instantaneous.

    Args:
        name: Name of the extension.
        sources: Source files to compile.
        cuda_version_specifiers: Specifier (e.g. ">=11.8,<12") for CUDA versions required to enable the extension.
        fail_msg: Additional message to display if the extension fails to load.
        raise_if_failed: Raise an exception if the extension fails to load.
        **load_kwargs: Keyword arguments to torch.utils.cpp_extension.load().
    """
    ext = None
    print(f"Loading extension {name}...")
    start = time()

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        try:
            device_capability = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{device_capability[0]}.{device_capability[1]}"
        except Exception:
            warnings.warn("GPU not detected. Please unset `TORCH_CUDA_ARCH_LIST` env variable.")

    if torch.version.cuda is None:
        fail_msg = f"Skipping extension {name} because CUDA is not available."
    elif cuda_version_specifiers and Version(torch.version.cuda) not in SpecifierSet(
        cuda_version_specifiers
    ):
        fail_msg = (
            f"Skipping extension {name} because the current CUDA version {torch.version.cuda}"
            f" does not satisfy the specifiers {cuda_version_specifiers}."
        )
    else:
        try:
            ext = load(name, sources, **load_kwargs)
        except Exception as e:
            if not fail_msg:
                fail_msg = f"Unable to load extension {name} and falling back to CPU version."
            fail_msg = f"{e}\n{fail_msg}"
            # RuntimeError can be raised if there are any errors while compiling the extension.
            # OSError can be raised if CUDA_HOME path is not set correctly.
            # subprocess.CalledProcessError can be raised on `-runtime` images where c++ is not installed.

    if ext is None:
        if raise_if_failed:
            raise RuntimeError(fail_msg)
        else:
            warnings.warn(fail_msg)
    else:
        print(f"Loaded extension {name} in {time() - start:.1f} seconds")
    return ext
