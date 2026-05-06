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

"""Optional dependency availability flags for model backends.

Usage in model files::

    from .._deps import WAN_AVAILABLE

    if WAN_AVAILABLE:
        from wan.modules.model import WanModel

        ...
"""

import importlib.machinery
import importlib.util
import sys
import types


def _available(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


class _LazyStubModule(types.ModuleType):
    """Module stub that returns a dummy for any public attribute access.

    Allows ``from missing_pkg import SomeClass`` to succeed at import time.
    Any use of the imported names at *call* time will fail with a clear error.
    Sets ``__path__`` so it can act as a package for sub-module imports.
    Dunder attributes fall through to the normal ModuleType behavior.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__: list[str] = []
        self.__file__ = f"<stub:{name}>"
        self.__loader__ = None
        self.__spec__ = importlib.machinery.ModuleSpec(name, None, origin=f"<stub:{name}>")

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _unavailable(*args, **kwargs):
            raise ImportError(
                f"Attempted to use '{name}' from stub module '{self.__name__}'. "
                f"Install '{self.__name__}' to use this functionality."
            )

        _unavailable.__name__ = name
        _unavailable.__qualname__ = f"{self.__name__}.{name}"
        return _unavailable


def _stub_missing_wan_deps() -> None:
    """Inject stubs for optional wan sub-dependencies (decord, etc.).

    The wan package's __init__.py unconditionally imports speech2video,
    animate, etc. which pull in decord, sam2, and other heavy packages
    that are only needed for specific tasks. Stubbing them lets us import
    the parts of wan we actually use (configs, modules, textimage2video)
    without installing the full s2v/animate dependency stack.
    """
    _stubs = (
        "decord",
        "sam2",
        "sam2.build_sam",
        "sam2.modeling",
        "sam2.modeling.sam2_base",
        "sam2.sam2_video_predictor",
        "sam2.utils",
        "sam2.utils.misc",
        "sam_utils",
    )
    for mod_name in _stubs:
        if mod_name not in sys.modules and not _available(mod_name):
            sys.modules[mod_name] = _LazyStubModule(mod_name)


WAN_AVAILABLE = _available("wan")
if WAN_AVAILABLE:
    _stub_missing_wan_deps()

LTX_CORE_AVAILABLE = _available("ltx_core")
LTX_TRAINER_AVAILABLE = _available("ltx_trainer")
