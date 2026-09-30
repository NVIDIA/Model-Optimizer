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

"""Handles suppressing import errors for third-party modules that may or may not be available."""

import sys
from contextlib import contextmanager

from .logging import warn_rank_0


@contextmanager
def import_plugin(plugin_name, msg_if_missing=None, verbose=True, success_msg=None):
    """Context manager to import a plugin and suppress ModuleNotFoundError."""
    try:
        yield
        if verbose and success_msg is not None:
            warn_rank_0(success_msg)
    except ModuleNotFoundError:
        if verbose and msg_if_missing is not None:
            warn_rank_0(msg_if_missing)
    except Exception as e:
        if verbose:
            # Capture the ``with import_plugin(...)`` call site so warnings point at the plugin
            # that actually failed rather than at this helper. When ``__enter__`` runs the
            # generator body, the stack is [0]=here, [1]=contextlib.__enter__, [2]=the caller.
            caller = sys._getframe(2)
            warn_rank_0(
                f"Failed to import modelopt {plugin_name} plugin "
                f"(from {caller.f_code.co_filename}:{caller.f_lineno}) due to: {e!r}. "
                "You may ignore this warning if you do not need this plugin."
            )
