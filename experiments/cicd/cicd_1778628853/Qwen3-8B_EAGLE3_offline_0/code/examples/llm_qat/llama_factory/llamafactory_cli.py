# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from wrapt import register_post_import_hook


def patch_launcher_module(module):
    """Patch the launcher module to override its __file__ attribute."""
    # Override the __file__ attribute to point to our custom file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module.__file__ = f"{current_dir}/llama_factory.py"


if __name__ == "__main__":
    # Register the post-import hook for the launcher module
    register_post_import_hook(patch_launcher_module, "llamafactory.launcher")

    from llamafactory.cli import main

    main()
