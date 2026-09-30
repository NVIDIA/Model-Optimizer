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

"""CLI launcher for the vendored Qwen-Image preprocessing.

Builds the VAE + text-embed ``.pt`` cache that the DMD2 training dataloader reads, using only
stock ``nemo_automodel`` (no dependency on the un-packaged AutoModel ``tools/`` tree).

Mirrors ``dmd2_finetune.py``: it puts this example directory on ``sys.path`` so the
``preprocess`` package imports cleanly from a source checkout, then dispatches to the vendored
driver's ``main`` (argparse with ``image`` / ``video`` subcommands; this example uses ``image``
with ``--processor qwen_image``).

Example::

    python examples/diffusers/fastgen/preprocess_qwen_image.py image \\
        --image_dir <raw images> --output_dir <cache dir> --processor qwen_image \\
        --caption_format meta_json
"""

from __future__ import annotations

import os
import sys

# Make the ``preprocess`` package importable as a top-level package regardless of the current
# working directory (same seam as dmd2_finetune.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from preprocess.preprocessing_multiprocess import main  # noqa: E402

if __name__ == "__main__":
    main()
