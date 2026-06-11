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

"""Self-contained DMD2 dataloaders for the fastgen example.

These modules are vendored from NeMo-AutoModel (``nemo_automodel`` @ e42584e3, Apache-2.0)
so the published example's data path does not depend on local *modifications* to AutoModel.
The unpatched upstream helpers (``sampler``, ``base_dataset``, ``text_to_video_dataset``)
are still imported from the installed stock ``nemo_automodel`` package; only the patched
files (collate / dataset reader / mock loader) are carried here.

The training configs reference these builders via ``_target_: fastgen_data.build_*`` once
``dmd2_finetune.py`` has put this directory on ``sys.path`` (source-checkout flow).
"""

from .collate_fns import (
    build_text_to_image_multiresolution_dataloader,
    collate_fn_text_to_image,
)
from .mock_dataloader import build_mock_t2i_dataloader
from .text_to_image_dataset import TextToImageDataset

__all__ = [
    "TextToImageDataset",
    "build_mock_t2i_dataloader",
    "build_text_to_image_multiresolution_dataloader",
    "collate_fn_text_to_image",
]
