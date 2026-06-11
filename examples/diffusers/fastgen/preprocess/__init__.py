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

"""Self-contained Qwen-Image preprocessing for the fastgen example.

Vendored from NeMo-AutoModel's ``tools/diffusion`` (Apache-2.0, @ e42584e3) so an external
user can build the VAE + text-embed cache from raw images using only stock ``nemo_automodel``
— the un-packaged AutoModel ``tools/`` tree is not required. Trimmed to the Qwen-Image
processor; ``MultiTierBucketCalculator`` is imported from the stock ``nemo_automodel`` package.

Produces ``.pt`` cache items byte-compatible with what the training dataloader
(``fastgen_data``) reads. Run via the ``preprocess_qwen_image.py`` launcher one directory up.
"""

# Runtime soft-guard: the vendored driver imports the UNPATCHED upstream bucketing helper
# ``nemo_automodel.components.datasets.diffusion.multi_tier_bucketing.MultiTierBucketCalculator``.
# Surface a missing/moved helper as an actionable message (named version range) rather than a
# raw ImportError deep inside the driver.
try:
    from nemo_automodel.components.datasets.diffusion.multi_tier_bucketing import (
        MultiTierBucketCalculator,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "fastgen preprocessing requires a stock nemo_automodel>=0.4.0,<1.0 install providing "
        "nemo_automodel.components.datasets.diffusion.multi_tier_bucketing. Install the example "
        "dependencies with:\n"
        "    pip install -r examples/diffusers/fastgen/requirements.txt\n"
        f"Underlying import error: {exc!r}"
    ) from exc
