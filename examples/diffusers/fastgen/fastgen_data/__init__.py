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

# Runtime soft-guard: the vendored modules import UNPATCHED upstream helpers
# (``nemo_automodel.components.datasets.diffusion.{sampler,base_dataset,text_to_video_dataset}``).
# Convert a missing-helper ImportError into an actionable message naming the supported range.
try:
    from .collate_fns import (
        build_text_to_image_multiresolution_dataloader,
        collate_fn_text_to_image,
    )
    from .mock_dataloader import build_mock_t2i_dataloader
    from .text_to_image_dataset import TextToImageDataset
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "fastgen_data could not import its dependencies. It requires a stock "
        "nemo_automodel>=0.4.0,<1.0 install (it imports the unpatched upstream helpers "
        "nemo_automodel.components.datasets.diffusion.{sampler,base_dataset,text_to_video_dataset}). "
        "Install the example dependencies with:\n"
        "    pip install -r examples/diffusers/fastgen/requirements.txt\n"
        f"Underlying import error: {exc!r}"
    ) from exc

__all__ = [
    "TextToImageDataset",
    "build_mock_t2i_dataloader",
    "build_text_to_image_multiresolution_dataloader",
    "collate_fn_text_to_image",
]


def _warn_if_unsupported_upstream() -> None:
    """Soft-warn (never raise) if the installed ``nemo_automodel`` is outside the tested range.

    The vendored data/preprocessing code imports unpatched upstream helpers (``sampler``,
    ``base_dataset``, ``multi_tier_bucketing``); an out-of-range version may have moved them.
    This complements the hard import guard above with a clear, non-fatal version signal.
    """
    import logging

    try:
        import nemo_automodel

        raw = str(getattr(nemo_automodel, "__version__", "") or "")
        nums = []
        for tok in raw.split(".")[:3]:
            digits = "".join(ch for ch in tok if ch.isdigit())
            nums.append(int(digits) if digits else 0)
        while len(nums) < 3:
            nums.append(0)
        version = tuple(nums[:3])
        if not ((0, 4, 0) <= version < (1, 0, 0)):
            logging.getLogger(__name__).warning(
                "fastgen_data: installed nemo_automodel %s is outside the tested range "
                "(>=0.4.0,<1.0). The vendored data/preprocessing code imports unpatched upstream "
                "helpers (sampler, base_dataset, multi_tier_bucketing); if imports fail or behavior "
                "drifts, pin nemo_automodel to the supported range.",
                raw or "<unknown>",
            )
    except Exception:  # pragma: no cover - never block import on a version probe
        pass


_warn_if_unsupported_upstream()
