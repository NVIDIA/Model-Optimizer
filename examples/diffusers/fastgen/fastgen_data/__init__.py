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

"""Self-contained shared dataloaders for the FastGen diffusion examples.

The data path builds on stock ``nemo_automodel==0.5.0`` where it is model-agnostic and implements
the example-owned batch contract locally, so the published example does not depend on AutoModel
source modifications:

* ``collate_fns.py`` — the collate functions + dataloader builder. It reuses the upstream
  ``SequentialBucketSampler`` and emits either the ordinary latent-conditioned batch or a
  prompt-only batch for data-free PDD, including the optional broadcast negative-prompt
  embedding. It deliberately does **not** call upstream ``collate_fn_production``, which stacks
  model-specific token keys
  (``clip_tokens`` / ``t5_tokens``) that the Qwen-Image cache does not produce.
* ``text_to_image_dataset.py`` — a faithful vendored copy of the upstream dataset reader (built
  on the upstream ``BaseMultiresolutionDataset``); its change emits ``prompt_embeds_mask``
  interleaved with cache loading, so it is carried verbatim rather than wrapped.

The training configs reference these via ``_target_: fastgen_data.build_*`` once
``dmd2/finetune.py`` has put the FastGen directory on ``sys.path`` (source-checkout flow).
"""

import re

# Runtime soft-guard: the data path imports unmodified upstream helpers
# (``nemo_automodel.components.datasets.diffusion.{sampler,base_dataset}``).
# Convert a missing-helper ImportError into an actionable message naming the supported release.
try:
    from . import collate_fns as _collate_fns
    from . import paths as _paths
    from . import resume as _resume
    from . import splits as _splits
    from . import text_to_image_dataset as _text_to_image_dataset
    from .collate_fns import *
    from .paths import *
    from .resume import *
    from .splits import *
    from .text_to_image_dataset import *
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "fastgen_data could not import its dependencies. It requires a stock "
        "nemo_automodel==0.5.0 install (it imports the unmodified upstream helpers "
        "nemo_automodel.components.datasets.diffusion.{sampler,base_dataset}). "
        "Install the example dependencies with:\n"
        "    pip install -r examples/diffusers/fastgen/requirements.txt\n"
        f"Underlying import error: {exc!r}"
    ) from exc

__all__: list[str] = []
for _module in (
    _collate_fns,
    _paths,
    _resume,
    _splits,
    _text_to_image_dataset,
):
    __all__.extend(_module.__all__)


def _warn_if_unsupported_upstream() -> None:
    """Soft-warn (never raise) if the installed ``nemo_automodel`` is outside the tested range.

    The vendored data/preprocessing code imports unpatched upstream helpers (``sampler``,
    ``base_dataset``, ``multi_tier_bucketing``); an out-of-range version may have moved them.
    This complements the hard import guard above with a clear, non-fatal signal.
    """
    import logging

    try:
        import nemo_automodel

        raw = str(getattr(nemo_automodel, "__version__", "") or "")
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw)
        version = tuple(int(part) for part in match.groups()) if match else ()
        if version != (0, 5, 0):
            logging.getLogger(__name__).warning(
                "fastgen_data: installed nemo_automodel %s does not match the tested release "
                "(==0.5.0). The vendored data/preprocessing code imports unmodified upstream "
                "helpers (sampler, base_dataset, multi_tier_bucketing); if imports "
                "fail or behavior drifts, pin nemo_automodel to the supported release.",
                raw or "<unknown>",
            )
    except Exception:  # pragma: no cover - never block import on a version probe
        pass


_warn_if_unsupported_upstream()
