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

"""CI check: every models/<org>/<model_id>/ptq/ entry names a real Hugging Face Hub repo.

Not a pre-commit hook -- this makes one network call per entry, and pre-commit hooks
(tools/precommit/) have to stay usable offline. Wired into CI instead; see
.github/workflows/check_model_hub_orgs.yml, which only runs it when
modelopt_recipes/models/** changes.

A recipe's own text can't catch a wrong org -- it's typically self-consistent with
the (wrong) directory it lives in, since both were written by the same mistake. Hub
existence is the one fact that isn't derivable from the recipe itself.

Scoped to the ptq/ task, matching tests/unit/recipe/test_recipe_docs.py's
test_every_model_specific_ptq_dir_is_mentioned: a models/<org>/<model_id>/ entry with
only e.g. an auto_quantize/ recipe (Muse-Glimmer-30B) is exempt from the ptq.md
membership check for the same reason it should be exempt here -- models/README.md
allows a checkpoint mirror for a *planned*, not-yet-released checkpoint, and ptq/ is
where the backfilled, already-published aliases this check targets actually live.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

MODELS_ROOT = Path(__file__).resolve().parents[2] / "modelopt_recipes" / "models"


def _checkpoint_ids() -> list[str]:
    """<org>/<model_id> for every models/<org>/<model_id>/ptq/ entry, deduplicated."""
    ids = {
        f"{p.parent.parent.parent.name}/{p.parent.parent.name}"
        for p in MODELS_ROOT.glob("*/*/ptq/*.yaml")
    }
    return sorted(ids)


def _error_if_missing(api: HfApi, repo_id: str) -> str | None:
    try:
        api.model_info(repo_id)
    except RepositoryNotFoundError:
        return (
            f"{repo_id}: no such Hugging Face Hub repo. modelopt_recipes/models/<org>/"
            "<model_id>/ must be keyed by the SOURCE checkpoint's hub path (what you pass "
            "to from_pretrained(...)), not a quantized derivative published elsewhere -- "
            "see modelopt_recipes/models/README.md."
        )
    return None


def main() -> int:
    """Verify every checkpoint id against the Hub, exit 1 if any is missing."""
    checkpoint_ids = _checkpoint_ids()
    api = HfApi()
    errors = [e for e in (_error_if_missing(api, r) for r in checkpoint_ids) if e is not None]

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Verified {len(checkpoint_ids)} models/<org>/<model_id>/ptq/ hub paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
