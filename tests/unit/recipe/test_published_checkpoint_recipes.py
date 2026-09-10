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

"""Check the shipped recipes against NVIDIA's published quantized checkpoints.

``tools/recipe_backfill/published_checkpoints.json`` records, for every model in the
`Inference Optimized Checkpoints
<https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer>`_
collection, which format each of its modules ships in. ``recipe_map.json`` names the
recipe that reproduces each one. These tests replay the recipe's ``quant_cfg`` over the
recorded module names and assert the result matches, so a recipe edit that stops
reproducing a released checkpoint fails here rather than in a re-quantization run.

Everything runs offline against the checked-in snapshot -- no Hub access and no weights.
Regenerate the snapshot with ``tools/recipe_backfill/scan_collection.py``.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKFILL_DIR = REPO_ROOT / "tools" / "recipe_backfill"

pytestmark = pytest.mark.skipif(
    not (BACKFILL_DIR / "published_checkpoints.json").is_file(),
    reason="tools/recipe_backfill is not available in this checkout (e.g. an installed wheel)",
)

sys.path.insert(0, str(BACKFILL_DIR))


def _load(name: str) -> dict:
    return json.loads((BACKFILL_DIR / name).read_text(encoding="utf-8"))


def _snapshot_entries() -> dict[str, dict]:
    return {e["id"]: e for e in _load("published_checkpoints.json")["checkpoints"]}


def _recipe_map() -> dict:
    return _load("recipe_map.json")


def _spec(value) -> dict:
    return {"recipe": value} if isinstance(value, str) else value


def _mapped_checkpoints() -> list[tuple[str, dict]]:
    return [(cid, _spec(v)) for cid, v in sorted(_recipe_map()["recipes"].items())]


@pytest.mark.parametrize(("checkpoint_id", "spec"), _mapped_checkpoints())
def test_recipe_reproduces_published_checkpoint(checkpoint_id: str, spec: dict):
    """The mapped recipe must assign every module the format the checkpoint ships."""
    from verify_recipes import verify

    entry = _snapshot_entries().get(checkpoint_id)
    assert entry is not None, (
        f"{checkpoint_id} is in recipe_map.json but not in published_checkpoints.json; "
        "re-run tools/recipe_backfill/scan_collection.py."
    )
    if "modules" not in entry:
        pytest.skip(f"{checkpoint_id}: {entry.get('skipped', 'no module map')}")

    ok, detail = verify(entry, spec)
    if not ok and spec.get("approximate"):
        pytest.xfail(f"{checkpoint_id} is a known approximate mirror: {spec['approximate']}")
    assert ok, (
        f"{spec['recipe']} no longer reproduces {checkpoint_id}:\n{detail}\n"
        "Either the recipe changed, or the mapping in tools/recipe_backfill/recipe_map.json "
        "needs updating."
    )


def test_every_published_checkpoint_is_accounted_for():
    """Every scanned checkpoint is either mapped to a recipe or explicitly unmapped."""
    mapping = _recipe_map()
    known = set(mapping["recipes"]) | set(mapping["unmapped"])
    missing = sorted(cid for cid in _snapshot_entries() if cid not in known)
    assert not missing, (
        "Published checkpoints with no entry in tools/recipe_backfill/recipe_map.json: "
        f"{missing}. Add the recipe that reproduces each one, or list it under "
        "'unmapped' with the reason it has none."
    )


def test_mapped_recipes_resolve():
    """Every recipe path in the map must load."""
    from modelopt.recipe import load_recipe

    def loads(recipe: str) -> str | None:
        try:
            load_recipe(recipe)
        except ValueError as exc:  # pragma: no cover - only on a broken mapping
            return str(exc)
        return None

    broken = {
        checkpoint_id: error
        for checkpoint_id, spec in _mapped_checkpoints()
        if (error := loads(spec["recipe"]))
    }
    assert not broken, f"recipe_map.json points at recipes that do not load: {broken}"


def test_unmapped_checkpoints_have_a_reason():
    """An unmapped checkpoint must say why, so the gap is a decision and not an oversight."""
    blank = sorted(cid for cid, reason in _recipe_map()["unmapped"].items() if not reason.strip())
    assert not blank, f"'unmapped' entries with no reason: {blank}"


def test_published_checkpoint_index_is_up_to_date():
    """``modelopt_recipes/published_checkpoints.md`` is generated; keep it in sync."""
    from render_index import OUTPUT, render

    assert OUTPUT.is_file(), f"{OUTPUT} is missing; run tools/recipe_backfill/render_index.py"
    assert OUTPUT.read_text(encoding="utf-8") == render(), (
        f"{OUTPUT.name} is out of date with tools/recipe_backfill/recipe_map.json. "
        "Re-run `python tools/recipe_backfill/render_index.py`."
    )


def test_approximate_mirrors_are_still_approximate():
    """A recipe marked ``approximate`` that now matches exactly should lose the marker."""
    from verify_recipes import verify

    entries = _snapshot_entries()
    fixed = []
    for checkpoint_id, spec in _mapped_checkpoints():
        if not spec.get("approximate"):
            continue
        entry = entries.get(checkpoint_id)
        if entry is None or "modules" not in entry:
            continue
        if verify(entry, spec)[0]:
            fixed.append(checkpoint_id)
    assert not fixed, (
        f"These recipes now reproduce their checkpoint exactly: {fixed}. Drop the "
        "'approximate' note from tools/recipe_backfill/recipe_map.json."
    )
