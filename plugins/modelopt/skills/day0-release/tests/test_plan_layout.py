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

"""Unit tests for plan_layout.py on synthetic profiles with known unit costs (stdlib only)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from plan_layout import fit_costs, load_profile, main, plan

# Ground truth: 4 stages x 16 layers, 8 micro-batches per pipeline (Qwen3.8-27B-like at 16K).
RECOMPUTED_MS, STORED_MS, LAST_EXTRA_MS = 80.0, 60.0, 450.0
STORED_LAYER_GIB, STATIC_LAYER_GIB, DYNAMIC_GIB = 5.0, 4.0, 10.0
PP, LAYERS, MICROBATCHES = 4, 16, 8


def _write_profile(directory, recomputed):
    directory.mkdir()
    for k in range(PP):
        stored = LAYERS - recomputed[k]
        record = {
            "pp_rank": k,
            "microbatches": MICROBATCHES,
            "fwd_ms": 0.0,
            "bwd_ms": MICROBATCHES
            * (
                RECOMPUTED_MS * recomputed[k]
                + STORED_MS * stored
                + (LAST_EXTRA_MS if k == PP - 1 else 0)
            ),
            "layers": LAYERS,
            "recomputed_layers": recomputed[k],
            "static_gib": STATIC_LAYER_GIB * LAYERS,
            "peak_alloc_gib": STATIC_LAYER_GIB * LAYERS
            + DYNAMIC_GIB
            + (PP - k) * STORED_LAYER_GIB * stored,
            "device_gib": 280.0,
        }
        # The first line is the warm-up iteration, which load_profile skips.
        warmup = dict(record, bwd_ms=1e9)
        (directory / f"rank{k}.jsonl").write_text(f"{json.dumps(warmup)}\n{json.dumps(record)}\n")
    return directory


@pytest.fixture
def profiles(tmp_path):
    full = _write_profile(tmp_path / "full", [LAYERS] * PP)
    partial = _write_profile(tmp_path / "partial", [8] * PP)
    return load_profile(full), load_profile(partial), full, partial


def test_fit_recovers_unit_costs(profiles):
    full, partial, *_ = profiles
    costs = fit_costs(full, partial)
    assert costs["recomputed_ms"] == pytest.approx(RECOMPUTED_MS)
    assert costs["stored_ms"] == pytest.approx(STORED_MS)
    assert costs["stored_layer_gib"] == pytest.approx(STORED_LAYER_GIB)
    assert costs["extra_ms"][-1] == pytest.approx(LAST_EXTRA_MS)


def test_plan_unloads_last_stage_and_respects_memory_cap(profiles):
    full, partial, *_ = profiles
    cap = 200.0
    layout = plan(full, fit_costs(full, partial), cap)
    layers = [s["layers"] for s in layout["stages"]]
    recompute = [s["recompute"] for s in layout["stages"]]
    assert sum(layers) == PP * LAYERS
    assert layers[-1] < LAYERS  # the last stage runs the output layers and KD loss
    assert recompute == sorted(recompute, reverse=True)  # earlier stages hold more micro-batches
    assert all(s["peak_gib"] <= cap for s in layout["stages"])
    assert max(s["ms_per_mb"] for s in layout["stages"]) < LAYERS * RECOMPUTED_MS + LAST_EXTRA_MS


def test_plan_returns_none_when_nothing_fits(profiles):
    full, partial, *_ = profiles
    assert plan(full, fit_costs(full, partial), cap=10.0) is None


def test_mismatched_profiles_are_rejected(profiles, tmp_path, capsys):
    *_, full_dir, _ = profiles
    assert main(["--full", str(full_dir), "--partial", str(full_dir)]) == 2
    assert "fewer layers" in capsys.readouterr().out
