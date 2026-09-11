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

"""The vLLM dump resolves aux layers without modelopt -- so it can drift from it.

``compute_hidden_states_vllm.py`` runs in a stock vLLM container where importing
``modelopt.torch`` fails, so it carries its own copy of the preset logic. A copy that
nothing compares against is a copy that silently diverges: the 'eagle' preset -- which
is ``add_aux_layers_args``' *default* -- was missing from it entirely, so the documented
invocation died on ``int('eagle')`` before the dump began.

These tests pin the standalone resolver to the shared implementation it mirrors.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[3] / "examples" / "speculative_decoding"
_COLLECT = _DIR / "collect_hidden_states"
sys.path.insert(0, str(_COLLECT))

_SPEC = importlib.util.spec_from_file_location(
    "chs_vllm", _COLLECT / "compute_hidden_states_vllm.py"
)
assert _SPEC is not None and _SPEC.loader is not None
chs_vllm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chs_vllm)

resolve = chs_vllm._resolve_aux_layers_standalone

# Spans tiny models through 80-layer frontier stacks; includes counts small enough that
# the formula's max(0, ...) clamps collapse ids together.
LAYER_COUNTS = [4, 6, 8, 12, 24, 28, 32, 36, 48, 52, 61, 80]


def _reference_eagle(num_layers):
    """Copy of hf_eagle.default_eagle_aux_layer_ids, which cannot be imported here."""
    return sorted({1, max(0, num_layers // 2 - 1), max(0, num_layers - 4)})


@pytest.mark.parametrize("num_layers", LAYER_COUNTS)
def test_eagle_preset_matches_the_shared_helper(num_layers):
    """The standalone copy must agree with modelopt's own EAGLE layer selection.

    Disagreement is silent: the dump writes plausible-looking hidden states from the
    wrong layers, and only a poor acceptance rate much later reveals it.
    """
    assert resolve("eagle", num_layers) == _reference_eagle(num_layers)


def test_eagle_is_resolvable_because_it_is_the_documented_default():
    """Regression for nvbugs/6753684: `--aux-layers eagle` raised ValueError.

    ``add_aux_layers_args`` defaults to 'eagle', so omitting the flag entirely used to
    crash the dump before it started.
    """
    assert resolve("eagle", 24) == [1, 11, 20]


def test_dflash_preset_still_resolves():
    assert resolve("dflash", 32, num_draft=5) == [1, 8, 15, 22, 29]


def test_explicit_id_list_still_resolves():
    assert resolve("2,5,8", 32) == [2, 5, 8]


@pytest.mark.parametrize("spec", ["bogus", "EAGLE3", "", "eagle3"])
def test_unknown_spec_explains_what_is_accepted(spec):
    """A bare ``invalid literal for int()`` hides what the caller should have passed."""
    with pytest.raises(ValueError, match="'eagle' / 'dflash' presets"):
        resolve(spec, 32)


def test_out_of_range_ids_are_rejected():
    with pytest.raises(ValueError, match="out of range"):
        resolve("2,99", 32)
