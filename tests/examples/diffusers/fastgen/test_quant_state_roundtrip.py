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

"""Regression test for the DMD2 QAT (restore-only) quantizer-state restore.

The fastgen QAT path NEVER calibrates during training -- it RESTORES a ModelOpt quantizer
state (recipe + frozen amax) saved by the calibration example and re-applies it on every
(re)start. This test pins the guarantee the recipe depends on, on a tiny CPU model (no
GPU, milliseconds): ``dmd2_recipe.restore_quantizer_state`` onto a *fresh* model with
DIFFERENT weights reproduces the amax bit-identically and leaves that model's weights
untouched -- i.e. amax stays exactly as calibrated and the warm-started student weights
are preserved.

The on-disk state is built here with ModelOpt's own idiom (the same one the calibration
example's ``--quantized-torch-ckpt-save-path`` uses), so the test also pins format
compatibility with that file.

Dependency-guarded with ``importorskip`` so it skips where torch / modelopt are absent.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
if str(_FASTGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTGEN_DIR))

torch = pytest.importorskip("torch")
mtq = pytest.importorskip("modelopt.torch.quantization")
mto = pytest.importorskip("modelopt.torch.opt")
dmd2_recipe = pytest.importorskip("dmd2_recipe")

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.core_utils import get_quantizer_state_dict


def _tiny_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))


def _amax_by_name(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.amax.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, TensorQuantizer) and module.amax is not None
    }


def test_restore_quantizer_state_is_bit_identical_and_weight_free(tmp_path):
    # Calibrate a tiny model (this is the ONLY place quantize/calibration happens -- the
    # calibration example; the trainer never does this) and save its quantizer state in the
    # weight-free format the calibration example writes (mto.modelopt_state + amax).
    model = _tiny_model(seed=0)
    calib = torch.randn(16, 8)
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(calib))
    src_amax = _amax_by_name(model)
    assert src_amax, "expected at least one calibrated TensorQuantizer amax"

    state = mto.modelopt_state(model)
    state["modelopt_state_weights"] = get_quantizer_state_dict(model)
    path = tmp_path / "transformer.pt"
    torch.save(state, str(path))

    # Restore onto a FRESH model with DIFFERENT weights; amax must come back
    # bit-identically and the fresh model's weights must be untouched.
    fresh = _tiny_model(seed=999)
    before = {n: p.detach().clone() for n, p in fresh.named_parameters()}
    dmd2_recipe.restore_quantizer_state(fresh, str(path))

    restored_amax = _amax_by_name(fresh)
    assert set(restored_amax) == set(src_amax)
    for name, amax in src_amax.items():
        assert torch.equal(restored_amax[name], amax), f"amax mismatch at {name}"

    for n, p in fresh.named_parameters():
        if n in before:
            assert torch.equal(p.detach(), before[n]), f"restore changed weight {n}"
