# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""End-to-end smoke test for the streaming quantile calibrator on Qwen3-8B.

Runs ``mtq.quantize`` with ``algorithm={"method": "quantile", ...}`` and
``*input_quantizer.calibrator = "quantile"`` on a tiny calibration set, then
verifies:

1. Every enabled input quantizer ends with a finite, positive ``_amax``.
2. At least one ``QuantileCalibrator`` instance exists in the model and holds
   four ``P2QuantileEstimator`` objects.
3. ``save_quantile_data`` dumps a non-empty JSON checkpoint that round-trips.
4. A post-quantization forward pass produces non-NaN logits.

Requires CUDA. Run manually from a GPU node:

    python tests/gpu/torch/quantization/test_quantile_qwen3_e2e.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

import pytest
import torch

MODEL_PATH = os.environ.get(
    "QUANTILE_E2E_MODEL", "/home/scratch.omniml_data_2/fridah/model/Qwen/Qwen3-8B"
)


def _run_smoke() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available — skipping E2E smoke", file=sys.stderr)
        return 0
    if not os.path.isdir(MODEL_PATH):
        print(f"Model path missing: {MODEL_PATH}", file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.calib.quantile import (
        P2QuantileEstimator,
        QuantileCalibrator,
        save_quantile_data,
    )
    from modelopt.torch.quantization.nn import TensorQuantizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    prompts = [
        "The quick brown fox jumps over the lazy dog. " * 16,
        "Quantization reduces the precision of neural-network weights. " * 16,
        "Calibration estimates the dynamic range of activations. " * 16,
        "Streaming algorithms use bounded memory. " * 16,
    ] * 8  # 32 prompts

    batches = []
    for p in prompts:
        enc = tokenizer(
            p, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
        )
        batches.append({k: v.to("cuda") for k, v in enc.items()})

    def forward_loop(m):
        for b in batches:
            with torch.no_grad():
                m(**b, use_cache=False)

    qcfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    qcfg["algorithm"] = {
        "method": "quantile",
        "quantiles": [0.99, 0.999, 0.9999, 0.99999],
    }
    # Append-only override of the activation-side calibrator. modelopt's
    # ``quant_cfg`` is a list of {"quantizer_name": pattern, "cfg": {...}}
    # entries applied in order; later entries override earlier ones.
    qcfg["quant_cfg"].append(
        {"quantizer_name": "*input_quantizer", "cfg": {"calibrator": "quantile"}}
    )

    model = mtq.quantize(model, qcfg, forward_loop)

    # --- Assertion 1: input quantizer amaxes are finite and non-zero ---
    bad: list[str] = []
    n_input_q = 0
    for name, m in model.named_modules():
        if not isinstance(m, TensorQuantizer):
            continue
        if not m.is_enabled or "input_quantizer" not in name:
            continue
        n_input_q += 1
        amax = getattr(m, "amax", None)
        if amax is None:
            bad.append(f"{name}: amax is None")
            continue
        if not torch.isfinite(amax).all():
            bad.append(f"{name}: amax has non-finite values")
            continue
        if (amax == 0).all():
            bad.append(f"{name}: amax all zero")
    assert n_input_q > 0, "no enabled input quantizers found"
    assert not bad, f"{len(bad)} bad input quantizers; first: {bad[:3]}"
    print(f"[OK] {n_input_q} enabled input quantizers all have finite non-zero amax")

    # --- Assertion 2: QuantileCalibrator instances exist with 4 P2 estimators ---
    n_cals = 0
    for _name, m in model.named_modules():
        cal = getattr(m, "_calibrator", None)
        if isinstance(cal, QuantileCalibrator):
            n_cals += 1
            assert len(cal._estimators) == 4
            assert all(isinstance(e, P2QuantileEstimator) for e in cal._estimators.values())
    assert n_cals > 0, "no QuantileCalibrator instances installed"
    print(f"[OK] {n_cals} QuantileCalibrator instances; each holds 4 P^2 estimators")

    # --- Assertion 3: save_quantile_data round-trips ---
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        out_path = f.name
    n_saved = save_quantile_data(model, out_path)
    assert n_saved == n_cals
    with open(out_path) as f:
        payload = json.load(f)
    assert len(payload) == n_cals
    # Spot-check one entry
    first = next(iter(payload.values()))
    assert len(first) == 4  # four quantile levels
    print(f"[OK] save_quantile_data wrote {n_saved} entries that round-trip via JSON")

    # --- Assertion 4: post-quantization forward returns non-NaN logits ---
    probe = tokenizer("Hello world.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**probe, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    assert torch.isfinite(logits).all(), "post-quantization forward returned non-finite logits"
    print(f"[OK] post-quantization forward produced finite logits {tuple(logits.shape)}")

    return 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason=f"model not at {MODEL_PATH}")
def test_quantile_qwen3_smoke():
    assert _run_smoke() == 0


if __name__ == "__main__":
    sys.exit(_run_smoke())
