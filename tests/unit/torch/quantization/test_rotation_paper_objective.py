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

"""Tests for the paper-protocol objective extensions: per-token ASYM min-max
activation fake-quant (``QuantObjective.a_asym``) and the training-graph-only online R4
down_proj Hadamard (``QuantObjective.r4_in_graph``).

Plain test_* functions with asserts: collectable by pytest, and also runnable without it
via ``python test_rotation_paper_objective.py``. CPU-only, tiny models, seconds per test.
"""

import sys
import traceback

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from modelopt.torch.quantization.rotation import (
    W16A4_ASYM_R4G_OBJECTIVE,
    QuantObjective,
    learn_rotations,
)
from modelopt.torch.quantization.rotation.learn import _fq_act, _fq_act_asym, _walsh_hadamard

VOCAB = 128
HIDDEN = 64
HEAD_DIM = 32
N_LAYERS = 2


def _randomize_rmsnorm_gains(model):
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)


def _tiny_llama(intermediate=2 * HIDDEN):
    torch.manual_seed(1234)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,  # default 128 = 2^7: power-of-2 R4 seam
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _calib_batches(n_batches=2, bs=2, seq=16, seed=7):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (bs, seq)) for _ in range(n_batches)]


# --------------------------------------------------------------------------------------
# 1. Asym activation fake-quant: official ActQuantizer numerics, exactly
# --------------------------------------------------------------------------------------


def _official_asym_reference(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Line-for-line transcription of the official SpinQuant ActQuantizer
    (utils/quant_utils.py) at sym=False, clip_ratio=1, groupsize=-1: find_params'
    zero-inclusive per-token range + asym_quant_dequant."""
    maxq = float(2**bits - 1)
    flat = x.reshape(-1, x.shape[-1])
    tmp = torch.zeros(flat.shape[0], dtype=x.dtype)
    xmin = torch.minimum(flat.min(1)[0], tmp)
    xmax = torch.maximum(flat.max(1)[0], tmp)
    degen = (xmin == 0) & (xmax == 0)
    xmin[degen] = -1
    xmax[degen] = +1
    scale = ((xmax - xmin) / maxq).unsqueeze(1)
    zero = torch.round(-xmin.unsqueeze(1) / scale)
    q = torch.clamp(torch.round(flat / scale) + zero, 0, maxq)
    return (scale * (q - zero)).reshape(x.shape)


def test_asym_act_quant_matches_official_reference():
    """_fq_act_asym reproduces the official asym recipe bit-for-bit on hard cases:
    mixed-sign tokens, an ALL-POSITIVE token (zero-inclusion changes the range), an
    all-negative token, and an all-zero token (the [-1, 1] degenerate fallback)."""
    torch.manual_seed(0)
    x = torch.randn(5, 4, 32)
    x[0, 0] = x[0, 0].abs() + 0.5  # all-positive token: xmin must clamp to 0
    x[1, 1] = -x[1, 1].abs() - 0.5  # all-negative token: xmax must clamp to 0
    x[2, 2] = 0.0  # degenerate all-zero token
    for bits in (4, 8):
        got = _fq_act_asym(x, bits)
        want = _official_asym_reference(x, bits)
        assert torch.equal(got, want), (
            f"bits={bits}: asym fake-quant deviates from the official recipe "
            f"(max |diff| = {(got - want).abs().max().item():.3e})"
        )


def test_asym_ste_gradient_is_identity():
    torch.manual_seed(1)
    x = torch.randn(3, 8, 16, requires_grad=True)
    _fq_act_asym(x, 4).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x)), "STE gradient must be identity"


def test_asym_beats_sym_on_shifted_activations():
    """The reason the paper uses asym (A.4): a positively-shifted activation (post-SiLU
    regime) wastes half the sym grid. Same tensor, same bits: asym error must be well
    below sym error."""
    torch.manual_seed(2)
    x = torch.randn(4, 16, 64) + 3.0  # strong positive shift
    bits = 4
    qpos = float(2 ** (bits - 1) - 1)
    s = (x.abs().amax(dim=-1, keepdim=True) / qpos).clamp_min(1e-12)
    err_sym = (_fq_act(x, s, bits) - x).norm()
    err_asym = (_fq_act_asym(x, bits) - x).norm()
    assert err_asym < 0.7 * err_sym, (
        f"asym ({err_asym:.4f}) should clearly beat sym ({err_sym:.4f}) on shifted data"
    )


# --------------------------------------------------------------------------------------
# 2. Walsh-Hadamard helper
# --------------------------------------------------------------------------------------


def test_walsh_hadamard_properties():
    for n in (1, 2, 8, 128):
        H = _walsh_hadamard(n)
        assert H.shape == (n, n) and H.dtype == torch.float32
        assert torch.equal(H, H.t()), "Sylvester Hadamard must be symmetric"
        err = (H @ H.t() - torch.eye(n)).abs().max().item()
        assert err < 1e-6, f"n={n}: |H H^T - I| = {err:.3e}"
    for bad in (0, 3, 48, 6144):  # 6144 = Qwen3-1.7B intermediate — documented unsupported
        try:
            _walsh_hadamard(bad)
            raise AssertionError(f"n={bad} should have raised NotImplementedError")
        except NotImplementedError:
            pass


# --------------------------------------------------------------------------------------
# 3. r4_in_graph: functional identity in the graph, absent from the output
# --------------------------------------------------------------------------------------


def test_r4_in_graph_is_functional_identity():
    """With ALL quantizers off, the r4 pair (input hook x @ H + weight cols @ H) must be
    a functional identity: at lr=0 the step-0 loss equals the objective=None loss, and
    the returned rotations equal the objective=None run's rotations (same seed draws +
    final retraction) — i.e. no H leaks into the deployable output."""
    batches = _calib_batches()
    r4_only = QuantObjective(
        name="r4_only", w_bits=None, w_group=None, a_bits=None, r4_in_graph=True
    )
    rs_r4 = learn_rotations(
        _tiny_llama(), batches, steps=1, lr=0.0, objective_cfg=r4_only, seed=3, log_every=0
    )
    rs_ref = learn_rotations(
        _tiny_llama(), batches, steps=1, lr=0.0, objective_cfg=None, seed=3, log_every=0
    )
    l_r4, l_ref = rs_r4.history[0]["loss"], rs_ref.history[0]["loss"]
    assert abs(l_r4 - l_ref) < 1e-3 * max(1.0, abs(l_ref)), (
        f"r4-only step-0 loss {l_r4} != objective-None loss {l_ref} — the H pair is not "
        "a functional identity"
    )
    for k in rs_ref.rotations:
        d = (rs_r4.rotations[k] - rs_ref.rotations[k]).abs().max().item()
        assert d < 1e-5, f"{k}: rotations differ by {d:.3e} — H leaked into the output"


def test_r4_rejects_non_pow2_seam():
    model = _tiny_llama(intermediate=96)  # 96 = 3 * 32: not a power of 2
    with pytest.raises(NotImplementedError, match="power of 2"):
        learn_rotations(
            model,
            _calib_batches(),
            steps=1,
            lr=0.0,
            objective_cfg=QuantObjective(name="r4", w_bits=None, a_bits=None, r4_in_graph=True),
            seed=0,
            log_every=0,
        )


# --------------------------------------------------------------------------------------
# 4. The paper preset trains: loss decreases on a tiny overfit case, meta records flags
# --------------------------------------------------------------------------------------


def test_w16a4_asym_r4g_preset_trains_and_records_meta():
    torch.manual_seed(11)
    batch = [torch.randint(0, VOCAB, (2, 16))]  # single repeated batch: overfit regime
    rs = learn_rotations(
        _tiny_llama(),
        batch,
        steps=12,
        lr=0.5,
        objective_cfg=W16A4_ASYM_R4G_OBJECTIVE,
        seed=5,
        log_every=0,
    )
    first, last = rs.history[0]["loss"], rs.history[-1]["loss"]
    assert last < first, f"loss did not decrease: {first} -> {last}"
    obj = rs.meta["objective"]
    assert obj["a_asym"] is True and obj["r4_in_graph"] is True and obj["w_bits"] is None
    audit = rs.ortho_audit()
    assert max(audit.values()) < 1e-4, f"rotations left the manifold: {audit}"


def test_validation_errors():
    for kwargs in (
        {"a_asym": True, "a_bits": None},  # asym needs a_bits
        {"a_asym": True, "a_bits": 8, "a_mode": "per_tensor_static"},  # asym is per-token
    ):
        try:
            QuantObjective(name="bad", w_bits=None, w_group=None, **kwargs)
            raise AssertionError(f"{kwargs} should have raised ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
