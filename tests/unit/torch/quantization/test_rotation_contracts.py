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

"""Input-contract and state-hygiene regression tests for the rotation module.

Each test pins one contract whose violation is SILENT (wrong numbers, a mutated caller
model, a dropped user input) rather than loud, i.e. exactly the class of defect the rest
of the suite's equivalence gates cannot catch.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only unit tests: never claim a GPU

import sys
import traceback

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from modelopt.torch.quantization.rotation import QuantObjective, fold_rotations, learn_rotations

VOCAB, HIDDEN, HEAD_DIM = 128, 64, 32

TINY_W4A4 = QuantObjective(
    name="tiny_w4a4", w_bits=4, w_group=16, a_bits=4, a_mode="per_token_dynamic"
)


def _tiny_llama():
    """Two-layer Llama with non-unit RMSNorm gains (so norm fusion is observable)."""
    torch.manual_seed(1234)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)
    return model


def _batches(n=1, bs=2, seq=16, seed=7):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (bs, seq)) for _ in range(n)]


def _orthogonal(n, seed):
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    return q.contiguous()


# --------------------------------------------------------------------------------------
# 1. KD objective contracts
# --------------------------------------------------------------------------------------


def test_kd_term_is_per_token_not_per_sequence():
    """The KD term must not scale with calibration sequence length.

    ``kl_div(reduction="batchmean")`` on unflattened [bs, seq, vocab] logits divides by bs
    alone, making the KD term seq_len times the per-token KL — so the documented
    ``(1-alpha)*CE + alpha*T^2*KL`` mix would silently change meaning when the caller
    changes the calibration sequence length. ``kd_alpha=1.0`` makes the recorded loss
    exactly the KD term, so the two lengths are directly comparable.
    """

    def kd_only_loss(seq):
        torch.manual_seed(21)
        teacher = _tiny_llama()
        for p in teacher.parameters():
            p.data.add_(0.05 * torch.randn_like(p.data))
        teacher.eval().requires_grad_(False)
        torch.manual_seed(4)
        ids = torch.randint(0, VOCAB, (2, seq))
        rs = learn_rotations(
            _tiny_llama(),
            [ids],
            steps=1,
            lr=0.0,
            objective_cfg=None,
            seed=3,
            log_every=0,
            teacher=teacher,
            kd_alpha=1.0,
            kd_temp=2.0,
        )
        return rs.history[0]["loss"]

    small, big = kd_only_loss(8), kd_only_loss(32)
    assert big < 2.0 * small, (
        f"KD term scales with sequence length: seq=32 -> {big:.4f} vs seq=8 -> {small:.4f} "
        "(a per-token mean keeps these comparable)"
    )


def test_self_teacher_rejected():
    """teacher=model must raise, not silently degenerate to a scaled plain-CE run.

    The teacher forward runs inside the student's reparametrization with the student's
    activation-quant hooks attached, so a self-teacher yields identical logits and KL == 0
    while ``meta["kd"]`` still advertises KD as active.
    """
    model = _tiny_llama()
    with pytest.raises(ValueError, match="distinct module"):
        learn_rotations(
            model,
            _batches(),
            steps=1,
            lr=0.0,
            objective_cfg=TINY_W4A4,
            seed=3,
            log_every=0,
            teacher=model,
            kd_alpha=0.5,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kd_temp": 0.0}, "kd_temp"),
        ({"kd_alpha": 1.5}, "kd_alpha"),
        ({"kd_alpha": -0.1}, "kd_alpha"),
    ],
)
def test_kd_hyperparameters_validated(kwargs, match):
    """kd_temp <= 0 (division by zero -> NaN much later) and kd_alpha outside [0, 1]
    (which silently flips the CE term's sign) must be rejected at the call, not surface as
    a NaN in the closing SVD."""
    with pytest.raises(ValueError, match=match):
        learn_rotations(
            _tiny_llama(),
            _batches(),
            steps=1,
            lr=0.0,
            objective_cfg=None,
            seed=3,
            log_every=0,
            teacher=_tiny_llama(),
            **kwargs,
        )


# --------------------------------------------------------------------------------------
# 2. Calibration-batch contracts
# --------------------------------------------------------------------------------------


def test_attention_mask_is_honored_in_dict_batches():
    """Padding must affect neither attention nor the loss.

    The documented dict-batch form carries ``attention_mask``; dropping it makes the CE a
    function of how much padding the tokenizer happened to add, so identical real tokens
    with different padding would train against different objectives.
    """
    torch.manual_seed(11)
    real = torch.randint(1, VOCAB, (2, 8))

    def padded(n_pad):
        ids = torch.cat([real, torch.zeros(2, n_pad, dtype=real.dtype)], dim=1)
        mask = torch.cat(
            [torch.ones(2, 8, dtype=torch.long), torch.zeros(2, n_pad, dtype=torch.long)], dim=1
        )
        return [{"input_ids": ids, "attention_mask": mask}]

    losses = [
        learn_rotations(
            _tiny_llama(), padded(n_pad), steps=1, lr=0.0, objective_cfg=None, seed=3, log_every=0
        ).history[0]["loss"]
        for n_pad in (2, 16)
    ]
    rel = abs(losses[0] - losses[1]) / max(abs(losses[0]), 1e-9)
    assert rel < 0.02, (
        f"attention_mask ignored: 2 vs 16 pad tokens on identical real tokens give "
        f"{losses[0]:.4f} vs {losses[1]:.4f} ({rel:.1%} apart)"
    )


def test_dict_batch_without_mask_matches_tensor_batch():
    """The documented dict form must be equivalent to the plain-tensor form when there is
    no padding (the Mapping branch of the batch extractor is otherwise untested)."""
    ids = _batches()[0]
    as_tensor = learn_rotations(
        _tiny_llama(), [ids], steps=1, lr=0.0, objective_cfg=None, seed=3, log_every=0
    ).history[0]["loss"]
    as_dict = learn_rotations(
        _tiny_llama(),
        [{"input_ids": ids}],
        steps=1,
        lr=0.0,
        objective_cfg=None,
        seed=3,
        log_every=0,
    ).history[0]["loss"]
    assert as_tensor == pytest.approx(as_dict, rel=1e-9)


def test_unsupported_batch_type_raises():
    """A batch that is neither a tensor nor a mapping must name the accepted forms."""
    with pytest.raises(TypeError, match="input_ids"):
        learn_rotations(
            _tiny_llama(),
            [["not", "ids"]],
            steps=1,
            lr=0.0,
            objective_cfg=None,
            seed=3,
            log_every=0,
        )


def test_empty_calib_loader_raises():
    """An exhausted/empty loader must be reported, not loop forever or divide by zero."""
    with pytest.raises(ValueError, match="no batches"):
        learn_rotations(_tiny_llama(), [], steps=1, lr=0.0, objective_cfg=None, seed=3, log_every=0)


# --------------------------------------------------------------------------------------
# 3. QuantObjective validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["w_bits", "a_bits"])
def test_bit_width_below_two_rejected(field):
    """b=1 gives scale = amax/(2^0 - 1) = amax/0 = inf and then 0*inf = NaN for every
    value, so the whole objective silently becomes NaN; it must be rejected instead."""
    with pytest.raises(ValueError, match=field):
        QuantObjective(name="bad", **{field: 1})


def test_a_mode_validated_even_without_a_bits():
    """An unknown a_mode must be caught regardless of a_bits, so a typo cannot sit
    dormant in a config until activations are switched on."""
    with pytest.raises(ValueError, match="a_mode"):
        QuantObjective(name="bad", a_mode="per_tensor_dynamic")


# --------------------------------------------------------------------------------------
# 4. External-input hygiene: no aliasing, no silent key collapse
# --------------------------------------------------------------------------------------


def test_returned_rotations_do_not_alias_caller_buffers():
    """The returned provenance record must be a copy.

    Every conversion in the accept path (as_tensor/detach/to/cpu) is a no-op for an
    already-float64 CPU input, so without an explicit clone the audited matrix would keep
    changing whenever the caller reuses its buffer — after the orthogonality gate passed.
    """
    r1 = _orthogonal(HIDDEN, seed=9)
    rots = fold_rotations(_tiny_llama(), R1=r1, use_r2=False)
    returned = rots["R1"]
    assert returned.data_ptr() != r1.data_ptr()
    before = returned.clone()
    r1.mul_(2.0)
    assert torch.equal(returned, before)


def test_duplicate_r2_layer_keys_rejected():
    """An int key and the R.bin-convention key naming the SAME layer must raise.

    Both normalize to one index, so last-writer-wins silently discards one user-supplied
    rotation while the per-layer completeness check still passes — a wrong-rotation
    checkpoint with no diagnostic.
    """
    model = _tiny_llama()
    r2 = {
        0: _orthogonal(HEAD_DIM, seed=1),
        "model.layers.0.self_attn.R2": _orthogonal(HEAD_DIM, seed=2),
        1: _orthogonal(HEAD_DIM, seed=3),
    }
    with pytest.raises(ValueError, match="more than once"):
        fold_rotations(model, R1=_orthogonal(HIDDEN, seed=4), R2=r2)


def test_warm_start_gate_checks_both_residual_forms():
    """A warm start must be rejected up front by the same criterion the exit audit uses.

    ``R^T R - I`` and ``R R^T - I`` share eigenvalues but not entries, so a matrix can
    pass a one-sided gate and then fail the closing two-sided audit *after* the whole run
    has been paid for.
    """
    n = HIDDEN
    torch.manual_seed(13)
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    skew = torch.zeros(n, n, dtype=torch.float64)
    skew[0, 1] = 1.2e-4
    r = q @ (torch.eye(n, dtype=torch.float64) + skew)
    eye = torch.eye(n, dtype=torch.float64)
    assert (r @ r.T - eye).abs().max().item() < 1e-4 <= (r.T @ r - eye).abs().max().item()

    model = _tiny_llama()
    init = {"R1": r}
    for i in range(model.config.num_hidden_layers):
        init[f"model.layers.{i}.self_attn.R2"] = _orthogonal(HEAD_DIM, seed=100 + i)
    with pytest.raises(ValueError, match="not orthogonal"):
        learn_rotations(
            model,
            _batches(),
            steps=0,
            lr=0.0,
            objective_cfg=None,
            seed=3,
            log_every=0,
            init_rotations=init,
        )


# --------------------------------------------------------------------------------------
# 5. State hygiene on the failure path
# --------------------------------------------------------------------------------------


def test_failed_call_leaves_no_activation_hooks():
    """A setup failure must not leave fake-quant hooks on the caller's model.

    Hooks registered before the guarded region survive the exception, so every later
    forward of that model silently quantizes activations and a retry double-quantizes.
    """
    model = _tiny_llama()
    ids = _batches()[0]
    ref = model(input_ids=ids).logits.clone()

    bad = QuantObjective(
        name="bad_seam",
        w_bits=4,
        w_group=16,
        a_bits=4,
        a_mode="per_token_dynamic",
        learn_seam_diag=True,
    )
    model.config.intermediate_size += 8  # provoke a seam-shape assert during setup
    with pytest.raises(Exception):
        learn_rotations(model, [ids], steps=1, lr=0.0, objective_cfg=bad, seed=3, log_every=0)

    after = model(input_ids=ids).logits
    assert torch.allclose(ref, after, atol=1e-5), (
        "activation fake-quant hooks survived a failed learn_rotations call: "
        f"max logit delta {(ref - after).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            if hasattr(fn, "pytestmark"):  # parametrized: run via pytest instead
                print(f"SKIP {name} (parametrized)", flush=True)
                continue
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)} checked; FAILED: {failed}" if failed else "\nall passed")
    sys.exit(1 if failed else 0)
