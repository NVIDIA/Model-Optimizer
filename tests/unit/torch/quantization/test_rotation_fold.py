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

"""Tests for modelopt.torch.quantization.rotation.fold_rotations (offline R1+R2 folding).

Plain test_* functions with asserts: collectable by pytest, and also runnable without it
via ``python test_rotation_fold.py`` (the __main__ driver runs every test function and
exits nonzero on any failure).
"""

import sys
import traceback

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from modelopt.torch.quantization.rotation import fold_rotations

VOCAB = 128
HIDDEN = 64
# Deliberately decoupled: HEAD_DIM != HIDDEN // num_attention_heads (32 vs 64//4 = 16), like
# Qwen3-0.6B (128 vs 1024//16 = 64). A coincident config would let a regression of the
# head_dim resolution to the wrong fallback formula pass every test (the R2 shape check in
# test_returned_rotations_orthonormal is the only one that can catch it, and only if the two
# values differ). This also makes o_proj non-square ([64, 128]), covering that trap too.
HEAD_DIM = 32
N_LAYERS = 2

# FP-equivalence tolerance for fp32 round-trip: fold_rotations does all math in float64 and
# error enters only through the float64 -> float32 cast of each rewritten weight (<= 2^-24
# relative per weight), propagated through 2 decoder layers of an exactly-equivalent
# reparametrization. Measured max |delta logit| on these models is ~3e-7 against logits of
# magnitude ~0.6; atol=1e-4 keeps >100x headroom (also for deeper/bigger real models where
# the cast noise accumulates) while still catching any real transform bug — a wrong rotation
# orientation or a missed norm fusion perturbs logits by O(1).
ATOL_FP32 = 1e-4


def _randomize_rmsnorm_gains(model):
    """Set every RMSNorm gain to a random non-one value. Fresh HF models initialize all norm
    weights to ones, which would make both the norm-fusion math and the fused-to-ones /
    q_norm-untouched checks vacuous."""
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data = 1.0 + 0.1 * torch.randn_like(module.weight.data)


def _tiny_llama(tie=False):
    torch.manual_seed(1234)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _tiny_qwen3(tie=False):
    torch.manual_seed(1234)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(cfg).eval()
    _randomize_rmsnorm_gains(model)
    return model


def _logits(model):
    torch.manual_seed(99)
    ids = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        return model(ids).logits


def test_fp_equivalence():
    """(a) Logits before vs after fold agree within the fp32 round-trip tolerance."""
    for build in (_tiny_llama, _tiny_qwen3):
        model = build()
        before = _logits(model)
        fold_rotations(model, mode="hadamard", seed=0, use_r2=True)
        after = _logits(model)
        max_diff = (after - before).abs().max().item()
        assert torch.allclose(after, before, rtol=0, atol=ATOL_FP32), (
            f"{build.__name__}: max |delta logit| = {max_diff:.3e} > {ATOL_FP32}"
        )


def test_fused_norm_weights_are_ones():
    """(b) All fused RMSNorm gains (started random) are exactly ones after fold."""
    for build in (_tiny_llama, _tiny_qwen3):
        model = build()
        fold_rotations(model)
        for layer in model.model.layers:
            assert torch.all(layer.input_layernorm.weight == 1)
            assert torch.all(layer.post_attention_layernorm.weight == 1)
        assert torch.all(model.model.norm.weight == 1)


def test_qwen3_qk_norm_bitwise_untouched():
    """(c) Qwen3 per-head q_norm/k_norm are bitwise identical after fold."""
    model = _tiny_qwen3()
    before = {
        n: p.data.clone() for n, p in model.named_parameters() if "q_norm" in n or "k_norm" in n
    }
    assert len(before) == 2 * N_LAYERS
    # Sanity: gains were randomized, so "untouched" is not trivially "still ones".
    assert all(not torch.all(p == 1) for p in before.values())
    fold_rotations(model)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p.data, before[n]), f"{n} changed"


def test_tied_embeddings_untied_and_lm_head_correct():
    """(d) Tied embeddings: config flag off after fold, storage untied, and lm_head equals
    the final-norm-fused + R1-rotated copy of the original shared weight (while embed_tokens
    is rotated WITHOUT the norm gain)."""
    model = _tiny_qwen3(tie=True)
    embed = model.model.embed_tokens
    assert model.lm_head.weight.data_ptr() == embed.weight.data_ptr()  # really tied

    before = _logits(model)
    shared = embed.weight.data.clone()
    gamma_final = model.model.norm.weight.data.clone()
    rotations = fold_rotations(model)

    assert model.config.tie_word_embeddings is False
    assert model.lm_head.weight.data_ptr() != embed.weight.data_ptr()

    # Replicate fold's exact op sequence (fp64 fuse -> fp32 cast -> fp64 rotate -> fp32).
    fused = (shared.double() * gamma_final.double()).to(shared.dtype)
    expected_head = (fused.double() @ rotations["R1"]).to(shared.dtype)
    expected_embed = (shared.double() @ rotations["R1"]).to(shared.dtype)
    assert torch.allclose(model.lm_head.weight.data, expected_head, rtol=0, atol=1e-7)
    assert torch.allclose(embed.weight.data, expected_embed, rtol=0, atol=1e-7)

    after = _logits(model)
    assert torch.allclose(after, before, rtol=0, atol=ATOL_FP32)


def test_returned_rotations_orthonormal():
    """(e) Returned R1 + per-layer R2 are float64 CPU and orthonormal to < 1e-10."""
    model = _tiny_qwen3()
    rotations = fold_rotations(model)
    expected_keys = {"R1"} | {f"model.layers.{i}.self_attn.R2" for i in range(N_LAYERS)}
    assert set(rotations) == expected_keys
    for name, mat in rotations.items():
        size = HIDDEN if name == "R1" else HEAD_DIM
        assert mat.dtype == torch.float64 and mat.device.type == "cpu"
        assert mat.shape == (size, size)
        err = (mat @ mat.T - torch.eye(size, dtype=torch.float64)).abs().max().item()
        assert err < 1e-10, f"{name}: max |R R^T - I| = {err:.3e}"


def test_r2_actually_applied_to_v_and_o_weights():
    """(g) R2 is really folded into the weights, not just returned. R2 folding is a
    functional identity, so fp-equivalence (test a) can never catch a silent no-op in
    _rotate_v_proj_r2/_rotate_o_proj_r2. Fold two identically-built models with the same
    seed, one with use_r2=False (R1 is drawn before any R2, so R1 is identical) and one
    with use_r2=True, then check the use_r2=True weights equal the use_r2=False weights
    with the returned R2 applied: v_proj rows R2^T per KV-head block, o_proj columns @ R2
    per Q-head block (replicating fold's exact fp64 op sequence)."""
    for build in (_tiny_llama, _tiny_qwen3):
        model_no_r2 = build()
        model_r2 = build()  # same construction seed -> bitwise-identical weights
        rot_no_r2 = fold_rotations(model_no_r2, mode="hadamard", seed=0, use_r2=False)
        rotations = fold_rotations(model_r2, mode="hadamard", seed=0, use_r2=True)
        assert set(rot_no_r2) == {"R1"}
        assert torch.equal(rot_no_r2["R1"], rotations["R1"])

        for idx in range(N_LAYERS):
            R2 = rotations[f"model.layers.{idx}.self_attn.R2"]
            attn0 = model_no_r2.model.layers[idx].self_attn
            attn1 = model_r2.model.layers[idx].self_attn

            # v_proj: per-KV-head row blocks W_h <- R2^T @ W_h (as W^T blocks @ R2).
            v0 = attn0.v_proj.weight.data
            out_f, in_f = v0.shape
            vt = v0.to(torch.float64).t()
            vt = (vt.reshape(in_f, out_f // HEAD_DIM, HEAD_DIM) @ R2).reshape(in_f, out_f)
            expected_v = vt.t().contiguous().to(v0.dtype)
            got_v = attn1.v_proj.weight.data
            assert not torch.equal(got_v, v0), f"layer {idx}: v_proj unchanged by use_r2=True"
            assert torch.allclose(got_v, expected_v, rtol=0, atol=1e-7), (
                f"layer {idx}: v_proj does not carry the returned R2"
            )

            # o_proj: per-Q-head column blocks W[:, h*d:(h+1)*d] <- W[:, h*d:(h+1)*d] @ R2.
            o0 = attn0.o_proj.weight.data
            out_f, in_f = o0.shape
            expected_o = (
                (o0.to(torch.float64).reshape(out_f, in_f // HEAD_DIM, HEAD_DIM) @ R2)
                .reshape(out_f, in_f)
                .to(o0.dtype)
            )
            got_o = attn1.o_proj.weight.data
            assert not torch.equal(got_o, o0), f"layer {idx}: o_proj unchanged by use_r2=True"
            assert torch.allclose(got_o, expected_o, rtol=0, atol=1e-7), (
                f"layer {idx}: o_proj does not carry the returned R2"
            )


def test_unsupported_arch_raises():
    """(f) A model class outside the registry raises NotImplementedError."""

    class NotADecoderLM(torch.nn.Module):
        pass

    with pytest.raises(NotImplementedError, match="NotADecoderLM"):
        fold_rotations(NotADecoderLM())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(
        f"\n{len(tests) - len(failed)}/{len(tests)} tests passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    sys.exit(1 if failed else 0)
