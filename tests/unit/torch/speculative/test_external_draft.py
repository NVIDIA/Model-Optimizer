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

"""Unit tests for the ``external`` speculative decoding mode."""

import json

import pytest
import torch

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp

transformers = pytest.importorskip("transformers")

VOCAB = 64
HIDDEN = 32


def _tiny_causal_lm(vocab_size=VOCAB):
    cfg = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(cfg)


def _convert(model, **overrides):
    return mtsp.convert(model, [("external", overrides)])


def test_convert_and_restore_roundtrip(tmp_path):
    """convert -> save -> restore preserves the draft weights."""
    model = _convert(_tiny_causal_lm())
    before = {k: v.clone() for k, v in model.state_dict().items()}

    mto.save(model, tmp_path / "draft.pth")
    restored = mto.restore(_tiny_causal_lm(), tmp_path / "draft.pth")

    after = restored.state_dict()
    assert set(before) == set(after)
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"weight changed across save/restore: {k}"


def test_vocab_mismatch_raises_and_names_both_sizes():
    """A draft indexing a different vocabulary must fail loudly, not train."""
    model = _convert(_tiny_causal_lm(vocab_size=VOCAB))
    with pytest.raises(ValueError, match="share a vocabulary") as exc:
        model.validate_against_base(base_vocab_size=999)
    msg = str(exc.value)
    assert str(VOCAB) in msg and "999" in msg
    model.validate_against_base(base_vocab_size=VOCAB)


@pytest.mark.parametrize("bad", ["not_a_loss", "not_an_objective"])
def test_invalid_objective_is_rejected(bad):
    with pytest.raises(ValueError, match="external_loss"):
        _convert(_tiny_causal_lm(), external_loss=bad)


def test_draft_does_not_share_base_parameters():
    """The point of the mode: the draft owns its embeddings and lm_head."""
    draft = _convert(_tiny_causal_lm())
    base = _tiny_causal_lm()
    assert (
        draft.get_input_embeddings().weight.data_ptr()
        != base.get_input_embeddings().weight.data_ptr()
    )
    assert draft.lm_head.weight.data_ptr() != base.lm_head.weight.data_ptr()


def test_soft_ce_matches_eagle_formula_and_reports_accuracy():
    """``soft_ce`` must reproduce the Eagle offline objective exactly."""
    model = _convert(_tiny_causal_lm(), external_loss="soft_ce", external_report_acc=True)
    torch.manual_seed(0)
    draft_logits = torch.randn(2, 5, VOCAB)
    base_logits = torch.randn(2, 5, VOCAB)
    loss_mask = torch.ones(2, 5)

    loss, _ = model.compute_loss(draft_logits, base_logits, loss_mask)
    expected = -torch.sum(
        loss_mask[:, :, None]
        * torch.softmax(base_logits.float(), dim=-1)
        * torch.log_softmax(draft_logits.float(), dim=-1)
    ) / (loss_mask.sum() + 1e-5)
    assert torch.allclose(loss, expected, atol=1e-6)

    _, acc = model.compute_loss(base_logits.clone(), base_logits, loss_mask)
    assert torch.allclose(acc, torch.tensor(1.0))


def test_loss_mask_excludes_masked_positions():
    """Masked positions must not contribute; prompt tokens are masked in practice."""
    model = _convert(_tiny_causal_lm(), external_loss="soft_ce", external_report_acc=False)
    torch.manual_seed(0)
    draft_logits = torch.randn(1, 4, VOCAB)
    base_logits = torch.randn(1, 4, VOCAB)

    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss_masked, _ = model.compute_loss(draft_logits, base_logits, mask)
    loss_subset, _ = model.compute_loss(draft_logits[:, :2], base_logits[:, :2], torch.ones(1, 2))
    assert torch.allclose(loss_masked, loss_subset, atol=1e-6)


@pytest.mark.parametrize("identical", [False, True])
def test_tvd_deploy_is_filtered_tvd(identical):
    """TVD charges the draft for mass off the teacher's support; both sides are filtered,
    so identical logits score exactly zero."""
    model = _convert(
        _tiny_causal_lm(),
        external_loss="tvd_deploy",
        external_top_k=8,
        external_top_p=0.95,
        external_report_acc=False,
    )
    torch.manual_seed(0)
    base_logits = torch.randn(1, 3, VOCAB)
    draft_logits = base_logits.clone() if identical else torch.randn(1, 3, VOCAB)

    loss, _ = model.compute_loss(draft_logits, base_logits, torch.ones(1, 3))
    if identical:
        assert loss.abs() < 1e-5, f"identical logits must score 0, got {loss}"
    else:
        # Independent reference: plain per-row top-k then nucleus, no call into the
        # code under test. Comparing against _deployment_probs would only assert that
        # the function equals itself.
        def _reference(logits, k, top_p):
            out = torch.zeros_like(logits, dtype=torch.float)
            flat, ref = logits.float().reshape(-1, logits.shape[-1]), out.reshape(-1, out.shape[-1])
            for row in range(flat.shape[0]):
                probs = torch.softmax(flat[row], dim=-1)
                vals, idx = torch.topk(probs, k)
                vals = vals / vals.sum()
                running = 0.0
                for j in range(k):
                    if running >= top_p:
                        vals[j] = 0.0
                    running += float(vals[j])
                vals = vals / vals.sum()
                ref[row].scatter_(0, idx, vals)
            return out

        p = _reference(base_logits, 8, 0.95)
        q = _reference(draft_logits, 8, 0.95)
        assert torch.allclose(loss, (0.5 * (p - q).abs().sum(-1)).mean(), atol=1e-5)


class _FakeTok:
    def __init__(self, vocab):
        self._v = vocab

    def get_vocab(self):
        return self._v


def test_saved_config_names_the_original_class_not_the_dynamic_subclass(tmp_path):
    """vLLM and TRT-LLM resolve config.architectures against their own registries.

    transformers writes type(self).__name__, which after conversion is the registry's
    dynamic subclass, and nothing downstream knows that name.
    """
    model = _convert(_tiny_causal_lm())
    assert "ExternalDraft" in type(model).__name__, "precondition: conversion renamed the class"
    out = tmp_path / "ckpt"
    model.save_pretrained(out)
    written = json.loads((out / "config.json").read_text())["architectures"]
    assert written == ["LlamaForCausalLM"], written


def test_non_main_process_does_not_rewrite_the_config(tmp_path):
    """Every rank enters save_pretrained under FSDP; a read-modify-write from all races."""
    model = _convert(_tiny_causal_lm())
    out = tmp_path / "ckpt"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"architectures": ["Sentinel"]}))
    # positionally, as transformers' own callers pass it
    model.save_pretrained(out, False)
    assert json.loads((out / "config.json").read_text())["architectures"] == ["Sentinel"]


def test_attaching_the_base_head_does_not_touch_the_callers_modules():
    """The head belongs to the base model and may be tied to its embeddings.

    Freezing or moving it in place would silently disable training of the base's output
    head for anything that uses it later.
    """
    base = _tiny_causal_lm()
    head, norm = base.get_output_embeddings(), base.model.norm
    draft = _convert(_tiny_causal_lm())
    draft.attach_base_lm_head(head, norm)

    assert all(p.requires_grad for p in head.parameters()), "caller's head was frozen"
    assert all(p.requires_grad for p in norm.parameters()), "caller's norm was frozen"
    assert draft._base_lm_head is not head, "attached head must be our own copy"
    assert not any(p.requires_grad for p in draft._base_lm_head.parameters())


def test_push_to_hub_is_refused_rather_than_publishing_a_broken_config(tmp_path):
    """config.json is uploaded from inside super().save_pretrained, before we can fix it.

    Both call shapes: transformers declares push_to_hub as the fifth positional
    parameter, so a kwargs-only guard would let the positional form through.
    """
    model = _convert(_tiny_causal_lm())
    with pytest.raises(ValueError, match="push_to_hub"):
        model.save_pretrained(tmp_path / "ckpt", push_to_hub=True)
    with pytest.raises(ValueError, match="push_to_hub"):
        model.save_pretrained(tmp_path / "ckpt", True, None, True)


def test_a_parameterless_final_norm_does_not_abort_the_forward():
    """elementwise_affine=False norms have no parameters; probing with next() raises."""
    base = _tiny_causal_lm()
    norm = torch.nn.LayerNorm(base.config.hidden_size, elementwise_affine=False)
    assert next(norm.parameters(), None) is None, "precondition: no parameters"
    draft = _convert(_tiny_causal_lm())
    draft.attach_base_lm_head(base.get_output_embeddings(), norm)

    hidden = torch.randn(1, 4, base.config.hidden_size)
    # would raise StopIteration -> "generator raised RuntimeError" inside the forward
    draft._teacher_logits(hidden, {"base_model_hidden_states": hidden})


def test_tokenizer_check_sees_tokens_only_the_draft_has():
    """Walking the base alone misses a draft that adds ids the base leaves unused."""
    model = _convert(_tiny_causal_lm(vocab_size=VOCAB))
    base = _FakeTok({"a": 0, "b": 1})
    superset = _FakeTok({"a": 0, "b": 1, "<|extra|>": 2})
    with pytest.raises(ValueError, match="different tokenizers"):
        model.validate_tokenizer_against_base(superset, base)


def test_same_size_different_tokenizers_are_rejected():
    """Equal vocab_size is not enough: misaligned ids train against the wrong columns.

    Shaped like byte-level BPE: the 256 byte symbols occupy ids 0-255 in the same order
    in both tokenizers, and they diverge only above that. A low-id sample would compare
    exactly the region where unrelated tokenizers agree by construction and pass.
    """
    model = _convert(_tiny_causal_lm(vocab_size=VOCAB))
    byte_syms = {chr(i): i for i in range(256)}
    base = _FakeTok({**byte_syms, "Ġthe": 256, "Ġand": 257})
    model.validate_tokenizer_against_base(_FakeTok(dict(base.get_vocab())), base)
    diverges_above_the_bytes = _FakeTok({**byte_syms, "Ġthe": 257, "Ġand": 256})
    with pytest.raises(ValueError, match="different tokenizers"):
        model.validate_tokenizer_against_base(diverges_above_the_bytes, base)
