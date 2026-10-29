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

"""The cached-teacher example preserves full-vocabulary loss and decoder gradients."""

import importlib
import shutil
from pathlib import Path

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_tokenizer

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/hf_ptq"))
    return importlib.import_module("hf_gsq_rco")


@pytest.mark.parametrize("objective_kind", ["kl", "ce"])
def test_cached_teacher_callback_matches_direct_loss_and_rejects_stale_cache(
    example, tmp_path, objective_kind
):
    model = get_tiny_llama(dtype=torch.float32).eval()
    batch = {
        "sample_id": 0,
        "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "loss_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]]),
    }
    batches = [
        batch,
        {
            "sample_id": 1,
            "input_ids": torch.tensor([[4, 3]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "loss_mask": torch.tensor([[0, 1]]),
        },
    ]
    with torch.no_grad():
        teacher_logits = [
            model(
                input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False
            ).logits
            for b in batches
        ]
    objective = example.CachedTeacherKL.prepare(
        model, batches, tmp_path, fingerprint="model-and-data-a", chunk_tokens=3
    )
    with torch.no_grad():
        weight = model.model.layers[0].mlp.down_proj.weight
        weight.add_(torch.randn_like(weight) * 0.2)
    direct = 0
    for b, teacher in zip(batches, teacher_logits, strict=True):
        student = model(
            input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False
        ).logits
        if objective_kind == "kl":
            loss = torch.nn.functional.kl_div(
                student.log_softmax(-1), teacher.log_softmax(-1), log_target=True, reduction="none"
            ).sum(-1)
            direct = direct + (loss * b["loss_mask"]).sum()
        else:
            loss = torch.nn.functional.cross_entropy(
                student[:, :-1].flatten(0, 1), b["input_ids"][:, 1:].flatten(), reduction="none"
            ).reshape_as(b["loss_mask"][:, 1:])
            direct = direct + (loss * b["loss_mask"][:, 1:] * b["attention_mask"][:, :-1]).sum()
    count = sum(
        int(
            (
                b["loss_mask"]
                if objective_kind == "kl"
                else b["loss_mask"][:, 1:] * b["attention_mask"][:, :-1]
            ).sum()
        )
        for b in batches
    )
    direct = direct / count
    direct.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    callback = objective if objective_kind == "kl" else objective.cross_entropy
    actual = sum(callback(model, b) for b in batches) / len(batches)
    torch.testing.assert_close(actual, direct.detach(), atol=1e-7, rtol=1e-5)
    assert not actual.requires_grad
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(
                parameter.grad / len(batches), expected[name], atol=1e-7, rtol=1e-4
            )
    assert model.lm_head.weight.grad is None
    if objective_kind == "kl":
        report = example._evaluate(model, batches, objective)
        nll, tokens = 0.0, 0
        with torch.no_grad():
            for b in batches:
                valid = b["loss_mask"][:, 1:] * b["attention_mask"][:, :-1]
                labels = b["input_ids"].clone()
                labels[:, 0] = -100
                labels[:, 1:].masked_fill_(~valid.bool(), -100)
                count = int(valid.sum())
                loss = model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                    labels=labels,
                    use_cache=False,
                ).loss
                nll += float(loss) * count
                tokens += count
        assert report["teacher_kl"] == pytest.approx(float(direct.detach()), rel=1e-5, abs=1e-7)
        assert report["perplexity"] == pytest.approx(
            float(torch.tensor(nll / tokens).exp()), rel=1e-5
        )
        assert report["prediction_tokens"] == tokens
    expected_metric = callback(model, batches[0])
    model.requires_grad_(False)
    torch.testing.assert_close(callback(model, batches[0]), expected_metric)
    with pytest.raises(ValueError):
        example.CachedTeacherKL.prepare(
            model, [batch], tmp_path, fingerprint="different-data", chunk_tokens=3
        )


def test_frozen_samples_round_trip_with_masks_and_reject_changed_identity(example, tmp_path):
    records = [{"text": "This is a test sentence.", "url": "local:0"}] * 3
    tokenizer = get_tiny_tokenizer()
    identity = {
        "dataset": "local-fixture",
        "revision": "abc",
        "tokenizer_revision": "def",
        "tokenizer": "original-location",
    }
    samples, manifest = example.freeze_samples(
        records, tokenizer, tmp_path, identity, count=2, max_length=4, seed=7
    )
    restored, same = example.freeze_samples(
        [],
        tokenizer,
        tmp_path,
        {**identity, "tokenizer": "relocated"},
        count=2,
        max_length=4,
        seed=7,
    )
    assert manifest == same
    assert len(restored) == 2
    for original, saved in zip(samples, restored, strict=True):
        assert saved["input_ids"].shape == (1, 4)
        torch.testing.assert_close(saved["input_ids"], original["input_ids"])
        torch.testing.assert_close(saved["attention_mask"], torch.ones((1, 4), dtype=torch.long))
        torch.testing.assert_close(saved["loss_mask"], saved["attention_mask"])
    with pytest.raises(ValueError):
        example.freeze_samples(
            [],
            tokenizer,
            tmp_path,
            {**identity, "revision": "changed"},
            count=2,
            max_length=4,
            seed=7,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_teacher_autoquant_callback_and_ordinary_restore(example, tmp_path, dtype):
    model = get_tiny_llama(dtype=dtype).eval()
    batch = {
        "sample_id": 0,
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    objective = example.CachedTeacherKL.prepare(model, [batch], tmp_path / "teacher", "fixture", 2)
    config = example._formats("max")[1]
    model, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 9},
        quantization_formats=[config],
        data_loader=[batch],
        forward_step=lambda m, b: m.model(
            input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False
        ),
        forward_backward_step=objective.cross_entropy,
        num_calib_steps=1,
        num_score_steps=1,
        method={
            "method": "gradient",
            "data_fingerprint": "fixture",
            "preserve_candidate_weights": True,
        },
        disabled_layers=["lm_head"],
    )
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 9
    assert any(
        float(score) > 0 for stat in state["candidate_stats"].values() for score in stat["scores"]
    )
    with torch.no_grad():
        expected = model(input_ids=batch["input_ids"], use_cache=False).logits
    mto.save(model, tmp_path / "modelopt.pt")
    restored = mto.restore(get_tiny_llama(dtype=dtype).eval(), tmp_path / "modelopt.pt")
    with torch.no_grad():
        torch.testing.assert_close(
            restored(input_ids=batch["input_ids"], use_cache=False).logits, expected
        )
    published = tmp_path / "published"
    example.export_hf_checkpoint(
        model,
        export_dir=published,
        max_shard_size="4KB",
        extra_state_dict={"removed_layer.weight": torch.ones(2, 2)},
    )
    del model, restored, state
    (tmp_path / "modelopt.pt").unlink()
    shutil.rmtree(tmp_path / "teacher")
    packed = example._reload_packed_for_validation(published, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(
            packed(input_ids=batch["input_ids"], use_cache=False).logits,
            expected,
            rtol=0,
            atol=1e-7 if dtype == torch.float32 else 0,
        )
    assert packed.model.layers[0].mlp.down_proj.weight.dtype == dtype
    assert packed.model.rotary_emb.inv_freq.dtype == torch.float32


def test_example_gsq_search_traverses_layerwise_capture_and_preserves_head(example, tmp_path):
    torch.manual_seed(5)
    model = get_tiny_llama(dtype=torch.float32).eval()
    original_head = model.lm_head.weight.detach().clone()
    batch = {
        "sample_id": 0,
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    objective = example.CachedTeacherKL.prepare(model, [batch], tmp_path / "teacher", "fixture", 2)
    config = example._formats("max", {"steps": 1, "data_fingerprint": "fixture"})[1]
    selected, state = example._search(
        None,
        model,
        [batch],
        objective,
        [config],
        {"method": "gradient", "data_fingerprint": "fixture", "preserve_candidate_weights": True},
        tmp_path / "candidates.pt",
        9,
    )
    assert state["best"]["is_satisfied"]
    assert state["best"]["constraints"]["effective_bits"] <= 9
    assert state["quantizer_states"]
    for layer in selected.model.layers:
        for region in ("self_attn", "mlp"):
            assert layer.gsq[region]["best_loss"] <= layer.gsq[region]["initial_loss"]
        assert not hasattr(layer, "_original_forward")
    assert not hasattr(selected, "_original_forward")
    torch.testing.assert_close(selected.lm_head.weight, original_head, rtol=0, atol=0)
    assert any(
        float(score) > 0 for stat in state["candidate_stats"].values() for score in stat["scores"]
    )
    assert torch.isfinite(objective.metric(selected, batch))
    expected = selected(input_ids=batch["input_ids"], use_cache=False).logits
    mto.save(selected, tmp_path / "modelopt.pt")
    restored = mto.restore(get_tiny_llama(dtype=torch.float32).eval(), tmp_path / "modelopt.pt")
    torch.testing.assert_close(
        restored(input_ids=batch["input_ids"], use_cache=False).logits, expected, rtol=0, atol=0
    )
