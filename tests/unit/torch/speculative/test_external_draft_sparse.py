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

"""Sparse teacher-policy path for the ``external`` speculative decoding mode."""

import gzip
import json
import math

import pytest
import torch

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.external.sparse_data import (
    SparsePolicyCollator,
    SparsePolicyDataset,
)

transformers = pytest.importorskip("transformers")

VOCAB = 64
HIDDEN = 32
TOPK = 8


def _causal_lm(hidden=HIDDEN, vocab_size=VOCAB):
    cfg = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(cfg)


def _draft(**overrides):
    m = _causal_lm()
    mtsp.convert(
        m, [("external", {"external_offline": True, "external_loss": "tvd_deploy", **overrides})]
    )
    return m


def _write_records(path, n=4, n_prompt=3, n_gen=5):
    torch.manual_seed(0)
    with gzip.open(path, "wt") as f:
        for i in range(n):
            lp = torch.randn(n_gen, TOPK).log_softmax(-1)
            f.write(
                json.dumps(
                    {
                        "id": f"r{i}",
                        "prompt_ids": torch.randint(0, VOCAB, (n_prompt,)).tolist(),
                        "gen_ids": torch.randint(0, VOCAB, (n_gen,)).tolist(),
                        "topk_tok": torch.randint(0, VOCAB, (n_gen, TOPK)).tolist(),
                        "topk_lp": lp.tolist(),
                    }
                )
                + "\n"
            )


def test_dataset_masks_response_positions_and_pads_to_train_len(tmp_path):
    """Only response positions carry a teacher policy, and top_k is read from the data.

    A config top_k narrower than the records would silently truncate the policy.
    """
    p = tmp_path / "s.jsonl.gz"
    _write_records(p, n=3, n_prompt=3, n_gen=5)
    ds = SparsePolicyDataset([str(p)], max_length=32)
    assert ds.top_k == TOPK

    item = ds[0]
    assert item["input_ids"].shape[0] == 8
    assert item["loss_mask"][:3].sum() == 0
    assert item["loss_mask"][3:8].sum() == 5

    batch = SparsePolicyCollator(train_len=16)([ds[i] for i in range(3)])
    assert batch["input_ids"].shape == (3, 16)
    assert batch["teacher_topk_tok"].shape == (3, 16, TOPK)
    assert batch["teacher_topk_prob"].shape == (3, 16, TOPK)


def test_sparse_loss_is_next_token_aligned():
    """A draft that predicts the NEXT position's teacher policy must score ~0 loss.

    Without the shift this same input scores near 1: the bug is invisible in the loss
    curve, so it is pinned here instead.
    """
    model = _draft(external_report_acc=False)
    seq = 6
    # Teacher at position t is one-hot on token t*3; draft at t-1 must predict it.
    tok = torch.zeros(1, seq, TOPK, dtype=torch.long)
    prob = torch.zeros(1, seq, TOPK)
    tok[0, :, 0] = torch.arange(seq) * 3
    prob[0, :, 0] = 1.0

    draft_logits = torch.full((1, seq, VOCAB), -30.0)
    for t in range(seq - 1):
        draft_logits[0, t, (t + 1) * 3] = 30.0

    loss, _ = model.compute_sparse_loss(draft_logits, tok, prob, torch.ones(1, seq))
    assert loss < 1e-3, f"aligned draft should score ~0, got {loss}"

    # Displacing the draft by one position destroys that agreement.
    misaligned, _ = model.compute_sparse_loss(
        draft_logits.roll(1, dims=1), tok, prob, torch.ones(1, seq)
    )
    assert misaligned > 0.5 > loss


def test_sparse_tvd_matches_one_minus_acceptance():
    """The sparse path must compute the same acceptance objective as the dense one."""
    model = _draft(external_report_acc=False)
    torch.manual_seed(0)
    draft_logits = torch.randn(1, 4, VOCAB)
    tok = torch.randint(0, VOCAB, (1, 4, TOPK))
    prob = torch.randn(1, 4, TOPK).softmax(-1)

    loss, _ = model.compute_sparse_loss(draft_logits, tok, prob, torch.ones(1, 4))

    # Teacher position t is scored against draft position t-1; see compute_sparse_loss.
    # p is zero off its support, so TVD over the full vocabulary is the on-support
    # difference plus everything the draft placed elsewhere.
    p = prob[:, 1:] / prob[:, 1:].sum(-1, keepdim=True)
    q = model._deployment_probs(
        draft_logits[:, :-1], model.external_top_k, model.external_top_p
    ).gather(-1, tok[:, 1:])
    expected = 0.5 * ((p - q).abs().sum(-1) + (1.0 - q.sum(-1)).clamp_min(0.0))
    assert torch.allclose(loss, expected.mean(), atol=1e-5)


def test_sparse_tvd_is_zero_when_the_draft_is_the_teacher_policy():
    """tvd leaves the draft untruncated, so its optimum is the stored policy itself."""
    model = _draft(external_loss="tvd", external_report_acc=False)
    torch.manual_seed(0)
    seq = 3
    # a draft whose softmax IS the teacher policy: mass only on the support
    tok = torch.arange(TOPK).view(1, 1, TOPK).expand(1, seq, TOPK).contiguous()
    logits = torch.full((1, seq, VOCAB), -60.0)
    logits[..., :TOPK] = torch.randn(1, seq, TOPK)
    q = torch.softmax(logits.float(), -1)[..., :TOPK]
    # position t is scored against draft position t-1
    prob = torch.cat([q[:, :1], q[:, :-1]], dim=1)

    loss, _ = model.compute_sparse_loss(logits, tok, prob, torch.ones(1, seq))
    assert loss.abs() < 1e-4, f"draft == stored policy should score 0, got {loss}"

    # tvd_deploy filters the draft too, so the same inputs score differently
    deploy, _ = _draft(external_loss="tvd_deploy", external_report_acc=False).compute_sparse_loss(
        logits, tok, prob, torch.ones(1, seq)
    )
    assert not torch.allclose(loss, deploy), "filtering the draft must change the loss"


def test_tvd_deploy_penalises_mass_outside_the_teacher_support():
    """Probability the draft places off the teacher's support must cost something."""
    torch.manual_seed(0)
    tok = torch.arange(TOPK).view(1, 1, TOPK).expand(1, 2, TOPK).contiguous()
    prob = torch.full((1, 2, TOPK), 1.0 / TOPK)

    logits = torch.full((1, 2, VOCAB), -30.0)
    logits[..., :TOPK] = 0.0
    leaky = logits.clone()
    leaky[..., TOPK : 2 * TOPK] = 0.0  # half the mass now sits off-support

    m = _draft(external_loss="tvd_deploy", external_report_acc=False)
    tight, _ = m.compute_sparse_loss(logits, tok, prob, torch.ones(1, 2))
    leak, _ = m.compute_sparse_loss(leaky, tok, prob, torch.ones(1, 2))
    assert tight < 1e-3, f"aligned draft should score ~0, got {tight}"
    assert leak > 0.4, f"leaked mass must be penalised, got {leak}"


def test_every_sparse_capable_loss_runs_and_backprops(tmp_path):
    """The guard in main.py and the implementations here must not drift apart, and the
    sparse path must train without a teacher lm_head."""
    from modelopt.torch.speculative.plugins.hf_external import SPARSE_CAPABLE_LOSSES

    p = tmp_path / "s.jsonl.gz"
    _write_records(p, n=2, n_prompt=2, n_gen=4)
    ds = SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK)
    batch = SparsePolicyCollator(train_len=8)([ds[0], ds[1]])

    for loss_fn in SPARSE_CAPABLE_LOSSES:
        model = _draft(external_loss=loss_fn, external_report_acc=False)
        assert getattr(model, "_base_lm_head", None) is None, "sparse path needs no teacher head"
        out = model(**batch)
        assert torch.isfinite(out.loss), f"{loss_fn} produced {out.loss}"
        out.loss.backward()
        assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in model.parameters())

    # soft_ce needs full teacher logits, so it must NOT be listed
    assert "soft_ce" not in SPARSE_CAPABLE_LOSSES


@pytest.mark.parametrize(
    ("kind", "expect"),
    [("truncated", "warn"), ("unparseable", "warn"), ("widespread", "raise"), ("empty", "raise")],
)
def test_damaged_shards(tmp_path, kind, expect):
    """One bad shard must not discard an hour of parsing, but losing most of the corpus
    is a broken dump rather than a truncation to tolerate."""
    files = []
    if kind == "empty":
        p = tmp_path / "empty.jsonl.gz"
        gzip.open(p, "wt").close()
        files = [str(p)]
    elif kind == "unparseable":
        p = tmp_path / "s.jsonl.gz"
        _write_records(p, n=2)
        with gzip.open(p, "at") as f:
            f.write("{not json}\n")
        files = [str(p)]
    else:
        good = tmp_path / "good.jsonl.gz"
        _write_records(good, n=3 if kind == "truncated" else 1)
        files = [str(good)]
        for i in range(1 if kind == "truncated" else 6):
            bad = tmp_path / f"bad{i}.jsonl.gz"
            bad.write_bytes(b"\x9f?not a gzip stream at all")
            files.append(str(bad))

    if expect == "raise":
        match = "No sparse-policy records" if kind == "empty" else "unreadable"
        with pytest.raises(ValueError, match=match):
            SparsePolicyDataset(files, max_length=32)
    else:
        with pytest.warns(UserWarning):
            ds = SparsePolicyDataset(files, max_length=32)
        assert len(ds) == (3 if kind == "truncated" else 2)


def test_sparse_data_path_selects_mode_and_rejects_conflicts():
    from modelopt.torch.speculative.plugins.hf_training_args import DataArguments

    assert DataArguments(sparse_data_path="/tmp/x").mode == "sparse"
    assert DataArguments(offline_data_path="/tmp/x").mode == "offline"
    assert DataArguments().mode == "online"
    with pytest.raises(ValueError, match="ambiguous"):
        DataArguments(sparse_data_path="/tmp/x", offline_data_path="/tmp/y")


def test_export_keeps_the_original_architecture(tmp_path):
    """vLLM and TRT-LLM resolve config.architectures against their own registries."""
    model = _causal_lm()
    original = type(model).__name__
    mtsp.convert(model, [("external", {"external_offline": True})])
    assert type(model).__name__ != original  # conversion really did rename the class

    model.save_pretrained(tmp_path / "ckpt")
    config = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    assert config["architectures"] == [original]


def test_cache_rebuilds_over_an_existing_one(tmp_path):
    """A stale cache must be replaced, not crash the rebuild with ENOTEMPTY."""
    a = tmp_path / "a.jsonl.gz"
    _write_records(a, n=3)
    first = SparsePolicyDataset([str(a)], max_length=32)
    assert len(first) == 3

    # a second shard changes the signature, so the cache is stale and rebuilds
    b = tmp_path / "b.jsonl.gz"
    _write_records(b, n=2)
    second = SparsePolicyDataset([str(a), str(b)], max_length=32)
    assert len(second) == 5
    assert not (tmp_path / ".sparse_cache.stale").exists()


def test_warns_when_stored_policy_is_a_truncated_nucleus(tmp_path):
    """tvd treats the stored policy as complete; a partial one must not pass silently."""
    p = tmp_path / "partial.jsonl.gz"
    torch.manual_seed(0)
    with gzip.open(p, "wt") as f:
        # deliberately store only half the mass, as a too-small top-k would
        lp = (torch.rand(5, TOPK) * 0.01).log() + math.log(0.5 / TOPK)
        f.write(
            json.dumps(
                {
                    "id": "r0",
                    "prompt_ids": [1, 2],
                    "gen_ids": [3, 4, 5, 6, 7],
                    "topk_tok": torch.randint(0, VOCAB, (5, TOPK)).tolist(),
                    "topk_lp": lp.tolist(),
                }
            )
            + "\n"
        )
    with pytest.warns(UserWarning, match="stored teacher policy sums to"):
        SparsePolicyDataset([str(p)], max_length=32)


def test_tvd_ce_adds_a_ranking_term_to_tvd():
    """CE must change the loss and must reward the base's own top-1 token."""
    torch.manual_seed(0)
    seq = 3
    logits = torch.randn(1, seq, VOCAB)
    tok = torch.randint(0, VOCAB, (1, seq, TOPK))
    prob = torch.rand(1, seq, TOPK).softmax(-1)
    mask = torch.ones(1, seq)

    plain, _ = _draft(external_loss="tvd", external_report_acc=False).compute_sparse_loss(
        logits, tok, prob, mask
    )
    mixed, _ = _draft(external_loss="tvd_ce", external_report_acc=False).compute_sparse_loss(
        logits, tok, prob, mask
    )
    assert not torch.allclose(plain, mixed)

    # putting all mass on the base's top-1 must beat putting it elsewhere
    m = _draft(external_loss="tvd_ce", external_report_acc=False)
    best = tok.gather(-1, prob.argmax(-1, keepdim=True)).squeeze(-1)
    good = torch.full((1, seq, VOCAB), -30.0).scatter_(-1, best.unsqueeze(-1), 30.0)
    bad = torch.full((1, seq, VOCAB), -30.0).scatter_(-1, tok[..., -1:], 30.0)
    lg, _ = m.compute_sparse_loss(good, tok, prob, mask)
    lb, _ = m.compute_sparse_loss(bad, tok, prob, mask)
    assert lg < lb, f"agreeing with the base's top-1 must score better: {lg} vs {lb}"


def _write_sourced(path, sources):
    """One record per entry in ``sources``, tagged with that source."""
    torch.manual_seed(0)
    with gzip.open(path, "wt") as f:
        for i, src in enumerate(sources):
            lp = torch.randn(4, TOPK).log_softmax(-1)
            f.write(
                json.dumps(
                    {
                        "id": f"r{i}",
                        "source": src,
                        "prompt_ids": torch.randint(0, VOCAB, (3,)).tolist(),
                        "gen_ids": torch.randint(0, VOCAB, (4,)).tolist(),
                        "topk_tok": torch.randint(0, VOCAB, (4, TOPK)).tolist(),
                        "topk_lp": lp.tolist(),
                    }
                )
                + "\n"
            )


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (None, 4),  # unweighted: one entry per record
        ({"a": 3.0}, 3 * 2 + 2),  # 2 'a' records tripled, 2 'b' untouched
        ({"a": 2.0, "b": 0.0}, 4),  # a zero weight drops a source entirely
    ],
)
def test_source_weights_oversample_by_source(tmp_path, weights, expected):
    p = tmp_path / "s.jsonl.gz"
    _write_sourced(p, ["a", "a", "b", "b"])
    plain = SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK)
    ds = SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK, source_weights=weights)
    assert len(ds) == expected
    # oversampling repeats index entries; it must not corrupt what they point at
    base = {tuple(plain[i]["input_ids"].tolist()) for i in range(len(plain))}
    for i in range(len(ds)):
        assert tuple(ds[i]["input_ids"].tolist()) in base


def test_source_weights_reject_an_unknown_source(tmp_path):
    p = tmp_path / "s.jsonl.gz"
    _write_sourced(p, ["a", "b"])
    # silently ignoring a typo would leave the corpus unweighted and the run looking fine
    with pytest.raises(ValueError, match="not in the data"):
        SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK, source_weights={"typo": 2.0})


def test_source_weights_are_deterministic_for_fractional_weights(tmp_path):
    """A fractional weight draws per record; every rank and restart must agree."""
    p = tmp_path / "s.jsonl.gz"
    _write_sourced(p, ["a"] * 20)
    a = SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK, source_weights={"a": 2.5})
    b = SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK, source_weights={"a": 2.5})
    assert len(a) == len(b)
    assert (a.index == b.index).all()
    assert 20 * 2 <= len(a) <= 20 * 3


@pytest.mark.parametrize(
    ("bad", "kept"),
    [
        # Empty policy: width inference used to raise IndexError and abort the build.
        ({"topk_tok": [], "topk_lp": []}, 1),
        # topk_lp shorter than topk_tok: the ragged fallback used to index past its end.
        ({"topk_lp": [[0.0] * TOPK]}, 2),
    ],
)
def test_malformed_policy_records_never_abort_the_build(tmp_path, bad, kept):
    """A damaged record is dropped or truncated, never fatal to the whole pass."""
    p = tmp_path / "s.jsonl.gz"
    torch.manual_seed(0)
    good = {
        "id": "ok",
        "prompt_ids": torch.randint(0, VOCAB, (3,)).tolist(),
        "gen_ids": torch.randint(0, VOCAB, (4,)).tolist(),
        "topk_tok": torch.randint(0, VOCAB, (4, TOPK)).tolist(),
        "topk_lp": torch.randn(4, TOPK).log_softmax(-1).tolist(),
    }
    with gzip.open(p, "wt") as f:
        f.write(json.dumps({**good, "id": "bad", **bad}) + "\n")
        f.write(json.dumps(good) + "\n")
    ds = SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK)
    assert len(ds) == kept
    # the intact record must still be intact
    assert ds[len(ds) - 1]["input_ids"].shape[0] == len(good["prompt_ids"]) + len(good["gen_ids"])
