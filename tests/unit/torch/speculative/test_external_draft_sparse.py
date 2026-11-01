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

import contextlib
import gzip
import json
import math
import threading
from unittest import mock

import pytest
import torch

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.external import sparse_data
from modelopt.torch.speculative.external.sparse_data import (
    SparsePolicyCollator,
    SparsePolicyDataset,
)

transformers = pytest.importorskip("transformers")

# Below the guard: the plugins package registers these behind an import_plugin
# guard, but importing the submodules directly does not, so on an install
# without transformers this module would fail to collect rather than skip.
from modelopt.torch.speculative.plugins.hf_external import SPARSE_CAPABLE_LOSSES
from modelopt.torch.speculative.plugins.hf_training_args import DataArguments

VOCAB = 64
HIDDEN = 32
TOPK = 8
# Marks a key the parametrization removes from a record rather than overwrites.
_DROP = object()


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
    mtsp.convert(m, [("external", {"external_loss": "tvd_deploy", **overrides})])
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


# Named rather than taken from SPARSE_CAPABLE_LOSSES, so a later objective cannot
# silently join this parametrization and be scored against the wrong reference.
@pytest.mark.parametrize("loss_fn", ["tvd", "tvd_deploy", "tvd_ce"])
def test_sparse_tvd_matches_one_minus_acceptance(loss_fn):
    """The sparse path must compute the same acceptance objective as the dense one.

    The reference below is the whole contract: it applies the one-position shift, the
    per-objective draft filter, and the out-of-support term. Agreeing with it on random
    inputs therefore also pins next-token alignment, the zero at draft == stored policy,
    and the penalty on mass the draft places off the teacher's support.
    """
    model = _draft(external_loss=loss_fn, external_report_acc=False)
    torch.manual_seed(0)
    draft_logits = torch.randn(1, 4, VOCAB)
    # Distinct ids per position. randint would collide most rows out of 8-of-64, and
    # _tvd_sparse deduplicates a repeated id -- which the reference below deliberately
    # does not model, because the dedup has its own test. (tvd_deploy hides the
    # difference: its filter zeroes the duplicate slots anyway.)
    tok = torch.stack([torch.randperm(VOCAB)[:TOPK] for _ in range(4)]).unsqueeze(0)
    prob = torch.randn(1, 4, TOPK).softmax(-1)

    loss, _ = model.compute_sparse_loss(draft_logits, tok, prob, torch.ones(1, 4))

    # Teacher position t is scored against draft position t-1; see compute_sparse_loss.
    # p is zero off its support, so TVD over the full vocabulary is the on-support
    # difference plus everything the draft placed elsewhere.
    p = prob[:, 1:] / prob[:, 1:].sum(-1, keepdim=True)
    q_full = (
        model._deployment_probs(draft_logits[:, :-1], model.external_top_k, model.external_top_p)
        if loss_fn == "tvd_deploy"
        else torch.softmax(draft_logits[:, :-1].float(), dim=-1)
    )
    q = q_full.gather(-1, tok[:, 1:])
    expected = 0.5 * ((p - q).abs().sum(-1) + (1.0 - q.sum(-1)).clamp_min(0.0)).mean()
    if loss_fn == "tvd_ce":
        # one assertion pins the CE target, the shift and both weights
        top1 = tok[:, 1:].gather(-1, prob[:, 1:].argmax(-1, keepdim=True)).squeeze(-1)
        ce = torch.nn.functional.cross_entropy(draft_logits[0, :-1], top1[0])
        expected = model.external_tvd_alpha * expected + model.external_ce_alpha * ce
    assert torch.allclose(loss, expected, atol=1e-5)


def test_every_sparse_capable_loss_runs_and_backprops(tmp_path):
    """The guard in main.py and the implementations here must not drift apart, and the
    sparse path must train without a teacher lm_head."""
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
    assert DataArguments(sparse_data_path="/tmp/x").mode == "sparse"
    assert DataArguments(offline_data_path="/tmp/x").mode == "offline"
    assert DataArguments().mode == "online"
    with pytest.raises(ValueError, match="ambiguous"):
        DataArguments(sparse_data_path="/tmp/x", offline_data_path="/tmp/y")


def test_an_explicit_top_k_rebuilds_a_cache_built_at_another_width(tmp_path):
    """A changed top_k must invalidate the cache, not be silently dropped.

    The width is part of the objective: it truncates or pads the stored policy. The
    staleness check keys on the shard list, which does not change when only the
    requested width does, so without this the run keeps the old width and
    ``self.top_k`` reports the cache's rather than the caller's.
    """
    a = tmp_path / "a.jsonl.gz"
    _write_records(a, n=2)
    first = SparsePolicyDataset([str(a)], max_length=32, top_k=TOPK)
    assert first.top_k == TOPK

    # Widening is the direction the builder allows: it pads with zero-probability
    # entries. Narrowing would drop teacher mass and raises instead, so it cannot
    # reach the cache check this test is about.
    wider = SparsePolicyDataset([str(a)], max_length=32, top_k=TOPK + 4)
    assert wider.top_k == TOPK + 4, "the cache ignored the requested width"
    # an unset top_k still accepts whatever the cache holds
    assert SparsePolicyDataset([str(a)], max_length=32).top_k == TOPK + 4
    # a non-builder rank must wait for the requested width, not take the one on disk
    with pytest.raises(TimeoutError):
        sparse_data._wait_for_cache(tmp_path / ".sparse_cache", [str(a)], TOPK, stall_s=0)


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
    assert not list(tmp_path.glob(".sparse_cache.stale*"))


def test_wait_for_cache_ignores_a_stale_one(tmp_path):
    """A non-builder rank must wait for *these* shards, not for any meta.json.

    The only way to reach the wait with a cache on disk is that it was found stale, so
    returning on mere existence hands the waiter the previous dump.
    """
    a = tmp_path / "a.jsonl.gz"
    _write_records(a, n=3)
    SparsePolicyDataset([str(a)], max_length=32)  # publishes a cache for [a]
    cache = tmp_path / ".sparse_cache"
    assert (cache / "meta.json").exists()

    b = tmp_path / "b.jsonl.gz"
    _write_records(b, n=2)
    # [a, b] has a different signature, so the cache on disk is stale for it: the wait
    # must time out rather than accept the one already published.
    with pytest.raises(TimeoutError):
        sparse_data._wait_for_cache(cache, [str(a), str(b)], stall_s=0.1)
    # and it returns promptly for the shards it was actually built from
    sparse_data._wait_for_cache(cache, [str(a)], stall_s=5.0)


def test_losing_a_publish_race_to_different_shards_is_refused(tmp_path):
    """Two runs over different subsets of one dump share the default cache path.

    The loser of the publish rename returns from the build normally, having adopted the
    winner's cache without ever comparing it against its own shards. It must refuse that
    rather than train a whole run on the wrong corpus.
    """
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    _write_records(a, n=3)
    _write_records(b, n=2)
    # publish a cache for {a, b}, then ask for one over {a} alone
    SparsePolicyDataset([str(a), str(b)], max_length=32)
    cache = tmp_path / ".sparse_cache"
    meta = json.loads((cache / "meta.json").read_text())

    # simulate losing the race: the build is skipped and the other job's cache is in place
    with (
        mock.patch.object(sparse_data, "_build_cache", return_value=None),
        pytest.raises(RuntimeError, match="concurrent job"),
    ):
        SparsePolicyDataset([str(a)], max_length=32)
    # the published cache is untouched by the refusal
    assert json.loads((cache / "meta.json").read_text()) == meta


def test_a_non_builder_rank_waits_on_the_files_not_a_barrier(tmp_path):
    """With a process group already up, rank != 0 must still take the file wait.

    HfTrainingArguments builds the process group -- with ddp_timeout, 1800s by default --
    while parsing arguments, so by the time any dataset is constructed a collective here
    is both reachable and bounded. A first build over a large corpus outlasts that bound
    and every non-builder rank dies at the barrier. _wait_for_cache needs no group and
    bounds time without progress instead, so it is the branch both cases must take.
    """
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    _write_records(a, n=3)
    _write_records(b, n=2)
    SparsePolicyDataset([str(a)], max_length=32)  # a cache for [a], stale for [a, b]

    waited = []
    real_build = sparse_data._build_cache

    def spy(cache, files, *args, **kwargs):
        waited.append(list(files))
        real_build(cache, files, None)  # stand in for the rank-0 process

    barrier = mock.Mock()
    with (
        mock.patch.object(sparse_data.torch.distributed, "is_initialized", return_value=True),
        mock.patch.object(sparse_data.torch.distributed, "get_rank", return_value=1),
        mock.patch.object(sparse_data.torch.distributed, "barrier", barrier),
        mock.patch.object(sparse_data, "_wait_for_cache", spy),
    ):
        ds = SparsePolicyDataset([str(a), str(b)], max_length=32)

    assert waited == [[str(a), str(b)]], "rank 1 did not wait on the files"
    barrier.assert_not_called()
    assert len(ds) == 5


def test_ragged_row_whose_support_includes_id_zero():
    """A padded row whose support holds id 0 still scores ~0 for a matching draft.

    The builder pads a ragged row by repeating id 0, so a row legitimately containing
    id 0 has it twice, and ``p`` is not deduplicated -- collapsing to the pad instead of
    the real occurrence would pair the teacher's probability with the pad's ``q``.

    The collapse keys on stored probability rather than on token id, so the pad slots
    are identified directly and the real occurrence is never the one dropped.
    """
    tok = torch.tensor([[[0, 7, 0, 0]]])
    prob = torch.tensor([[[0.6, 0.4, 0.0, 0.0]]])
    # a draft that matches the teacher exactly on this row
    logits = torch.full((1, 1, VOCAB), -30.0)
    logits[0, 0, 0] = math.log(0.6) + 30.0
    logits[0, 0, 7] = math.log(0.4) + 30.0
    m = _draft(external_loss="tvd", external_report_acc=False)
    # one scored position needs two columns: the policy at t is scored against draft t-1
    tok2 = torch.cat([tok, tok], dim=1)
    prob2 = torch.cat([prob, prob], dim=1)
    logits2 = torch.cat([logits, logits], dim=1)
    loss, _ = m.compute_sparse_loss(logits2, tok2, prob2, torch.ones(1, 2))
    assert loss.item() < 0.05, f"matching the teacher should score ~0, got {loss.item()}"


def test_concurrent_builders_do_not_corrupt_the_cache(tmp_path):
    """Several drafts train from one dumped dataset, so two jobs can build at once.

    They must not share a scratch directory: deleting files the other still holds open
    leaves it writing to unlinked inodes and publishing an index whose offsets no longer
    match its data, which reads back as a complete cache full of nonsense.
    """
    shard = tmp_path / "a.jsonl.gz"
    _write_records(shard, n=6)
    files = [str(shard)]
    cache = tmp_path / ".sparse_cache"

    # Hold both builders at the first shard read, so each has created its scratch
    # directory before either starts writing -- the interleaving that loses data.
    barrier = threading.Barrier(2)
    real_open = gzip.open

    def synced_open(*args, **kwargs):
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=30)
        return real_open(*args, **kwargs)

    errors = []

    def build():
        try:
            sparse_data._build_cache(cache, files, None)
        except BaseException as exc:
            errors.append(exc)

    with mock.patch.object(sparse_data.gzip, "open", synced_open):
        threads = [threading.Thread(target=build) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
    assert not errors, errors

    ds = SparsePolicyDataset(files, max_length=32)
    assert len(ds) == 6
    for i in range(len(ds)):
        item = ds[i]
        scored = item["loss_mask"].bool()
        # Garbage offsets show up here: a row read from the wrong place does not
        # sum to 1, and an unwritten one is all zeros.
        assert torch.allclose(
            item["teacher_topk_prob"][scored].sum(-1),
            torch.ones(int(scored.sum())),
            atol=1e-2,
        )
    assert not list(tmp_path.glob(".sparse_cache.tmp*"))
    assert not list(tmp_path.glob(".sparse_cache.stale*"))


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


def test_tvd_ce_ignores_rows_the_dump_left_empty():
    """An empty policy row has no argmax; CE must not read that as a target of id 0.

    The ragged fallback zero-fills a short row while ``__getitem__`` still scores the
    position, and TVD only degrades to its bounded out-of-support term there. CE's
    penalty is unbounded, so an unguarded target would drag the draft toward token 0.
    """
    torch.manual_seed(0)
    seq = 4
    tok = torch.randint(1, VOCAB, (1, seq, TOPK))
    prob = torch.rand(1, seq, TOPK).softmax(-1)
    prob[0, 2] = 0.0  # the row the dump never filled
    tok[0, 2] = 0
    mask = torch.ones(1, seq)
    m = _draft(external_loss="tvd_ce")

    # neither the loss nor the accuracy may depend on what the draft predicts there.
    # b spikes on a different id: with both uniform their argmax is also id 0, so the
    # accuracies would agree even with the guard removed.
    logits = torch.randn(1, seq, VOCAB)
    a = logits.clone()
    a[0, 1] = torch.full((VOCAB,), -30.0).scatter(-1, torch.tensor([0]), 30.0)
    b = logits.clone()
    b[0, 1] = torch.full((VOCAB,), -30.0).scatter(-1, torch.tensor([1]), 30.0)
    # shift: position 2 of the policy is scored against draft position 1
    la, acc_a = m.compute_sparse_loss(a, tok, prob, mask)
    lb, acc_b = m.compute_sparse_loss(b, tok, prob, mask)
    assert torch.allclose(la, lb), f"empty row still steers the CE term: {la} vs {lb}"
    assert acc_a == acc_b, "empty row still counts toward accuracy"
    assert torch.isfinite(la)


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


@pytest.mark.parametrize(
    ("bad", "n_bad", "kept"),
    [
        # Empty policy: width inference used to raise IndexError and abort the build.
        ({"topk_tok": [], "topk_lp": []}, 1, 1),
        # topk_lp shorter than topk_tok: the ragged fallback used to index past its end.
        ({"topk_lp": [[0.0] * TOPK]}, 1, 2),
        # A renamed field drops every record it touches. One dropped record is a
        # truncation to tolerate, but a dump-wide rename is schema drift: publishing
        # whatever happened to parse would train a whole run on a handful of rows
        # against an ordinary-looking loss curve, so the drop rate has to fail it.
        ({"topk_lp": _DROP, "logprobs": [[0.0] * TOPK]}, 40, "raises"),
    ],
)
def test_malformed_policy_records_never_abort_the_build(tmp_path, bad, n_bad, kept):
    """A damaged record is dropped or truncated, never fatal to the whole pass -- until
    enough of them are damaged that the dump, not the record, is what is wrong."""
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
        for i in range(n_bad):
            record = {**good, "id": f"bad{i}", **bad}
            f.write(json.dumps({k: v for k, v in record.items() if v is not _DROP}) + "\n")
        f.write(json.dumps(good) + "\n")

    if kept == "raises":
        with pytest.raises(ValueError, match="failed to parse"):
            SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK)
        return

    ds = SparsePolicyDataset([str(p)], max_length=32, top_k=TOPK)
    assert len(ds) == kept
    # the intact record must still be intact
    assert ds[len(ds) - 1]["input_ids"].shape[0] == len(good["prompt_ids"]) + len(good["gen_ids"])


def test_sample_size_draws_across_sources_not_a_prefix(tmp_path):
    """`limit` must sample the weighted index, not take index.npy's prefix.

    index.npy is written in shard-then-line order, so a prefix is a prefix of the shard
    list: with one source per shard -- the natural layout -- it can hold a single source,
    and the configured mixture is then applied to whatever survived.
    """
    shards = []
    for name in ("alpha", "beta"):
        path = tmp_path / f"{name}.jsonl"
        with path.open("w") as fh:
            for i in range(40):
                fh.write(
                    json.dumps(
                        {
                            "id": f"{name}-{i}",
                            "source": name,
                            "prompt_ids": [1, 2],
                            "gen_ids": [3, 4],
                            "topk_tok": [[3, 5], [4, 6]],
                            "topk_lp": [[-0.1, -2.0], [-0.1, -2.0]],
                        }
                    )
                    + "\n"
                )
        shards.append(str(path))

    ds = SparsePolicyDataset(shards, max_length=64, limit=20)
    assert ds.sources is not None
    assert len(ds.sources) == len(ds), "sources desynchronised from index"
    kept = {ds.source_names[s] for s in ds.sources}
    assert kept == {"alpha", "beta"}, f"sample came from {kept}, not both shards"


def test_a_wider_row_refuses_rather_than_losing_teacher_mass(tmp_path):
    """Width comes from the first record; a later wider row must not be silently clipped.

    With top_p < 1 the stored nucleus varies per position, so a corpus whose later rows
    are wider than the first record's is ordinary -- and truncating them drops real
    teacher probability that no stored-mass check can see afterwards.
    """
    shard = tmp_path / "ragged.jsonl"
    rows = [
        {
            "id": "narrow",
            "prompt_ids": [1, 2],
            "gen_ids": [3, 4],
            "topk_tok": [[3, 5], [4, 6]],
            "topk_lp": [[-0.1, -2.0], [-0.1, -2.0]],
        },
        {
            "id": "wide",
            "prompt_ids": [1, 2],
            "gen_ids": [3, 4],
            "topk_tok": [[3, 5, 7, 9], [4, 6, 8, 10]],
            "topk_lp": [[-0.1, -2.0, -3.0, -4.0], [-0.1, -2.0, -3.0, -4.0]],
        },
    ]
    with shard.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="sized at"):
        SparsePolicyDataset([str(shard)], max_length=64)
