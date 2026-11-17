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

"""CPU stage differentials; only the vLLM/model boundary is mocked."""

import json
import os
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_collect_mask_reuse_cli as fixtures

from modelopt.torch.sparsity.attention_sparsity.calibration.staged_capture import (
    CaptureStageFiles,
    verify_controller_bridge,
)

resume_case = fixtures.resume_case
whole_resume_case = fixtures.whole_resume_case
collector = fixtures.collect_mask_reuse_cli


def _mutate_journal(path, mutation):
    payload, before = path.read_bytes(), path.stat()
    if mutation == "symlink":
        saved = path.with_suffix(".saved")
        path.rename(saved)
        path.symlink_to(saved)
    elif mutation == "replace":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(payload + b" ")
        replacement.replace(path)
    elif mutation == "noncanonical":
        path.write_bytes(
            b"".join(json.dumps(json.loads(line)).encode() + b"\n" for line in payload.splitlines())
        )
    elif mutation == "valid_changed_record":
        records = [json.loads(line) for line in payload.splitlines()]
        records[0]["manifest"]["candidate_cell_count"] += 1
        previous = records[0]["previous_sha256"]
        for record in records:
            record["previous_sha256"] = previous
            unsigned = {k: v for k, v in record.items() if k != "record_sha256"}
            previous = sha256(collector._canonical_capture_line(unsigned)).hexdigest()
            record["record_sha256"] = previous
        path.write_bytes(b"".join(collector._canonical_capture_line(record) for record in records))
    else:
        path.write_bytes(
            {
                "append": payload + b" ",
                "truncate": payload[:-1],
                "same_size": b"[" + payload[1:],
                "same_size_restored_mtime": b"[" + payload[1:],
            }[mutation]
        )
        if mutation == "same_size_restored_mtime":
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


@pytest.fixture
def validated_journal(tmp_path):
    request, invocations = {"fixture": "private CPU journal"}, [{"index": 0}]
    journal = collector._CaptureJournal(tmp_path / "journal", request, invocations)
    journal.append(
        {"invocation": invocations[0]},
        {"invocation": invocations[0], "candidate_cell_count": 1},
    )
    validated = collector._CaptureJournal.existing(journal.directory, request, invocations)
    validated.count, validated.previous = journal.count, journal.previous
    return validated


def _next_phase(args, phase, receipt):
    args.phase = phase
    args.resume_from = None
    args.source_bridge = args.source_bridge_sha256 = None
    args.stage_receipt = receipt
    args.stage_receipt_sha256 = sha256(receipt.read_bytes()).hexdigest()
    return collector.run(args)


@pytest.mark.parametrize("interrupted", [False, True])
def test_staged_matches_monolithic_bytes(whole_resume_case, tmp_path, monkeypatch, interrupted):
    args = whole_resume_case
    args.output = str(tmp_path / "monolithic/captures.jsonl")
    expected, expected_manifest = collector.run(args)
    journal_bytes = (expected.parent / "capture.progress.jsonl").read_bytes()
    args.output = str(tmp_path / "staged/captures.jsonl")
    args.phase = "prepare"
    before = len(fixtures._LLM.instances)
    _, prepared = collector.run(args)
    assert len(fixtures._LLM.instances) == before
    monkeypatch.setattr(fixtures._StreamLLM, "fail_after_chunk", 0 if interrupted else None)
    if interrupted:
        with pytest.raises(RuntimeError, match="interrupted"):
            _next_phase(args, "collect", prepared)
        collected = prepared.parent / "capture.collected.json"
    else:
        _, collected = _next_phase(args, "collect", prepared)
    assert not Path(args.output).exists()
    before = len(fixtures._LLM.instances)
    actual, manifest = _next_phase(args, "finalize", collected)
    assert len(fixtures._LLM.instances) == before
    if interrupted:
        assert manifest.name == "capture.partial.json"
        prior = prepared.parent
        args.phase = "prepare"
        args.stage_receipt = args.stage_receipt_sha256 = None
        args.resume_from = prior
        args.output = str(tmp_path / "resumed/captures.jsonl")
        _, prepared = collector.run(args)
        monkeypatch.setattr(fixtures._StreamLLM, "fail_after_chunk", None)
        _, collected = _next_phase(args, "collect", prepared)
        actual, manifest = _next_phase(args, "finalize", collected)
    assert actual.read_bytes() == expected.read_bytes()
    assert manifest.read_bytes() == expected_manifest.read_bytes()
    assert (actual.parent / "capture.progress.jsonl").read_bytes() == journal_bytes
    assert json.loads(manifest.read_bytes())["whole_prefill"]["dense_exclusion_count"] == 4


@pytest.mark.parametrize(
    "mutation", ["journal", "normalized", "request", "receipt_pin", "runtime", "targets"]
)
def test_stage_rejects_changed_prepared_input(whole_resume_case, tmp_path, monkeypatch, mutation):
    args = whole_resume_case
    args.phase = "prepare"
    _, receipt = collector.run(args)
    args.phase = "collect"
    args.stage_receipt = receipt
    args.stage_receipt_sha256 = sha256(receipt.read_bytes()).hexdigest()
    if mutation == "receipt_pin":
        args.stage_receipt_sha256 = "0" * 64
    elif mutation == "targets":
        args.target_sparsities = [0.5]
    elif mutation == "runtime":
        # Independently test a trusted producer receipt with a conflicting expected runtime.
        data = json.loads(receipt.read_bytes())
        data["expected_runtime"] = {"kv_page_tokens": 4160}
        receipt.write_bytes(collector._canonical_capture_line(data))
        args.stage_receipt_sha256 = sha256(receipt.read_bytes()).hexdigest()
    else:
        name = {
            "journal": "capture.progress.jsonl",
            "normalized": CaptureStageFiles.NORMALIZED,
            "request": "capture.request.json",
        }[mutation]
        with (receipt.parent / name).open("ab") as handle:
            handle.write(b" ")
    before = len(fixtures._LLM.instances)
    with pytest.raises(ValueError):
        collector.run(args)
    assert len(fixtures._LLM.instances) == before + int(mutation == "runtime")
    assert not (receipt.parent / "capture.complete.json").exists()


def test_migration_requires_exact_controller_bridge(whole_resume_case, tmp_path, monkeypatch):
    args = whole_resume_case
    args.output = str(tmp_path / "old/captures.jsonl")
    old_output, _ = collector.run(args)
    old_request = json.loads((old_output.parent / "capture.request.json").read_bytes())
    original_identity = collector._controller_identity
    changed = deepcopy(original_identity())
    changed["files"]["examples/vllm_serve/collect_mask_reuse.py"] = "f" * 64
    monkeypatch.setattr(collector, "_controller_identity", lambda: deepcopy(changed))
    args.phase = "prepare"
    args.resume_from = old_output.parent
    args.output = str(tmp_path / "new/captures.jsonl")
    with pytest.raises(ValueError, match="pinned source bridge"):
        collector.run(args)
    inputs = collector._prepare_capture_inputs(args)
    request, _ = collector._capture_request(args, inputs, changed)
    path = "examples/vllm_serve/collect_mask_reuse.py"
    bridge = {
        "schema_version": 1,
        "old_request_sha256": sha256(collector._canonical_capture_line(old_request)).hexdigest(),
        "new_request_sha256": sha256(collector._canonical_capture_line(request)).hexdigest(),
        "controller_file_changes": [
            {
                "path": path,
                "before": old_request["controller"]["files"][path],
                "after": changed["files"][path],
            }
        ],
    }
    args.source_bridge = tmp_path / "bridge.json"
    args.source_bridge.write_bytes(collector._canonical_capture_line(bridge))
    args.source_bridge_sha256 = sha256(args.source_bridge.read_bytes()).hexdigest()
    _, receipt = collector.run(args)
    normalized = receipt.parent / CaptureStageFiles.NORMALIZED
    assert normalized.read_bytes() == old_output.read_bytes()
    new_journal = (receipt.parent / "capture.progress.jsonl").read_bytes()
    assert new_journal != (old_output.parent / "capture.progress.jsonl").read_bytes()
    assert (
        json.loads(receipt.read_bytes())["migration"]["bridge_receipt_sha256"]
        == args.source_bridge_sha256
    )
    altered = deepcopy(request)
    altered["engine_kwargs"]["seed"] = 999
    with pytest.raises(ValueError, match="capture semantics"):
        verify_controller_bridge(
            old_request,
            altered,
            args.source_bridge,
            args.source_bridge_sha256,
            parse=collector._strict_progress_json,
            serialize=collector._canonical_capture_line,
        )


def test_collection_contract_failure_cannot_finalize(whole_resume_case, monkeypatch):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)
    monkeypatch.setattr(fixtures._StreamLLM, "tamper_completion", True)
    with pytest.raises(ValueError, match="completion receipt"):
        _next_phase(args, "collect", prepared)
    collected = prepared.parent / "capture.collected.json"
    with pytest.raises(ValueError, match="runtime contract"):
        _next_phase(args, "finalize", collected)
    assert not (prepared.parent / "capture.partial.json").exists()


def test_prepare_recovers_partial_spool_without_model_and_collect_does_not_replay(
    whole_resume_case, tmp_path, monkeypatch
):
    args = whole_resume_case
    monkeypatch.setattr(fixtures._StreamLLM, "fail_after_chunk", 0)
    with pytest.raises(RuntimeError, match="interrupted"):
        collector.run(args)
    prior = Path(args.output).parent
    # Simulate a process kill between durable TP writes and journal commitment.
    (prior / "capture.progress.jsonl").write_bytes(b"")
    (prior / "capture.failure.json").unlink()
    spool = next((prior / "capture-spools").glob("*/rank-0.jsonl"))
    with spool.open("ab") as handle:
        handle.write(b'{"interrupted":')
    before = {p: p.read_bytes() for p in prior.rglob("*") if p.is_file()}
    args.phase = "prepare"
    args.resume_from = prior
    args.output = str(tmp_path / "prepared/captures.jsonl")
    models = len(fixtures._LLM.instances)
    _, prepared = collector.run(args)
    assert len(fixtures._LLM.instances) == models
    assert json.loads(prepared.read_bytes())["completed_capture_count"] == 1
    assert all(p.read_bytes() == value for p, value in before.items())
    monkeypatch.setattr(fixtures._StreamLLM, "fail_after_chunk", None)
    with monkeypatch.context() as context:

        def forbidden(*args, **kwargs):
            raise AssertionError("GPU stage parsed inherited journal or rebuilt stream plan")

        context.setattr(collector._CaptureJournal, "_records", forbidden)
        context.setattr(collector, "build_capture_stream", forbidden)
        _, collected = _next_phase(args, "collect", prepared)
    result, manifest = _next_phase(args, "finalize", collected)
    assert len(result.read_bytes().splitlines()) == 8
    assert json.loads(manifest.read_bytes())["whole_prefill"]["dense_exclusion_count"] == 4


def test_noncanonical_prior_is_canonicalized_without_overwriting(whole_resume_case, tmp_path):
    args = whole_resume_case
    output, _ = collector.run(args)
    prior = output.parent
    journal = prior / "capture.progress.jsonl"
    canonical = journal.read_bytes()
    noncanonical = b"".join(
        json.dumps(json.loads(line), sort_keys=False).encode() + b"\n"
        for line in canonical.splitlines()
    )
    journal.write_bytes(noncanonical)
    receipt = prior / "capture.complete.json"
    value = json.loads(receipt.read_bytes())
    value["journal_sha256"] = sha256(noncanonical).hexdigest()
    receipt.write_bytes(collector._canonical_capture_line(value))
    args.phase = "prepare"
    args.resume_from = prior
    args.output = str(tmp_path / "canonicalized/captures.jsonl")
    normalized, prepared = collector.run(args)
    assert journal.read_bytes() == noncanonical
    assert (prepared.parent / "capture.progress.jsonl").read_bytes() == canonical
    assert normalized.read_bytes() == output.read_bytes()


def test_model_load_failure_is_not_a_resumable_runtime_prefix(whole_resume_case, monkeypatch):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)

    def fail_load(**kwargs):
        raise RuntimeError("model load failed before runtime binding")

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=fail_load, SamplingParams=fixtures._SamplingParams)
    )
    with pytest.raises(RuntimeError, match="model load failed"):
        _next_phase(args, "collect", prepared)
    collected = prepared.parent / "capture.collected.json"
    receipt = json.loads(collected.read_bytes())
    assert receipt["runtime_bound"] is False and receipt["resumable_prefix"] is False
    assert not (prepared.parent / "capture.runtime.json").exists()
    with pytest.raises(ValueError, match="no validated runtime"):
        _next_phase(args, "finalize", collected)


def test_finalization_recovers_interrupted_normalized_append(whole_resume_case, monkeypatch):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)

    def interrupted_append(self, capture):
        with self.normalized.open("ab") as handle:
            handle.write(b'{"partial":')
        raise OSError("interrupted normalized append")

    with monkeypatch.context() as context:
        context.setattr(CaptureStageFiles, "append_normalized", interrupted_append)
        with pytest.raises(OSError, match="interrupted normalized"):
            _next_phase(args, "collect", prepared)
    collected = prepared.parent / "capture.collected.json"
    original = (prepared.parent / CaptureStageFiles.NORMALIZED).read_bytes()
    normalized, partial = _next_phase(args, "finalize", collected)
    assert partial.name == "capture.partial.json"
    assert normalized.name == "capture.normalized.recovered.jsonl"
    assert (prepared.parent / CaptureStageFiles.NORMALIZED).read_bytes() == original
    assert len(normalized.read_bytes().splitlines()) == 1
    sidecar = json.loads((prepared.parent / "capture.finalized.json").read_bytes())
    assert sidecar["typed_receipt_sha256"] == sha256(partial.read_bytes()).hexdigest()
    assert sidecar["collected_receipt_sha256"] == sha256(collected.read_bytes()).hexdigest()


def test_collect_budget_includes_pre_model_preparation(whole_resume_case, monkeypatch):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)
    args.max_wall_seconds = 10
    real_prepare = collector._prepare_capture_inputs
    clock = [100.0]

    def slow_prepare(*a, **kw):
        result = real_prepare(*a, **kw)
        clock[0] += 11
        return result

    monkeypatch.setattr(collector, "_prepare_capture_inputs", slow_prepare)
    monkeypatch.setattr(collector.time, "monotonic", lambda: clock[0])
    before = len(fixtures._LLM.instances)
    with pytest.raises(ValueError, match="budget exhausted before engine load"):
        _next_phase(args, "collect", prepared)
    assert len(fixtures._LLM.instances) == before
    assert not (prepared.parent / "capture.collection-started.json").exists()


def test_retry_preserves_interrupted_cpu_recovery(whole_resume_case, monkeypatch):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)

    def interrupted_append(self, capture):
        with self.normalized.open("ab") as handle:
            handle.write(b'{"partial":')
        raise OSError("interrupted normalized append")

    with monkeypatch.context() as context:
        context.setattr(CaptureStageFiles, "append_normalized", interrupted_append)
        with pytest.raises(OSError, match="interrupted normalized"):
            _next_phase(args, "collect", prepared)
    collected = prepared.parent / "capture.collected.json"
    real_publish = CaptureStageFiles.publish_derived

    def interrupted_recovery(self, path, chunks, expected_sha256):
        if Path(path).name == "capture.normalized.recovered.jsonl":

            def partial_chunks():
                yield next(iter(chunks))[:20]
                raise OSError("CPU recovery interrupted")

            return real_publish(self, path, partial_chunks(), expected_sha256)
        return real_publish(self, path, chunks, expected_sha256)

    with monkeypatch.context() as context:
        context.setattr(CaptureStageFiles, "publish_derived", interrupted_recovery)
        with pytest.raises(OSError, match="CPU recovery interrupted"):
            _next_phase(args, "finalize", collected)
    failed = list(prepared.parent.glob(".capture.normalized.recovered.jsonl.*.partial"))
    assert len(failed) == 1 and len(failed[0].read_bytes()) == 20
    failed_bytes = failed[0].read_bytes()
    actual, typed = _next_phase(args, "finalize", collected)
    assert typed.name == "capture.partial.json"
    assert len(actual.read_bytes().splitlines()) == 1
    assert failed[0].read_bytes() == failed_bytes
    # Reuse only complete, exact recovered bytes; the retry is idempotent.
    assert _next_phase(args, "finalize", collected) == (actual, typed)


@pytest.mark.parametrize("partial", [False, True])
def test_retry_after_typed_receipt_before_sidecar(whole_resume_case, monkeypatch, partial):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)
    monkeypatch.setattr(fixtures._StreamLLM, "fail_after_chunk", 0 if partial else None)
    if partial:
        with pytest.raises(RuntimeError, match="interrupted"):
            _next_phase(args, "collect", prepared)
        collected = prepared.parent / "capture.collected.json"
    else:
        _, collected = _next_phase(args, "collect", prepared)
    real_write = CaptureStageFiles.write_derived

    def fail_before_sidecar(self, name, value):
        if name == "capture.finalized.json":
            raise OSError("interrupted before finalized sidecar")
        return real_write(self, name, value)

    with monkeypatch.context() as context:
        context.setattr(CaptureStageFiles, "write_derived", fail_before_sidecar)
        with pytest.raises(OSError, match="before finalized sidecar"):
            _next_phase(args, "finalize", collected)
    typed = prepared.parent / f"capture.{'partial' if partial else 'complete'}.json"
    original = typed.read_bytes()
    assert not (prepared.parent / "capture.finalized.json").exists()
    result = _next_phase(args, "finalize", collected)
    assert typed.read_bytes() == original
    sidecar = json.loads((prepared.parent / "capture.finalized.json").read_bytes())
    assert sidecar["typed_receipt_sha256"] == sha256(original).hexdigest()
    assert _next_phase(args, "finalize", collected) == result


@pytest.mark.parametrize(
    "path",
    [
        "modelopt/torch/sparsity/attention_sparsity/calibration/mask_reuse.py",
        "modelopt/torch/sparsity/attention_sparsity/plugins/mask_reuse_capture.py",
    ],
)
def test_pinned_bridge_cannot_authorize_non_host_change(tmp_path, path):
    old = {"controller": {"versions": {}, "files": {path: "1" * 64}}, "invocations": []}
    new = deepcopy(old)
    new["controller"]["files"][path] = "2" * 64
    bridge = {
        "schema_version": 1,
        "old_request_sha256": sha256(collector._canonical_capture_line(old)).hexdigest(),
        "new_request_sha256": sha256(collector._canonical_capture_line(new)).hexdigest(),
        "controller_file_changes": [{"path": path, "before": "1" * 64, "after": "2" * 64}],
    }
    location = tmp_path / "bridge.json"
    location.write_bytes(collector._canonical_capture_line(bridge))
    with pytest.raises(ValueError, match="non-host controller changes"):
        verify_controller_bridge(
            old,
            new,
            location,
            sha256(location.read_bytes()).hexdigest(),
            parse=collector._strict_progress_json,
            serialize=collector._canonical_capture_line,
        )


@pytest.mark.parametrize("mode", ["complete", "partial", "empty"])
def test_finalization_uses_one_parsed_pass_and_fresh_stable_hash(
    whole_resume_case, monkeypatch, mode
):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)
    monkeypatch.setattr(fixtures._StreamLLM, "fail_after_chunk", 0 if mode == "partial" else None)
    if mode == "empty":

        def interrupted(*a, **kw):
            raise RuntimeError("interrupted before first chunk")

        monkeypatch.setattr(fixtures._StreamLLM, "generate", interrupted)
    if mode != "complete":
        with pytest.raises(RuntimeError, match="interrupted"):
            _next_phase(args, "collect", prepared)
        collected = prepared.parent / "capture.collected.json"
    else:
        _, collected = _next_phase(args, "collect", prepared)
    calls = {"parsed": 0, "verify": 0, "inherited": 0, "stable": 0}
    records = collector._CaptureJournal._records
    verify = CaptureStageFiles.verify
    inherited = CaptureStageFiles.verify_inherited_bytes
    stable = collector.stable_file_sha256

    def counted_records(path):
        calls["parsed"] += 1
        yield from records(path)

    def counted_verify(self, *a, **kw):
        calls["verify"] += 1
        return verify(self, *a, **kw)

    def counted_inherited(self, *a, **kw):
        calls["inherited"] += 1
        return inherited(self, *a, **kw)

    def counted_stable(path, **kw):
        calls["stable"] += 1
        assert Path(path) == prepared.parent / "capture.progress.jsonl"
        return stable(path, **kw)

    monkeypatch.setattr(collector._CaptureJournal, "_records", staticmethod(counted_records))
    monkeypatch.setattr(CaptureStageFiles, "verify", counted_verify)
    monkeypatch.setattr(CaptureStageFiles, "verify_inherited_bytes", counted_inherited)
    monkeypatch.setattr(collector, "stable_file_sha256", counted_stable)
    models = len(fixtures._LLM.instances)
    _next_phase(args, "finalize", collected)
    assert len(fixtures._LLM.instances) == models
    assert calls == {"parsed": 1, "verify": 2, "inherited": 1, "stable": 1}


@pytest.mark.parametrize(
    ("boundary", "name", "mutation"),
    [
        ("summary", "capture.progress.jsonl", "append"),
        ("summary", "capture.progress.jsonl", "same_size"),
        ("summary", CaptureStageFiles.NORMALIZED, "same_size"),
        ("recheck", "capture.progress.jsonl", "replace"),
        ("recheck", CaptureStageFiles.NORMALIZED, "truncate"),
        ("recheck", CaptureStageFiles.NORMALIZED, "symlink"),
        ("finish", "capture.progress.jsonl", "same_size"),
        ("finish", "capture.progress.jsonl", "append"),
        ("finish", "capture.progress.jsonl", "truncate"),
        ("finish", "capture.progress.jsonl", "replace"),
        ("finish", "capture.progress.jsonl", "symlink"),
        ("finish", "capture.progress.jsonl", "same_size_restored_mtime"),
        ("finish", "capture.progress.jsonl", "noncanonical"),
        ("finish", "capture.progress.jsonl", "valid_changed_record"),
    ],
)
def test_fused_finalization_retains_mutation_rejection(
    whole_resume_case, monkeypatch, boundary, name, mutation
):
    args = whole_resume_case
    args.phase = "prepare"
    _, prepared = collector.run(args)
    monkeypatch.setattr(fixtures._StreamLLM, "fail_after_chunk", 0)
    with pytest.raises(RuntimeError, match="interrupted"):
        _next_phase(args, "collect", prepared)
    collected = prepared.parent / "capture.collected.json"

    def mutate():
        _mutate_journal(prepared.parent / name, mutation)

    if boundary == "summary":
        validate = collector._CaptureJournal._validate_prefix

        def after_summary(self, *a, **kw):
            result = validate(self, *a, **kw)
            if kw.get("on_record") is not None:
                mutate()
            return result

        monkeypatch.setattr(collector._CaptureJournal, "_validate_prefix", after_summary)
    elif boundary == "recheck":
        verify = collector._verify_capture_inputs

        def after_inputs(*a, **kw):
            result = verify(*a, **kw)
            mutate()
            return result

        monkeypatch.setattr(collector, "_verify_capture_inputs", after_inputs)
    else:
        finish = collector._CaptureJournal.finish

        def before_finish(self, *a, **kw):
            mutate()
            return finish(self, *a, **kw)

        monkeypatch.setattr(collector._CaptureJournal, "finish", before_finish)
    with pytest.raises(ValueError):
        _next_phase(args, "finalize", collected)
    for terminal in ("capture.partial.json", "capture.complete.json", "capture.finalized.json"):
        assert not (prepared.parent / terminal).exists()


@pytest.mark.parametrize(
    "field",
    [
        "path",
        "request",
        "request_sha",
        "invocations",
        "resume_count",
        "resume_digest",
        "count",
        "previous",
    ],
)
def test_stable_finish_rejects_mismatched_in_process_identity(validated_journal, field):
    journal = validated_journal
    if field == "path":
        replacement = journal.path.with_suffix(".copy")
        replacement.write_bytes(journal.path.read_bytes())
        journal.path = replacement
    elif field == "request":
        journal.request.update(changed=True)
    elif field == "invocations":
        journal.invocations.append({"index": 1})
    elif field in ("count", "resume_count"):
        setattr(journal, field, getattr(journal, field) + 1)
    else:
        setattr(journal, field, "0" * 64)
    with pytest.raises(ValueError, match=r"validated prefix|in-process prefix"):
        journal.finish(
            "complete", identity_verified=True, stage="collection", _validated_bytes=True
        )
    assert not (journal.directory / "capture.complete.json").exists()


@pytest.mark.parametrize("mode", ["fresh", "prepared", "failed_revalidation"])
def test_stable_finish_requires_successful_local_full_scan(validated_journal, tmp_path, mode):
    old = validated_journal
    if mode == "fresh":
        journal = collector._CaptureJournal(tmp_path / "fresh", old.request, old.invocations)
    elif mode == "prepared":
        # A durable receipt, even with matching digest/count, is not a local scan.
        journal = collector._CaptureJournal.existing(
            old.directory,
            old.request,
            old.invocations,
            prepared={
                "completed_capture_count": old.count,
                "chain_tail_sha256": old.previous,
                "validated_prefix": list(map(str, old._validated_prefix)),
            },
        )
    else:
        journal = old
        journal.path.write_bytes(b"invalid\n")
        with pytest.raises(ValueError):
            journal._validate_prefix(journal.path)
    with pytest.raises(ValueError, match="in-process prefix"):
        journal.finish(
            "complete", identity_verified=True, stage="collection", _validated_bytes=True
        )
    assert not (journal.directory / "capture.complete.json").exists()


def test_monolithic_finish_keeps_semantic_revalidation(validated_journal, monkeypatch):
    journal = validated_journal
    scans = []
    records = collector._CaptureJournal._records

    def counted(path):
        scans.append(path)
        yield from records(path)

    def forbidden(*a, **kw):
        pytest.fail("ordinary finish must not use the staged byte shortcut")

    monkeypatch.setattr(collector._CaptureJournal, "_records", staticmethod(counted))
    monkeypatch.setattr(collector, "stable_file_sha256", forbidden)
    journal.finish("complete", identity_verified=True, stage="collection")
    assert scans == [journal.path]


@pytest.mark.parametrize("mutation", ["append", "replace", "symlink", "same_size_restored_mtime"])
def test_stable_finish_rejects_mutation_during_hash(validated_journal, monkeypatch, mutation):
    from modelopt.torch.sparsity.attention_sparsity.calibration import checkpoint_manifest

    journal = validated_journal
    read = checkpoint_manifest.os.read
    injected = []

    def altered_read(fd, size):
        if not injected:
            injected.append(True)
            _mutate_journal(journal.path, mutation)
        return read(fd, size)

    monkeypatch.setattr(checkpoint_manifest.os, "read", altered_read)
    with pytest.raises(ValueError):
        journal.finish(
            "complete", identity_verified=True, stage="collection", _validated_bytes=True
        )
    assert injected and not (journal.directory / "capture.complete.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "append",
        "truncate",
        "replace",
        "symlink",
        "same_size_restored_mtime",
        "noncanonical",
        "valid_changed_record",
    ],
)
def test_prepare_rechecks_prior_exact_bytes_at_last_boundary(
    whole_resume_case, tmp_path, monkeypatch, mutation
):
    args = whole_resume_case
    collector.run(args)
    prior = Path(args.output).parent
    args.phase, args.resume_from = "prepare", prior
    args.output = str(tmp_path / "prepared-next/captures.jsonl")
    recheck = collector._CaptureJournal._recheck_validated_prefix

    def altered(self):
        assert self.path == prior / "capture.progress.jsonl"
        _mutate_journal(self.path, mutation)
        return recheck(self)

    monkeypatch.setattr(collector._CaptureJournal, "_recheck_validated_prefix", altered)
    with pytest.raises(ValueError):
        collector.run(args)
    assert not (Path(args.output).parent / "capture.prepared.json").exists()
