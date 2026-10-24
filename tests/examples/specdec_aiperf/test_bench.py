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

import json
import subprocess
import sys
from itertools import pairwise

import bench
import pytest

# Per turn: acceptance histogram (accepted drafts -> verify steps), streamed reply chunks.
_TURNS = [({"0": 1, "3": 1}, ["hel", "lo"]), ({"7": 2}, ["wor", "ld"])]


def _fake_server(stats):
    def request_json(url, payload=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "target"}]}
        if url.endswith("/v1/chat/completions"):
            return {"choices": [{"message": {"content": "1 2 3"}}], "metrics": stats}
        return {"version": "test"}

    return request_json


def _fake_aiperf(cmd, check, env, stdin):
    """Stand-in for `aiperf profile`: answers every session in the input file."""
    assert "HF_HUB_OFFLINE" not in env and "TRANSFORMERS_OFFLINE" not in env
    opt = {k: v for k, v in pairwise(cmd) if k.startswith("--")}
    if opt["--concurrency"] == "1":
        raise subprocess.CalledProcessError(1, cmd)
    extra = json.loads(opt["--extra-inputs"])
    assert extra == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 64,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"reasoning_effort": "high"},
    }
    out = opt["--output-artifact-dir"]
    with (
        open(opt["--input-file"]) as f,
        open(f"{out}/profile_export.jsonl", "w") as rec,
        open(f"{out}/profile_export_raw.jsonl", "w") as raw,
    ):
        for n, line in enumerate(f):
            session = json.loads(line)
            assert [t["text"] for t in session["turns"]] == ["q1", "q2"]
            for turn, (histogram, chunks) in enumerate(_TURNS):
                meta = {
                    "conversation_id": session["session_id"],
                    "turn_index": turn,
                    "request_start_ns": (2 * n + turn + 1) * 10**9,
                }
                metrics = {
                    "time_to_first_token": {"value": 100.0, "unit": "ms"},
                    "inter_chunk_latency": {"value": [10.0, 10.0], "unit": "ms"},
                    "output_sequence_length": {
                        "value": 1 + sum((int(j) + 1) * c for j, c in histogram.items()),
                        "unit": "tokens",
                    },
                }
                stats = {"acceptance_histogram": histogram}
                row = {"metadata": meta, "metrics": metrics, "spec_decode_acceptance": stats}
                rec.write(json.dumps(row) + "\n")
                packets = [
                    {"name": "data", "value": json.dumps({"choices": [{"delta": {"content": c}}]})}
                    for c in chunks
                ] + [{"name": "data", "value": "[DONE]"}]
                raw.write(
                    json.dumps({"metadata": meta, "responses": [{"packets": packets}]}) + "\n"
                )


def _argv(tmp_path, concurrency):
    questions = tmp_path / "question.jsonl"
    questions.write_text(
        "".join(
            json.dumps({"question_id": 81 + i, "category": c, "turns": ["q1", "q2"]}) + "\n"
            for i, c in enumerate(["writing", "math"])
        )
    )
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        "sampling_kwargs: {temperature: 0}\nchat_template_args: {reasoning_effort: high}\n"
    )
    argv = f"bench.py --url http://server:8000 --tokenizer {tmp_path} --mtbench {questions}"
    argv += f" --runtime_params {runtime} --temperature 1 --concurrency {concurrency}"
    argv += f" --output_length 64 --num_gpus 4 --save_dir {tmp_path / 'results'}"
    return argv.split()


def test_sweep_writes_specdec_bench_files_and_survives_a_failed_point(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    stats = {"speculative_decoding": {"num_spec_steps": 9}}
    monkeypatch.setattr(bench, "_request_json", _fake_server(stats))
    monkeypatch.setattr(bench.subprocess, "run", _fake_aiperf)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "1 2"))

    with pytest.raises(SystemExit, match="t1_c1"):
        bench.main()

    out = tmp_path / "results" / "t1_c2"
    acceptance = json.loads((out / "acceptance_rate.json").read_text())[0]
    steps = [1, 1, 4, 1, 8, 8]  # the prefill token, then 1 + accepted drafts per verify step
    assert acceptance["Average_AL"] == pytest.approx(sum(steps) / len(steps))
    assert list(acceptance["Request_AL"]) == ["0", "1"]  # MT-Bench keys by request index
    assert acceptance["Category_AL"].keys() == {"writing", "math"}
    timing = json.loads((out / "timing.json").read_text())[0]
    # 4 requests starting at 1..4 s, each ending 120 ms later.
    assert timing["Output TPS"] == pytest.approx(2 * sum(steps) / (4.12 - 1.0))
    assert timing["Output TPS/gpu"] == pytest.approx(timing["Output TPS"] / 4)
    responses = [json.loads(line) for line in open(out / "mtbench_responses.jsonl")]
    assert [r["choices"][0]["turns"] for r in responses] == [["hello", "world"]] * 2


def test_server_without_spec_decode_stats_fails_before_aiperf(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_request_json", _fake_server(None))
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: pytest.fail("AIPerf ran"))
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "2"))

    with pytest.raises(RuntimeError, match="per-request-spec-decode-metrics"):
        bench.main()


def test_matches_specdec_bench(tmp_path):
    datasets = pytest.importorskip("datasets")
    from specdec_bench import metrics
    from specdec_bench.datasets.speed import SPEEDBench

    # SPEED-Bench subsampling picks the same rows.
    rows = [{"question_id": i, "category": "abc"[i % 7 % 3], "turns": ["x"]} for i in range(20)]
    reference = SPEEDBench._stratified_select(datasets.Dataset.from_list(rows), 8)
    assert [r["question_id"] for r in bench._stratified(rows, 8)] == reference["question_id"]

    # Acceptance and timing fields match on the same steps and timestamps.
    requests = [
        bench.Request(i, 81 + i, "writing" if i < 10 else "roleplay", None, ["a", "b"])
        for i in range(12)
    ]
    results = [
        [
            bench.TurnResult(
                [1, 1 + i % 4, 8], [2.0 * i, 2 * i + 0.2, 2 * i + 0.25], 9 + i % 5, ""
            ),
            bench.TurnResult(
                [1, 2, 3 + i % 2, 8], [2 * i + 1.0, 2 * i + 1.3, 2 * i + 1.31], 12, ""
            ),
        ]
        for i in range(12)
    ]
    acceptance, timing = metrics.MTBench(), metrics.Timing(4)
    for metric in (acceptance, timing):
        metric.update_directory(str(tmp_path))
    for i, turns in enumerate(results):
        for turn, result in enumerate(turns):
            chunks = [[0] * n for n in result.steps]
            acceptance.process_step({"output_ids": [chunks]}, i, turn)
            tokens = [[[0] * result.num_tokens]]
            timing.process_step({"output_ids": tokens, "token_times": result.times}, i, turn)
    texts = [[{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}]] * 12
    acceptance.process_final(texts)
    timing.process_final(texts)

    ours = bench.acceptance_metrics(requests, results)
    for key, value in acceptance.out.items():
        assert ours[key] == pytest.approx(value), key
    ours = bench.timing_metrics(results, 4)
    assert ours.keys() == timing.out.keys()
    for key, value in timing.out.items():
        assert ours[key] == (pytest.approx(value) if isinstance(value, float) else value), key
