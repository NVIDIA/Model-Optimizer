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
from itertools import pairwise
from types import SimpleNamespace

import pytest
from specdec_bench import aiperf_client, datasets, metrics
from specdec_bench.models.vllm import VLLMModel


def test_serve_command_matches_in_process_engine():
    engine_kwargs = VLLMModel.engine_kwargs(
        "/ckpt/target",
        32,
        speculative_algorithm="DFLASH",
        draft_model_dir="/ckpt/draft",
        speculative_num_draft_tokens=7,
        draft_sample_method="probabilistic",
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        tokenizer_path="/ckpt/target",
    )
    cmd = aiperf_client.vllm_serve_command(
        engine_kwargs, 8123, ["--max-num-batched-tokens", "8192"]
    )

    assert cmd[:5] == ["vllm", "serve", "/ckpt/target", "--port", "8123"]
    assert cmd[cmd.index("--per-request-spec-decode-metrics") + 1] == "summary"
    assert json.loads(cmd[cmd.index("--speculative-config") + 1]) == {
        "method": "dflash",
        "model": "/ckpt/draft",
        "num_speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
    }
    assert cmd[cmd.index("--max-num-seqs") + 1] == "224"
    for flag in ("--no-enable-prefix-caching", "--enable-expert-parallel", "--async-scheduling"):
        assert flag in cmd
    assert "--max-model-len" not in cmd
    assert cmd[-2:] == ["--max-num-batched-tokens", "8192"]


# Per turn: acceptance histogram, streamed reply chunks.
_TURNS = [({"0": 1, "3": 1}, ["hel", "lo"]), ({"7": 2}, ["wor", "ld"])]


def _fake_aiperf(cmd, check, env, stdin):
    """Stand-in for `aiperf profile`: answers every session in the input file."""
    assert "HF_HUB_OFFLINE" not in env and "TRANSFORMERS_OFFLINE" not in env
    opt = {k: v for k, v in pairwise(cmd) if k.startswith("--")}
    extra = json.loads(opt["--extra-inputs"])
    assert extra["temperature"] == 1 and extra["top_k"] == 0 and extra["max_tokens"] == 64
    assert extra["chat_template_kwargs"] == {"reasoning_effort": "high"}
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
                num_tokens = 1 + sum((int(j) + 1) * c for j, c in histogram.items())
                metrics_ = {
                    "time_to_first_token": {"value": 100.0, "unit": "ms"},
                    "inter_chunk_latency": {"value": [10.0, 10.0], "unit": "ms"},
                    "output_sequence_length": {"value": num_tokens, "unit": "tokens"},
                }
                stats = {"acceptance_histogram": histogram}
                row = {"metadata": meta, "metrics": metrics_, "spec_decode_acceptance": stats}
                rec.write(json.dumps(row) + "\n")
                packets = [
                    {"name": "data", "value": json.dumps({"choices": [{"delta": {"content": c}}]})}
                    for c in chunks
                ] + [{"name": "data", "value": "[DONE]"}]
                raw.write(
                    json.dumps({"metadata": meta, "responses": [{"packets": packets}]}) + "\n"
                )


def test_run_reports_in_run_py_format(tmp_path, monkeypatch):
    questions = tmp_path / "question.jsonl"
    questions.write_text("".join(json.dumps({"turns": ["q1", "q2"]}) + "\n" for _ in range(2)))
    dataset = datasets.MTBench(str(questions))
    metrics_list = [metrics.MTBench(), metrics.Timing(4)]
    for metric in metrics_list:
        metric.update_directory(str(tmp_path))
    args = SimpleNamespace(
        save_dir=str(tmp_path),
        output_length=64,
        ignore_eos=False,
        aiperf_bin="aiperf",
        aiperf_url="http://server:8000",
        tokenizer=str(tmp_path),
        concurrency=2,
        trust_remote_code=False,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(aiperf_client, "server_info", lambda url: {"served_model": "target"})
    monkeypatch.setattr(aiperf_client.subprocess, "run", _fake_aiperf)

    aiperf_client.run(args, dataset, metrics_list, {"temperature": 1}, {"reasoning_effort": "high"})

    # Counted like run.py's stream chunks: the prefill token, then 1 + accepted per verify step.
    steps = [1, 1, 4, 1, 8, 8]
    acceptance = json.loads((tmp_path / "acceptance_rate.json").read_text())[-1]
    assert acceptance["Average_AL"] == pytest.approx(sum(steps) / len(steps))
    responses = [json.loads(line) for line in open(tmp_path / "mtbench_responses.jsonl")]
    assert [r["choices"][0]["turns"] for r in responses] == [["hello", "world"]] * 2
    timing = json.loads((tmp_path / "timing.json").read_text())[-1]
    # 4 requests starting at 1..4 s, each ending 120 ms later.
    assert timing["Output TPS"] == pytest.approx(2 * sum(steps) / (4.12 - 1.0))
