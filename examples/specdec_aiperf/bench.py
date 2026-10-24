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

"""Measure speculative-decoding acceptance length and throughput of a served model with AIPerf.

Results use specdec_bench's file names, fields and acceptance-length definition, so they can be
compared with in-process specdec_bench runs.
"""

import argparse
import contextlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import traceback
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path

import numpy as np
import yaml

MTBENCH_TOPICS = [
    "writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities",
]  # fmt: skip
# What examples/specdec_bench/prepare_data.py leaves in SPEED-Bench turns it could not resolve.
SPEED_PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"


@dataclass(frozen=True)
class Request:
    key: int | str  # this request's Request_AL key, as specdec_bench writes it for the dataset
    question_id: int | str
    category: str | None
    system_prompt: str | None
    turns: list[str]


@dataclass(frozen=True)
class TurnResult:
    steps: list[int]  # tokens per step, counted like specdec_bench's streamed chunks
    times: list[float]  # request start, first token, then every later chunk, in seconds
    num_tokens: int
    reply: str


@dataclass(frozen=True)
class DatasetSpec:
    read_rows: Callable[[str, int | None], list[dict]]  # (path, num_requests) -> rows in run order
    responses_file: str
    key_by_index: bool = (
        False  # MT-Bench keys Request_AL by request index, the others by question_id
    )


@dataclass(frozen=True)
class Setup:
    spec: DatasetSpec
    requests: list[Request]
    model: str
    tokenizer: str
    temperatures: list[float]
    request_fields: dict  # sent with every request, besides the temperature


# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------


def _read_jsonl(path, num_requests=None):
    with open(path) as f:
        return [json.loads(line) for line in f][:num_requests]


def _read_mtbench(path, num_requests):
    rows = _read_jsonl(path, num_requests)
    for i, row in enumerate(rows):
        row.setdefault("category", MTBENCH_TOPICS[i // 10])
    return rows


def _stratified(rows, n):
    """Pick ``n`` rows round-robin across categories, as specdec_bench samples SPEED-Bench."""
    by_category = defaultdict(list)
    for row in rows:
        by_category[row.get("category")].append(row)
    if n >= len(rows) or len(by_category) <= 1:
        return rows[:n]
    queues = list(by_category.values())
    picked = []
    for i in range(max(map(len, queues))):
        picked += [queue[i] for queue in queues if i < len(queue)]
    return picked[:n]


def _read_speed(path, num_requests):
    """A SPEED-Bench split prepared by specdec_bench's prepare_data.py."""
    import pyarrow.parquet as pq

    path = Path(path)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    rows = [row for f in files for row in pq.read_table(f).to_pylist()]
    if any(turn.startswith(SPEED_PLACEHOLDER) for row in rows for turn in row["turns"]):
        raise ValueError(f"{path} has unresolved turns; run specdec_bench's prepare_data.py")
    return rows if num_requests is None else _stratified(rows, num_requests)


DATASETS = {
    "mtbench": DatasetSpec(_read_mtbench, "mtbench_responses.jsonl", key_by_index=True),
    "specbench": DatasetSpec(_read_jsonl, "specbench_responses.jsonl"),
    "speed": DatasetSpec(_read_speed, "specbench_responses.jsonl"),
}


def load_requests(spec, path, num_requests=None):
    return [
        Request(
            key=i if spec.key_by_index else row.get("question_id", i),
            question_id=row.get("question_id", i),
            category=row.get("category"),
            system_prompt=row.get("system_prompt"),
            turns=list(row["turns"] if "turns" in row else row["prompt"]),
        )
        for i, row in enumerate(spec.read_rows(path, num_requests))
    ]


# ---------------------------------------------------------------------------------------------
# Server and AIPerf
# ---------------------------------------------------------------------------------------------


def _request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def check_server(url, model):
    """Fail in seconds, not after a full AIPerf run, when the server reports no spec-decode stats."""
    message = {"role": "user", "content": "Count from 1 to 20."}
    reply = _request_json(
        f"{url}/v1/chat/completions", {"model": model, "messages": [message], "max_tokens": 32}
    )
    # vLLM puts the stats at the response root, TensorRT-LLM on the choice.
    stats = (reply.get("metrics") or {}).get("speculative_decoding")
    if not (stats or reply["choices"][0].get("speculative_decoding")):
        raise RuntimeError(
            f"{url} returns no per-request spec-decode stats, so acceptance cannot be measured. "
            "Serve with speculative decoding and --per-request-spec-decode-metrics summary "
            "(vLLM v0.29.0 or later)."
        )


def _versions(url):
    versions = {"aiperf_version": None, "server_version": None}
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        versions["aiperf_version"] = importlib.metadata.version("aiperf")
    with contextlib.suppress(OSError):
        versions["server_version"] = _request_json(f"{url}/version").get("version")
    return versions


def write_input(requests, path):
    """Write an AIPerf multi_turn file with one session per request, keyed by request index."""
    with open(path, "w") as f:
        for i, request in enumerate(requests):
            turns = [{"text": turn} for turn in request.turns]
            if request.system_prompt is not None:
                turns.insert(0, {"role": "system", "text": request.system_prompt})
            f.write(json.dumps({"session_id": str(i), "turns": turns}) + "\n")


def run_aiperf(args, setup, extra, concurrency, artifact_dir):
    cmd = [
        args.aiperf_bin, "profile",
        "--model", setup.model, "--tokenizer", setup.tokenizer, "--url", args.url,
        "--endpoint-type", "chat", "--streaming",
        "--input-file", str(artifact_dir / "input.jsonl"),
        "--custom-dataset-type", "multi_turn",
        "--dataset-sampling-strategy", "sequential",
        # Counts sessions; without it AIPerf sends 10 requests, all first turns.
        "--num-conversations", str(len(setup.requests)),
        "--concurrency", str(concurrency),
        "--extra-inputs", json.dumps(extra),
        "--use-server-token-count",
        "--export-level", "raw",
        "--output-artifact-dir", str(artifact_dir),
        "--ui", "none",
        "--no-gpu-telemetry",
    ]  # fmt: skip
    if args.trust_remote_code:
        cmd.append("--tokenizer-trust-remote-code")
    env = dict(os.environ)
    if os.path.isdir(setup.tokenizer):
        # In HF offline mode AIPerf 0.13 looks every tokenizer up in the Hub cache, local paths too.
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(var, None)
    subprocess.run(cmd, check=True, env=env, stdin=subprocess.DEVNULL)
    return cmd


def _key(record):
    return int(record["metadata"]["conversation_id"]), record["metadata"]["turn_index"]


def _reply(raw):
    """Concatenate the streamed content of one raw-exported request."""
    parts = []
    for response in raw["responses"]:
        for packet in response.get("packets", []):
            if packet.get("name") != "data" or packet["value"] == "[DONE]":
                continue
            choices = json.loads(packet["value"]).get("choices", [])
            parts.extend((choice.get("delta") or {}).get("content") or "" for choice in choices)
    return "".join(parts)


def _seconds(metric):
    # All timings are derived from these; another unit would scale every one of them silently.
    assert metric["unit"] == "ms", f"expected AIPerf timings in ms, got {metric['unit']}"
    value = metric["value"]
    return [v / 1e3 for v in value] if isinstance(value, list) else value / 1e3


def collect(artifact_dir, requests):
    """Read AIPerf's exports back into one TurnResult per request turn."""
    records = {_key(r): r for r in _read_jsonl(artifact_dir / "profile_export.jsonl")}
    replies = {_key(r): _reply(r) for r in _read_jsonl(artifact_dir / "profile_export_raw.jsonl")}
    results = []
    for i, request in enumerate(requests):
        turns = []
        for turn in range(len(request.turns)):
            record = records.get((i, turn))
            if record is None or "spec_decode_acceptance" not in record:
                raise RuntimeError(
                    f"Request {i} turn {turn} failed; see AIPerf's logs in {artifact_dir}"
                )
            histogram = record["spec_decode_acceptance"]["acceptance_histogram"]
            metrics = record["metrics"]
            start = record["metadata"]["request_start_ns"] / 1e9
            first = start + _seconds(metrics["time_to_first_token"])
            gaps = (
                _seconds(metrics["inter_chunk_latency"]) if "inter_chunk_latency" in metrics else []
            )
            turns.append(
                TurnResult(
                    # The prefill token, then 1 + accepted drafts for every verify step.
                    steps=[1] + [int(j) + 1 for j, n in histogram.items() for _ in range(n)],
                    times=[start, first] + [first + t for t in accumulate(gaps)],
                    num_tokens=metrics["output_sequence_length"]["value"],
                    reply=replies[(i, turn)],
                )
            )
        results.append(turns)
    return results


# ---------------------------------------------------------------------------------------------
# specdec_bench's result files
# ---------------------------------------------------------------------------------------------


def acceptance_metrics(requests, results):
    """specdec_bench's acceptance_rate.json: a request's AL pools its turns; averages are per request."""
    out = {"Request_AL": {}}
    per_category = defaultdict(list)
    histogram = defaultdict(int)
    for request, turns in zip(requests, results):
        lengths = [length for turn in turns for length in turn.steps]
        out["Request_AL"][request.key] = sum(lengths) / len(lengths)
        per_category[request.category].append(sum(lengths) / len(lengths))
        for length in lengths:
            histogram[length] += 1
    if None not in per_category:
        out["Category_AL"] = {c: float(np.mean(v)) for c, v in per_category.items()}
    out["Average_AL"] = float(np.mean(list(out["Request_AL"].values())))
    out["Acceptance_Length_Histogram"] = dict(sorted(histogram.items()))
    out["Conditional_Acceptance_Rate"], out["Joint_Acceptance_Rate"] = {}, {}
    total = remaining = sum(histogram.values())
    previous = joint = 1.0
    for length, count in out["Acceptance_Length_Histogram"].items():
        rate = remaining / total / previous
        joint *= rate
        out["Conditional_Acceptance_Rate"][length] = rate
        out["Joint_Acceptance_Rate"][length] = joint
        previous = remaining / total
        remaining -= count
    return out


def _stats(data):
    data = np.asarray(data, dtype=float)
    quantiles = np.percentile(data, [25, 50, 75])
    return {
        "min": f"{data.min():.4f}",
        "max": f"{data.max():.4f}",
        "mean": f"{data.mean():.4f}",
        "std": f"{data.std():.4f}",
        "quantiles": {q: f"{v:.4f}" for q, v in zip(["0.25", "0.5", "0.75"], quantiles)},
    }


def timing_metrics(results, num_gpus):
    """specdec_bench's timing.json."""
    turns = [turn for request in results for turn in request]
    span = max(t.times[-1] for t in turns) - min(t.times[0] for t in turns)
    out = {"Output TPS": sum(t.num_tokens for t in turns) / span}
    out["Output TPS/gpu"] = out["Output TPS"] / num_gpus
    e2e, ttft, step_times, rates = [], [], [], []
    for turn in turns:
        t = turn.times
        if len(t) > 1:
            e2e.append(t[-1] - t[0])
            ttft.append(t[1] - t[0])
        if len(t) > 2:
            rates.append((turn.num_tokens - 1) / (t[-1] - t[1]))
            step_times.extend(np.diff(t))
    out["E2E Request Time"] = _stats(e2e)
    out["TTFT Time"] = _stats(ttft)
    if step_times:
        out["Request Generation Step Time"] = _stats(step_times)
        out["Request Generation Tokens Per Second"] = _stats(rates)
    out["Number of Output Tokens"] = _stats([t.num_tokens for t in turns])
    return out


def _write_metric(path, metric):
    # specdec_bench stores each metric file as a list of runs.
    with open(path, "w") as f:
        json.dump([metric], f, indent=4)


def write_results(out, setup, results, config, num_gpus):
    acceptance = acceptance_metrics(setup.requests, results)
    _write_metric(out / "acceptance_rate.json", acceptance)
    _write_metric(out / "timing.json", timing_metrics(results, num_gpus))
    with open(out / setup.spec.responses_file, "w") as f:
        for request, turns in zip(setup.requests, results):
            line = {"question_id": request.question_id, "category": request.category}
            line["turns"] = request.turns
            line["choices"] = [{"index": 0, "turns": [turn.reply for turn in turns]}]
            f.write(json.dumps(line) + "\n")
    with open(out / "configuration.json", "w") as f:
        json.dump(config, f, indent=4)
    return acceptance


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Server base URL, e.g. http://localhost:8000")
    parser.add_argument("--model", help="Served model name (default: the first in /v1/models)")
    parser.add_argument("--tokenizer", help="Tokenizer for AIPerf (default: the model name)")
    parser.add_argument("--mtbench", help="Shorthand for --dataset mtbench --dataset_path PATH")
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument(
        "--dataset_path",
        help="JSONL questions, or for speed a split prepared by examples/specdec_bench/prepare_data.py",
    )
    parser.add_argument("--num_requests", type=int, help="Use the first N requests")
    parser.add_argument(
        "--runtime_params",
        help="specdec_bench runtime YAML; its sampling_kwargs and chat_template_args are sent",
    )
    parser.add_argument(
        "--temperature", type=float, nargs="+", help="Overrides sampling_kwargs.temperature"
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--output_length", type=int, default=4096)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPUs serving the model")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--aiperf_bin", help="AIPerf executable (default: next to this Python, else on PATH)"
    )
    parser.add_argument("--save_dir", default="results")
    args = parser.parse_args()
    if args.mtbench:
        args.dataset, args.dataset_path = "mtbench", args.mtbench
    if not (args.dataset and args.dataset_path):
        parser.error("pass --mtbench, or --dataset with --dataset_path")
    if args.aiperf_bin is None:
        sibling = Path(sys.executable).with_name("aiperf")
        args.aiperf_bin = str(sibling) if sibling.exists() else shutil.which("aiperf") or "aiperf"
    return args


def prepare(args):
    """Resolve everything the runs share, and check the server once before any of them."""
    runtime = {}
    if args.runtime_params:
        with open(args.runtime_params) as f:
            runtime = yaml.safe_load(f) or {}
    # specdec_bench's defaults: greedy without sampling_kwargs, else temperature 1.
    sampling = runtime.get("sampling_kwargs", {"temperature": 0})
    fields = {
        # Explicit, or vllm serve fills them from the model's generation_config.json.
        "top_p": sampling.get("top_p", 1.0),
        "top_k": sampling.get("top_k", 0),
        "max_tokens": args.output_length,
        # Turn 2 replays the turn-1 reply with special tokens, as specdec_bench does.
        "skip_special_tokens": False,
    }
    if runtime.get("chat_template_args"):
        fields["chat_template_kwargs"] = runtime["chat_template_args"]
    if args.ignore_eos:
        fields["ignore_eos"] = True
    model = args.model or _request_json(f"{args.url}/v1/models")["data"][0]["id"]
    check_server(args.url, model)
    spec = DATASETS[args.dataset]
    return Setup(
        spec=spec,
        requests=load_requests(spec, args.dataset_path, args.num_requests),
        model=model,
        tokenizer=args.tokenizer or model,
        temperatures=args.temperature or [sampling.get("temperature", 1.0)],
        request_fields=fields,
    )


def run_config(args, setup, temperature, concurrency):
    """Benchmark one temperature x concurrency and write it to <save_dir>/t<T>_c<C>/."""
    out = Path(args.save_dir) / f"t{temperature:g}_c{concurrency}"
    artifact_dir = out / "aiperf"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_input(setup.requests, artifact_dir / "input.jsonl")
    extra = {"temperature": temperature, **setup.request_fields}
    cmd = run_aiperf(args, setup, extra, concurrency, artifact_dir)
    results = collect(artifact_dir, setup.requests)
    config = {
        **vars(args),
        "temperature": temperature,
        "concurrency": concurrency,
        "served_model": setup.model,
        **_versions(args.url),
        # Required by specdec_bench's upload_to_s3.py; set from the env as specdec_bench does.
        "container_image": os.environ.get("CONTAINER_IMAGE") or None,
        "aiperf_command": cmd,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    acceptance = write_results(out, setup, results, config, args.num_gpus)
    print(f"{out}: Average_AL {acceptance['Average_AL']:.4f}", flush=True)


def main():
    args = parse_args()
    setup = prepare(args)
    failed = []
    for temperature in setup.temperatures:
        for concurrency in args.concurrency:
            try:
                run_config(args, setup, temperature, concurrency)
            except Exception:  # noqa: PERF203
                # One bad point must not cost the rest of a sweep; the exit code still reports it.
                traceback.print_exc()
                failed.append(f"t{temperature:g}_c{concurrency}")
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
