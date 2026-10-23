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

"""Benchmark a served model with AIPerf and report it through specdec_bench's metrics.

Acceptance comes from the server's per-request spec-decode stats (vLLM
``--per-request-spec-decode-metrics``), so one long-lived server can be measured at
several temperatures and concurrencies without rebuilding the engine.
"""

import contextlib
import json
import os
import subprocess
import urllib.request
from itertools import accumulate
from pathlib import Path

from .metrics import Timing


def vllm_serve_command(engine_kwargs, port, extra_args=()):
    """``vllm serve`` argv for the engine ``AsyncEngineArgs(**engine_kwargs)`` would build."""
    kwargs = dict(engine_kwargs)
    cmd = ["vllm", "serve", kwargs.pop("model"), "--port", str(port)]
    cmd += ["--per-request-spec-decode-metrics", "summary"]
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            cmd.append(flag if value else "--no-" + flag[2:])
        elif isinstance(value, dict):
            cmd += [flag, json.dumps(value)]
        else:
            cmd += [flag, str(value)]
    return cmd + list(extra_args)


def _get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def server_info(url):
    """Served model name and engine version, for configuration.json."""
    info = {"aiperf_url": url, "served_model": _get_json(f"{url}/v1/models")["data"][0]["id"]}
    with contextlib.suppress(OSError):
        info["server_version"] = _get_json(f"{url}/version").get("version")
    return info


def write_input(dataset, path):
    """Write the requests as an AIPerf multi_turn file, one session per request index."""
    with open(path, "w") as f:
        for i, request in enumerate(dataset.data):
            turns = [{"text": turn} for turn in request.turns]
            if request.system_prompt is not None:
                turns.insert(0, {"role": "system", "text": request.system_prompt})
            f.write(json.dumps({"session_id": str(i), "turns": turns}) + "\n")


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


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
    assert metric["unit"] == "ms", metric
    value = metric["value"]
    return [v / 1e3 for v in value] if isinstance(value, list) else value / 1e3


def unsupported_options(args):
    """run.py options the AIPerf path cannot honor."""
    options = {
        "--aa_timing": args.aa_timing,
        "--completions": args.completions,
        "--postprocess gptoss": args.postprocess != "base",
    }
    return [name for name, used in options.items() if used]


def run(args, dataset, metrics_list, sampling_kwargs, chat_template_args):
    """Run AIPerf over ``dataset`` against ``args.aiperf_url`` and feed ``metrics_list``."""
    out = Path(args.save_dir or ".") / "aiperf"
    out.mkdir(parents=True, exist_ok=True)
    write_input(dataset, out / "input.jsonl")
    extra = {
        # Explicit, or vllm serve fills them from the model's generation_config.json.
        "temperature": sampling_kwargs.get("temperature", 1.0),
        "top_p": sampling_kwargs.get("top_p", 1.0),
        "top_k": sampling_kwargs.get("top_k", 0),
        "max_tokens": args.output_length,
        # run.py replays turn-1 replies with tokenizer.decode(), which keeps special tokens.
        "skip_special_tokens": False,
    }
    if chat_template_args:
        extra["chat_template_kwargs"] = chat_template_args
    if args.ignore_eos:
        extra["ignore_eos"] = True
    cmd = [
        args.aiperf_bin, "profile",
        "--model", server_info(args.aiperf_url)["served_model"],
        "--tokenizer", args.tokenizer,
        "--url", args.aiperf_url,
        "--endpoint-type", "chat", "--streaming",
        "--input-file", str(out / "input.jsonl"),
        "--custom-dataset-type", "multi_turn",
        "--dataset-sampling-strategy", "sequential",
        # Counts sessions; without it AIPerf sends 10 requests, all first turns.
        "--num-conversations", str(len(dataset.data)),
        "--concurrency", str(args.concurrency),
        "--extra-inputs", json.dumps(extra),
        "--use-server-token-count",
        "--export-level", "raw",
        "--output-artifact-dir", str(out),
        "--ui", "none",
        "--no-gpu-telemetry",
    ]  # fmt: skip
    if args.trust_remote_code:
        cmd.append("--tokenizer-trust-remote-code")
    env = dict(os.environ)
    if os.path.isdir(args.tokenizer):
        # In HF offline mode AIPerf 0.13 looks every tokenizer up in the Hub cache, local paths too.
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(var, None)
    subprocess.run(cmd, check=True, env=env, stdin=subprocess.DEVNULL)

    records = {_key(r): r for r in _read_jsonl(out / "profile_export.jsonl")}
    replies = {_key(r): _reply(r) for r in _read_jsonl(out / "profile_export_raw.jsonl")}
    text_outputs = []
    for i, request in enumerate(dataset.data):
        messages = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        for turn_id, question in enumerate(request.turns):
            record = records.get((i, turn_id))
            if record is None or "spec_decode_acceptance" not in record:
                raise RuntimeError(
                    f"No per-request spec-decode stats for request {i} turn {turn_id}: the request "
                    "failed, or the server was started without --per-request-spec-decode-metrics."
                )
            # Steps as run.py counts streamed chunks: the prefill token, then each verify step.
            histogram = record["spec_decode_acceptance"]["acceptance_histogram"]
            steps = [[0]] + [[0] * (int(j) + 1) for j, n in histogram.items() for _ in range(n)]
            metrics = record["metrics"]
            start = record["metadata"]["request_start_ns"] / 1e9
            first = start + _seconds(metrics["time_to_first_token"])
            gaps = (
                _seconds(metrics["inter_chunk_latency"]) if "inter_chunk_latency" in metrics else []
            )
            times = [start, first] + [first + t for t in accumulate(gaps)]
            num_tokens = metrics["output_sequence_length"]["value"]
            for metric in metrics_list:
                if isinstance(metric, Timing):
                    step = {"output_ids": [[[0] * num_tokens]], "token_times": times}
                else:
                    step = {"output_ids": [steps]}
                metric.process_step(step, i, turn_id)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": replies[(i, turn_id)]})
        text_outputs.append(messages)
    for metric in metrics_list:
        metric.process_final(text_outputs)
