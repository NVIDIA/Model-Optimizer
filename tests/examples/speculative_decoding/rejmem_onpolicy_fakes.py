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
"""Stand-ins for the vLLM side of rejmem_onpolicy, to run its loop on CPU.

``serve``: ``/v1/completions`` that simulates speculative decoding and writes one valid record per
request, stamped with its weight version; ``POST /fake_push`` advances the version, except pushes
after ``--ignore_pushes_after``. The rollout is seeded by the prompt, so a prompt always gives the
same rollout. ``train``: ``train_onpolicy.py`` whose weight push is ``Pusher``.
"""

import argparse
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import torch

from modelopt.torch.speculative.plugins.rejmem.records import SCHEMA, RolloutRecord, save_record

EXAMPLE = Path(__file__).parents[3] / "examples/speculative_decoding/rejmem_onpolicy"


def rollout(args, prompt, max_tokens, tag, version):
    """Rounds until the worker committed past the API's end (EOS or max_tokens), plus one more."""
    g = torch.Generator().manual_seed(
        int(hashlib.sha256(bytes(json.dumps(prompt), "utf8")).hexdigest()[:15], 16)
    )

    def rand(n):
        return torch.randint(10, args.vocab, (n,), generator=g).tolist()

    api_len = int(torch.randint(1, max_tokens + 1, (1,), generator=g))
    tokens = [*prompt, *rand(1)]
    anchors, accepted, drafts, offsets, mem_tokens, rejected = [], [], [], [0], [], []
    while True:
        anchors.append(len(tokens) - 1)
        drafts.append(rand(args.ns))
        mem_tokens += rejected
        offsets.append(len(mem_tokens))
        if len(tokens) - len(prompt) >= api_len:
            accepted.append(-1)
            break
        k = int(torch.randint(0, args.ns + 1, (1,), generator=g))
        accepted.append(k)
        tokens += [*drafts[-1][:k], *rand(1)]
        rejected = drafts[-1][k:]
    rows, aux = len(tokens) - 1, len(args.aux_layers) * args.hidden
    tensors = {
        "token_ids": torch.tensor(tokens, dtype=torch.int32),
        "ctx_aux_hidden": torch.randn(rows, aux, generator=g).bfloat16(),
        "ctx_last_hidden": torch.randn(rows, args.hidden, generator=g).bfloat16(),
        "round_anchor_pos": torch.tensor(anchors),
        "round_index": torch.arange(len(anchors)),
        "round_num_accepted": torch.tensor(accepted, dtype=torch.int32),
        "round_draft_token_ids": torch.tensor(drafts, dtype=torch.int32),
        "round_mem_offset": torch.tensor(offsets),
        "mem_token_ids": torch.tensor(mem_tokens, dtype=torch.int32),
        "mem_aux_hidden": torch.randn(len(mem_tokens), aux, generator=g).bfloat16(),
    }
    meta = {
        "schema": SCHEMA,
        "tag": tag,
        "req_id": tag,
        "prompt_len": len(prompt),
        "num_spec_tokens": args.ns,
        "aux_layer_ids": args.aux_layers,
        "hidden_size": args.hidden,
        "weight_version_start": version,
        "weight_version_end": version,
        "memory_consumer": True,
        "num_rounds_total": len(anchors),
        "round_keep": "all",
    }
    save_record(str(Path(args.capture_dir) / f"{tag}.safetensors"), RolloutRecord(tensors, meta))
    output = tokens[len(prompt) : len(prompt) + api_len]
    return output, ("length" if api_len == max_tokens else "stop")


def serve(args):
    lock, state = threading.Lock(), {"version": 0, "pushes": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            with lock:
                self.reply(state)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
            with lock:
                if self.path == "/fake_push":
                    state["pushes"] += 1
                    if args.ignore_pushes_after < 0 or state["pushes"] <= args.ignore_pushes_after:
                        state["version"] += 1
                    return self.reply(state)
                if self.path == "/collective_rpc":  # rejmem_dropped_memory_rows
                    return self.reply({"results": [args.dropped_memory_rows]})
                version = state["version"]
            tag, prompt = body["vllm_xargs"]["rejmem_tag"], body["prompt"]
            out, finish = rollout(args, prompt, body["max_tokens"], tag, version)
            with lock, open(args.log, "a") as f:
                f.write(json.dumps({"tag": tag, "prompt": prompt, "version": version}) + "\n")
            choice = {"prompt_token_ids": prompt, "token_ids": out, "finish_reason": finish}
            self.reply({"choices": [choice]})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


class Pusher:
    """``DraftWeightPusher``'s interface, for the fake servers."""

    def __init__(self, server_urls):
        self.urls = server_urls
        self.weight_version = httpx.get(f"{server_urls[0]}/").json()["version"]

    def push(self, model):
        for url in self.urls:
            httpx.post(f"{url}/fake_push").raise_for_status()
        self.weight_version += 1
        return self.weight_version

    def close(self):
        pass


if __name__ == "__main__":
    if sys.argv[1] == "train":
        sys.argv.pop(1)
        sys.path.insert(0, str(EXAMPLE))
        import train_onpolicy

        train_onpolicy.main(connect=Pusher)
    else:
        p = argparse.ArgumentParser()
        p.add_argument("command", choices=["serve"])
        for name in ("port", "ns", "hidden", "vocab"):
            p.add_argument(f"--{name}", type=int, required=True)
        p.add_argument("--aux_layers", type=json.loads, required=True)
        p.add_argument("--capture_dir", required=True)
        p.add_argument("--log", required=True)
        p.add_argument("--ignore_pushes_after", type=int, default=-1)
        p.add_argument("--dropped_memory_rows", type=int, default=0)
        serve(p.parse_args())
