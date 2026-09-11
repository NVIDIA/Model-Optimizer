#!/usr/bin/env python3
"""Build the streaming-safe messages-only corpus from Spec-Decoding-Dataset v1.

The published file is gzip despite the .jsonl name. It may also carry a prompt-only
`messages` field alongside the real reply in `conversations`; hf_streaming_dataset
prefers `messages`, so that combination gives an empty answer span and a silent hang.
Emit `messages`-only records built from `conversations`, and refuse to produce a file
whose records lack an assistant turn.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
src, out = dest / "default.jsonl", dest / "default-msgs.jsonl"

opener = gzip.open if src.open("rb").read(2) == b"\x1f\x8b" else open
print(f"source: {src} ({'gzip' if opener is gzip.open else 'plain'})")

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def norm(turns):
    out = []
    for t in turns:
        role = t.get("role") or t.get("from")
        content = t.get("content") if "content" in t else t.get("value")
        out.append({"role": ROLE_MAP.get(role, role), "content": content})
    return out


fields = collections.Counter()
roles_msgs = collections.Counter()
roles_conv = collections.Counter()
n = kept = 0

with opener(src, "rt", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        fields.update(r.keys())
        for t in r.get("messages") or []:
            roles_msgs[t.get("role") or t.get("from")] += 1
        conv = norm(r.get("conversations") or [])
        for t in conv:
            roles_conv[t["role"]] += 1
        if any(t["role"] == "assistant" and t["content"] for t in conv):
            g.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
            kept += 1
        if n % 200000 == 0:
            print(f"  ... {n} records, {kept} kept", flush=True)

print(f"\n  records               : {n}")
print(f"  top-level fields      : {dict(fields)}")
print(f"  roles in `messages`   : {dict(roles_msgs) or 'field absent'}")
print(f"  roles in `conversations`: {dict(roles_conv)}")
print(f"  written with a real assistant turn: {kept}/{n}")
if kept == 0:
    raise SystemExit("FATAL: no record had an assistant reply -- wrong source file")
if kept < n * 0.9:
    print(f"  WARNING: dropped {n - kept} records ({100 * (n - kept) / n:.1f}%)")
print(f"  -> {out}")
