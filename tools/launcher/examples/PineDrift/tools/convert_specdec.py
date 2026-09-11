#!/usr/bin/env python3
"""Build the streaming-safe corpus from Spec-Decoding-Dataset v1.

Two things the published file needs fixing for:

1. It is gzip despite the .jsonl name, and stores the reply under `conversations`.
   It may also carry a prompt-only `messages` field alongside; hf_streaming_dataset
   prefers `messages`, so that combination gives an empty answer span and a silent
   hang. So: emit `messages`-only records built from `conversations`.

2. It has **no id field at all**. `hf_streaming_dataset._tokenize_entry` starts with
   `cid = entry.get("conversation_id") or entry.get("uuid")` and drops the entry when
   that is None -- silently, as an "unfit entry", so the run dies much later with
   "no fetchable sample found in the entire corpus" and no hint about why (job
   405859). So: synthesize `conversation_id` from the output index.

The gate at the end mirrors what _tokenize_entry actually requires, not just the
assistant turn.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
smoke_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
src = dest / "default.jsonl"
out = dest / "default-msgs.jsonl"
smoke = dest / f"default-msgs-smoke{smoke_n // 1000}k.jsonl"

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
roles_conv = collections.Counter()
n = kept = 0

with opener(src, "rt", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g, \
        smoke.open("w", encoding="utf-8") as s:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n += 1
        fields.update(r.keys())
        conv = norm(r.get("conversations") or [])
        for t in conv:
            roles_conv[t["role"]] += 1
        if not any(t["role"] == "assistant" and t["content"] for t in conv):
            continue
        # Index the OUTPUT, not the source line, so a prefix of this file is a valid
        # standalone corpus with the same ids (that is what the smoke shard is).
        rec = json.dumps({"conversation_id": f"v1-{kept}", "messages": conv},
                         ensure_ascii=False)
        g.write(rec + "\n")
        if kept < smoke_n:
            s.write(rec + "\n")
        kept += 1
        if n % 200000 == 0:
            print(f"  ... {n} records, {kept} kept", flush=True)

print(f"\n  records                 : {n}")
print(f"  top-level fields        : {dict(fields)}")
print(f"  roles in `conversations`: {dict(roles_conv)}")
print(f"  written                 : {kept}/{n}")
if kept == 0:
    raise SystemExit("FATAL: no record had an assistant reply -- wrong source file")
if kept < n * 0.9:
    print(f"  WARNING: dropped {n - kept} records ({100 * (n - kept) / n:.1f}%)")

# Gate: replay _tokenize_entry's own preconditions on what we just wrote.
for path in (out, smoke):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            cid = d.get("conversation_id") or d.get("uuid")
            msgs = d.get("conversations") or d.get("messages")
            assert cid, f"{path}:{i + 1} has no conversation_id/uuid"
            assert msgs and isinstance(msgs, list), f"{path}:{i + 1} has no turns"
            assert any(t["role"] == "assistant" and t["content"] for t in msgs), \
                f"{path}:{i + 1} has no assistant turn"
            if i >= 999:
                break
    print(f"  -> {path}  (first 1000 records pass the _tokenize_entry preconditions)")
