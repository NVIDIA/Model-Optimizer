#!/usr/bin/env python3
"""Find a token id that is safe to use as DFlash's mask token on PineDrift.

Safe means: inside config.vocab_size (so lm_head/embed have a row for it), never
produced by the tokenizer on real text, and not one of the chat template's control
tokens. Picking a live id is silent -- training runs and the drafter learns to
predict a token the base actually uses.
"""
import json

D = "/home/haoguo/lustre/hf-local/pinedrift-820b-a42b-nvfp4_vv3"
cfg = json.load(open(f"{D}/config.json"))
tc = json.load(open(f"{D}/tokenizer_config.json"))
tok = json.load(open(f"{D}/tokenizer.json"))

vocab_size = cfg["vocab_size"]
print("config vocab_size :", vocab_size)
print("bos/eos/pad       :", cfg["bos_token_id"], cfg["eos_token_id"], cfg["pad_token_id"])

# every id the tokenizer can emit: base BPE vocab + added tokens
base_vocab = tok["model"]["vocab"]
added = tok.get("added_tokens", [])
added_ids = {a["id"]: a["content"] for a in added}
base_ids = set(base_vocab.values())
print("\nbase BPE vocab    :", len(base_ids), "ids, max", max(base_ids))
print("added_tokens      :", len(added_ids), "ids,",
      (f"range {min(added_ids)}..{max(added_ids)}" if added_ids else "none"))

used = base_ids | set(added_ids)
free = [i for i in range(vocab_size) if i not in used]
print("\nids inside vocab_size that the tokenizer can NEVER emit:", len(free))
if free:
    print("  first 10:", free[:10])
    print("  last 10 :", free[-10:])

print("\n=== added tokens (the chat-template control set) ===")
for i in sorted(added_ids):
    c = added_ids[i]
    print(f"  {i:6d}  {c!r}")

# Reserved/unused-looking added tokens are the conventional choice (Qwen3 uses 151669,
# an unused reserved special) because they have trained embedding rows.
reservedish = [
    i for i, c in sorted(added_ids.items())
    if any(k in c.lower() for k in ("reserved", "unused", "extra", "placeholder"))
]
print("\nreserved/unused-looking added tokens:", reservedish[:20] or "none")
