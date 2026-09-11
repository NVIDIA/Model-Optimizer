#!/usr/bin/env python3
"""Pick and justify DFlash's mask token id for PineDrift."""
import json
import re

D = "/home/haoguo/lustre/hf-local/pinedrift-820b-a42b-nvfp4_vv3"
cfg = json.load(open(f"{D}/config.json"))
tok = json.load(open(f"{D}/tokenizer.json"))
tmpl = open(f"{D}/chat_template.jinja").read()

vocab_size = cfg["vocab_size"]
base_ids = set(tok["model"]["vocab"].values())
added = {a["id"]: a["content"] for a in tok.get("added_tokens", [])}

print(f"config.vocab_size      : {vocab_size}")
print(f"base BPE vocab         : {len(base_ids)} ids, max {max(base_ids)}")
print(f"added tokens           : {len(added)} ids, {min(added)}..{max(added)}")
print(f"ids the tokenizer can never emit (inside vocab_size): "
      f"{vocab_size - len(base_ids | set(added))}")

# Which control tokens does the chat template actually write?
in_template = sorted(i for i, c in added.items() if c in tmpl)
print(f"\ncontrol tokens used by chat_template.jinja ({len(in_template)}):")
for i in in_template:
    print(f"  {i:6d}  {added[i]!r}")

reserved = sorted(i for i, c in added.items() if re.fullmatch(r"<\|reserved_special_token_\d+\|>", c))
print(f"\nreserved_special_token ids: {len(reserved)}, {reserved[0]}..{reserved[-1]}")

named = sorted(i for i, c in added.items()
               if i not in reserved and not re.fullmatch(r"<\|reserved_special_token_\d+\|>", c))
print(f"\nnamed (non-reserved) added tokens ({len(named)}):")
for i in named:
    flag = "  <-- used by chat template" if i in in_template else ""
    print(f"  {i:6d}  {added[i]!r}{flag}")

# The pick: highest reserved id that is (a) inside vocab_size, (b) absent from the
# chat template, (c) not a config-declared special. Highest = furthest from the
# low-numbered specials a future release is most likely to start assigning.
declared = {cfg["bos_token_id"], cfg["pad_token_id"]}
declared |= set(cfg["eos_token_id"] if isinstance(cfg["eos_token_id"], list) else [cfg["eos_token_id"]])
candidates = [i for i in reserved if i < vocab_size and i not in in_template and i not in declared]
pick = candidates[-1]
print(f"\nPICK: {pick}  {added[pick]!r}")
print(f"  inside vocab_size      : {pick < vocab_size}")
print(f"  in chat template       : {pick in in_template}")
print(f"  declared special in cfg: {pick in declared}")
print(f"  has an embedding row   : yes (id < vocab_size, so embed_tokens/lm_head cover it)")
