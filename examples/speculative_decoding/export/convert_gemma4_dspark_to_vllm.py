#!/usr/bin/env python3
"""Convert a ModelOpt DSpark drafter export into the layout vLLM's
Gemma4DSparkForCausalLM expects.

Three classes of mismatch, all on the ModelOpt side (vLLM needs no patch):

1. config.json
   - architectures: DFlashDraftModel -> Gemma4DSparkModel   (registry key)
   - model_type:    qwen3            -> gemma4_text          (Gemma4DSparkAttention
     reads layer_types/head_dim/global_head_dim off a Gemma4 text config)
   - target_layer_ids / markov_rank are read as TOP-LEVEL attrs by
     Gemma4DSparkModel.__init__, but ModelOpt nests them under dflash_config.

2. weight names
   - markov_w1/markov_w2 -> markov_head.markov_w1/markov_head.markov_w2
     (DSparkMarkovHead registers them under a `markov_head` submodule)

3. missing tensors  <-- the one that silently destroys AL
   - Gemma4DSparkForCausalLM builds its OWN ParallelLMHead and VocabParallelEmbedding
     and its load_weights() only fills names it finds; anything absent stays
     RANDOMLY INITIALIZED with no error. The ModelOpt export ships neither
     lm_head nor embed_tokens, so both must be baked in from the base model.
     Gemma 4 is tie_word_embeddings=true, so both come from the same tensor:
     model.language_model.embed_tokens.weight
"""

import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--drafter", required=True, help="ModelOpt export dir")
ap.add_argument("--base", required=True, help="base Gemma-4-E4B-it dir (for lm_head/embed)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
cfg = json.load(open(os.path.join(args.drafter, "config.json")))
df = cfg.get("dflash_config", {}) or {}

new = dict(cfg)
new["architectures"] = ["Gemma4DSparkModel"]
new["model_type"] = "gemma4_text"
# promote what Gemma4DSparkModel.__init__ reads off the top level
new["target_layer_ids"] = df.get("target_layer_ids", cfg.get("target_layer_ids"))
new["markov_rank"] = df.get("markov_rank", 256)
new["mask_token_id"] = df.get("mask_token_id")
new["block_size"] = cfg.get("block_size")
new.setdefault("draft_vocab_size", cfg["vocab_size"])
# Gemma4 attention knobs consulted by gemma4_layer_config()/Gemma4DSparkAttention
# Do NOT default these: Gemma4 sizes attention per layer, and quietly falling back to
# the sliding-layer values rebuilds the draft with the wrong q/k/o and q/k-norm shapes.
# They must come from the training config (see hf_spec_export.py).
for _req in ("global_head_dim", "attention_k_eq_v"):
    if _req not in cfg:
        raise SystemExit(
            f"drafter config.json is missing {_req!r}. Re-export with a ModelOpt that "
            "propagates the Gemma4 per-layer attention fields, or add it by hand."
        )
if cfg.get("attention_k_eq_v") and "num_global_key_value_heads" not in cfg:
    raise SystemExit(
        "attention_k_eq_v is set but num_global_key_value_heads is missing; "
        "vLLM would size k_proj with the sliding-layer KV head count."
    )
new.setdefault("sliding_window", 512)
new.setdefault("final_logit_softcapping", None)
json.dump(new, open(os.path.join(args.out, "config.json"), "w"), indent=2)
print(
    "config: architectures={} model_type={} target_layer_ids={} markov_rank={}".format(
        new["architectures"], new["model_type"], new["target_layer_ids"], new["markov_rank"]
    )
)

sd = load_file(os.path.join(args.drafter, "model.safetensors"))
print(f"loaded {len(sd)} drafter tensors")

out = {}
renamed = 0
for k, v in sd.items():
    nk = k
    if k.startswith(("markov_w1", "markov_w2")):
        nk = "markov_head." + k
        renamed += 1
    out[nk] = v
print(f"renamed {renamed} markov tensors -> markov_head.*")

# --- bake in lm_head + embed_tokens from the base (tied on Gemma 4) ---
idx_path = os.path.join(args.base, "model.safetensors.index.json")
if os.path.exists(idx_path):
    wm = json.load(open(idx_path))["weight_map"]
    key = next(k for k in wm if k.endswith("language_model.embed_tokens.weight"))
    base_sd = load_file(os.path.join(args.base, wm[key]))
else:
    base_sd = load_file(os.path.join(args.base, "model.safetensors"))
    key = next(k for k in base_sd if k.endswith("language_model.embed_tokens.weight"))
emb = base_sd[key]
print(f"base embed_tokens {tuple(emb.shape)} from {key!r}")
assert emb.shape[0] == cfg["vocab_size"] and emb.shape[1] == cfg["hidden_size"], (
    f"base embed {tuple(emb.shape)} does not match draft vocab/hidden "
    f"({cfg['vocab_size']},{cfg['hidden_size']})"
)

out["lm_head.weight"] = emb.clone()
out["embed_tokens.weight"] = emb.clone()
print("baked lm_head.weight + embed_tokens.weight (tie_word_embeddings=true on Gemma 4)")

save_file(out, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
for f in ("tokenizer.json", "tokenizer_config.json"):
    src = os.path.join(args.base, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(args.out, f))
print(f"wrote {len(out)} tensors -> {args.out}")
print()
print("Serve with (note the draft attention backend):")
_spec = (
    f'{{"model": "{args.out}", "num_speculative_tokens": 3, '
    '"method": "dspark", "attention_backend": "FLASHINFER"}'
)
print(f"  vllm serve <base> --speculative-config '{_spec}'")
print(
    "  FLASHINFER is REQUIRED: the draft re-runs backend auto-selection and lands on\n"
    "  FLASH_ATTN, whose FA2 kernel caps head dimension at 256, but Gemma4 full-attention\n"
    "  layers use global_head_dim=512. The VLLM_ATTENTION_BACKEND env var does NOT reach\n"
    "  the draft -- it is read from speculative_config.attention_backend."
)
