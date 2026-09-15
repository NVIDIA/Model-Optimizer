# NVFP4 Model-Card Sampling Reference

Published `temperature` / `top_p` / max generation length for the **2026 NVFP4
checkpoints under [huggingface.co/nvidia](https://huggingface.co/nvidia/models)
whose cards disclose them** — 25 rows, collected 2026-08-20. All 69 NVFP4
checkpoints in the org were read; absent are those published before 2026-01-01,
those whose cards disclose nothing usable, and **speculative-decoding variants
(`-DSpark`, `-DFlash`)** — spec decoding is verified against the target and does
not change its output distribution, so those checkpoints share their base
checkpoint's row. A miss here means "read the card", not "not yet checked".

Use this dated snapshot to cross-check published settings, not to fill gaps in
silent or ambiguous cards. Read the exact card and verify each field's evaluation
provenance — see `model-card-research.md`.

## Lookup

**Explicit user/task requirements take precedence over model-card settings.**

1. **Exact row, resolved per field.** `eval` → verify the card explicitly ties
   that field to evaluation/benchmarking for the applicable task/mode, then cite
   the statement. A row can mix evaluation settings and general recommendations
   in its notes; the row label does not authorize every value. `rec` and `—` do
   not justify overrides.
2. **Missing or ambiguous evidence** → preserve `config.json`,
   `generation_config.json`, and vLLM defaults. No same-family inference or
   generic fallback. Verify evaluator defaults and actual requests as described
   in SKILL.md Step 3; omission and explicit `null` are not interchangeable.
3. **Card vs. table** → re-read the card and surface discrepancies; only verified
   evaluation settings qualify. Quickstarts, supported limits, and general
   "Recommended Sampling" rows do not qualify without an explicit evaluation tie.
4. **Scope overrides to the stated tasks/modes**, including output caps. Do not
   promote a task-specific or headline maximum to a suite-wide value. Apply
   verified settings consistently to baseline and candidate when reproducing
   their published comparison; otherwise confirm effective config parity.

`provenance` — **`eval`** (20 rows): recorded as tied to evaluation; verify each
field against the card. **`rec`** (5 rows): inference recommendations only, not
an override source. `max_num_tokens` records generation length, corresponding
to `nemo_evaluator_config.config.params.max_new_tokens`, not context length.

| Model card ID | temp | top_p | max_num_tokens | prov | notes |
| --- | --- | --- | --- | --- | --- |
| `nvidia/DeepSeek-V4-Flash-NVFP4` | 1.0 | **1.0** | 384000 | eval | `top_p=1.0`, unlike every other row here |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 1.0 | 0.95 | 131072 | eval | SciCode used `temperature=0.6` |
| `nvidia/Qwen3.6-27B-NVFP4` | 1.0 | 0.95 | 81920 | eval | SciCode `0.6`; τ²-Bench Telecom `0.0` / `top_p=1.0` |
| `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` | 0.6 | 0.95 | 64000 | eval | τ²-Bench Telecom used `128000` |
| `nvidia/Qwen3.5-397B-A17B-NVFP4` | 0.6 | 0.95 | 64000 | eval | |
| `nvidia/Qwen3.5-122B-A10B-NVFP4` | 0.6 | 0.95 | 64000 | eval | |
| `nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4` | **0.0** | **1.0e-05** | 16384 | eval | greedy — instruct variant |
| `nvidia/GLM-5.2-NVFP4` | 1.0 | 0.95 | 64000 | eval | GPQA Diamond used `100000` |
| `nvidia/GLM-5.1-NVFP4` | 1.0 | 0.95 | 64000 | eval | benchmarked on `vllm/vllm-openai:v0.19.1` |
| `nvidia/GLM-5-NVFP4` | 1.0 | 0.95 | 131072 | eval | |
| `nvidia/GLM-4.7-NVFP4` | 1.0 | 0.95 | 131072 | eval | |
| `nvidia/Kimi-K3-NVFP4` | 1.0 | 0.95 | 65536 | eval | **uncapped for Terminal-Bench**; card also recommends `top_p=1.0` agentic, `n=1`, `presence_penalty=0`, `frequency_penalty=0` |
| `nvidia/Kimi-K2.7-Code-NVFP4` | 1.0 | 0.95 | 64000 | eval | |
| `nvidia/Kimi-K2.6-NVFP4` | 1.0 | 0.95 | 128000 | eval | |
| `nvidia/MiniMax-M3-NVFP4` | 1.0 | 0.95 | 65536 | eval | baseline is native MXFP8 |
| `nvidia/MiniMax-M2.5-NVFP4` | 1.0 | 0.95 | 64000 | eval | |
| `nvidia/Gemma-4-31B-IT-NVFP4` | 1.0 | 0.95 | 131072 | eval | |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 1.0 | 0.95 | 131072 | eval | |
| `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | upstream | upstream | `null` (uncapped) | eval | defers to `google/diffusiongemma-26B-A4B-it`; diffusion decoding, serve with `--override-generation-config '{"max_new_tokens": null}'`. Uncapped is deliberate — do **not** substitute a numeric fallback |
| `nvidia/Ising-Calibration-1.5-31B-NVFP4` | 0.2 | — | 8192 zero-shot / 32767 ICL | rec | domain model (Gemma-4-31B derivative) |
| `nvidia/Mistral-Medium-3.5-128B-NVFP4` | 0.7 | 0.95 | — | eval | benchmarked with `reasoning_effort="high"` |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 1.0 | 0.95 | — | rec | its spec-decode siblings' cards say *Benchmarked with* these same values; eval recipes live in NeMo Gym, client examples use `max_tokens=16000` |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 1.0 | 0.95 | — | rec | card: use across **all** tasks and serving backends |
| `nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4` | 1.0 | 1.0 | — | rec | reasoning tasks |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | 0.6 think / 0.2 instruct | 0.95 think / — | 20480 think / 1024 instruct | rec | think adds `reasoning_budget=16384`, `grace_period=1024`; instruct sets `top_k=1` |

## Refreshing

Built from the HF API, verified to match the website pagination page for page
(918 repos across `p=0..31`, identical NVFP4 sets).

```bash
curl -s "https://huggingface.co/api/models?author=nvidia&limit=1000" -o all.json
python3 -c "
import json, re
KEEP = re.compile(r'-NVFP4(-V\d+|-QAD)?\$', re.I)   # target checkpoints only
for m in json.load(open('all.json')):
    if KEEP.search(m['id']) and m.get('createdAt', '') >= '2026-01-01':
        print(m['id'])
" > ids.txt

mkdir -p cards
# 404 = repo ships no card; 401 = gated, fetch with 'hf download <id> README.md'
# (never interpolate the HF token into a curl argument)
while read id; do curl -sfL "https://huggingface.co/$id/raw/main/README.md" \
  -o "cards/${id//\//_}.md"; done < ids.txt

grep -ihnE "benchmark(ed|ing) (parameters|with)|were evaluated with|we evaluate the model using|evaluation settings" cards/*.md
grep -ihnE "max OSL|for evals?|including benchmarking" cards/*.md
```

The second grep is **not optional**: some cards give the cap only as a footnote
under the accuracy table (*"\*Max OSL for evals can be as high as 64K"*), and
DeepSeek states sampling in its `## Input:` usage block — neither is reachable
from the first.

`KEEP` matches a **target checkpoint's** name shape — `…-NVFP4`, plus the
`-V2`-style revision and `-QAD` (a quantization recipe, so still a target). It
therefore drops, by construction, every repo class that would only pollute the
table: `-DSpark` / `-DFlash` speculative-decoding variants (verified against the
target, so identical accuracy — they duplicate the base row), `-Eagle3` draft
heads, and `-MLPerf-Inference-Closed-*` submission snapshots (which ship no
card). `re.I` matters: DeepSeek spells its revisions `-v2`, not `-V2`. The one
blind spot is a repo named `-FP4-*` whose `hf_quant_config.json` says NVFP4 —
rare, and none currently in scope; check the `fp4` tag if you need certainty.
