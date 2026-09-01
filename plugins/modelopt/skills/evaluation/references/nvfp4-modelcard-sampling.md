# NVFP4 Model-Card Sampling Reference

Published `temperature` / `top_p` / max generation length for the **2026 NVFP4
checkpoints under [huggingface.co/nvidia](https://huggingface.co/nvidia/models)
whose cards disclose them** — 25 rows, collected 2026-08-20. All 69 NVFP4
checkpoints in the org were read; absent are those published before 2026-01-01,
those whose cards disclose nothing usable, and **speculative-decoding variants
(`-DSpark`, `-DFlash`)** — spec decoding is verified against the target and does
not change its output distribution, so those checkpoints share their base
checkpoint's row. A miss here means "read the card", not "not yet checked".

Use it to reproduce a published NVFP4 number, and as the cross-check when a card
is silent or ambiguous. It does not replace reading the card — see
`model-card-research.md`.

## Lookup

**The card is the source of truth; this table is a reference, not a constraint.**
Use it to confirm what you read, to fill a gap when the card is silent, and as a
sanity check when you are unsure — never to override a value the card states.

1. **Exact row, resolved per field.** `eval` → use it, cite the row. `rec` → use
   it, but note in the config comment that it is recommended sampling, not a
   stated eval setting; if a same-family `eval` row disagrees, surface both.
   `—` → that field is unpublished; resolve **it alone** via step 2.
   `max_num_tokens` is the card's *headline* cap — where a note names a higher
   per-task cap (GLM-5.2 GPQA `100000`, Qwen3.5-397B-V2 τ²-Telecom `128000`,
   Kimi-K3 uncapped for Terminal-Bench) and that task is in your suite, SKILL.md
   Step 3's take-the-highest rule governs the single top-level value, not the
   column.
2. **No row** (new or unreleased variant, non-NVIDIA baseline, pre-2026) → take
   the nearest same-family rows as the expected value.
3. **Card vs. table.** Agree → proceed. Card silent + family consistent → adopt
   the family value and cite this file in a line comment; that beats SKILL.md
   Step 3's generic 65536 / 16384. **Card disagrees → the card wins**, but
   surface it — defaults shift between generations, so a mismatch means re-read,
   not auto-correct.
4. **Baseline and candidate share one setting.** Cards report both precisions
   measured under the single setting listed; use the NVFP4 row for both.

**Per-task sampling is precedent, not mandate.** Some notes record a
benchmark-specific `temperature` / `top_p` (Qwen3.6 SciCode `0.6`; Qwen3.6-27B
τ²-Bench Telecom `0.0` / `top_p=1.0`; Kimi-K3 `top_p=1.0` agentic). Engineers do
tune sampling per benchmark, so **follow the card you are working from** and use
these as the cross-check. Where the two disagree, **escalate to the user on a
regime change, not a nudge** — greedy (`temperature ≤ 0.1` or `top_p ≤ 1e-4`)
versus sampled flips the regime and materially moves both score and variance;
`0.95` vs `1.0` does not. NEL accepts per-task `temperature` / `top_p` under
`evaluation.tasks.*.nemo_evaluator_config`; only `max_new_tokens` is barred
(SKILL.md Step 3).

> **Never take sampling from a card's quickstart snippet.**
> `SamplingParams(temperature=0.8, top_p=0.95)` and `max_tokens=32` are
> boilerplate, repeated verbatim across unrelated models. Only *Benchmarked
> with…* / *evaluated with…* / *We evaluate the model using…* sentences,
> "Recommended Sampling" rows, and footnotes under the accuracy table count.

`provenance` — **`eval`** (20 rows): card ties the values to its accuracy table,
authoritative. **`rec`** (5 rows): card recommends them for inference without
that tie. `max_num_tokens` is the max generation length, i.e.
`nemo_evaluator_config.config.params.max_new_tokens`.

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

## Priors (verify against the card)

- **`1.0 / 0.95` is the house default** — GLM 4.7–5.2, Kimi K2.6–K3, MiniMax
  M2.5–M3, Gemma 4, Nemotron 3/3.5, Qwen3.6. Best guess when a recent card is
  silent.
- **DeepSeek is carved out of it** — its row uses `top_p=1.0`, not `0.95`.
  Never carry the house default onto an unlisted DeepSeek variant.
- **Qwen splits by variant** — thinking `0.6 / 0.95` at Qwen3.5, raised to `1.0`
  at Qwen3.6; instruct/coder near-greedy `0 / 1e-5` with `16384`.
- **GLM** — `1.0 / 0.95` throughout; cap fell from 131072 (4.7, 5) to 64000
  (5.1, 5.2).
- **Caps cluster at 64000 / 65536 / 81920 / 128000 / 131072**, 64000 most
  common. `16384` appears only with greedy instruct Qwen; DeepSeek-V4-Flash's
  `384000` is a long-context outlier.
- **Per-task overrides are narrow** — SciCode (lower temperature), τ²-Bench
  Telecom (greedy or larger cap), GPQA Diamond (larger cap), Terminal-Bench
  (uncapped). SKILL.md Step 3 forbids per-task `max_new_tokens`, so when a card
  lists two caps **take the maximum** as the single top-level value and note the
  split in a comment.

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
