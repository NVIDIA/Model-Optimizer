# Benchmarking a Deployment with AIPerf

[AIPerf](https://github.com/ai-dynamo/aiperf) (Apache-2.0, NVIDIA; the successor
to GenAI-Perf) is a client-side benchmark for any OpenAI-compatible endpoint. It
reports the latency/throughput metrics that matter for serving — TTFT, ITL,
output token throughput, and per-user throughput — under controlled concurrency
and token shapes.

Use this once you have a healthy endpoint (see the main `SKILL.md` "Verify the
deployment" step). Benchmarking gibberish is meaningless, so always run the
**coherence gate** below first.

## 1. Install

Install into a **clean venv**. Installing `aiperf` directly into some inference
images fails on a `blinker` distutils conflict (`Cannot uninstall blinker ...`):

```bash
python3 -m venv aiperf-venv
aiperf-venv/bin/pip install -q aiperf
AIPERF=aiperf-venv/bin/aiperf
```

## 2. Coherence gate (run before benchmarking)

AIPerf measures throughput on incoherent output just as happily as on correct
output. A wrong KV-cache dtype, a missing serving patch, or a mis-wired
activation can produce *fluent nonsense* that only a real generation check (or an
accuracy eval) catches — the perf numbers will look fine. Assert sane output
first:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model>","messages":[{"role":"user",
       "content":"What is the capital of France? Then compute 17*23."}],
       "max_tokens":128}' \
  | python3 -c "import json,sys; t=json.load(sys.stdin)['choices'][0]['message']['content']; \
print('COHERENCE:',repr(t[:200])); sys.exit(0 if 'Paris' in t else 1)" \
  || { echo 'INCOHERENT OUTPUT — wrong KV dtype / missing serving patch?'; exit 1; }
```

## 3. Run a sweep

```bash
$AIPERF profile -m <model> --endpoint-type chat --streaming -u localhost:8000 \
    --synthetic-input-tokens-mean $ISL --output-tokens-mean $OSL \
    --concurrency $C --request-count $RC \
    --tokenizer <model> --tokenizer-trust-remote-code \
    --extra-inputs ignore_eos:true \
    --random-seed 42 --artifact-dir $ARTIFACT_DIR
```

Critical flags:

- **`--extra-inputs ignore_eos:true`** — forces generation to run to
  `output-tokens-mean` so the realized output length equals the target. Without
  it, models that stop early give throughput computed over truncated outputs, and
  results aren't comparable across precisions. **Do not omit.**
- **`--streaming`** — required for a meaningful TTFT / inter-token-latency split.
- **`--tokenizer ... --tokenizer-trust-remote-code`** — point the tokenizer at
  the same repo so synthetic token counts match the model's tokenizer.
- **`--random-seed 42`** — reproducible synthetic prompts across runs.

Sweep `--concurrency` over a range (e.g. `1 4 16 64 128 256 512`) to trace the
latency-vs-throughput curve; set `--request-count` to a few × concurrency so each
point reaches steady state.

### Suggested token shapes

Run more than one shape so the deployment is characterized under different
prefill/decode and prefix-cache conditions:

| Shape       | Input (ISL) | Output (OSL) | Notes                                  |
|-------------|-------------|--------------|----------------------------------------|
| chat        | ~8000       | ~1000        | typical chat turn                      |
| agentic     | ~64000      | ~400         | long-context, short completion         |

To exercise prefix caching, AIPerf's prefix-prompt options (e.g.
`--prefix-prompt-length` with a small pool) reuse a shared prefix across requests
so a known fraction hits the KV cache — useful when the server has
`--enable-prefix-caching`.

## 4. Compare the results

Each run writes `profile_export_aiperf.json` to `--artifact-dir`. Key fields:

- `time_to_first_token`
- `inter_token_latency`
- `output_token_throughput`
- `output_token_throughput_per_user`
- `output_sequence_length`

First **sanity-check `output_sequence_length` == target OSL** — this confirms
`ignore_eos` took effect. Then compare per concurrency point (e.g. quantized vs
baseline precision of the same model) at matched ISL/OSL.

## 5. Gotchas

- **KV-cache dtype is per-model.** Some attention kernels are fp8-capable, others
  are bf16-only — passing `--kv-cache-dtype fp8` to a bf16-only kernel can break
  generation (caught by the coherence gate). When unsure, omit the flag (defaults
  to the model dtype) and confirm coherence. This is a *serving* flag, not an
  AIPerf flag — set it on `vllm serve` / the launch command.
- **`ignore_eos` is mandatory for fair comparison** — see §3.
- **Match the serving config across compared runs** — image, tensor/expert
  parallelism, KV dtype, and token shapes must be identical, or the comparison
  isn't apples-to-apples.
- **NVFP4 on Blackwell sm_103 needs a cu130 image** — see the vLLM note in the
  main `SKILL.md`; a wrong image serves nothing (or the coherence gate fails).
