# Doom Loops: Stop Repeating Reasoning Early

A damaged (often quantized) reasoning model can repeat one block verbatim until
`max_new_tokens`. Each such sample burns the full output budget and scores wrong
anyway. vLLM can end the request as soon as its output tail repeats.

## When to add it

- **User asks** ("avoid doom loops", "stop looping reasoning", "repetition
  detection"): add it to every task they asked to run.
- **Suspected only**: under a long `max_new_tokens` (≥32K) a canary or short task
  runs far longer than its sample count implies, the vLLM log keeps a few requests
  `Running` after the rest finish, or many responses stop with
  `finish_reason: length` and logged ones end in one block repeated verbatim (long
  but varied reasoning is not a loop). Show the evidence and ask first: it changes
  the eval protocol.

## Config

vLLM ≥ 0.17.0 accepts a per-request `repetition_detection` field. Add it beside the
existing entries in the top-level `target.api_endpoint.adapter_config.params_to_add`:

```yaml
        adapter_config:
          params_to_add:
            repetition_detection:
              max_pattern_size: 256   # longest repeating chunk checked, in tokens; sets the CPU cost
              min_pattern_size: 32    # shortest chunk; the false-fire lever
              min_count: 3            # back-to-back copies that stop the request
```

On `ns_aa_lcr`, set `max_pattern_size: 1024` under the same path in the task's own
`nemo_evaluator_config`. Confirm each task's values in the dry-run's resolved config.
nel-next: the same dict under `proxy.extra_body.repetition_detection`.

On a hit vLLM ends the whole request with `finish_reason: "repetition"`, so the
sample gets no final answer. Most loops run to the cap anyway, but a badly damaged
model sometimes escapes one and answers (see Guards).

**Use 256/32/3, and 1024/32/3 on AA-LCR.** From replays of vLLM's detector on
finished eval outputs, including IQ1/IQ2-quantized Qwen3.8-Flash-Next:

- **Healthy models:** no fire on 1,152 healthy DeepSeek-R1 traces and one on 7,471
  non-looping Qwen3.8-Flash-Next BF16 generations, on an answer that was wrong
  anyway. 512/4/3 fired on 4.2% of correct R1 math and 12.2% of complete code
  traces (`1,000,000,000`, `0.333333333333`, `5 \times 5 \times 5`), each a correct
  answer turned wrong. Don't lower `min_pattern_size` to catch more.
- **Most loops are short.** Over half have a period of 1-4 tokens; on SciCode and
  GPQA the 99th percentile is under 160. There 256 catches 98-100% of IQ1 loops
  and saves the same 18-64% of generated tokens as 512-4096 (within 0.2 pt).
- **AA-LCR loops restate whole paragraphs**, with periods up to ~4K tokens: 256
  catches 77-86%, 512 85-90%, 1024 91-93%, saving 57-73% of IQ1's tokens. At 1024,
  BF16 and IQ2 fire on exactly the generations they fire on at 512. 4096 catches
  all, but cuts at least twice as many correct IQ1 answers as 512.
- **Cost** is vLLM scheduler CPU time per running request per decode step: ~50 µs
  at 256, 120 µs at 512, 250 µs at 1024, 1 ms at 4096. The default adds ~13 ms
  per step at 256 requests; 1024 adds 4-8 ms at AA-LCR's 16-32.

Tokenizers that group digits (Llama 3) shift these numbers.

## Guards

- **Backend.** vLLM < 0.17.0 silently ignores the field (debug log only), so the
  run looks protected but isn't. Check the deploy image's vLLM version; probe an
  external endpoint (expect `repetition`; `length` means ignored):

  ```bash
  curl -s "$BASE_URL/v1/completions" -H 'Content-Type: application/json' -d '{
    "model": "<served_model_name>", "prompt": "ha ha ha ha ha ha", "max_tokens": 64,
    "temperature": 0, "repetition_detection":
      {"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 2}}' \
    | jq -r '.choices[0].finish_reason'
  ```

  SGLang and TRT-LLM don't support it: tell the user and don't add the field.
- **Matched runs.** Baseline and candidate use the identical setting. Published
  scores ran without it; say so when comparing against them.
- **Damaged checkpoints score lower with it.** Ending the request also cuts loops
  the model would have escaped and answered. IQ1 checkpoints lost up to 0.75 pt per
  benchmark (GPQA is the same at every size; AA-LCR at 1024 falls between 512's
  0.75 pt and 4096's 1.5 pt), IQ2 at most 0.13 pt and BF16 nothing. Report such
  scores as detector-on.
- **Record it.** Append `repetition_detection=256/32/3, AA-LCR 1024/32/3` (the
  values used) to `export.mlflow.description` and add it as a tag; it is not
  otherwise queryable.
- **Validation.** In run-validation's Output limits row and Response gate, count
  `repetition` stops with `length` stops but report them separately. A candidate
  rate well above the baseline's is a quantization finding.

## End only the reasoning (unmerged vLLM PR)

vLLM PR #52677 (Model Runner V2 only; check whether it has merged) instead forces
`reasoning_end_str` on a loop so the model still answers, configured server-side
with `--reasoning-config` keys `loop_break_max_pattern_size`,
`loop_break_min_pattern_size` and `loop_break_min_count`. That can lift a damaged
model above its real score. Use it only when the user asks and their vLLM build
carries the PR, matched on both sides. Put a short transition phrase before
`</think>` in `reasoning_end_str`: a bare `</think>` sent Qwen3.8-27B-NVFP4 back
into the loop inside the answer.
