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
existing entries in the top-level `target.api_endpoint.adapter_config.params_to_add`,
and confirm every task keeps it in the dry-run's resolved config:

```yaml
        adapter_config:
          params_to_add:
            repetition_detection:
              max_pattern_size: 512   # longest repeating chunk checked, in tokens
              min_pattern_size: 32    # shortest chunk; the false-fire lever
              min_count: 3            # back-to-back copies that stop the request
```

nel-next: the same dict under `proxy.extra_body.repetition_detection`.

On a hit vLLM ends the whole request with `finish_reason: "repetition"`. No final
answer is produced, so the sample scores as it would after looping to the cap.

**Keep 512/32/3.** Replaying vLLM's detector on 1,152 healthy DeepSeek-R1 traces,
512/32/3 fired on none and still caught the degenerate loops; 512/4/3 fired on 4.2% of
correct math and 12.2% of complete code traces (`1,000,000,000`, `0.333333333333`,
`5 \times 5 \times 5`), each a correct answer turned wrong. Don't lower
`min_pattern_size` to catch more. Tokenizers that group digits (Llama 3) shift
these numbers. The check costs ~30 ms of scheduler time per decode step at 256
running requests; `max_pattern_size: 256` halves it if throughput drops.

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
- **Record it.** Append `repetition_detection=512/32/3` to
  `export.mlflow.description` and add it as a tag; it is not otherwise queryable.
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
