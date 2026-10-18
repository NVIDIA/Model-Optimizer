# Doom Loops: Stop Repeating Reasoning Early

Damaged (often quantized) reasoning models can repeat one block verbatim until
`max_new_tokens`, burning the full budget on a sample that scores wrong anyway.
vLLM ≥ 0.17.0 ends such a request through the per-request `repetition_detection` field.

- **User asks** to avoid doom loops or looping reasoning: add it to every task.
- **Suspected** (under a long `max_new_tokens` a canary runs far longer than its
  sample count implies, a few requests stay `Running` after the rest finish, or
  `finish_reason: length` responses end in one block repeated verbatim): show the
  evidence and ask first, since it changes the eval protocol.

## Config

Add beside the existing entries in the top-level
`target.api_endpoint.adapter_config.params_to_add` (nel-next: `proxy.extra_body`):

```yaml
            repetition_detection:
              max_pattern_size: 256   # longest chunk checked, in tokens; sets the CPU cost
              min_pattern_size: 32    # shortest chunk; lower fires on text like 1,000,000,000
              min_count: 3            # back-to-back copies that end the request
```

On `ns_aa_lcr`, override `max_pattern_size: 1024` in the task's own `adapter_config`:
its loops restate paragraphs up to ~4K tokens long. Confirm every task's values in
the dry-run's resolved config. A hit ends the whole request with
`finish_reason: "repetition"` and no answer.

**Keep these values.** In replays on BF16 and IQ1/IQ2 Qwen3.8-Flash-Next outputs,
256 caught 98-100% of SciCode/GPQA loops and 1024 caught 91-93% of AA-LCR's, with
one fire on 7,471 non-looping BF16 generations. Larger sizes add scheduler CPU
(~1 ms per request per decode step at 4096) and cut more escaped-loop answers.

## Guards

- **Backend.** vLLM < 0.17.0 silently ignores the field. Check the image's vLLM
  version; probe an external endpoint (expect `repetition`, not `length`):

  ```bash
  curl -s "$BASE_URL/v1/completions" -H 'Content-Type: application/json' -d '{
    "model": "<served_model_name>", "prompt": "ha ha ha ha ha ha", "max_tokens": 64,
    "temperature": 0, "repetition_detection":
      {"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 2}}' \
    | jq -r '.choices[0].finish_reason'
  ```

  SGLang and TRT-LLM don't support it: tell the user and don't add the field.
- **Matched and recorded.** Use identical values for baseline and candidate, and add
  them to `export.mlflow.description` and the tags.
- **Report scores as detector-on.** Ending the request also cuts loops a damaged
  model would have escaped: IQ1 lost up to 0.75 pt per benchmark at 512 (1.5 pt on
  AA-LCR at 4096), IQ2 at most 0.13 pt, BF16 nothing. Published scores ran without it.

Ending only the reasoning, so the model still answers, needs unmerged vLLM PR #52677
(server-side `--reasoning-config` `loop_break_*` keys, Model Runner V2 only). It can
lift a damaged model above its real score: use it only when asked, matched on both
sides, with a transition phrase before `</think>` in `reasoning_end_str` (a bare
`</think>` re-entered the loop).
