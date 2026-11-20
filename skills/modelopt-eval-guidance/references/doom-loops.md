# Doom Loops: Stop Repeating Reasoning Early

Damaged (often quantized) reasoning models can repeat one block verbatim until the output-token limit, burning
the whole budget on a sample that scores wrong anyway. vLLM 0.17.0 and later can end such a request through a
per-request `repetition_detection` field.

- **The user asks** to avoid doom loops or looping reasoning: turn it on for every task.
- **Suspected** (under a long output limit a canary runs far longer than its sample count implies, a few
  requests stay `Running` after the rest finish, or `finish_reason: length` responses end in one block repeated
  verbatim): show the evidence and ask first, because it changes the evaluation protocol. Long but varied
  reasoning, for example from a high reasoning effort, is not a loop.

## The setting

```yaml
repetition_detection:
  max_pattern_size: 256   # longest chunk checked, in tokens; sets the CPU cost
  min_pattern_size: 32    # shortest chunk; lower fires on text like 1,000,000,000
  min_count: 3            # back-to-back copies that end the request
```

Keep these values; larger sizes add scheduler CPU (about 1 ms per request per decode step at 4096) and cut
more escaped-loop answers. AA-LCR loops restate paragraphs up to about 4K tokens long, so for AA-LCR alone set
`max_pattern_size: 1024`. A hit ends the request with a repetition stop reason and keeps the text so far; a loop
inside the reasoning leaves no final answer.

**How the field reaches the server depends on the tool.** It goes into the request body the tool sends to the
serving endpoint. The tool's reference says where it goes in that tool's config and what else the tool's client
needs; if the tool cannot pass the field, say so instead of editing the evaluation.

## Guards

- **Backend.** vLLM older than 0.17.0 silently ignores the field. Check the serving image's version; for an
  external endpoint run this probe, where `logit_bias` forces one token every step so the output repeats at once.
  `repetition_detected` means the field works; `null` means it was ignored.

  ```bash
  curl -s "$BASE_URL/v1/completions" -H 'Content-Type: application/json' -d '{
    "model": "<served_model_name>", "prompt": "Hi", "max_tokens": 16,
    "temperature": 0, "logit_bias": {"1000": 100}, "repetition_detection":
      {"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 2}}' \
    | jq -r '.choices[0].stop_reason'
  ```

  SGLang and TRT-LLM do not support it: tell the user and do not add the field.
- **Matched and recorded.** Use identical values for the baseline and the candidate, and record them in the
  run description and tags.
- **Report scores as detector-on.** Ending the request also cuts loops a damaged model would have escaped:
  IQ1 lost up to 0.75 points per benchmark at 512 (1.5 points on AA-LCR at 4096), IQ2 at most 0.13, BF16
  nothing. Published scores ran without it.
- Ending only the reasoning (an unmerged vLLM PR) lets the model still answer but can inflate damaged-model
  scores: use it only when the user asks, matched on both sides.
