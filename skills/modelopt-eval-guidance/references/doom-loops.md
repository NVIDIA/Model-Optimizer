# Doom Loops: Stop Repeating Reasoning Early

Damaged (often quantized) reasoning models can repeat one block verbatim until the output-token limit, burning
the whole budget on a sample that scores wrong anyway. vLLM 0.17.0 and later can end such a request through a
per-request `repetition_detection` field. Turning it on changes the evaluation protocol, so it is a decision, not
a default.

- **The user asks** to avoid doom loops or looping reasoning: turn it on for every task.
- **Suspected** (under a long output limit a canary runs far longer than its sample count implies, a few
  requests stay `Running` after the rest finish, or `finish_reason: length` responses end in one block repeated
  verbatim): show the evidence and ask first. Long but varied reasoning, for example from a high reasoning
  effort, is not a loop.

**The setting and where it goes belong to the tool.** Its values and the place in its config (and what its client
needs) are in the tool's reference; if the tool cannot pass the field, say so instead of editing the evaluation.

## Guards

- **Backend.** vLLM older than 0.17.0 silently ignores the field: check the serving image's version (it is part of
  the serving record). For an external endpoint run this probe, where `logit_bias` forces one token every step;
  `repetition_detected` means the field works, `null` means it was ignored.

  ```bash
  curl -s "$BASE_URL/v1/completions" -H 'Content-Type: application/json' -d '{
    "model": "<served_model_name>", "prompt": "Hi", "max_tokens": 16,
    "temperature": 0, "logit_bias": {"1000": 100}, "repetition_detection":
      {"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 2}}' \
    | jq -r '.choices[0].stop_reason'
  ```

  SGLang and TRT-LLM do not support it: tell the user and do not add the field.
- **Matched and recorded.** Identical values for the baseline and the candidate, recorded in the run description.
- **Report scores as detector-on.** Ending the request also cuts loops a damaged model would have escaped: IQ1 lost
  up to 0.75 points per benchmark at 512 (1.5 points on AA-LCR at 4096), IQ2 at most 0.13, BF16 nothing. Published
  scores ran without it.
- Ending only the reasoning (an unmerged vLLM PR) lets the model still answer but can inflate damaged-model
  scores: use it only when the user asks, matched on both sides.
