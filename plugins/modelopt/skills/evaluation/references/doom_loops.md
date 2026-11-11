# Doom Loops: Stop Repeating Reasoning Early

Damaged (often quantized) reasoning models can repeat one block verbatim until
`max_new_tokens`, burning the full budget on a sample that scores wrong anyway.
vLLM ≥ 0.17.0 ends such a request through the per-request `repetition_detection` field.

- **User asks** to avoid doom loops or looping reasoning: add it to every task.
- **Suspected** (under a long `max_new_tokens` a canary runs far longer than its
  sample count implies, a few requests stay `Running` after the rest finish, or
  `finish_reason: length` responses end in one block repeated verbatim): show the
  evidence and ask first, since it changes the eval protocol. Long but varied
  reasoning, e.g. from a high reasoning effort, is not a loop.

## Config

Add beside the existing entries in
`evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config.params_to_add`
(nel-next: `services.<service>.proxy.extra_body`):

```yaml
            repetition_detection:
              max_pattern_size: 256   # longest chunk checked, in tokens; sets the CPU cost
              min_pattern_size: 32    # shortest chunk; lower fires on text like 1,000,000,000
              min_count: 3            # back-to-back copies that end the request
```

AA-LCR loops restate paragraphs up to ~4K tokens long, so on `ns_aa_lcr` set only
`repetition_detection.max_pattern_size: 1024` at the same path in the task's
`nemo_evaluator_config`. NEL deep-merges task config over the top level, so the
other entries survive; check the dry-run's `config_ef.yaml` block. A hit ends the
request with `finish_reason: "repetition"` (`length` on `ns_*`, see below), keeping
the text so far; a loop inside the reasoning leaves no final answer.

**Keep these values**; larger sizes add scheduler CPU (~1 ms per request per decode
step at 4096) and cut more escaped-loop answers.

**nemo-skills tasks (`ns_*`) need the hit reported as `length`.** Their litellm client
rejects `"repetition"` (pydantic `literal_error`) and retries the request, resampling
the loop away and inflating the score. Remap it in the serving container (append to
any existing `deployment.pre_cmd`) and submit with `NEMO_EVALUATOR_TRUST_PRE_CMD=1`:

```yaml
deployment:
  pre_cmd: |-
    F=$(python3 -c 'import vllm; print(vllm.__path__[0])')/v1/engine/__init__.py
    sed -i 's/"abort", "error", "repetition")/"abort", "error", "length")/' "$F"
    grep -q '"error", "length")' "$F" || { echo "finish_reason remap failed"; exit 1; }
```

A hit then scores like a `max_new_tokens` truncation. Confirm the canary's client logs
show no `literal_error`.

## Guards

- **Backend.** vLLM < 0.17.0 silently ignores the field. Check the image's vLLM
  version; for an external endpoint, run this probe. `logit_bias` forces one token
  every step, so the output repeats at once: `repetition_detected` means the field
  works (remapped or not), `null` means it was ignored.

  ```bash
  curl -s "$BASE_URL/v1/completions" -H 'Content-Type: application/json' -d '{
    "model": "<served_model_name>", "prompt": "Hi", "max_tokens": 16,
    "temperature": 0, "logit_bias": {"1000": 100}, "repetition_detection":
      {"max_pattern_size": 4, "min_pattern_size": 1, "min_count": 2}}' \
    | jq -r '.choices[0].stop_reason'
  ```

  SGLang and TRT-LLM don't support it: tell the user and don't add the field.
- **Matched and recorded.** Use identical values for baseline and candidate, and add
  them to `export.mlflow.description` and the tags.
- **Report scores as detector-on.** Ending the request also cuts loops a damaged
  model would have escaped: IQ1 lost up to 0.75 pt per benchmark at 512 (1.5 pt on
  AA-LCR at 4096), IQ2 at most 0.13 pt, BF16 nothing. Published scores ran without it.

Ending only the reasoning (unmerged vLLM PR #52677) lets the model still answer but
can inflate damaged-model scores: use it only when the user asks, matched on both sides.
