# NeMo Gym

Details of the NeMo Gym path. The model onboarding, the run and its recovery are done by the Gym skills; this
file only says where they cover a duty of this guidance and where they do not.

## Skills

`add-model-eval-config` (model onboarding), `run-and-babysit-eval` (submit, babysit, inspect, recover) and
`nemo-gym-score-triage` (a score that does not match a reference, two runs that disagree, or whether a
baseline-versus-candidate difference is real).

## Health verdict and faults

Take the tool's own verdict (`SUCCESS` with health `PASS`; its 2% limit counts evaluation-side faults). A wrong,
truncated or empty answer, or a clean agent timeout, is the model's result and counts as zero; do not add a
response-level invalidation of your own. `UNVERIFIED` means the run finished but some grading was not recorded:
report the score as unverified.

## Settings

- **Model card reading:** `add-model-eval-config` (sampling, the reasoning switch, parsers, context length,
  parallelism, the serving image). It does not cover ARM64 or non-standard GPU images.
- **A quantized variant** is served with its baseline's config except for the fields the quantization requires
  (`add-model-eval-config`, section 9, "A quantized variant"); the same skill accepts a deployment derived elsewhere.
- **Output budget, repeats and the score definition** come from the benchmark's configuration.
- **Investigating a baseline that does not match its published score:** `nemo-gym-score-triage`.

## Which metric is the headline

A run's headline in the tool's summary is `<agent>/mean/reward`, but for some benchmarks that is not the metric the vendor or NEL report. Compare like with like, and say which metric you used:

| Benchmark | `mean/reward` is | The vendor and NEL headline | Gym metric to use for it |
|---|---|---|---|
| IFBench | strict prompt-level grade | prompt-level loose | `<agent>/mean/follow_all_instructions_loose` (a full run: strict 75.4 vs loose 79.6, NEL 78.7 to 79.9, vendor 79.5) |
| SciCode | whole-problem accuracy | subtask accuracy | `<agent>/mean/subtask_accuracy` (a full run: problem 12.7 to 14.0, subtask 44.1 to 46.6, NEL subtask 44 to 49) |
| AA-LCR | the judged accuracy of the config that ran | depends on the config: `bench.yaml` uses the same Qwen 235B judge and dataset as NEL; the v1.1 files use a corrected dataset and a different judge and give a different number | the config you ran; do not compare v1.1 with NEL's score |

A baseline and its candidate always use the same metric, so a delta is unaffected; the choice matters when a score is compared with a vendor number or with another tool.

## Doom loops

Gym counts `repetition_stops` (the model's own loops, scored). **To turn the detection on**, put the detector in the
policy model's `sampling_overrides` (not `extra_body`, which only supplies defaults that a request can replace):

```yaml
policy_model.responses_api_models.vllm_model.sampling_overrides:
  repetition_detection: {max_pattern_size: 256, min_pattern_size: 32, min_count: 3}
```

Larger sizes cut more escaped loops at more scheduler CPU; AA-LCR loops restate paragraphs up to about 4K tokens, so for AA-LCR alone use `max_pattern_size: 1024`.
Under the benchmark's `run` block of the submit document this key is the literal dotted mapping key. It is forced onto
every outbound request after the request body is merged. A forced-loop canary (vLLM 0.26) confirmed it end to end:
the engine stopped every request with `repetition_detected`, Gym recorded the native stop reason, normalized the
finish reason to `stop` (so no client remap is needed, unlike a client that rejects `repetition`), and counted
the stops. The saved model-call capture records the request before the override, so to see the engine-bound body
set `NEMO_GYM_VLLM_TRANSPORT_LOG` to a mounted path before submitting.

## MLflow

- Runs carry the tags `precision`, `pair_id`, `stage`, `requested_task`.
- The headline score is the metric `<agent>/mean/reward` on a 0 to 1 scale (multiply by 100; on every checked run it
  equalled the run summary score). Token averages are `<agent>/mean/output_tokens`; sample counts are
  `coverage/expected` and `coverage/scored`; evaluation health is under `health/*` and `coverage/*`.
- One run is one benchmark of one submission, so a combined score averages equal-repeat runs, and the task status
  comes from the tool's verdict (a benchmark's run can be `FINISHED` while its submission failed overall).
