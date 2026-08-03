# Day-0 Release Report

## Outcome

- Decision: <ACCEPT, REGRESSION, ANOMALOUS, or INFEASIBLE>
- Publish recommendation: <publish or do not publish>
- Summary: <concise rationale>

## Run metadata

| Field | Value |
| --- | --- |
| Source model and revision | <value> |
| Quantized checkpoint | <absolute path> |
| Recipe | <qformat or recipe path> |
| Workspace | <absolute path> |
| Execution environment | <value> |
| Agent session ID | <value or none> |
| MLflow experiment | <name and URL or none> |
| MLflow runs | <baseline and candidate run IDs/URLs or none> |

## Checkpoint validation

| Check | Result | Evidence |
| --- | --- | --- |
| Source size | <value> | <path or command> |
| Output size | <value> | <path or command> |
| Output/source ratio | <value> | <calculation> |
| Quantization coverage | <value> | <validation artifact> |
| Metadata consistency | <value> | <validation artifact> |
| Serving canary | <value> | <endpoint, command, and response summary> |

## External baseline sanity

| Task | Measured baseline | External source | Comparable | Finding |
| --- | ---: | --- | --- | --- |
| <task> | <score> | <citation or none> | <yes, no, or unverified> | <finding> |

## Final comparison

Use score degradation (`Baseline - Quantized`) on each task's documented score
scale. Apply the acceptance criterion recorded in `PROGRAM.md`.

| Task | Version/config | Score field | Scale | Baseline | Baseline SD | Quantized | Quantized SD | Delta (B - Q) | Criterion | Pass |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| <task> | <config> | <field> | <scale> | <score> | <value or none> | <score> | <value or none> | <delta> | <threshold> | <yes or no> |

## Performance results

<Include when performance measurement was requested; otherwise write `Not requested`.>

Record the endpoint model name, framework and version, container image or
environment, hardware, accelerator count, parallelism, cache configuration,
launch command, benchmark tool version, and benchmark command. Confirm observed
sequence lengths match the requested workload.

| Shape | Input length | Output length | Concurrency | Requests | TTFT | ITL | Output tokens/s | Per-user tokens/s | Observed output length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <shape> | <value> | <value> | <value> | <value> | <value> | <value> | <value> | <value> | <value> |

## Artifacts

Include absolute paths or stable URLs for checkpoints, validation summaries,
baseline and quantized evaluation results, generated configurations, run and
job IDs, deployment logs, container images, and performance exports.

| Artifact | Location or ID | Notes |
| --- | --- | --- |
| <artifact> | <absolute path, stable URL, or ID> | <notes> |

## Notable issues and workarounds

| Issue | Workaround | Effect on result |
| --- | --- | --- |
| <issue> | <workaround> | <effect> |

## Follow-ups

- <follow-up or none>

## Self-assessment

| Field | Value |
| --- | --- |
| Agent score (1–10) | <value> |
| Rationale | <value> |
