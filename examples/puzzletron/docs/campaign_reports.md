# Puzzletron results and campaign progress

Puzzletron stores each run's evidence in one structured JSON result. That
document drives live and detached status, portable result export, the central
results catalog, and the optional HTML summary. The HTML is a replaceable view:
it does not contain evidence that is absent from the structured result.

Use the generated [results catalog](../reports/catalog.yaml) to find every
retained run. It is the single central listing and points to each structured
result and any available human-readable summary. Regenerate it after
adding or updating checked-in result leaves:

```bash
python examples/puzzletron/generate_results_catalog.py
```

The catalog is YAML so it is easy to scan and review. Run results remain JSON
because their canonical bytes are validated, hashed, and atomically replaced.

## Inspect a running or detached campaign

The controller atomically refreshes `<run-root>/results/result.json`. It
contains the run, DAG stages, attempts, completed and total work, native
evaluator dimensions, timing, freshness, and qualified ETA. Detached
inspection reads that same file and does not require the original controller
process:

```bash
python examples/puzzletron/puzzletron.py results inspect /shared/puzzle_runs/my_campaign
python examples/puzzletron/puzzletron.py results inspect /shared/puzzle_runs/my_campaign --json
```

Treat `attachment: detached` separately from scheduler state: work may still be
running after its launching terminal exits. Check `freshness` before acting on
a status. An ETA is qualified only after the producer has a stable total and
observed progress; otherwise its reason explains why no estimate is shown.
Completed progress remains visible so engineers can reconstruct what finished,
not only what is active now.

Evaluator records distinguish repetitions, evaluator iterations, tasks,
samples, and optimizer steps. The controller polling count is never presented
as evaluator work. A failed evaluator with zero processed samples stays a
failure with its diagnostic artifacts; it cannot become a numeric score.

## Export and refresh

After clean completion, Puzzletron finalizes `result.json` before it generates
any presentation. The result is already portable. Validate and locate it
without rerunning model work:

```bash
python examples/puzzletron/puzzletron.py results export /shared/puzzle_runs/my_campaign
```

Regenerate the optional HTML from that structured evidence:

```bash
python examples/puzzletron/puzzletron.py results refresh /shared/puzzle_runs/my_campaign
```

The output is
`<run-root>/artifacts/campaign_report/campaign_report.html`; its neighboring
`report_manifest.json` records `source_result_digest`, the renderer revision,
output digest, and validation status. The summary visibly includes
run and subject identity, teacher and candidate roles, heterogeneous
architecture axes, execution and attachment state, freshness, timing, DAG
parents and phases, active and completed progress, metric values and qualified
comparisons, artifacts, provenance, and limitations. If HTML generation fails,
the sealed structured result remains valid and usable.

## Metric and comparison boundaries

Teacher, candidate, and control checkpoints use the same subject, architecture,
metric, artifact, and limitation fields. Each teacher/candidate metric pair
with the same name and producer execution produces a comparison entry that
names both source metric IDs. A numeric delta is emitted only when the unit,
direction, aggregation, workload contract, task, row manifest, prompt template,
decoding contract, and dimensions match and both values are numeric. Dimensions
carry denominators, evaluator repetitions, and sample counts when producers
record them. Otherwise, the entry records explicit exclusion reasons.

Language-model loss is `quality.lm_loss` in `nats_per_target_token` and uses
`lower_is_better`. Producers must distinguish a target-token-weighted mean over
all unmasked target tokens from the current scoring route's unweighted mean of
per-sample token means. Those aggregation and denominator contracts are not
interchangeable. Record teacher loss and candidate loss under the same frozen
workload when both were explicitly measured; do not infer teacher loss from a
candidate loss or teacher-relative metric. Token accuracy follows the same
rule: preserve whether ratios are token-weighted or averaged per sample.
`training.effective_tokens` is cumulative loss-bearing exposure after masking
and packing. It records tokenizer and data identity and whether it was measured
or derived; it is not inferred from optimizer steps times maximum sequence
length. Requested input/output tokens, observed sequence lengths, aggregate
output-token throughput, per-user throughput, token accuracy, examples,
samples, steps, latency, and GPU-hours remain distinct measures.

## Historical evidence

Older retained runs use the provisional
`modelopt.puzzletron-result-record/v1` schema and retain their historical
`result_record.json` filenames. Their summaries and structured files remain
available through the central catalog, but the catalog marks them as qualified
historical evidence and does not translate nested historical values into new
tidy metrics. This preserves the original claim boundaries, including bespoke
or superseded selection policies, unmatched teacher/student conditions,
missing row manifests, missing repetitions, and incomplete runtime provenance.

The two standalone historical HTML reports have structured legacy wrappers.
Those wrappers carry their producer, reproduction, support, current-config
relationship, and known limitations. They do not infer missing values from the
HTML. A current configuration linked from a legacy record is a migration or
reconstruction starting point, not proof of the executed configuration.

Run summaries remain useful derived explanations of recorded results and
limitations. There are no reports-level or campaign-level README indexes;
navigation belongs to `reports/catalog.yaml`, and this guide owns the shared
operational and interpretation instructions.
