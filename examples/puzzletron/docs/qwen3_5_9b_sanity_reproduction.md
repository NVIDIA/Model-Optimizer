# Qwen3.5-9B sanity campaign reproduction

This is the current-code reconstruction of the completed `Qwen/Qwen3.5-9B`
sanity campaign. Its canonical entry config is:

```text
examples/puzzletron/configs/clean/families/qwen3_5/qwen3_5_9b/sanity_reproduction.yaml
```

That file composes the model, family, full-pipeline, sanity, and full-width-MIP
overlays. It also records the manual runtime overrides found in the completed
campaign manifests. Do not replace it with `full_pipeline.yaml`; the latter is
the source campaign whose immutable activation, depth, vLLM, scoring, and bypass
evidence bundles were imported into the sanity run.

## 1. Artifact-verified experiment contract

| Item | Exact value |
|---|---|
| Teacher | `Qwen/Qwen3.5-9B`, revision `main` |
| Sanity run | `qwen3_5/qwen3_5_9b/sanity_check` under `PUZZLETRON_RUN_ROOT` |
| Source run | `qwen3_5/qwen3_5_9b/full_pipeline` under the same root |
| Train cache | 65,536 fixed samples × 16,384 tokens, seed 444 |
| Validation cache | 128 fixed samples × 16,384 tokens, seed 445 |
| Width/depth parent topology | TP1, CP4, PP2, DP1 on eight GPUs |
| Embedding search | Disabled; hidden width is always 4096 |
| Depth search | Sublayer granularity, removals 0 through 6 |
| MIP profile | `latency-095`, at most 95% of the original teacher runtime |
| Zero-shot evaluation | 128 samples × 16,384 tokens |
| AIPerf | teacher + `h4096-d4`; ISL 16,384, OSL 4,096; concurrency 1/4/16/64 |
| KD overfit | `h4096-d4`, 8 samples, 128 steps, LR `1e-4`, PP8 |
| Full KD | 256 steps, global batch 16, LR `1e-5`, CP4 × PP2 × DP8 |
| Post-KD evaluation | 128 samples × 16,384 tokens |

The MIP grid contains exactly seven width/depth scenarios. Depths 0–2 are
infeasible and depths 3–6 are feasible. The recorded zero-shot LM losses are:

| Solution | Removed sublayers | LM loss |
|---|---:|---:|
| Teacher | 0 | 0.8567323713 |
| `h4096-d3` | 3 | 0.8721836959 |
| `h4096-d4` | 4 | 0.8113155472 |
| `h4096-d5` | 5 | 0.8235977609 |
| `h4096-d6` | 6 | 0.8392351726 |

This is why `h4096-d4`, not the teacher, is the fixed AIPerf/KD candidate.

## 2. Environment and paths

Install the common Puzzletron environment exactly as described in
[`examples/puzzletron/README.md`](../README.md). Run every payload below inside
that same compute-node container and virtual environment.

```bash
export PUZZLETRON_RUN_ROOT=/absolute/shared/path/puzzle_runs
export CONFIG=examples/puzzletron/configs/clean/families/qwen3_5/qwen3_5_9b/sanity_reproduction.yaml
export SOURCE=${PUZZLETRON_RUN_ROOT}/qwen3_5/qwen3_5_9b/full_pipeline
export RUN=${PUZZLETRON_RUN_ROOT}/qwen3_5/qwen3_5_9b/sanity_check
```

Keep the model revision, dataset revision, source checkout revisions, CUDA
image, and GPU SKU in the experiment evidence. Exact vLLM/AIPerf timings are
hardware- and build-specific.

## 3. Produce or import the source bundles

First run the source campaign with `full_pipeline.yaml`, or reuse its completed
immutable bundles. The reference sanity campaign used the latter path:

```bash
python examples/puzzletron/import_campaign_artifacts.py \
  --source-root "${SOURCE}" \
  --destination-root "${RUN}" \
  --receipt "${RUN}/manifests/imports/campaign_artifacts.json" \
  --target-config "${CONFIG}" \
  --artifact activation \
  --artifact depth \
  --artifact vllm_stats \
  --artifact scoring \
  --artifact bypass_evidence
```

The receipt must list exactly those five bundles. Bypass is evidence-only in
the sanity campaign; it is not used as the scoring checkpoint.

## 4. Run the sanity DAG

Use the normal stage runner through the compute launcher documented in the main
README. Imported completion receipts cause the expensive source stages to be
reused; the remaining enabled stages execute in dependency order:

```bash
python examples/puzzletron/main.py \
  --config "${CONFIG}" \
  --stage full \
  --gpus-per-node 8
```

For a staged launch, use the same config for every command and run at least the
following payloads in order: convert, tokenize data, sort, forward/reverse sort
sanity, width/slicing sanity, bypass overfit sanity, MIP, zero-shot evaluation,
AIPerf, KD overfit, full KD, and post-KD evaluation. Never regenerate imported
activation/depth/runtime/scoring artifacts inside the sanity root.

The exact MIP payload in current code is:

```bash
python examples/puzzletron/run_width_depth_mips.py --config "${CONFIG}"
```

It realizes the four feasible checkpoints because this reference experiment
predates the Nano campaign's online-only MIP evaluation contract.

## 5. AIPerf and distillation

AIPerf evaluates only `teacher` and `h4096-d4`. The completed matrix has 24
rows: two models × three two-GPU topologies × four concurrency settings. The
topologies are TP2, PP2, and DP2; the exact matrix is in the reproduction config.

The 8-sample overfit must be healthy before full KD. The production KD topology
is CP4 × PP2 × DP8, so it requires eight eight-GPU nodes; it is not a one-node
payload. Preserve checkpoints at steps 128 and 256, then run the configured
128-sample post-KD LM-loss evaluation.

## 6. Report and verification

Regenerate the single-file report without rerunning model work:

```bash
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir "${RUN}" \
  --model-name Qwen/Qwen3.5-9B
```

The reference report is:

```text
../puzzle_runs/qwen3_5/qwen3_5_9b/sanity_check/artifacts/campaign_report/campaign_report.html
```

The current verified report contains 282,112 nested-bypass observations,
51,200 referenced candidates, 64 units, and all 19 pipeline stages. Its report
generator uses per-section snapshots, so unchanged sections are reused.
