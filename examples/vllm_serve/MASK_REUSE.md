# Offline cross-layer mask-reuse calibration

This workflow builds a **candidate** policy from dense-reference statistics.
It does not promote a serving policy or establish task accuracy. The capture
provider is a separate runtime package; install and validate that provider
before collecting real-model data. Decode remains dense.

Install the optional exact solver from the repository root:

```bash
pip install -e '.[mask-reuse]'
```

Run the following commands from `examples/vllm_serve`. First calibrate vanilla
skip-softmax through `calibrate_sparse_attn.py`, as described in the main
README. Its exported fit must include observed sparsity bounds. Mask-reuse
trials must stay within those bounds.

## Bind the checkpoint and capture provider

```bash
python create_checkpoint_manifest.py <CHECKPOINT> --model-id <MODEL-ID> \
  --checkpoint-manifest checkpoint-manifest.json
python create_fa4_source_witness.py <PROVIDER-CHECKOUT> \
  --source-kind mask-reuse-reference --expected-commit <COMMIT> \
  --archive-output provider.tar --manifest-output provider-manifest.json
mkdir provider
tar -xf provider.tar -C provider
```

Keep the printed hashes separately. The reference provider archive contains
`src/attention_backend/`; its package is under `src/attention_backend/src/`.
Checkpoint files and provider source must remain unchanged throughout capture.
An existing checkpoint manifest should be reused, rather than replaced.

## Capture the entire prefill

```bash
python collect_mask_reuse.py <CHECKPOINT> --model-id <MODEL-ID> \
  --checkpoint-manifest checkpoint-manifest.json \
  --checkpoint-manifest-sha256 <CHECKPOINT-MANIFEST-SHA256> \
  --plan <CAPTURE-PLAN> \
  --capture-provider reference --capture-scope whole-prefill \
  --provider-source provider --provider-source-manifest provider-manifest.json \
  --provider-source-manifest-sha256 <PROVIDER-MANIFEST-SHA256> \
  --provider-commit <COMMIT> \
  --prompts-jsonl prompts.jsonl --vanilla-config sparse_attention_config.json \
  --target-sparsities 0.5 0.6 0.7 --tensor-parallel-size 8 \
  --max-model-len <MAXIMUM-PROMPT-LENGTH-PLUS-ONE> --output captures.jsonl
```

Each prompt JSONL record has `split`, `partition`, `inner_fold`, `prompt_id`,
`source`, `source_group_sha256`, `prompt`, `min_kv_tokens`, and `max_kv_tokens`.
Use `calibration` / `development` with a nonnegative fold, or `heldout` /
`outer_test` with a null fold. A source group cannot cross folds or partitions.
Context ranges are inclusive and cannot overlap, including at their endpoints.

The collector tokenizes and validates the plan before loading vLLM. It forces
batch-one eager execution and BF16 KV, preserving checkpoint weight precision.
Engine overrides are restricted to an explicit runtime allowlist. Raw counts
and dropped masses are merged across tensor-parallel ranks; ratios are not
averaged across unequal shards.

Whole-prefill capture is the default. Every chunk is accounted for, including
explicit dense exclusions when a threshold disables sparsity or a chunk is too
short. These exclusions contribute to workload accounting but are not measured
sparse observations. A target met within each calibrated range does not imply
the same saving over an arbitrary request mixture.

Progress is journaled. Resume into a new output directory with
`--resume-from <PREVIOUS-DIRECTORY>`; source, model, prompt, and runtime identities
must match. Completed chunks are retained when a later request fails.

The FA4 provider's final-chunk capture remains diagnostic. Calibrators reject
it unless `--diagnostic-final-chunk` is supplied. That flag does not make the
data representative of the entire prefill. `--validate-dense-output` is a
FA4-only bitwise shadow check; the reference provider needs its own accuracy
validation.

## Refit a fixed topology

```bash
python calibrate_mask_reuse.py --checkpoint <CHECKPOINT> \
  --checkpoint-manifest checkpoint-manifest.json \
  --compact-captures captures.jsonl --capture-manifest captures.jsonl.manifest.json \
  --vanilla-config sparse_attention_config.json --topology topology.json \
  --calibration-plan calibration-plan.json --family-registry family-registry.json \
  --grouped-fit grouped-fit.json --outer-report outer-report.json \
  --anchor-dropped-mass-report-threshold 0.02 \
  --reuse-dropped-mass-report-threshold 0.02 --target-bmm1-skip-ratio 0.50 \
  --output-policy candidate.json --output-report report.json
```

The topology must cover exactly the checkpoint's full-attention layers and
query heads. Optional `--topology-sha256` pins the topology bytes. Evidence
files must be nonempty JSON; their hashes bind the candidate to those files.
This structural check does not replace the end-to-end quality protocol.

For each context range, the selector meets the model-wide BMM1 skip target,
then minimizes worst-prompt anchor-plus-consumer dropped mass. It subsequently
minimizes tile cost and uses deterministic lexicographic tie-breaks. The two
reported mass thresholds are diagnostics, not hard acceptance caps. The
optional `--max-anchor-dropped-mass` selects the older capped formulation.

HiGHS is the default; `--solver cbc` selects the hardened fallback.
`--solver-time-limit-seconds` bounds the complete selection session. Every stage
requires a proved optimum with zero native MIP gap; a feasible incumbent alone
is insufficient. Unresolved solves return exit code 3 and a report. An
infeasible BMM1 target returns code 2 and a report, without exporting a policy.
`--allow-unmet-target` explicitly requests a diagnostic maximum-saving candidate.
A zero target is valid and may produce a dense policy.

Held-out observations evaluate the frozen choices without retuning them.
Outputs retain `promotion_status="candidate_only"` and
`deployment_geometry_validated=false`. Serving must reject these candidates
until the independent quality and runtime-validation gates pass.
