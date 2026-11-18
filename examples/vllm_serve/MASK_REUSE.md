# Offline cross-layer mask-reuse calibration

This workflow builds a **candidate** policy from dense-reference statistics.
It does not promote a serving policy or establish task accuracy. The capture
provider is a separate runtime package; install and validate that provider
before collecting real-model data. Decode remains dense.

Install the optional MILP solver from the repository root:

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

## Discover anchors, then freeze the topology

For discovery, collect with a `*_topology_discovery` plan and an explicit
`--max-reuse-span`. The span bounds the number of consecutive consumers between
anchors; it does not change the objective within that search space.

```bash
python calibrate_mask_reuse_topology.py --checkpoint <CHECKPOINT> \
  --checkpoint-manifest checkpoint-manifest.json \
  --captures discovery.jsonl --capture-manifest discovery.jsonl.manifest.json \
  --vanilla-config sparse_attention_config.json --prompt-plan prompts.jsonl \
  --reuse-dropped-mass-report-threshold 0.02 --target-bmm1-skip-ratio 0.50 \
  --output topology.json
```

Discovery enforces the BMM1 target in **every context range**, rather than only
over pooled prompts. It rejects duplicate full prompt sources and inconsistent
layer sets or spans. It uses the same risk objective and solver optimality checks
as refitting. A pooled maximum-saving diagnostic is not a per-range guarantee.
Both calibrators verify each whole-prefill observation against its parent plan.
Different-length prompts may share prefixes within one source group; all planned
observations retain their original multiplicities.

Freeze the exported `anchors` and `nearest` assignments. Pass the discovery
capture file and its manifest to the refit command below, with the complete
discovery artifact as `--topology`. Refit streams only the chosen anchor and
consumer matrices from those captures; no second GPU capture is required.
The candidate records both source and topology hashes. Checkpoint identity,
head count, layer coverage, captured span, and target status are checked before
refitting. Discovery never selects using held-out data.

## Refit a fixed topology

```bash
python calibrate_mask_reuse.py --checkpoint <CHECKPOINT> \
  --checkpoint-manifest checkpoint-manifest.json \
  --compact-captures captures.jsonl --capture-manifest captures.jsonl.manifest.json \
  --vanilla-config sparse_attention_config.json --topology topology.json \
  --anchor-dropped-mass-report-threshold 0.02 \
  --reuse-dropped-mass-report-threshold 0.02 --target-bmm1-skip-ratio 0.50 \
  --output-policy candidate.json --output-report report.json
```

The topology must cover exactly the checkpoint's full-attention layers and
query heads. Context ranges need finite maximum lengths. Optional
`--topology-sha256` pins the topology bytes. Capture, checkpoint, fit, and
topology identities are mandatory. The four additional provenance JSON flags
are optional; supplying them records their hashes without certifying their
contents or benchmark quality.

For each context range, the selector meets the model-wide BMM1 skip target,
then minimizes worst-prompt anchor-plus-consumer dropped mass. It subsequently
minimizes tile cost and uses deterministic lexicographic tie-breaks. The two
reported mass thresholds are diagnostics, not hard acceptance caps. The
optional `--max-anchor-dropped-mass` selects the older capped formulation.

HiGHS is the default; `--solver cbc` selects the hardened fallback. Every
stage requires optimal status with zero native MIP gap. The returned assignment
must reproduce the reported objective. HiGHS starts with presolve disabled;
an inconsistent assignment or nonzero gap triggers a fresh, unseeded solve with
presolve enabled. A second failure remains unresolved rather than accepting a
feasible incumbent or a positive gap.

Fixed risk is rechecked using exact arithmetic on the captured coefficients,
with an eight-ULP allowance for expression accumulation. Distinct donor risk
vectors separated by at most native feasibility precision (`1e-9` in raw
per-head mass) remain unresolved; they are not rounded into ties. Native
optimal status is not a rational-arithmetic proof of arbitrary sub-tolerance
ordering.

`--solver-time-limit-seconds` is a shared cooperative selection budget checked
during model construction. Native operations may overrun it; use a
process or scheduler deadline when a hard wall-time limit is required.
Unresolved solves return exit code 3 and a report. An infeasible BMM1 target
returns code 2 and a report without a policy. `--allow-unmet-target` explicitly
requests a diagnostic maximum-saving candidate. A zero target is valid and may
produce a dense policy.

Held-out observations evaluate the frozen choices without retuning them.
Outputs retain `promotion_status="candidate_only"` and
`deployment_geometry_validated=false`. Serving must reject these candidates
until the independent quality and runtime-validation gates pass. Production-size
discovery completion remains unqualified.
