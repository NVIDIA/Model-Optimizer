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
layer sets or spans. It uses the same risk objective and two-pass selection rule
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
  --compact-captures discovery.jsonl --capture-manifest discovery.jsonl.manifest.json \
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

The refit searches each context range independently. Every exported candidate must
meet the requested model-wide BMM1 skip ratio in each range. For each target
sparsity it first minimizes the largest calibration-prompt dropped mass, summed
over anchor and consumer heads and normalized by the number of attention heads
across all full-attention layers. The second pass minimizes combined tile cost
under one ceiling shared by the targets: the smallest observed risk plus
`1e-7` in normalized units. Equal tile cost favors the smaller target.
Discovery applies the same rule to target tuples sharing one anchor layout.
The optional legacy anchor cap keeps its consumer-only risk definition.

With HiGHS, discovery first prices the two things that couple the layers: the
worst-prompt maximum and each range's work target. The priced value is a lower
bound on the risk of every policy. It also yields an exactly replayed starting
policy, and the integer program omits every anchor edge and donor option that
no policy at or below that policy's risk can use. A target tuple whose bound
exceeds the ceiling is not solved. The cost pass prices the risk rows the same
way. When the solver cannot prove the smallest risk within its budget but the
bound is within `1e-7` of the best replayed policy, the ceiling is the bound
plus `1e-7`, and the report records `risk_certificate: lower_bound` instead of
`optimum`.

HiGHS uses one thread. The solver checks the objective against its returned
assignment, and the calibrator replays the exported policy against the capture.
The default refit budget is 1,600 seconds per range. Discovery shares a
600-second budget across all target tuples. `--solver-time-limit-seconds`
overrides the corresponding budget. `--previous-candidate` can supply an initial
assignment for a matching compact capture. Omit it to measure a fresh refit.

The report records available primal and dual bounds, completed stages, and
the policy's recounted risk and work. A feasible policy can be exported on a
timeout. A within-tolerance result requires bounds supporting both the risk
ceiling and the cost comparison across qualifying targets. An unresolved run
without a verified candidate returns exit 3. An established unmet work target
returns exit 2 with a report. `--allow-unmet-target` permits diagnostic output.

For qualification, evaluate the pinned policy against dense attention on the
same checkpoint with the same benchmark configuration. The accuracy gate
requires every designated benchmark to regress by at most one percentage point,
as well as whole-prefill replay and the requested work saving in every range.

Held-out observations evaluate the frozen choices without retuning them.
Outputs retain `promotion_status="candidate_only"` and
`deployment_geometry_validated=false`. Serving must reject these candidates
until the independent quality and runtime-validation gates pass. Production-size
discovery completion remains unqualified.
