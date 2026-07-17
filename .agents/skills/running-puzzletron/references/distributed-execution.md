# Distributed Execution

## Persist the Execution Contract

Before any task, record the container image/digest, mounts, repository root,
venv, required site setup, cache roots, sibling revisions, and experiment-
scoped environment overrides. Apply the same ordered prologue to every Python
command, including CPU conversion, tokenization, reports, and tests. Enforce any
cluster policy that permits only shell utilities or Python standard library on
login nodes.

Hash this contract into resumable identities. Source site setup before venv
activation when it sets caches or compiler state. Reuse a persistent allocation
for short iterations; use interactive capacity for unstable debugging and batch
for validated work. Bind disjoint `CUDA_VISIBLE_DEVICES` for partial-node tasks
and avoid exclusive nodes when GPUs are intentionally unused.

## Parallelism and Model Instances

Ask for TP, CP, PP, EP, DP, and sequence parallelism separately for smoke and
production. Validate backend constraints rather than assuming all named degrees
multiply independently.

For AutoModel FSDP/HSDP, record the realized mesh as
`(PP, DP_REPLICATE, DP_SHARD, CP, TP)`:

- `DP_SHARD` is the FSDP shard group; it collectively owns one model replica.
- `DP_REPLICATE` is the HSDP replica axis and creates sample-parallel lanes.
- EP overlays `DP_SHARD`; it is not another multiplicative allocation axis.
- Logical sample DP is `DP_REPLICATE × (DP_SHARD / EP)` when EP fully overlays
  the shard axis.
- Allocation is `PP × DP_REPLICATE × DP_SHARD × CP × TP`.

For example, `(2, 8, 4, 1, 1)` with EP4 consumes 64 GPUs and has logical DP8.
Verify the realized mesh in logs. Derive PP microbatch divisibility and update
counts from logical sample DP, not raw shards or GPU count. Increase batch only
after measuring memory headroom and preserving sample/identity semantics.

Classify each stage by model instance:

- Width importance is one coordinated instance with one global hook/reduction
  set. Scale through logical DP/HSDP; never merge independent width models.
- RPC depth uses multiple independent persistent model instances, each scoring
  different cumulative replacement requests.
- Sharded vLLM statistics may use many independent one-GPU instances because
  each immutable layout measurement is self-contained.

## Execute the DAG

Use `modelopt/torch/puzzletron/stages/graph.py` as dependency authority.
Conversion/tokenization may overlap vLLM stats; depth may overlap coordinated
width importance; vLLM may overlap bypass when allocations and writers are
disjoint. Only one writer publishes a canonical identity.

## RPC Iterative Depth

Use `modelopt.torch.puzzletron.distributed_eval`. One worker group is one
persistent distributed model instance that loads the sorted teacher and target
cache once, then evaluates many cumulative no-op requests. A group may use part
of a node, one node, or multiple nodes. Use the coordinator; do not add a second
filesystem scheduler.

Depth iterations are sequential; candidates within one iteration are parallel:

1. Load the durable selected prefix and cached results.
2. Submit the prefix baseline and each `prefix + candidate` request. Merge
   multiple removed sublayers in one layer into one replacement.
3. Apply cumulative replacements in temporary prune contexts, score, and
   restore the resident model before another request.
4. Rank the complete iteration atomically, publish trajectory, then continue.

Identity every request by model, data, metric, precision, cumulative removals,
and evaluator revision. Retry only missing/transient requests. The coordinator
alone writes rankings and trajectory.

When multiple groups share a node, assign disjoint CUDA devices, unique HTTP
ports, rendezvous IDs/ports, and worker IDs. All groups in one campaign must
match its declared world size/topology. Drain and terminate them explicitly.

## Resumable Expensive Stages

Bypass, KD, scoring, and vLLM stats use immutable shards and transactional
checkpoints. A training checkpoint stores model/optimizer shards, scheduler,
scaler, global step, Python/NumPy/CPU/CUDA/sampler/per-rank RNG, dataloader
cursor/sample order, topology/world size, config/code/model/data identities,
manifest, and atomic completion marker.

Write to a temporary transaction, validate expected shards, publish atomically,
then update `latest`. Quarantine incomplete transactions. Smoke must interrupt
and resume at least bypass and global KD, verifying exact step/sample continuity
and report validity.

Any semantic change—LR, token budget, estimator schema, physical construction,
workload, or execution contract—creates a new identity and invalidates selected
downstream dependencies. Never merge old/new shards based only on matching
filenames or shapes.

## Slurm

Derive launch commands from the cluster guide. Capture job IDs/logs with
`pipefail`. Checkpoint well before wall time and queue compatible resumable
continuations. Cancel redundant continuations after valid completion.

Use persistent allocations when startup dominates short tests, but release them
when GPUs would idle. Run stable paths on batch and debug unresolved paths on
interactive. For sharded single-GPU work, request one GPU per task, avoid
exclusive nodes, and verify scheduler binding inside every worker.

## Bare Metal

Verify passwordless SSH, paths, clocks, ports, GPU visibility, and environment
compatibility on every host. Choose one rendezvous host/port, create
deterministic rank mappings, launch one process group, record PIDs/logs/hashes,
health-check all ranks, and terminate all peers after a local failure. Shared
storage does not prove identical Python/CUDA environments.

