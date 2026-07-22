# Puzzletron Explicit Task Topology Design

## Problem

Puzzletron attempts currently describe allocated nodes and total GPUs, while the
executors infer how many scheduler tasks to launch. The Slurm executor equates
nodes with tasks and repeats the same command once per node. That is insufficient
to distinguish one distributed instance spanning several nodes from several
independent instances sharing an allocation. It produced a real failure in which
two nodes each started an identical rank-zero `torchrun` agent and waited forever.

The same ambiguity exists in the local and bare-metal executors. Bare metal also
leases GPUs from only one host, even when an attempt requests several nodes.

## Goals

- Represent an allocation and its task/process-group structure without inference.
- Support `N >= 1` nodes, `K >= 1` GPUs per node, `M >= 1` tasks, and `k >= 1`
  GPUs per task.
- Support one or more independent distributed groups within the allocation.
- Give every task a disjoint `k`-GPU slice.
- Preserve independent-instance execution: independent tasks must never join the
  same process group merely because they share nodes or an allocation.
- Apply the same validated topology to Slurm, bare-metal, and local execution.
- Keep existing one-task attempts backward compatible.

## Non-goals

- Changing Puzzletron model meshes or algorithm semantics.
- Combining work items that the adapters currently submit as separate attempts.
- Replacing stage-owned launchers in persistent-pool scripts.
- Adding topology inference from stage names, instance counts, or command text.

## Execution Model

For one attempt:

- `N` is the number of allocated nodes.
- `K` is the GPU capacity requested per node.
- `M` is the number of scheduler tasks launched across the allocation.
- `k` is the number of GPUs assigned to each task.
- `G` is the number of tasks in each distributed group.

The derived capacity and group count are:

```text
tasks_per_node = floor(K / k)
task_capacity = N * tasks_per_node
group_count = M / G
```

The compiler rejects a topology unless:

```text
1 <= k <= K
1 <= M <= task_capacity
1 <= G <= M
M % G == 0
```

Unused node capacity is allowed and reported. The runner never silently changes
`N`, `K`, `M`, `k`, or `G` to improve utilization.

Examples:

| Purpose | N | K | M | k | G |
| --- | ---: | ---: | ---: | ---: | ---: |
| One eight-GPU model | 1 | 8 | 1 | 8 | 1 |
| One sixteen-GPU model | 2 | 8 | 2 | 8 | 2 |
| Eight independent workers | 1 | 8 | 8 | 1 | 1 |
| Two independent sixteen-GPU models | 4 | 8 | 4 | 8 | 2 |
| Four independent two-GPU workers | 1 | 8 | 4 | 2 | 1 |

## Schema

Add a frozen `TaskTopology` value to `AttemptSpec` with backward-compatible
defaults:

- `task_count` (`M`, default `1`)
- `gpus_per_task` (`k`, default derived from the attempt's allocated GPUs for the
  legacy one-task case)
- `tasks_per_group` (`G`, default `1`)
- `launcher`, either `direct` or `torchrun` (default `direct`)
- `placement`, initially only deterministic `block`

`K` remains the attempt's `gpus_per_node` execution metadata and `N` remains
`allocation_nodes`. A topology validation helper is the single source of truth
used by the compiler/adapters and all executors.

For a `torchrun` task, the runner launches one local process per assigned GPU, so
the distributed world size of one group is `G * k`. A `direct` task receives the
same GPU and task identity environment but owns its internal process model.

The application command in `CommandSpec.argv` is the payload. Executors add the
backend-specific distributed launcher only when `launcher == torchrun`; they do
not detect `torchrun` by parsing command strings.

## Task and Group Identity

Tasks are placed in deterministic block order and receive:

- global task index `0 .. M-1`
- node-local task index
- group index `task_index // G`
- group rank `task_index % G`
- group size `G`
- a task-local `CUDA_VISIBLE_DEVICES` containing exactly `k` devices

Each group receives a rendezvous identity derived from the attempt ID and group
index. Groups never share an identity. The rendezvous master is the host running
group rank zero. Ports are deterministically namespaced per attempt and group,
validated to be in range, and surfaced in logs. A bind conflict fails that group
with an actionable error rather than falling back to a different group or
silently joining unrelated workers.

## Slurm Executor

The batch allocation requests `N` nodes and `K` GPUs per node. The `srun` step
launches exactly `M` tasks with deterministic block placement and `k` GPUs per
task. Slurm performs GPU allocation; the task wrapper verifies that exactly `k`
devices are visible before starting the payload.

For `direct` tasks, the wrapper executes the payload unchanged after exporting
task/group identity.

For `torchrun` tasks, the wrapper adds the calculated node/group rank, shared
master address and port, group rendezvous ID, `G` agents, and `k` local processes.
This fixes a single distributed instance across nodes and also supports several
independent distributed groups in one allocation.

The executor must not add rendezvous arguments to persistent-pool or independent
direct tasks. Existing execution-contract ordering remains unchanged: site setup,
then virtual-environment activation, then the payload.

## Bare-metal Executor

Replace the single-host attempt lease with an atomic multi-task lease plan:

1. Select up to `N` inventory hosts in stable inventory order.
2. Allocate at most `floor(K / k)` task leases per selected host.
3. Assign a disjoint `k`-GPU slice to every task.
4. Launch remote tasks with the same task/group identities as Slurm.
5. Record one PID and log per task in the durable handle metadata.

If allocation or launch fails partway, terminate already-launched tasks and
release every lease from the attempt. Polling succeeds only when every task exits
successfully; cancellation terminates all task PIDs and releases the full lease
set.

## Local Executor

Local execution supports only `N == 1`. It validates `M * k <= K`, assigns
disjoint local GPU slices, and launches `M` subprocesses. It uses localhost
rendezvous endpoints for `torchrun` groups. The job handle owns all child
processes and logs; failure of any required task fails the attempt and terminates
remaining siblings. Cancellation terminates the complete task set.

## Adapter Mapping

Adapters declare topology explicitly:

- A canonical single distributed model instance uses `M = N`, `G = N`,
  `k = K`, and `launcher = torchrun`.
- A single-node distributed model uses `N = M = G = 1`, `k = K`, and
  `launcher = torchrun`.
- Independent sharded work remains separate attempts with `G = 1`. Sharing a
  node or allocation does not change group membership.
- Packed stage-owned launchers remain one `direct` task when their script owns
  child-process creation.
- Persistent-pool gang scripts use `M = N`, `G = 1`, and `launcher = direct`
  because the scripts own worker-group formation.
- CPU/control-plane tasks use one direct task and retain their existing GPU-slot
  compatibility behavior on GPU-only partitions.

## Validation and Observability

Before submission, validation reports `N`, `K`, `M`, `k`, `G`, group count,
capacity, and unused GPUs. Invalid topology fails before scheduler submission.

Every task logs one binding line containing host, task index, local task index,
visible GPU set, group index, group rank, group size, master endpoint, and
rendezvous ID. This makes wrong placement or accidental group merging evident
without enabling verbose model logs.

## Compatibility and Migration

The default topology represents one direct task, preserving existing CPU and
single-process attempts. Adapters are migrated individually to explicit
topologies. No executor infers distributed behavior from `allocation_nodes`,
`instances`, stage names, or the presence of `torch.distributed.run` in argv.

Attempt persistence includes the complete topology so recovery uses the same
task/group contract. The execution contract hash includes task-topology semantics
so incompatible old and new attempts are not resumed as equivalent.

## Failure Handling

- Invalid capacity or grouping fails before submission.
- Missing scheduler task identity fails before payload launch.
- Wrong visible-GPU count fails before importing the model.
- Rendezvous timeout or port conflict fails only the affected attempt and names
  its group and endpoint.
- Partial bare-metal/local launch rolls back sibling tasks and leases.
- Slurm uses kill-on-bad-exit for grouped attempts so one failed task drains the
  remaining tasks in that attempt.

## Focused Test Plan

1. Validate accepted and rejected `N/K/M/k/G` combinations.
2. Render and inspect a two-node, two-task, one-group Slurm launch.
3. Render multiple independent tasks and assert that no shared group is created.
4. Render two distributed groups in one allocation and assert distinct group
   identities/endpoints with correct ranks and GPU counts.
5. Assert persistent-pool direct tasks are not wrapped in `torchrun`.
6. Verify atomic multi-host bare-metal leasing, partial-launch rollback, polling,
   and cancellation.
7. Verify local multi-task GPU slicing and aggregate lifecycle behavior.
8. Verify the legacy one-direct-task default remains unchanged.

Only focused orchestration unit tests and a generated Slurm-script smoke check are
required for this change; model tests and unrelated repository-wide tests are out
of scope.
