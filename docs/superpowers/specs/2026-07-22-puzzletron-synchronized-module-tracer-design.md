# Puzzletron Synchronized Module Tracer Design

## Context

The Qwen3.5 MoE width-sanity stage consistently stops while scoring
`realized_0008`, which physically changes layer 4's shared-expert intermediate
width from 512 to 256. The hidden-width teacher and sliced-teacher passes and
the preceding eight realized candidates complete.

Batch tracing narrows the stop to the first candidate forward pass. Pipeline
stage 0 remains busy on GPUs 0-3, while pipeline stage 1 returns a hidden-state
capture and then waits in the existing `torch.cuda.synchronize`. This proves
that the orchestrator, metric code, and cache transfer are not the immediate
cause, but it does not identify which asynchronous operation on pipeline stage
0 fails to complete.

`TORCH_DISTRIBUTED_DEBUG=DETAIL` cannot provide that resolution in this
environment because its process-group wrapper makes DTensor LM-head
materialization fail with an unsupported
`allgather_into_tensor_coalesced` operation. The diagnostic rerun must continue
to use `TORCH_DISTRIBUTED_DEBUG=INFO`.

## Goal

Add a temporary, opt-in tracer that brackets selected layer 4 modules with CUDA
synchronization points. Its last emitted marker must identify the narrowest
module whose queued GPU work does not complete.

The tracer is diagnostic instrumentation, not a workaround. It must have no
behavioral or performance effect when disabled.

## Design

### Activation and scope

The tracer is enabled only when:

```text
PUZZLETRON_TRACE_MODULE_SYNCS=1
PUZZLETRON_TRACE_MODULE_LAYER=4
```

`PUZZLETRON_TRACE_MODULE_LAYER` is required when synchronized module tracing is
enabled and must be a non-negative integer. Requiring an explicit layer avoids
accidentally synchronizing every decoder block during a long-sequence run.

The scoring path installs hooks only around candidate capture iteration. Hook
handles are removed in a `finally` block (or an equivalent context manager), so
exceptions and early exits cannot leave instrumentation attached to the model.
Teacher capture and ordinary runs remain unchanged.

### Trace points

For the selected decoder layer, install hooks on the narrowest useful module
boundaries that exist in the Qwen3.5 AutoModel implementation:

- decoder block / distributed wrapper boundary;
- attention or GDN submodule;
- complete MoE MLP;
- shared experts;
- shared-expert `gate_proj`, `up_proj`, and `down_proj` projections;
- routed experts.

Module discovery must use the model's existing decoder-layer structure and
stable logical labels, rather than assuming that every wrapper exposes an
identical fully qualified module name. A missing optional trace point is
reported once and does not change model execution. Failure to locate the
requested decoder layer is an actionable error.

### Marker protocol

Each marker is a single flushed line using a stable prefix and fields:

```text
[solution/automodel/module-trace] rank=0 layer=4 module=shared_experts phase=enter
```

A forward pre-hook emits:

1. `phase=enter`
2. synchronizes the local CUDA device
3. `phase=inputs_synchronized`

A forward hook emits:

1. `phase=returned`
2. synchronizes the output tensor's CUDA device, or the current local CUDA
   device when the output is nested or contains no directly discoverable CUDA
   tensor
3. `phase=output_synchronized`

This ordering distinguishes three cases:

- `enter` without `inputs_synchronized`: work queued before this module did not
  complete, so attribution remains with the preceding module boundary;
- `inputs_synchronized` without `returned`: the current module call did not
  return from Python (including any synchronous communication it performs);
- `returned` without `output_synchronized`: the module queued GPU work that
  did not complete;
- `output_synchronized` followed by no child/sibling progress: the failure is
  after that module boundary.

Hooks must not introduce distributed collectives, tensor copies, or structured
logging dependencies. They write directly to the existing process output and
flush immediately. Synchronization is intentionally expensive because the
feature is diagnostic-only.

### Implementation shape

Keep environment parsing, device selection, marker emission, module discovery,
and hook lifetime in a small private helper near AutoModel solution scoring.
The scoring loop should need only one context-manager boundary around candidate
capture iteration. The helper should remain reusable by a focused unit test
without constructing a real Qwen model or distributed process group.

## Test contract

Develop the implementation test-first with a small CPU-only fake model that has
two decoder blocks and fake attention, shared-expert projections, and routed
experts. Mock CUDA synchronization and capture emitted markers.

Focused tests must establish that:

- disabled tracing installs no hooks and performs no synchronization;
- enabling layer 4 ignores other layers;
- selected module markers appear in the documented order;
- the selected synchronization device is propagated correctly;
- all hooks are removed after normal exit and after an exception;
- an invalid layer value or a missing requested layer raises a clear error;
- absent optional submodules are reported without preventing the forward pass.

The first test run must fail because the helper does not exist. After the
minimal implementation passes, run the surrounding Puzzletron AutoModel unit
tests and the repository's required formatting/static checks for touched files.

## Diagnostic rerun

After tests pass:

1. Verify no prior width-sanity controller or Slurm retry remains active.
2. Remove or quarantine only partial controller state and empty validation
   output for `realized_0008`; preserve completed teacher artifacts and physical
   checkpoints.
3. Dry-run the exact production experiment, runner, and execution configs.
4. Launch only `--stage width_sanity` through the user's orchestrator virtual
   environment with the two tracer variables above, existing batch tracing,
   NCCL diagnostics, and `TORCH_DISTRIBUTED_DEBUG=INFO`.
5. Monitor the log until the final synchronized-module marker identifies the
   stalled boundary, then cancel the diagnostic job rather than allowing
   automatic retries to consume resources.

The rerun uses the user's existing `PUZZLETRON_RUN_ROOT`, experiment, runner,
and execution files. It does not delete completed artifacts or mutate a model
checkpoint.

## Success criteria

- Normal execution is behaviorally unchanged when tracing is disabled.
- Unit tests verify activation, layer filtering, marker ordering, device
  synchronization, error handling, and cleanup.
- The isolated width-sanity rerun produces an unambiguous last marker at or
  below the layer 4 shared-expert/routed-expert boundary.
- The resulting evidence is sufficient to propose and test a root-cause fix;
  the tracer itself is not presented as that fix.
