# MCP scope policy — environment tooling, not workflow policy

This applies to `modelopt-mcp` and its sibling servers
(`nmm-sandbox-mcp`, `pensieve-intern-mcp`).

## The principle

These MCP servers host **environment tooling** — operations on the
cluster, launcher, or agent engine that are *universal across
workflows*. They do **not** host workflow-specific logic.

Workflow-specific logic — "run an EAGLE3 training cell", "publish a
specdec release" — lives in **SPEC text + agent reasoning**, composed
out of the environment primitives below.

## The test

A tool belongs in this MCP family if and only if it would be useful
on its own across *any* workflow that uses the same environment.

| ✅ Environment tooling | ❌ Workflow policy |
|---|---|
| `submit_job(yaml_path, ...)` | `bench_eagle3_against_baseline()` |
| `resolve_cluster_factory(name)` | `create_specdec_release_pr()` |
| `verify_setup(executor, ...)` | `run_qwen3_quantize_sweep()` |
| `open_draft_pr(target_repo, ...)` | `dispatch_intern_epic(workflow, ...)` |
| `read_cluster_artifact(experiment_id, path)` | `auto_skip_lts_failure()` |

## Symptoms a tool is misclassified

- Tool name contains a workload identifier (eagle3, specdec, a model
  family, a specific algorithm)
- Tool's job is "do these N steps in this order" rather than one
  well-defined operation
- Changes to a workflow's policy (a new model, a new sweep dimension,
  a renamed stage) require changing the tool's code
- Two unrelated workflows would *not* naturally compose the tool

If any of these is true, the abstraction belongs in the **composition
layer** (SPEC text + the agent's reasoning), not the **primitive
layer** (MCP).

## Why the line matters

Today's three MCPs deliberately host a small, closed set of
verb-shaped operations on the cluster, launcher, and engine.
That choice:

- Keeps the agent's tool catalog learnable → less hallucination,
  shorter reasoning chains
- Makes the MCPs **pre-knowledge** every workflow can rely on
  without per-SPEC opt-in (the SPEC stops carrying CLI invocation
  details; the tool description IS the documentation; the schema
  IS the contract)
- Collapses interface-drift blast radius: a launcher refactor →
  the MCP's bridge layer absorbs the change → SPECs are unaffected.
  (Compare to today's failure mode, where renaming `launch_train.sh
  --model` → `--config` silently broke every SPEC that hardcoded
  the flag form.)

If we cross the line and add workflow-specific MCP tools, the
catalog sprawls and the value collapses. Each new workflow wants
its own tools; old SPECs reference tools they no longer need;
the model spends its budget on tool discovery instead of reasoning;
the "pre-knowledge" promise breaks because new tools require
per-workflow opt-in.

## Practical guidance for adding new tools

Before adding a tool, ask:

1. **Would this tool exist whether or not workflow X existed?**
   If no, it's workflow policy. Compose it in a SPEC instead.
2. **Does the tool's signature contain any workflow-specific knobs?**
   If yes, those knobs *are* the workflow policy. Refactor to a
   primitive that takes generic args.
3. **Would two unrelated workflows naturally compose this tool?**
   If yes, it's environment tooling. Add it.
4. **Is the tool's output the same shape across all callers?**
   If no, the tool is doing workflow-specific shaping in disguise.

When a piece of work feels like it wants an MCP tool but fails
these tests, the right move is usually to add a *helper module*
(Python in `modules/Model-Optimizer/tools/...`) and let SPECs
invoke it via shell — or, better, to refactor the work so it
composes the existing environment tools.

## Current scope (reference)

The 14 tools currently in scope, by server:

- **modelopt-mcp** (9): `list_examples`, `verify_setup`, `submit_job`,
  `job_status`, `job_logs`, `wait_for_experiment`,
  `provision_passwordless_ssh_dry_run`, `read_cluster_artifact`,
  `open_draft_pr`
- **nmm-sandbox-mcp** (3): `list_internal_clusters`,
  `resolve_cluster_factory`, `submit_via_gitlab_ci`
- **pensieve-intern-mcp** (2): `clear_labels`, `report_verified`

Phase 3 plans (OMNIML-5133 NEL integration, OMNIML-5134 checkpoint
introspection) extend this set with more environment primitives;
both pass the test above.
