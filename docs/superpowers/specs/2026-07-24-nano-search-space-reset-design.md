# Nano Search-Space Reset Design

## Goal

Reset the Nano Puzzletron campaign to a clean, reproducible starting point and
reduce every enabled pruning axis to exactly two values. Preserve the campaign's
model, dataset, topology, MIP, post-MIP, runner, and execution settings.

## Canonical Search Space

Each enabled axis retains the teacher value and one pruned value:

| Axis | Values |
| --- | --- |
| `hidden_width` | `[2688, 2304]` |
| `kv_groups` | `[2, 1]` |
| `q_heads_per_group` | `[16, 12]` |
| `moe_experts` | `[128, 96]` |
| `moe_expert_intermediate` | `[1856, 1344]` |
| `moe_shared_expert_intermediate` | `[3712, 2176]` |
| `moe_top_k` | `[6, 5]` |
| `mamba_heads` | `[64, 56]` |
| `mamba_head_dim` | `[64, 56]` |

The model inventory continues to record every legal value. Only the selected
campaign search space is reduced.

## Bundle Regeneration

Update the selected values in
`../puzzle_runs/nano/answers.yaml`, then regenerate both `smoke` and
`production` bundles through the setup bundle renderer. Do not manually patch
only the generated experiment YAMLs, because `answers.yaml` is the canonical
wizard state.

Validate that:

- both generated experiment YAMLs contain exactly the values above;
- every enabled axis has exactly two selected values;
- `moe_top_k` is enabled with `[6, 5]`;
- the production and smoke experiment configs load successfully;
- both orchestrator plans compile in dry-run mode; and
- unrelated campaign settings remain unchanged.

## Artifact Reset

Before deletion, confirm that no active Slurm job is using the Nano campaign.
Delete the complete generated artifact root:

`../puzzle_runs/nano/results/`

This removes production and smoke checkpoints, caches, manifests, logs,
orchestration state, reports, and partial worker outputs. Preserve:

- `../puzzle_runs/nano/answers.yaml`;
- `../puzzle_runs/nano/README.md`;
- `../puzzle_runs/nano/smoke/`; and
- `../puzzle_runs/nano/production/`.

The next orchestrator invocation therefore has no durable completion state and
runs the campaign from its first required stage.

## Safety and Verification

Resolve and print the deletion target before removing it. Refuse deletion if
the canonical result root differs from
`../puzzle_runs/nano/results` or an active Nano Slurm job is found. After the
reset, verify that the result tree is absent and that all preserved bundle files
still exist.

No orchestrator job is submitted as part of this reset.
