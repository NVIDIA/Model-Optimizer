# <Workflow name>

## Goal

<One measurable outcome and its completion criteria.>

## Inputs

| Field | Value |
| --- | --- |
| Source model/checkpoint | <ID or absolute path> |
| Revision | <immutable revision or reason it is unavailable> |
| Quantization format/recipe | <value or selection policy> |
| Evaluation tasks and score fields | <exact list or none> |
| Acceptance criterion | <exact gate> |
| Execution target | <local or configured target name> |
| Workspace | <absolute path> |
| Runtime constraints | <limits or none> |

## Skills

| Skill | Purpose |
| --- | --- |
| <name> | <why selected> |

## Environment and credential requirements

| Requirement | Kind | Consumed by | Validation evidence |
| --- | --- | --- | --- |
| <name only> | <secret env/non-secret env/persisted credential/literal config> | <component> | <safe probe summary> |

## Tool preflight

| Tool/script/package | Required location | Version/path | Evidence |
| --- | --- | --- | --- |
| <name> | <agent/submission/compute/container> | <version or path> | <safe evidence> |
| Configuration validation | <execution location> | <command/config path> | <safe result> |

## Compute preflight

| Check | Evidence |
| --- | --- |
| Execution target | <safe connectivity evidence> |
| Accelerator compatibility | <hardware and constraint> |
| Resource route and limits | <scheduler/runtime evidence or none> |
| Workspace visible from compute | <safe evidence> |

## Execution plan

### Phase 1 — <name>

- Skill: `<skill>`
- Inputs: <concrete inputs>
- Planned command/config: <exact command or file that produces it>
- Gate: <pass condition>
- On failure: <stop/diagnose/retry rule>
- Artifacts: <required paths or IDs>

<Repeat for each phase.>

## Global stop and retry rules

- Stop when <blocking conditions>.
- Retry <transient operations and limits>.
- Never advance with <invalid/incomplete artifacts>.

## Required outputs

| Artifact/report | Required contents | Destination |
| --- | --- | --- |
| <artifact> | <contents> | <absolute path or destination> |

## Assumptions and non-blocking unknowns

- <assumption or `none`>

## Final report

Write `<absolute report path>` with the outcome, gate results, reproducibility
metadata, artifact locations, issues, and follow-ups.
