---
name: preflight
description: >-
  Prepare a Model Optimizer workflow before execution. Use when the user asks to
  run a quantization, evaluation, deployment, comparison, recipe-search, or
  day-0-release workflow and the request is not yet execution-ready. Interview
  the user, select and verify skills, functionally validate credentials, verify
  tools and compute, write PROGRAM.md, and provide a /goal handoff. Do not
  execute the workflow itself.
license: Apache-2.0
---

# Model Optimizer Workflow Preflight

Turn a broad request into a validated, durable program. Stop at an
execution-ready `PROGRAM.md`. Pause whenever user input or credential setup is
required. Never guess missing decisions or continue past a failed required
check.

## 1. Interview the user

Summarize the requested outcome, then ask only the unresolved questions in one
short, numbered group. Offer concrete choices where possible. Resolve:

1. **Scope** — PTQ only, checkpoint validation, matched evaluation, recipe
   search, deployment, performance measurement, or the full `day0-release` workflow.
2. **Model** — repository ID or checkpoint path.
3. **Quantization** — requested recipe or recipe search.
4. **Acceptance** — checkpoint requirements and, for comparisons, exact tasks,
   score fields, and maximum acceptable degradation.
5. **Execution target** — local or configured remote environment.
6. **Outputs and constraints** — workspace and publication requirements.

## 2. Build the skill plan

Map the agreed scope to the required skill set:

| Scope | Required ModelOpt skills |
| --- | --- |
| PTQ | `ptq` and common environment/workspace guidance |
| Recipe exploration | `quant-recipe-search`, `ptq` |
| Serving | `deployment` |
| Accuracy evaluation | `evaluation`, `launching-evals`, `monitor` |
| Comparison | `compare-results`; `accessing-mlflow` when MLflow is used |
| Day 0 Release | `day0-release` and its selected domain skills |
| Kernel performance | `benchmark-model-kernels` |

## 3. Verify skills

Verify the current agent can discover every selected skill exists in the
installed `modelopt` plugin or at its canonical path.

If a skill is unavailable, stop and report:

- skill name and why it is needed;
- expected plugin or canonical path;
- setup action the user must complete.

## 4. Verify credentials

Derive the credentials needed from the selected skills and
concrete run. Inspect as applicable:

- selected model and dataset access requirements;
- task recipes and environment-variable placeholders;
- launcher, deployment, and export configuration;
- container registry and execution-target authentication;
- judge, tracking, and artifact destinations;
- generated dry-run configuration.

### Check presence safely

Never ask the user to paste secrets into chat. Never use `cat`, file-reading
tools, `env`, `printenv`, `set`, `set -x`, or `echo $SECRET` on `.env` or secret
values. Do not open `.env` with Read, Write, or Edit tools.

If setup is needed, update the gitignored workspace `.env` and reply only when
done. Seed it from a public `env.example` when available, without overwriting
an existing file. Load it without output:

```bash
set -a
source .env
set +a
```

Resolve this skill's installation directory and run its bundled checker with
variable names only:

```bash
python <preflight-skill-dir>/scripts/check_env.py NAME [NAME ...]
```

The checker reports only `set` or `missing`. Repeat the check in the
environment that submits or runs remote work.

### Prove credentials work

Run the least-expensive, non-destructive probe for every required credential
and destination, using the exact resource needed by the program when practical.
For example:

- authenticate with the relevant CLI and check access to the selected model or
  dataset;
- query a documented identity, models, health, or minimal-request endpoint
  without printing credentials;
- verify registry credentials and exact-image access using
  `common/slurm-setup.md` and `deployment/SKILL.md`;
- test SSH and scheduler identity on the selected target;
- for evaluation credentials and environment propagation, use
  `evaluation/SKILL.md` Step 8: dry-run, then the Step 8.2 limited-sample canary
  before any full run;
- use the serving canary in `ptq/references/checkpoint-validation.md` after PTQ;
- test tracking/export connectivity when publishing is required.

Preflight may run non-workload authentication and dry-run checks. Put canaries
that require deployment, evaluation samples, or a quantized checkpoint into
`PROGRAM.md` as the first execution gate after `/goal`; do not run them during
preflight.

Never expose credential values. Record only the credential name, consumer, safe
probe, status, and diagnostic. Redact responses that may expose account data.

## 5. Verify tools

Verify each tool where the workflow consumes it; do not require every tool
locally. Classify tool placement as:

- **agent/control environment** — tools used to prepare files or connect to the
  target;
- **submission environment** — scheduler, launcher, and transfer tools used on
  a remote login or submission host;
- **compute/container environment** — Model Optimizer source, Python packages,
  evaluator, serving runtime, and benchmark tools used by the workload.

Using the selected skills, verify:

- each required CLI, script, and package exists in its consuming environment;
- versions are compatible across environment boundaries;
- planned container images contain the required runtime tools and packages;
- the workspace is writable and visible where each stage runs;
- planned commands pass syntax/configuration validation or dry-run checks in
  the environment that will execute them.

Record tool, required location, version, and safe command evidence. Do not
install the same tool in multiple environments unless the plan consumes it in
each. Missing or incompatible required tools are blockers.

## 6. Verify compute

Verify the selected execution target:

- local or remote connectivity;
- accelerator type, count, memory, and architecture;
- compatibility with the requested quantization, deployment, and evaluation;
- scheduler partition/account or equivalent resource route when applicable;
- expected capacity, runtime limit, and workspace accessibility from compute;
- container image compatibility with the target architecture and accelerator.

## 7. Write `PROGRAM.md`

After all required checks pass, write `PROGRAM.md` at the workspace root using
`references/program-template.md` from this skill. Replace every angle-bracket
placeholder before presenting the program. Use `none` where a field is
inapplicable. Never include credential values.

The program must contain:

- one measurable goal and explicit completion criteria;
- resolved inputs and execution target;
- selected skills and why each is needed;
- environment and credential requirements with safe validation evidence;
- tool and compute evidence;
- ordered phases, gates, stop conditions, and retry limits;
- exact planned commands or files that will produce them;
- required artifacts and final report location;
- unresolved non-blocking assumptions.

## 8. Handoff with `/goal`

If an active goal explicitly invoked preflight before execution, return control
to that goal after writing `PROGRAM.md`; do not emit a nested `/goal` or execute
the workflow inside this skill.

Otherwise, check whether `/goal` is available in the current agent. If it is,
end with a copyable prompt tailored to the program:

```text
/goal Follow PROGRAM.md and complete its goal. Stop at every failed gate, preserve the required artifacts, and write the specified final report.
```
