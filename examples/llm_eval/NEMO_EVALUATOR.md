# Run text benchmarks with NeMo Evaluator

This guide covers the Model Optimizer files that prepare NeMo Evaluator task
configurations. For general launcher setup and deployment options, use the
[NeMo Evaluator self-hosted quickstart](https://docs.nvidia.com/nemo/evaluator/latest/get-started/quickstart/index.html#self-hosted-options).
For Puzzletron backend selection, use the
[Puzzletron text evaluation guide](../puzzletron/evaluation/text/README.md).

## Task definitions

[`task_contracts.yaml`](task_contracts.yaml) is the machine-readable source for
the maintained NeMo task definitions. It owns task names, evaluation images,
sample or repeat counts, dataset settings, and required external-role fields.
[`nel_config.py`](nel_config.py) validates those definitions and adds selected
tasks to an existing launcher config.

The compiler defaults to the benchmarks missing from Puzzletron's pinned
`lmms-eval` revision:

- LiveCodeBench v6: `ns_livecodebench`
- SciCode: `ns_scicode`
- IFBench: `ns_ifbench`

The other definitions are explicit alternatives or tasks that require extra
configuration:

| Task | Evaluator image | Extra configuration |
| --- | --- | --- |
| `gpqa_diamond_aa_v3` | simple-evals | None |
| `mmlu_pro_aa_v3` | simple-evals | None |
| `AIME_2025_aa_v2` | simple-evals | Judge model and endpoint |
| `ns_hle_aa` | nemo-skills | Judge model and endpoint |
| `ns_aa_lcr` | nemo-skills | Judge model, endpoint, and task parallelism |
| `tau2_bench_telecom` | tau2-bench | User and judge models, endpoint, and task parallelism |

The task recipes under
[`../../.agents/skills/evaluation/recipes/tasks`](../../.agents/skills/evaluation/recipes/tasks)
contain operational requirements and score-extraction guidance. They do not
contain a second copy of the executable task YAML.

## Generate a config

Start with a launcher config that defines the deployment, model sampling,
execution resources, cache mounts, results location, and task-independent
evaluation settings.

Add the three default tasks:

```bash
python examples/llm_eval/nel_config.py \
  --base-config path/to/base_nel_config.yaml \
  --output path/to/text_benchmarks.yaml
```

Select tasks explicitly by repeating `--task`:

```bash
python examples/llm_eval/nel_config.py \
  --base-config path/to/base_nel_config.yaml \
  --task gpqa_diamond_aa_v3 \
  --task ns_scicode \
  --output path/to/reasoning_and_code.yaml
```

Run `python examples/llm_eval/nel_config.py --help` for the complete task list
and the options required by judge-scored or agentic tasks.

The compiler:

- preserves unrelated tasks and base-config fields;
- accepts an existing selected task only when it matches the maintained
  definition;
- rejects conflicting definitions and existing output files;
- does not launch jobs, download data, or accept secret values.

## External models and credentials

AIME uses `JUDGE_API_KEY`. HLE, AA-LCR, and Tau2 use
`INFERENCE_API_KEY`. Export the required credential in the submitting
environment; never put its value in YAML or a command-line argument. Generated
configs contain only the environment-variable name.

Judge and user-simulator model IDs and URLs are non-secret configuration. Pass
them through the compiler options so the generated config records the exact
external models used. Keep those values fixed when comparing checkpoints.

## Validate and run

Use one generated config for dry-run, smoke, and full execution. A reduced run
may add the launcher's global `limit_samples` override, but it must not change
the task definition, prompts, repeats, sampling, judge, sandbox, or scoring.

First validate the rendered configuration:

```bash
nel run --config path/to/text_benchmarks.yaml --dry-run
```

Then run a two-record smoke for one task:

```bash
nel run --config path/to/text_benchmarks.yaml \
  -t <task-name> \
  -o ++evaluation.nemo_evaluator_config.config.params.limit_samples=2
```

Inspect the deployment, client, and harness logs plus the response and result
artifacts. Require completed requests, non-empty outputs, the expected metric,
and no authentication, rate-limit, sandbox, or systematic truncation errors.
A successful scheduler exit alone is insufficient. A smoke validates the
runtime path; it is not an accuracy result.

After the smoke passes, run the same config without `limit_samples`:

```bash
nel run --config path/to/text_benchmarks.yaml
```

## Task-specific checks

- LiveCodeBench and SciCode execute generated programs. Use the sandbox
  provided by the evaluation environment and treat generated code as untrusted.
- SciCode needs enough deployment context for its background material. Follow
  its task recipe before launching.
- Repeated IFBench outputs ending at the generation limit indicate truncation;
  do not accept that run's score.
- Keep judge identity fixed for AIME, HLE, and AA-LCR. Start AA-LCR with
  conservative task parallelism because its long prompts and external judge can
  become bottlenecks.
- Tau2 requires automatic tool selection and a parser matching the checkpoint's
  chat template. Confirm that model turns contain parsed tool calls and that the
  user and judge endpoints are not throttling requests.

## Record results

Retain the following with every accepted result:

- immutable checkpoint identity;
- deployment and evaluation image identities;
- generated launcher config without secret values;
- evaluator, exact task name, and task definition;
- sample limit, judge, and user-simulator identity when applicable;
- launcher and scheduler identifiers;
- logs, responses, results, and cache artifacts;
- final metric name and value.

Keep results from different evaluators or task definitions separate. Do not
present smoke metrics as benchmark scores or combine them with full results.
