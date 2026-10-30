# Executable Markdown tests

The QAT and Megatron Bridge READMEs use the same parser, subprocess runner and
pytest adapter. Example-specific code only prepares inputs and verifies results.
The visible commands execute without command rewriting or mocked model operations.
These are trusted repository tests with the same privileges as pytest.

## Why a snippet runner

Pytest provides collection, fixtures, assertions and manual/resource markers, while
pytest-timeout provides an outer test deadline. The snippet layer adds Markdown
extraction, complete ID accounting, source locations and a worker deadline covering
setup, execution and verification, with process-group cleanup.

`tests/_test_utils/examples/run_command.py` runs individual example commands and
supports streaming output and Hub retries. It does not extract README fences or
preserve one shell/Python namespace across selected snippets, and it imports
Torch-specific distributed utilities. Keeping this small adapter separate lets
CPU-only documentation checks run without importing the ML stack. Existing example
helpers remain appropriate for tests that supply their own command arguments.

Stdlib doctest checks Python prompt/output examples. These READMEs also contain
shell pipelines, distributed training, background services and artifact assertions,
which need the scenario/handler contract described below.

## Fixture scope

The parser, runner and pytest adapter are generic. Helpers under `fixtures/`
implement example-specific inputs and output checks: use `llm_qat` or `bridge`
in helper names, and prefix standalone QAT fixture modules with `llm_qat_`.
Installation copies QAT requirements and checks its dependencies; serving expects
the QAT README's QLoRA checkpoint layout and adapter name. These are not general
installation or serving APIs. Add a shared abstraction only when another example
needs the same behavior with a clear parameter contract.

## Snippet IDs and Python scenarios

A README contains only one ID marker immediately before each fence (blank lines
are allowed). IDs are unique within the document and remain stable when prose or
snippets move. Code is extracted directly, without substitutions:

````markdown
<!-- modelopt-snippet: example-command -->
```bash
printf '%s' "$MESSAGE" > result.txt
```
````

The pytest module owns scenario definitions, execution order, resource metadata,
and exclusions. One snippet can be reused in multiple scenarios:

```python
SCENARIOS = [
    Scenario(
        "example-smoke",
        ["example-command"],
        timeout_seconds=30,
        handler="_test_utils.doc_tests.fixtures.example_readme:smoke",
    ),
]
EXCLUDED = {"interactive-container": "Provisioned outside pytest by the CI job."}

@pytest.mark.parametrize(
    "scenario", scenario_parameters(README, SCENARIOS, excluded=EXCLUDED)
)
def test_readme(scenario, tmp_path, monkeypatch):
    execute_scenario(scenario, REPO, tmp_path, monkeypatch)
```

An optional importable handler performs setup, calls `ctx.run()` exactly once,
then verifies results. Put it in an example-specific module under `fixtures/`:

```python
def smoke(ctx):
    ctx.cwd = ctx.tmp
    ctx.env["MESSAGE"] = "hello"
    ctx.run()
    assert (ctx.cwd / "result.txt").read_text() == "hello"
```

- `parse_markdown(path)` extracts named fences and source locations without
  executing them. `resolve_scenarios` requires an ID on **every** fence and
  validates that each ID is referenced or explicitly excluded with a reason.
  Missing/duplicate IDs, unknown references, stale exclusions, and tested/excluded
  overlaps fail collection. Accounting does **not** mean manual/excluded code passed.
- `Scenario` requires a unique scenario ID and a nonempty ordered list of snippet
  IDs. Defaults are `profile="cpu"` and `timeout_seconds=60`. Optional metadata:
  `profile="gpu"`, `min_gpus=2`, `requires=["megatron.bridge"]`, `manual=True`,
  `background=True`, and `handler="module:function"`.
- Handlers are imported only by the worker. Collection does not import their ML
  dependencies or discover GPUs. Keep optional ML imports in the worker path so
  CPU-only scenarios can still run independently. Without a handler, the worker
  simply runs the selected snippets.
- `ctx.repo` and `ctx.tmp` are absolute paths. `ctx.cwd` starts at the repository;
  `ctx.env` is mutable. Select a temporary workspace when isolation is needed.
- Shell snippets run in test-defined order in one Bash process with
  `set -Eeuo pipefail`. Variables, directory changes, and exported environment
  persist; the handler sees final `ctx.cwd` and `ctx.env` after `ctx.run()`.
- Python snippets share the dictionary supplied to `ctx.run(namespace)`, which is
  also returned for assertions. The worker supplies `ctx`, `__name__="__main__"`,
  and `__file__` pointing to the README. Before execution it applies `ctx.cwd` and
  `ctx.env`; afterward it records their final values. Keep Python and shell
  snippets in separate scenarios.
- `background=True` starts the **last shell snippet** asynchronously. A handler
  must verify readiness and behavior after `ctx.run()`. `ctx.background_pid`
  identifies its shell process. Background changes to directory/environment do
  not propagate back; configure server settings before running the snippets.
- The parent passes a JSON execution plan to the worker; the worker does not
  reparse the README or import pytest modules. The deadline covers handler import,
  setup, snippet execution, and verification. The process group is terminated on
  success, failure, or timeout. A zero exit status cannot bypass completion checks.
- Failures include the scenario ID and Markdown source location; execution logs
  identify each snippet ID. Python snippets preserve README traceback lines;
  handler errors point to the actual Python test helper. Internal control files
  are removed while workspace artifacts remain in pytest's temporary directory.
- Commands must not detach, disable error checking, or replace/exit the shell.
  These are trusted repository tests, not a security sandbox.
- The runner prepends the checkout and test utilities to `PYTHONPATH`. The separate
  installation test removes it and uses a disposable virtualenv to test the
  installed package without replacing the checkout environment.
- Manual scenarios use the existing `--run-manual` option. Missing declared
  modules or insufficient GPUs produce explicit skips. A GPU profile requires
  at least one device. Manual prerequisites remain the operator's responsibility.

The old `modelopt-doc-test:begin/setup/run/verify/end/skip` format is unsupported.
Test-only setup, assertions, metadata, and exclusion reasons belong in Python.

## Coverage and limits

| README | Default smoke scenarios | Opt-in integration scenarios | Exclusions |
| --- | --- | --- | --- |
| `llm_qat` | Recipes; QAT/QAD/QLoRA; simple Llama demo; three Python APIs; FSDP2 layer override; config copy; CLI learning-rate override; dataset cache | Installation; vLLM adapter serving; MMLU/WikiText evaluation | None |
| `megatron_bridge` | PTQ plus export; mock-data distillation; all-iteration export; manual pruning; distillation/pruning CLI help | Login; checkpoint import; MLflow PTQ; real-data distillation/QAD; three NAS targets; VLM pruning; vLLM generation | Interactive Docker launch; LoRA PTQ integration; incomplete VLM/inline-export/single-iteration templates; illustrative output tree |

QAT fixtures copy real scripts/configs and generate two-layer Qwen3 or Llama
models, with hidden/intermediate size 512 and 20 local text records. Training
configs use two optimizer steps, 128-token sequences, and one or two ranks via
`MODELOPT_DOC_TEST_GPUS=1|2`. QLoRA uses DDP; QAT/QAD use FSDP2. The simple demo
keeps its visible defaults. Python fragments receive real model, Trainer and data
objects. Dataset-cache and config-copy tests run without GPU execution.

Megatron smoke tests use a local two-layer Qwen3, four calibration records,
32-token sequences, two training steps and **two actual torchrun ranks**. Public
shell variables override inputs and scale while preserving full-scale defaults.
Export prerequisites are built by executing the README's actual training fence
inside the same timeout/process group. Pruning verifies the loaded model's reduced
intermediate dimension; multi-iteration export loads both saved HF checkpoints.

Checks cover quantization state, training progress/finite losses, changed QAT/QAD
weights, packed NVFP4 exports, adapter artifacts, dataset contents and loadable
Bridge outputs. These tests establish command/API composition; they do not establish
full-model accuracy, convergence, memory requirements or serving performance.
A copied FSDP2 config test checks the copy, not the subsequent human edit.

## Run and collect

Name README test modules `test_<example>_readme.py` so different examples can be
collected together without module-name collisions.

Run inside the appropriate example environment: PyTorch 26.07 with the QAT
requirements, or NeMo 26.08 for Megatron Bridge, as in the example CI workflow.
Keep the container's matching Torch/FlashAttention binaries when installing test
dependencies. Install the checkout and its `dev-test` dependencies first.

```bash
# Parser/runner tests: no Torch, GPU or repository-wide conftest required.
python -m pytest tests/unit/doc_tests --confcutdir=tests/unit/doc_tests -o addopts='' -q

# Collect and inspect skip reasons in the normal test environment.
python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --collect-only --no-cov
python -m pytest tests/examples/megatron_bridge/test_megatron_bridge_readme.py --collect-only --no-cov

# Bounded defaults; manual integrations are skipped.
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --no-cov -rs
CUDA_VISIBLE_DEVICES=0,1 MODELOPT_DOC_TEST_GPUS=2 \
  python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --no-cov -k quickstart -rs
CUDA_VISIBLE_DEVICES=0,1 \
  python -m pytest tests/examples/megatron_bridge/test_megatron_bridge_readme.py --no-cov -rs

# Select ONE manual integration with its prerequisites already supplied.
# MODELOPT_DOC_QLORA_CHECKPOINT: exported adapter directory containing base_model/.
python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --no-cov --run-manual -k llm-qlora-serve
# MODELOPT_DOC_EVAL_CHECKPOINT: fake-quantized QAT checkpoint; lm_eval and dataset access required.
python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --no-cov --run-manual -k llm-qat-evaluation
```

Installation downloads packages into a temporary virtualenv and may compile GPU
extensions. Bridge manual tests need the full models, gated-data access and the
GPU count declared in their scenario. Login requires `HF_TOKEN` and writes into a
temporary `HF_HOME`; no token belongs in Markdown. Real-data training needs
`DATA_PREFIX_1` and `DATA_PREFIX_2`; QAD also needs `QUANTIZED_CHECKPOINT`. MLflow
requires `MLFLOW_TRACKING_URI`; generation requires `PRUNED_CHECKPOINT`.
Manual tests can run for hours and are never enabled by the example CI command.

## CPU CLI drift checks

`test_readme_cli_options` compares the flag names in README Python script commands
against each script's `--help`. It checks direct `python`/`python3` commands and
extracts the script arguments after `torchrun` or `accelerate launch` options.
Help runs once per script per pytest process, with a 60-second timeout and CUDA
visibility disabled. Missing scripts, failed help commands and unknown flags
report the README path, snippet ID and command line number.

Only existing ID-marked fences are inspected; these checks add no README markers
and do not enforce fence coverage. The checks run in the installed example environment but do not launch training,
models or distributed workers. Imports still require the example dependencies.
The QAT evaluation and Bridge vLLM scripts need their optional manual environments;
their CLI checks use `--run-manual`. Values, config contents, launcher flags,
non-Python commands and general shell control flow are outside this check's scope.
Existing execution scenarios continue to validate behavior and output artifacts.

CI runs the CLI test module first and starts the example suite only if it passes.
A failed preflight preserves the failure status and skips the suite. The main run
excludes the CLI module, so help checks do not run twice. Direct pytest invocations
remain independent; use `&&` locally to apply the same gate:

```bash
python -m pytest tests/examples/llm_qat/test_llm_qat_readme_cli.py --no-cov -x && \
  python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --no-cov -v -rs
```

Use the corresponding `megatron_bridge` paths in the Bridge container.

```bash
python -m pytest tests/examples/llm_qat/test_llm_qat_readme_cli.py -k cli_options --no-cov -rs
python -m pytest tests/examples/megatron_bridge/test_megatron_bridge_readme_cli.py -k cli_options --no-cov -rs
```

## CI triggers

Both README test modules are collected by the existing example jobs. The
workflow runs nightly, on demand, and on pushes to copied PR branches
`pull-request/<number>` after unit tests pass. README-only changes explicitly
activate their respective Torch or Megatron lane despite the general Markdown
exclusion. Changes to shared test utilities activate every lane. Two-GPU Bridge
smokes skip on the single-GPU PR runner and run on the two-GPU nightly runner;
CLI-help scenarios still execute on the single-GPU runner.

## Advisory CI agent

The `README Testability Advice` workflow reviews new/updated `README*.md` and
`readme*.md` files in PRs from owners, members, and collaborators, following the
repository's existing Claude access convention. It uses the existing
`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `CLAUDE_MODEL` configuration.
The workflow must first land on the trusted base branch to run automatically.

The assessment suggests minimal snippet IDs, scenario definitions, fixtures,
assertions, prerequisites, and CI lane changes. It distinguishes new executable
coverage from illustrative/interactive exclusions and does not suggest tests for
prose-only changes that need no new coverage.

Assessment has a read-only repository token and never checks out PR code. A
separate job can only publish advisory comments; neither job has code-write
permission. It posts one proposal per PR, updates that comment on later heads,
and discards results if the PR head changed during assessment. An obsolete
proposal is superseded if no further changes are needed. Advice does not add a
required gate or relax existing tests.

The proposal asks before implementation. An authorized maintainer requests the
changes through the existing interactive workflow by posting:

```text
@claude implement the README testability proposal for commit <reviewed-head-sha>
```

That new comment authorizes the described README, test, and fixture work. The
advisory workflow itself never implements changes. If the head changed, reassess
against the new head before implementing the old proposal. The interactive
workflow remains subject to its own permissions and repository instructions.

The agent returns the action's documented
[structured output](https://github.com/anthropics/claude-code-action/blob/main/action.yml).
The deterministic publisher validates that output and asks for approval; it has
no API path for applying code. Its offline checks run with:

```bash
node --test .github/scripts/readme_testability.test.cjs
```

## File responsibilities

| File | Purpose in this extension |
| --- | --- |
| `parser.py` | Extract named fences and validate scenario references, resource metadata, and exclusions. |
| `runner.py` | Execute resolved plans with worker-side handlers, deadlines, shared state, and cleanup. |
| `pytest_utils.py` | Common collection, manual markers and resource checks, independent of the example. |
| `fixtures/llm_qat_readme.py`, `fixtures/megatron_bridge_readme.py` | Worker-side scenario setup and verification moved out of Markdown. |
| `fixtures/llm_qat.py` | Reduced QAT/QAD/QLoRA/Llama inputs, real prerequisite quantization, Python API inputs and artifact assertions. |
| `fixtures/megatron_bridge.py` | Local Bridge inputs, real distillation prerequisites and quantization/pruning/export assertions. |
| `fixtures/llm_qat_installation.py` | QAT requirements in a disposable venv and QAT dependency import verification. |
| `fixtures/llm_qat_serving.py` | QAT QLoRA checkpoint layout, adapter-server readiness and a real completion request. |
| `tests/unit/doc_tests/test_markdown.py` | CPU regression tests for the shared execution contract and complete fence accounting in both READMEs. |
| `tests/examples/llm_qat/test_llm_qat_readme.py` | Collect QAT scenarios through the common pytest adapter. |
| `tests/examples/megatron_bridge/test_megatron_bridge_readme.py` | Collect Bridge scenarios through the same adapter. |
| `examples/llm_qat/README.md` | Annotate the remaining snippets; correct QAD teacher construction and incomplete training commands. |
| `examples/megatron_bridge/README.md` | Annotate smoke/manual scenarios, expose input/scale overrides and account for non-executable templates. |
| `.github/workflows/example_tests.yml` | Trigger the Megatron lane for README-only changes. |
| `tests/examples/README.md` | Point example authors to the shared shell/Python test mechanism. |
| `tests/_test_utils/doc_tests/README.md` | Document the contract, scope, prerequisites, limits and CI behavior. |
