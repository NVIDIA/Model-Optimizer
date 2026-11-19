# Executable Markdown tests

The QAT and Megatron Bridge READMEs use Sybil's Markdown fence lexer with a shared
snippet-ID adapter, subprocess runner and pytest adapter. Example-specific code
only prepares inputs and verifies results.
Standalone snippets execute as written; existing end-to-end tests substitute explicit
CI inputs in the extracted command arguments. Neither path mocks model operations.
These are trusted repository tests with the same privileges as pytest.

## Why a snippet runner

Pytest provides collection, fixtures, assertions and GPU resource checks, while
pytest-timeout provides an outer test deadline. The snippet layer adds Markdown
extraction, named-snippet accounting, source locations and a worker deadline covering
setup, execution and verification, with process-group cleanup.

Sybil is a test-only dependency installed by `.[dev-test]`. It handles fence
boundaries, quoted blocks and source offsets; its collectors and evaluators are
not enabled. Existing pytest tests retain scenario selection and resource checks,
and the worker retains snippet execution, deadlines and process cleanup.
Unclosed fences extend to the end of the document, following Sybil's Markdown
semantics.

`tests/_test_utils/examples/run_command.py` runs individual example commands and
supports streaming output and Hub retries. It does not extract README fences or
preserve one shell/Python namespace across selected snippets, and it imports
Torch-specific distributed utilities. Keeping this small adapter separate lets
CPU-only documentation checks run without importing the ML stack. The migrated CLI workflows use that existing runner with arguments extracted
from README fences.

Stdlib doctest checks Python prompt/output examples. These READMEs also contain
shell pipelines, distributed training and artifact assertions,
which need the scenario/handler contract described below.

## Fixture scope

The parser, runner and pytest adapter are generic. Helpers under `fixtures/`
implement inputs and output checks for the remaining standalone QAT scenarios.
Bridge commands reuse the existing end-to-end tests and their fixtures. Add a shared
abstraction only when another example needs the same behavior with a clear contract.

## Snippet IDs and Python scenarios

Add one ID marker immediately before each fence selected for testing (blank
lines are allowed). Untagged fences are ignored and are outside test coverage.
IDs are unique within the document and remain stable when prose or snippets move.
Code is extracted directly, without substitutions:

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
  executing them. `resolve_scenarios` validates that every named fence is
  referenced or explicitly excluded with a reason. Untagged fences do not
  participate. Duplicate IDs, unknown references, unaccounted named fences, stale
  exclusions, and tested/excluded overlaps fail collection. Removing or renaming
  a referenced ID also fails collection. Accounting does **not** mean
  excluded code passed.
  Scenario metadata is trusted Python test configuration; the resolver does not
  duplicate type or format checks for resource fields and handler names.
- `Scenario` requires a unique scenario ID and a nonempty ordered list of snippet
  IDs. Defaults are `profile="cpu"` and `timeout_seconds=60`. Optional metadata:
  `profile="gpu"` and `handler="module:function"`.
- Handlers are imported only by the worker. Collection does not import their ML
  dependencies or discover GPUs. Keep optional ML imports in the worker path so
  CPU-only scenarios can still run independently. Without a handler, the worker
  simply runs the selected snippets.
- `ctx.repo` and `ctx.tmp` are absolute paths. `ctx.cwd` starts at the repository;
  `ctx.env` is mutable. Select a temporary workspace when isolation is needed.
- Shell snippets run in test-defined order in one Bash process with
  `set -Eeuo pipefail`. Variables, directory changes, and exported environment
  persist within that process. Shell state is not copied back into the handler.
- Python snippets share the dictionary supplied to `ctx.run(namespace)`, which is
  also returned for assertions. The worker supplies `ctx`, `__name__="__main__"`,
  and `__file__` pointing to the README. Before execution it applies `ctx.cwd` and
  `ctx.env`. Keep Python and shell snippets in separate scenarios.
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
- The runner prepends the checkout and test utilities to `PYTHONPATH`. A GPU
  profile requires at least one device; an unavailable GPU produces an explicit skip.

The old `modelopt-doc-test:begin/setup/run/verify/end/skip` format is unsupported.
Test-only setup, assertions, metadata, and exclusion reasons belong in Python.

## Coverage and limits

| README | Default smoke coverage | Exclusions |
| --- | --- | --- |
| `llm_qat` | QAT/QAD/QLoRA in existing end-to-end tests; recipes; simple Llama demo; three Python APIs; FSDP2 layer override; config copy; CLI learning-rate override; dataset cache | Installation, serving and evaluation integrations are outside this pilot; quickstarts are covered by `test_llm_qat.py` |
| `megatron_bridge` | PTQ/export, mock-data distill/export, parameter-target/manual pruning in existing end-to-end tests; CLI preflight | Login, full-scale conversion/training/pruning, tracking and generation integrations; Docker setup, LoRA integration, incomplete templates and illustrative output |

QAT/QAD/QLoRA quickstarts and dataset pre-tokenization run through `test_llm_qat.py` and its existing shared
PTQ model, dataset cache, backends and export checks. Commands are read from their
README fences, with explicit local model/output paths, test dataset inputs and
two training steps. The remaining QAT-only fragments retain their small workspace
and namespace setup: simple Llama, Python APIs, layer/learning-rate overrides,
configuration copy. The three Python API scenarios expose only the first device
in `CUDA_VISIBLE_DEVICES` (or device 0 when unset) to their worker. This prevents
Trainer from using threaded DataParallel, which races shared attention patching;
existing end-to-end tests retain distributed coverage. Dataset pre-tokenization
uses the existing dataset CLI test.
Installation, serving and evaluation integrations are explicitly excluded from
execution coverage; their former manual-only handlers are removed.

Bridge PTQ/export uses the `readme` case in `test_quantize_export.py`; mock-data
distillation and all-iteration export use `test_distill_llm_hf_export`; pruning
uses the existing `qwen3` case plus a `readme-manual` case in `test_prune_minitron.py`.
These tests use the existing `run_example_command` runner and its fixture inputs,
checkpoint assertions and GPU-count handling. There is no separate Bridge smoke
workspace or second distillation prerequisite. Other architecture and recipe
cases retain their existing commands. Full-scale integrations are excluded with
reasons in the Bridge CLI module, which also enforces named-snippet accounting.
CLI help is checked by the preflight.

`_test_utils.doc_tests.cli.readme_command` selects one Python invocation from an
existing named fence and returns an argument list for the existing command runner.
It substitutes explicit test overrides and literal shell defaults, preserving
other published flags. It neither evaluates shell expressions nor runs commands;
unresolved variables fail with the README location. Overrides are single-value
options; general shell workflows and Python fragments still use the snippet runner.
No new README markers are needed for these migrations.

Checks cover quantization state, training progress/finite losses, changed QAT/QAD
weights, packed NVFP4 exports, adapter artifacts, dataset contents and loadable
Bridge outputs. These tests establish command/API composition; they do not establish
full-model accuracy, convergence, memory requirements or serving performance.
A copied FSDP2 config test checks the copy, not the subsequent human edit.

## Run and collect

Name README test modules `test_<example>_readme.py` so different examples can be
collected together without module-name collisions.

Run inside the example CI environment: `nvcr.io/nvidia/pytorch:26.09-py3`
with the QAT requirements, or `nvcr.io/nvidia/nemo:26.10.00` for Megatron
Bridge. Apply the NeMo setup actions in `.github/workflows/_example_tests_runner.yml`
before installing dependencies, as the CI lane does.
Keep the container's matching Torch/FlashAttention binaries when installing test
dependencies. Install the checkout and its `dev-test` dependencies first.

```bash
# Parser/runner tests: no Torch, GPU or repository-wide conftest required.
python -m pytest tests/unit/doc_tests --confcutdir=tests/unit/doc_tests -o addopts='' -q

# Collect and inspect skip reasons in the normal test environment.
python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --collect-only --no-cov
python -m pytest tests/examples/megatron_bridge/test_megatron_bridge_readme_cli.py --collect-only --no-cov

# Bounded default execution coverage.
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/examples/llm_qat/test_llm_qat_readme.py --no-cov -rs
python -m pytest tests/examples/llm_qat/test_llm_qat.py --no-cov -k 'qat_nvfp4 or qad_nvfp4 or qlora_nvfp4' -rs
python -m pytest tests/examples/megatron_bridge/test_quantize_export.py --no-cov -k readme -rs
python -m pytest tests/examples/megatron_bridge/test_distill.py --no-cov -k test_distill_llm_hf_export -rs
python -m pytest tests/examples/megatron_bridge/test_prune_minitron.py --no-cov -k 'qwen3 or readme-manual' -rs

```

## CPU CLI drift checks

`test_readme_cli_options` compares the flag names in README Python script commands
against each script's `--help`. It checks direct `python`/`python3` commands and
extracts the script arguments after `torchrun` or `accelerate launch` options.
Help runs once per script per pytest process, with a 60-second timeout and CUDA
visibility disabled. Missing scripts, failed help commands and unknown flags
report the README path, snippet ID and command line number.

Only existing ID-marked fences are inspected; these checks add no README markers
and the Bridge module also accounts for named fences covered by existing tests
or excluded with a reason. The checks run in the installed example environment but do not launch training,
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

The CLI preflights and QAT standalone scenarios are collected by the existing example jobs. The
workflow runs nightly, on demand, and on pushes to copied PR branches
`pull-request/<number>` after unit tests pass. README-only changes explicitly
activate their respective Torch or Megatron lane despite the general Markdown
exclusion. Changes to shared test utilities activate every lane. Migrated Bridge
commands use the existing end-to-end tests' GPU-count fixture, including single-GPU
PR runners and two-GPU nightly runners.

## File responsibilities

| File | Purpose in this extension |
| --- | --- |
| `parser.py` | Extract named fences and check references, named-snippet accounting, and executable languages. |
| `runner.py` | Execute resolved plans with worker-side handlers, deadlines, shared state, and cleanup. |
| `pytest_utils.py` | Common collection and GPU availability checks, independent of the example. |
| `cli.py` | Extract README commands, substitute explicit end-to-end inputs, and validate flags against script help. |
| `fixtures/llm_qat_readme.py` | Worker-side setup and verification for default QAT standalone snippets. |
| `fixtures/llm_qat.py` | Inputs and assertions for the remaining QAT Python/API and standalone snippets. |
| `tests/unit/doc_tests/test_markdown.py` | Focused fence parsing, failure-location, and timeout-cleanup regressions. |
| `tests/examples/llm_qat/test_llm_qat_readme.py` | Collect QAT scenarios through the common pytest adapter. |
| `tests/examples/megatron_bridge/test_megatron_bridge_readme_cli.py` | Check CLI flags and account for Bridge fences covered elsewhere or excluded. |
| `examples/llm_qat/README.md` | Annotate the remaining snippets; correct QAD teacher construction and incomplete training commands. |
| `examples/megatron_bridge/README.md` | Name command fences for CLI and existing end-to-end coverage; expose input/scale overrides. |
| `.github/workflows/example_tests.yml` | Trigger the Megatron lane for README-only changes. |
| `tests/examples/README.md` | Point example authors to the shared shell/Python test mechanism. |
| `tests/_test_utils/doc_tests/README.md` | Document the contract, scope, prerequisites, limits and CI behavior. |
