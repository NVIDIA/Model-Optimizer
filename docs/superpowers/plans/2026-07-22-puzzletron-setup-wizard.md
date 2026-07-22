# Puzzletron Setup Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight `questionary` wizard that inspects a supported model configuration and emits validated, independently runnable smoke and production Puzzletron campaign bundles.

**Architecture:** A top-level `puzzletron_setup` package remains independent of `modelopt.torch`. Declarative model profiles normalize Hugging Face configuration dictionaries, a resumable question flow records decisions, and a bundle renderer deep-merges the canonical Puzzletron base/family fragments with generated model, MIP, post-MIP, runner, and execution settings.

**Tech Stack:** Python 3.10+, `questionary`, `transformers`, `huggingface_hub`, `PyYAML`, and the dependency-light `puzzletron_orchestrator` package.

## Global Constraints

- Do not import `torch`, `modelopt.torch`, `AutoModel`, or CUDA-backed libraries from setup code.
- Support only Nemotron 3 and Qwen 3.5/3.6 dense, MoE, text, and multimodal configurations.
- Support Slurm and SSH-managed bare metal; SSH to localhost replaces a local runner kind.
- A normal invocation starts fresh; only `--resume` restores prior answers.
- Persist every completed answer atomically.
- Generate both smoke and production bundles without making either conditional on the other.
- Validate and dry-run both bundles, but never submit jobs.
- Do not add an automated test suite in this iteration; use import, compile, scripted probe, schema, dry-run, and interactive checks.
- Preserve all unrelated worktree changes and stage only files changed by the current task.

---

## File Map

- `puzzletron_setup/__init__.py`: dependency-light public API.
- `puzzletron_setup/profiles.py`: supported-family recognition, normalized inventory, axes, and aligned option generation.
- `puzzletron_setup/inspection.py`: local/Hub configuration loading, revision resolution, and dataset-modality inference.
- `puzzletron_setup/state.py`: answer schema and atomic fresh/resume persistence.
- `puzzletron_setup/prompts.py`: small `questionary` adapter and reusable validators.
- `puzzletron_setup/wizard.py`: normal/detailed question flow and normalized answer construction.
- `puzzletron_setup/bundle.py`: experiment, runner, execution, README, validation, and dry-run rendering.
- `puzzletron_setup/cli.py`: argument parsing and top-level error/interrupt handling.
- `examples/puzzletron/puzzletron_setup.py`: executable repository entry point.
- `examples/puzzletron/requirements-setup.txt`: lightweight setup-only dependencies.
- `examples/puzzletron/README.md`: setup wizard installation and usage.

### Task 1: Lightweight package and CLI boundary

**Files:**
- Create: `puzzletron_setup/__init__.py`
- Create: `puzzletron_setup/cli.py`
- Create: `examples/puzzletron/puzzletron_setup.py`
- Create: `examples/puzzletron/requirements-setup.txt`

**Interfaces:**
- Produces: `puzzletron_setup.cli.main(argv: list[str] | None = None) -> int`
- Consumes later: `run_wizard(detailed: bool, resume: Path | None) -> Path` from Task 4.

- [ ] **Step 1: Add the dependency-light package surface and argument parser**

  Define `--detailed` and optional `--resume PATH`; do not add implicit resume or local-runner flags. Catch `KeyboardInterrupt` and the package's user-facing setup exception without printing a traceback.

  ```python
  def _parser() -> argparse.ArgumentParser:
      parser = argparse.ArgumentParser(description="Create a Puzzletron pruning campaign.")
      parser.add_argument("--detailed", action="store_true")
      parser.add_argument("--resume", type=Path)
      return parser


  def main(argv: list[str] | None = None) -> int:
      args = _parser().parse_args(argv)
      try:
          campaign = run_wizard(detailed=args.detailed, resume=args.resume)
      except KeyboardInterrupt:
          print("\nSetup interrupted. Re-run with --resume <campaign> to continue.")
          return 130
      except SetupError as error:
          print(f"Setup stopped: {error}")
          return 2
      print(f"Campaign written to {campaign}")
      return 0
  ```

- [ ] **Step 2: Add the repository entry point**

  Insert the repository root in `sys.path`, import `main`, and exit with its return code. Use the repository SPDX header.

- [ ] **Step 3: Add setup-only requirements**

  ```text
  questionary>=2.1,<3
  PyYAML>=6.0
  huggingface_hub>=0.24
  transformers>=4.56,<5.0
  ```

- [ ] **Step 4: Verify the lightweight import boundary**

  Run:

  ```bash
  python examples/puzzletron/puzzletron_setup.py --help
  python -c "import sys, puzzletron_setup; assert 'torch' not in sys.modules"
  ```

  Expected: help lists `--detailed` and `--resume`; the import assertion succeeds.

- [ ] **Step 5: Commit the package boundary**

  ```bash
  git add puzzletron_setup/__init__.py puzzletron_setup/cli.py \
    examples/puzzletron/puzzletron_setup.py examples/puzzletron/requirements-setup.txt
  git commit -s -S -m "feat: add Puzzletron setup entry point"
  ```

### Task 2: Declarative model profiles and inspection

**Files:**
- Create: `puzzletron_setup/profiles.py`
- Create: `puzzletron_setup/inspection.py`
- Modify: `puzzletron_setup/__init__.py`

**Interfaces:**
- Produces: `inspect_model(source: str, revision: str | None) -> InspectedModel`
- Produces: `resolve_profile(config: Mapping[str, Any]) -> ModelProfile`
- Produces: `ModelProfile.inventory(config: Mapping[str, Any]) -> ModelInventory`
- Produces: `AxisSpec.options(teacher: int, limit: int = 16) -> tuple[int, ...]`
- Produces: `infer_dataset_modality(source: str) -> ModalityFinding`

- [ ] **Step 1: Define normalized profile value objects**

  Add frozen dataclasses for `AxisSpec`, `ModelProfile`, `ModelInventory`, `InspectedModel`, and `ModalityFinding`. `ModelInventory` contains the descriptor, family, modality, MoE flag, layer count, sublayer count, normalized model facts, and supported axes.

  ```python
  @dataclass(frozen=True)
  class AxisSpec:
      axis_id: str
      label: str
      fields: tuple[str, ...]
      alignment: int = 1
      minimum: int = 1

      def options(self, teacher: int, limit: int = 16) -> tuple[int, ...]:
          aligned_half = max(
              self.minimum,
              (teacher // 2 // self.alignment) * self.alignment,
          )
          aligned_minimum = (
              (self.minimum + self.alignment - 1) // self.alignment * self.alignment
          )
          legal = list(range(aligned_minimum, teacher + 1, self.alignment))
          required = {teacher, aligned_half, legal[0]}
          if len(legal) <= limit:
              return tuple(reversed(legal))
          sampled = {
              legal[round(index * (len(legal) - 1) / (limit - 1))]
              for index in range(limit)
          }
          values = sorted(required | sampled, reverse=True)
          removable = [value for value in values if value not in required]
          while len(values) > limit:
              values.remove(removable.pop())
          return tuple(values)
  ```

  Generate descending aligned values from the teacher to the minimum, include the teacher, include the closest valid half value, deduplicate, and cap the final list at 16 while preserving both endpoints and half.

- [ ] **Step 2: Add initial supported profiles**

  Recognize nested `text_config` as well as top-level config fields. Include exact model types and architecture aliases already registered by Puzzletron:

  Define three entries in `SUPPORTED_PROFILES`: Nemotron with model types
  `nemotron_h`/`nemotron_h_v2` and architectures `NemotronHForCausalLM`/
  `NemotronHV2ForCausalLM`; dense Qwen with model types `qwen3_5`,
  `qwen3_5_text`, `qwen3_6`, and `qwen3_6_text`; and MoE Qwen with model types
  `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen3_6_moe`, and
  `qwen3_6_moe_text`. Both Qwen entries use
  `examples/puzzletron/configs/families/qwen3_5/family.yaml`; Nemotron uses
  `examples/puzzletron/configs/families/nemotron3/family.yaml`.

  Map dense and MoE axes using field aliases from the current model descriptors: hidden width, FFN intermediate width, KV groups, query heads per group, GDN groups/head dimensions, routed experts, expert/shared-expert intermediate widths, top-k, Mamba heads/head dimension, and latent width where present. Expert count alignment is 16; common channel widths use family-appropriate powers of 32/128/256.

- [ ] **Step 3: Load local and Hub configurations without weights**

  Use `AutoConfig.from_pretrained` without remote-code execution when the installed Transformers version recognizes the config. Fall back to `PretrainedConfig.get_config_dict` for config-only inspection of remote-code/unknown model types. For local sources force offline behavior. For Hub sources call `HfApi.model_info` with the requested revision, record its `sha`, and pass that immutable revision into configuration loading.

  Do not call `AutoModel`, `snapshot_download`, or model-weight APIs.

- [ ] **Step 4: Add dataset modality inference**

  Inspect local JSON/JSONL/YAML metadata keys and directory/file names for image, video, audio, or vision fields. For Hub dataset IDs inspect `HfApi.dataset_info` tags and feature metadata. Return `text`, `multimodal`, or `unknown` with a short evidence string; failures return `unknown` rather than aborting setup.

- [ ] **Step 5: Add unsupported-family handoff**

  Raise `UnsupportedModelError` carrying detected model types and architectures. Format it with the exact `.agents/skills/running-puzzletron/SKILL.md` path and an actionable descriptor/profile onboarding request.

- [ ] **Step 6: Run scripted profile probes**

  Run a Python probe containing representative dictionaries for Nemotron H, Qwen dense text, Qwen MoE, and nested multimodal Qwen. Assert the selected descriptor, modality, layer count, half-size inclusion, 16-option cap, and expert alignment.

  Expected: the probe prints one normalized inventory per family and exits zero without importing `torch`.

- [ ] **Step 7: Commit profiles and inspection**

  ```bash
  git add puzzletron_setup/__init__.py puzzletron_setup/profiles.py \
    puzzletron_setup/inspection.py
  git commit -s -S -m "feat: inspect supported Puzzletron models"
  ```

### Task 3: Atomic answer state and prompt adapter

**Files:**
- Create: `puzzletron_setup/state.py`
- Create: `puzzletron_setup/prompts.py`

**Interfaces:**
- Produces: `AnswerState.start(path: Path, detailed: bool) -> AnswerState`
- Produces: `AnswerState.resume(path: Path) -> AnswerState`
- Produces: `AnswerState.record(section: str, key: str, value: Any) -> None`
- Produces: `AnswerState.invalidate_after(section: str) -> None`
- Produces: `PromptSession.text`, `select`, `confirm`, `integer`, and `checkbox`
  methods with `message`, `default`, `description`, and `validate` parameters.

- [ ] **Step 1: Implement versioned atomic answer persistence**

  Store `schema_version`, `wizard_version`, `detailed`, `completed_section`, `model`, `inventory`, `answers`, and `updated_at`. Write YAML to a temporary sibling, flush and `os.fsync`, then `os.replace`. Resolve `--resume` directories to `<dir>/answers.yaml` and reject missing or incompatible state with a concise `SetupError`.

- [ ] **Step 2: Implement dependency invalidation**

  Define the ordered sections `model`, `data`, `pruning`, `runtime`, `mip`, `post_mip`, `infrastructure`, `output`. When a resumed value changes, remove later section mappings and set `completed_section` to the last retained section.

- [ ] **Step 3: Wrap questionary**

  Provide text, path, integer, select, confirm, and checkbox helpers. Each helper accepts a default, description, and validator and raises `KeyboardInterrupt` on cancellation. Keep questionary-specific objects out of `wizard.py`.

- [ ] **Step 4: Verify interruption-safe state**

  Run a short Python probe that creates a temporary answer file, records two answers, reloads it, invalidates the second section, and confirms the YAML remains parseable and contains only retained values.

- [ ] **Step 5: Commit state management**

  ```bash
  git add puzzletron_setup/state.py puzzletron_setup/prompts.py
  git commit -s -S -m "feat: persist resumable Puzzletron setup answers"
  ```

### Task 4: Normal and detailed question flow

**Files:**
- Create: `puzzletron_setup/wizard.py`
- Modify: `puzzletron_setup/cli.py`

**Interfaces:**
- Consumes: model inspection and state interfaces from Tasks 2 and 3.
- Produces: `run_wizard(detailed: bool, resume: Path | None) -> Path`
- Produces: one normalized `answers` mapping consumed by `build_bundles` in Task 5.

- [ ] **Step 1: Implement model, output, and resume bootstrap**

  Print `Welcome to Puzzletron — build a model-aware pruning campaign.` Ask for the campaign directory and model source/revision, inspect it, print the normalized inventory, and persist it. A fresh invocation refuses to overwrite an existing nonempty campaign directory; resume revalidates the saved model and invalidates dependents if its identity changed.

- [ ] **Step 2: Implement data and pruning questions**

  Ask the dataset source, print modality finding and evidence, allow correction when unknown, then ask fixed/packed/padded and sequence length. Ask layer versus sublayer with the deployment explanation and default the removal count to one quarter of the detected count. Show each supported axis as a maximum-16 checklist with teacher and half selected by default. Default full sorting on, sorting sanity off, and bypass sanity off. Default bypass itself on with sublayer granularity, 4096 bypass samples, and batch size 8.

- [ ] **Step 3: Implement importance and serving questions**

  Print detected block and subblock counts before asking replacement granularity. Default width-importance samples to `32768`. Ask whether to collect vLLM statistics and explain their approximation. When enabled ask ISL, OSL, and concurrency; otherwise retain a default serving workload based on calibration sequence length for later AIPerf.

- [ ] **Step 4: Implement multiple MIP run questions**

  Always ask constraint basis, percentage (75 by default), objective metric and direction, homogeneous inclusion, and whether to add another independent run. Generate stable run IDs from basis and percentage, resolving duplicates with a numeric suffix. Normal mode defaults search-space fields to `all`, homogeneous keep to 100, solver solutions to 2000, and Hamming distance to 3. Detailed mode asks additional constraints, depth/embedding loops, homogeneous ranking policy, solver backend, solution count, Hamming distance, and time limit.

- [ ] **Step 5: Implement post-MIP questions**

  Normal mode creates one flow per MIP run:

  ```yaml
  initial_filter: {type: filter, mode: top_k, metric: mip.score, direction: minimize}
  online_eval: {type: evaluation, input: initial_filter}
  best_kl: {type: filter, input: online_eval, mode: top_k,
            metric: online_eval.kl_div, direction: minimize, top_k: 32}
  materialized: {type: materialize, input: best_kl}
  serving: {type: aiperf, input: materialized}
  fastest: {type: filter, input: serving, mode: top_k,
            metric: serving.request_throughput, direction: maximize, top_k: 4}
  short_kd: {type: global_kd, input: fastest}
  final_eval: {type: evaluation, input: short_kd}
  best: {type: filter, input: final_eval, mode: top_k,
         metric: final_eval.kl_div, direction: minimize, top_k: 1}
  ```

  Split initial top-k as 100 homogeneous and 28 heterogeneous when homogeneous search is enabled; otherwise use scalar 128. Give every flow globally unique node IDs by prefixing subsequent flows with the run ID. Detailed mode iteratively asks node type, input, model source, and only the selected node type's relevant fields. Mark PTQ/downstream nodes as reserved and warn that current plan validation will reject them.

- [ ] **Step 6: Implement infrastructure questions and resource summary**

  Ask Slurm versus bare metal, repository, venv, container/image, mounts, pre-commands, result location, GPUs per node, common TP/CP/PP/EP/DP mesh, and separate bypass/global-KD meshes. Slurm asks account, partitions, time limit, QoS, and maximum nodes; bare metal repeatedly asks host and GPU count and supports `localhost`. Ask worker counts for sharded/pool stages. Validate mesh products and expert-parallel divisibility, derive nodes, and print a stage resource table before confirmation.

- [ ] **Step 7: Connect bundle generation**

  Call `build_bundles(state.path.parent, state.payload)` after final confirmation. Persist final normalized answers before rendering.

- [ ] **Step 8: Run an interactive prompt smoke**

  Use a local supported checkpoint config if available. Interrupt after the model section, confirm `answers.yaml` is valid, then re-run with `--resume` and confirm the wizard continues at the next section.

- [ ] **Step 9: Commit the wizard flow**

  ```bash
  git add puzzletron_setup/cli.py puzzletron_setup/wizard.py
  git commit -s -S -m "feat: add Puzzletron campaign question flow"
  ```

### Task 5: Self-contained experiment and orchestration rendering

**Files:**
- Create: `puzzletron_setup/bundle.py`
- Modify: `puzzletron_setup/wizard.py`

**Interfaces:**
- Produces: `build_bundles(campaign_dir: Path, state: Mapping[str, Any]) -> BundleResult`
- Produces: `render_experiment(state, budget: str) -> dict[str, Any]`
- Produces: `render_runner(state, budget: str) -> dict[str, Any]`
- Produces: `render_execution(state, experiment, budget: str) -> dict[str, Any]`

- [ ] **Step 1: Compose canonical experiment fragments**

  Load `examples/puzzletron/configs/base.yaml` and the selected profile's family YAML, remove Hydra package directives, and deep-merge them with generated model/data/search settings. Emit one self-contained YAML document with `defaults: [_self_]`; no generated file may reference the repository's config tree through Hydra defaults.

- [ ] **Step 2: Render model-aware pruning configuration**

  Fill `input_hf_model_path`, immutable Hub revision when present, `model_info`, `model.descriptor_override`, modality/layout/calibration, embedding widths, pruning axis lists, activation/bypass settings, depth budget, vLLM workload, and full `search_space.axes`. Preserve canonical family activation-pass fragments from the family YAML.

- [ ] **Step 3: Render improved MIP and post-MIP mappings**

  Write `mip.defaults`, `mip.workloads`, and separate `mip.runs` exactly in the current `mip_profiles.md` schema. Write `post_mip.flows` with each flow selecting one source run and its ordered node mapping. Disable legacy post-MIP stages so the dynamic graph is authoritative.

- [ ] **Step 4: Derive smoke from production answers**

  Keep graph topology and resource meshes identical. Bound smoke pruning/importance/evaluation samples, MIP solutions, AIPerf request counts, and KD steps; preserve production values unchanged. Put separate `puzzle_dir` values under the smoke and production output roots.

- [ ] **Step 5: Render runner YAML**

  Emit the existing `runner.kind: slurm` or `runner.kind: baremetal` schema with execution contract, image/container, mounts, pre/post commands, and scheduler/inventory details. Do not generate `runner.kind: local`.

- [ ] **Step 6: Render execution YAML and resource rows**

  Emit `execution.defaults` plus explicit stage entries for common fixed stages and each `post.<flow>.<node>` dynamic stage. Select single, sharded, or persistent-pool strategies from node type and user worker counts. Apply common, bypass, and global-KD parallel overrides to their respective stages.

- [ ] **Step 7: Write portable bundle files atomically**

  Write `experiment.yaml`, `runner.yaml`, and `execution.yaml` under both `smoke/` and `production/`, and preserve `answers.yaml` at campaign root.

- [ ] **Step 8: Run render probes**

  Feed one normalized Qwen and one Nemotron answer dictionary directly to `build_bundles` in a temporary directory. Parse every generated YAML, assert both bundle trees exist, confirm production values are unchanged, smoke budgets are bounded, and no YAML default references external config files.

- [ ] **Step 9: Commit rendering**

  ```bash
  git add puzzletron_setup/bundle.py puzzletron_setup/wizard.py
  git commit -s -S -m "feat: generate Puzzletron campaign bundles"
  ```

### Task 6: Validation, dry-run plans, and generated README

**Files:**
- Modify: `puzzletron_setup/bundle.py`
- Modify: `puzzletron_setup/cli.py`

**Interfaces:**
- Produces: `validate_bundle(bundle_dir: Path) -> BundleValidation`
- Produces: `dry_run_bundle(bundle_dir: Path) -> str`
- Produces: root `README.md` containing exact commands.

- [ ] **Step 1: Add dependency-light validation**

  Load generated YAML and call `load_runner_config`, `load_execution_config`, and `compile_campaign_plan` from `puzzletron_orchestrator.compiler`. Accumulate validation results independently for smoke and production so one failure never suppresses the other.

- [ ] **Step 2: Generate dry-run plan text**

  Compile the campaign, call `dry_run_plan`, and serialize the same stage/allocation/task fields printed by `examples/puzzletron/orchestrate.py --dry-run`. Write human-readable summary plus JSON detail to `dry-run-plan.txt`. Do not instantiate an executor or controller.

- [ ] **Step 3: Add runtime capability gate metadata**

  Set the experiment's preflight/descriptor validation controls so the full runtime re-resolves the descriptor, validates selected axes and activation hooks, and fails before GPU stages when the lightweight profile is stale.

- [ ] **Step 4: Generate campaign README**

  Include model/revision, modality finding, MIP run summary, resource table, validation status, exact dry-run and launch commands for smoke and production, explicit independence of the two bundles, and the resume command. Launch commands use `examples/puzzletron/orchestrate.py` and never include `--local`.

- [ ] **Step 5: Verify both bundle plans**

  Run the direct render probe from Task 5 and then invoke:

  ```bash
  python examples/puzzletron/orchestrate.py --experiment <smoke>/experiment.yaml \
    --runner <smoke>/runner.yaml --execution <smoke>/execution.yaml --dry-run
  python examples/puzzletron/orchestrate.py --experiment <production>/experiment.yaml \
    --runner <production>/runner.yaml --execution <production>/execution.yaml --dry-run
  ```

  Expected: both commands exit zero, print no submission action, and match the saved dry-run plan topology.

- [ ] **Step 6: Commit validation and handoff generation**

  ```bash
  git add puzzletron_setup/bundle.py puzzletron_setup/cli.py
  git commit -s -S -m "feat: validate generated Puzzletron campaigns"
  ```

### Task 7: User documentation and end-to-end verification

**Files:**
- Modify: `examples/puzzletron/README.md`

**Interfaces:**
- Consumes: final CLI and generated-bundle behavior.
- Produces: installation, fresh setup, detailed setup, resume, validation, and launch documentation.

- [ ] **Step 1: Document setup-only installation and commands**

  Add a concise Setup Wizard section:

  ```bash
  python -m pip install -r examples/puzzletron/requirements-setup.txt
  python examples/puzzletron/puzzletron_setup.py
  python examples/puzzletron/puzzletron_setup.py --detailed
  python examples/puzzletron/puzzletron_setup.py --resume /path/to/campaign
  ```

  Explain supported families, config-only inspection, smoke/production independence, SSH localhost behavior, and the unsupported-model skill handoff.

- [ ] **Step 2: Run formatting and static checks**

  Run:

  ```bash
  python -m compileall -q puzzletron_setup examples/puzzletron/puzzletron_setup.py
  ruff check puzzletron_setup examples/puzzletron/puzzletron_setup.py
  ruff format --check puzzletron_setup examples/puzzletron/puzzletron_setup.py
  git diff --check
  ```

  Expected: all commands exit zero.

- [ ] **Step 3: Run the setup flow with the user**

  Start fresh with one real supported checkpoint, complete normal mode, inspect both generated bundles and resource summary, and run both dry-runs. Repeat only the branches that expose issues; debug failures together as requested.

- [ ] **Step 4: Review changed files and commit documentation/fixes**

  Stage only the setup package, entry point, setup requirements, and the precise README hunk. Review `git diff --cached` before committing.

  ```bash
  git commit -s -S -m "docs: add Puzzletron setup wizard workflow"
  ```

- [ ] **Step 5: Final verification report**

  Report the campaign path used, detected model profile, smoke and production validation results, saved dry-run plans, static-check output, commits created, and any runtime-unimplemented post-MIP nodes intentionally left unavailable.
