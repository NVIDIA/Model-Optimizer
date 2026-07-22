# Puzzletron Setup Wizard Design

## Problem

Creating a Puzzletron pruning campaign requires model-specific experiment
configuration, orchestration configuration, resource topology, MIP searches, and
a post-MIP pipeline. The current workflow expects users to know these schemas and
their interactions. It is difficult to discover valid model axes and easy to
produce internally inconsistent resource or search settings.

Puzzletron needs a guided setup program that inspects a model configuration,
asks only the decisions that matter, generates independently runnable smoke and
production bundles, and leaves submission under the user's control.

## Goals

- Provide `python examples/puzzletron/puzzletron_setup.py` as an interactive
  `questionary` wizard.
- Inspect local checkpoints and Hugging Face repositories without loading model
  weights or importing PyTorch.
- Support Nemotron 3 and Qwen 3.5/3.6 dense, MoE, text, and multimodal variants.
- Generate model-aware depth and width choices with safe defaults.
- Configure multiple independent MIP searches and the configurable post-MIP DAG.
- Generate self-contained smoke and production campaign bundles.
- Support Slurm and SSH-managed bare-metal execution, including SSH to localhost.
- Save every answer atomically and resume only when explicitly requested.
- Validate and dry-run generated bundles without submitting jobs.
- Keep the setup environment lightweight and make future model-family support a
  small declarative addition.

## Non-goals

- Loading model weights, importing CUDA, or running pruning from the wizard.
- Supporting model families beyond the initial allowlist.
- Adding a first-class local orchestration backend; localhost uses SSH bare metal.
- Blocking production execution on successful smoke execution.
- Maintaining a separate catalogue of evaluation metrics.
- Automatically submitting any generated campaign.
- Building an automated test suite for the wizard in this iteration. The final
  flow will instead be exercised interactively and debugged from real output.

## Architecture

The entry point lives at `examples/puzzletron/puzzletron_setup.py`. Dependency-
light implementation code lives in a top-level `puzzletron_setup` package so
importing it cannot execute `modelopt.torch` package initialization.

The package has four primary responsibilities:

- `ModelProfile` recognizes a supported Hugging Face configuration and maps its
  fields to Puzzletron concepts.
- `AxisSpec` describes a pruning axis's teacher value, range, alignment, display
  name, and option-generation rule.
- `QuestionFlow` owns prompts, conditional defaults, normalized answers, and
  atomic progress persistence.
- `BundleBuilder` converts normalized answers into smoke and production campaign
  files and invokes validation and orchestration dry-run entry points.

Prompting is kept behind a small adapter. The model/profile and rendering logic
does not depend directly on `questionary`, which keeps the core implementation
straightforward to inspect and reuse.

The setup requirements are limited to:

- `questionary`
- `transformers`
- `huggingface_hub`
- `PyYAML`

The wizard must not import `torch`, `modelopt.torch`, `AutoModel`, or CUDA-backed
libraries.

## Model Inspection and Profiles

The model question accepts either a local checkpoint path or a Hugging Face URI.
Local inspection uses `AutoConfig.from_pretrained` with offline-only behavior.
Hub inspection accepts an optional revision and records the resolved immutable
commit when available. Only configuration and metadata files are downloaded.

After inspection, the wizard prints a concise inventory:

- detected family and architecture;
- dense or MoE;
- text-only or multimodal;
- layer and sublayer counts;
- hidden/intermediate/embedding dimensions;
- expert counts and active-expert settings when applicable;
- supported Puzzletron axes and their alignments.

Profiles are declarative capability metadata rather than complete experiment
templates. They contain model-type and architecture aliases, descriptor identity,
modality rules, dense/MoE capability flags, Hugging Face field aliases, layer and
sublayer count rules, supported axes, alignment/domain generators, and required
family configuration fragments.

Generic profile logic resolves these fields into one normalized model inventory.
Axis option generation and prompts operate only on that inventory. A profile can
therefore support configuration-field variations without duplicating an entire
campaign template.

Profiles are advisory at setup time. Every generated runtime configuration asks
the full Puzzletron environment to perform authoritative descriptor and
capability validation before GPU work begins. This catches drift between the
lightweight profile and runtime descriptors.

If a model is unsupported, the wizard saves the detected configuration metadata,
prints `.agents/skills/running-puzzletron/SKILL.md`, and provides a concise prompt
for onboarding the new family and descriptor. It exits cleanly without a Python
traceback or a partially runnable bundle.

## Wizard Flow

### Model and Dataset

The wizard begins with a short welcome message and the model inspection flow.
It then asks for a dataset source and reports whether the selected model and data
appear multimodal. The user chooses fixed, packed, or padded data and a sequence
length. Detailed mode exposes tokenizer, processor, and calibration overrides.

### Depth

The user chooses layer or sublayer pruning. The prompt explains that sublayer
search is more accurate but more expensive and may require heterogeneous
deployment support that is less available outside vLLM.

The default removal budget is one quarter of the detected layer or sublayer
count, rounded to a valid nonzero value without allowing the complete network to
be removed.

### Width Axes

Supported axes are derived from the model profile and shown one axis at a time as
checklists. Each axis displays at most 16 values. The teacher value is always
present, the closest valid half-size value is selected by default, and all values
respect profile-specific alignment. For example, expert-count options are
multiples of 16 where the family requires that alignment, while top-k can use
unaligned integer choices.

All supported axes are fully sorted by default. Detailed mode allows users to
disable axes or refine their option sets.

### Importance, Sorting, and Bypass

Width importance defaults to `32 * 1024` samples. Replacement granularity can be
block or subblock, and its prompt includes the detected number of each so users
can judge the cost.

Sorting sanity checks are disabled by default. Bypass is enabled by default at
sublayer granularity. When bypass is enabled, the wizard asks for its sample
count, sequence length, batch size, and separate device mesh. Defaults are 4096
samples and batch size 8.

### vLLM Runtime Statistics

vLLM statistics are optional and disabled by default. The prompt explains that
the estimate is imperfect and that parameter or memory objectives are often
sufficient. When enabled, the wizard asks once for ISL, OSL, and concurrency.
The default AIPerf node uses exactly the same workload settings.

### MIP Searches

Normal and detailed modes both support multiple independent MIP runs. Each run
asks for:

- a primary constraint basis of parameters, memory, or latency when vLLM
  statistics are enabled;
- a target percentage, defaulting to 75 percent;
- an objective field and minimize/maximize direction;
- whether to include homogeneous candidates, explaining their deployment
  advantage;
- optional additional constraints and search-space loops.

Each top-level run remains separate in configuration and results. Within a run,
specific variants can loop over embedding sizes, depth values, constraint bands,
or other supported choices.

Normal mode supplies defaults for solution count, Hamming distance, homogeneous
retention, and solver settings. Detailed mode exposes all supported solver
options, multiple objectives, exact search-space loops, additional constraints,
and independent homogeneous/heterogeneous retention behavior. The design assumes
that MIP may intentionally emit thousands of candidates for later filtering.

### Post-MIP Pipeline

Normal mode generates this pipeline for each applicable MIP result stream:

```text
MIP
 -> keep top 128 by MIP score
 -> online evaluation
 -> keep top 32 by KL divergence
 -> materialize checkpoints
 -> AIPerf with the vLLM workload
 -> keep top 4 by speed
 -> short global KD
 -> online evaluation
 -> select the best candidate
```

Detailed mode provides an iterative node builder: select a node type, answer
only that node's relevant fields, then add another node or finish. Generated
nodes use stable IDs, explicit dependencies, typed node configuration, checkpoint
requirements, and an explicit input source indicating previous-node output or
the original MIP solution.

Filters name metrics emitted by prior nodes directly. The wizard does not ask
users to define cases or maintain a metric registry. Candidate metadata remains
append-only as solutions pass through nodes, so later nodes and reports can still
inspect earlier evaluation, KD, PTQ, or runtime results.

PTQ and downstream-evaluation node shapes may be configured in detailed mode but
remain runtime-unimplemented until their corresponding executors are added.

## Infrastructure and Resources

The wizard supports Slurm and SSH-managed bare-metal runners. It asks for cluster
identity, image, pre-commands, results location, GPUs per node, and the common
TP/PP/EP/CP/DP mesh. Bypass and global KD receive separate meshes because their
execution needs commonly differ.

For sharded or pooled stages, the wizard asks for worker counts. It derives node
counts from mesh size, GPU capacity, and concurrency, validates divisibility and
expert-parallel constraints, and displays a final per-stage resource table.
Detailed mode permits per-stage overrides. Normal mode uses the common mesh and
derived resource defaults wherever a specialized mesh is unnecessary.

SSH to localhost is the supported single-host path in this iteration. No `local`
runner kind is generated.

## Normal and Detailed Modes

Normal mode still asks decisions that materially change results:

- model and data;
- depth strategy and pruning amount;
- selected width axes;
- each MIP run's constraint basis, percentage, and objective;
- whether more MIP runs should be added;
- infrastructure and output location.

It accepts defaults without prompting for low-level sampling, solver, node, and
stage tuning. `--detailed` exposes those controls and the post-MIP node builder.
Both modes record every explicit and defaulted value in the answer state.

## Persistence and Resume

Every invocation starts a new setup session by default. `--resume` accepts an
`answers.yaml` path or campaign directory and is the only way to continue a
previous session.

After every answer, state is written to a temporary sibling and atomically
replaced. The state records:

- wizard and schema versions;
- model path or URI and resolved revision;
- selected profile and normalized model inventory;
- explicit and defaulted answers;
- last completed section;
- generation timestamp.

On resume, the wizard revalidates model metadata before continuing. If a restored
answer changes, dependent answers from that section onward are discarded so stale
derived values cannot leak into generated configuration.

## Generated Bundle

The selected campaign directory contains:

```text
<campaign>/
|-- answers.yaml
|-- README.md
|-- smoke/
|   |-- experiment.yaml
|   |-- runner.yaml
|   |-- execution.yaml
|   `-- dry-run-plan.txt
`-- production/
    |-- experiment.yaml
    |-- runner.yaml
    |-- execution.yaml
    `-- dry-run-plan.txt
```

`README.md` contains exact validation, dry-run, launch, resume, and result paths.
Each subdirectory is independently portable and runnable.

Smoke and production bundles are generated from the same normalized answers and
preserve the same stage topology. Smoke applies bounded reductions to samples,
MIP solution counts, evaluation batches, and KD work. Production retains the
user's requested budgets. Neither bundle's execution is made conditional on the
other.

## Validation and Dry Run

Before completing, generation performs:

- profile and model-inventory validation;
- cross-field axis, depth, and mesh validation;
- post-MIP graph validation;
- runner and resource validation;
- full Puzzletron configuration validation when its lightweight entry point is
  available;
- an orchestration dry-run for smoke and production.

Dry-run output is saved in each bundle's `dry-run-plan.txt`. Failure in one
bundle does not skip validation of the other. The wizard never submits work.

The initial implementation does not add automated tests at the user's request.
Verification consists of syntax/import checks, scripted noninteractive setup
passes where practical, generated-schema validation, orchestration dry-runs, and
an interactive end-to-end run with the user.

## Failure Handling

- Invalid local paths or inaccessible Hub models return to the model prompt.
- Unsupported models exit with saved detection data and descriptor-onboarding
  instructions.
- Invalid axes or resource meshes explain the conflicting fields and return to
  the relevant question.
- A final validation failure preserves `answers.yaml` and generated diagnostics,
  reports the exact configuration path, and submits nothing.
- Interruption preserves the most recently completed atomic answer, but the user
  must explicitly pass `--resume` on the next invocation.

## Extensibility

Adding a model family should normally require one profile plus runtime descriptor
support, not changes to prompt control flow. Adding a post-MIP node should require
registering its node type and question provider; generic DAG composition and
rendering remain unchanged. New runners or resource strategies can be introduced
behind bundle rendering without changing normalized campaign answers.
