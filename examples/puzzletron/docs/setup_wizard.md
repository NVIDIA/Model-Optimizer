# Setup wizard

The Puzzletron setup wizard inspects a local checkpoint configuration or a
Hugging Face model configuration and generates self-contained smoke and
production bundles. It reads configuration metadata, not model weights, and
does not submit jobs. Use it after the maintained lifecycle smoke when you need
to select another model, dataset, search profile, or execution environment.

## Profiles

The guided flow offers three profiles:

- **Quick smoke** creates the smallest campaign for checking campaign shape.
- **Balanced pruning** provides the recommended defaults for a generated
  campaign after the maintained lifecycle smoke succeeds.
- **High-confidence search** spends more runtime on scoring and sanity checks.

The selected profile supplies pruning and search defaults from the detected
model family's `setup_v2_defaults.yaml`. The wizard then asks for the model,
dataset, worker environment, and cluster settings.

Model-specific defaults can also extend the recommended post-MIP flow. For
Qwen 3.5 0.8B, the detected dataset modality selects the complete post-MIP
route. Text bundles finish with pinned IFEval and GSM8K subsets. Multimodal
bundles use image-aware serving and VLM distillation, then compare the final
student and teacher on pinned RealWorldQA and MMMU subsets. Both comparisons
record measurements and samples without enforcing a quality threshold.

Adding another size from the same model family does not require another wizard
implementation. Add a model inventory match and model-specific defaults for
its pruning domains, resources, and pinned evaluator, then validate the
generated smoke and production plans for that checkpoint.

## Models and datasets

At the **Model** prompt, provide an existing local checkpoint or configuration
path, or a Hugging Face model URL or repository ID.

At the **Dataset** prompt, provide an existing local dataset path, a Hugging
Face dataset URL, or a repository ID. For a hosted dataset, setup records a
worker-visible output path. The generated campaign `README.md` contains the
exact acquisition command. Run that command from the worker environment before
launching the campaign. A local dataset is referenced directly.

## Defaults and advanced mode

Start the guided flow with the example defaults:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults examples/puzzletron/configs/setup/defaults.example.yaml
```

The example uses repository-relative values. Copy it and add site-specific
data, scheduler, and container settings before selecting it. The defaults file
is loaded only when passed explicitly and takes precedence over the selected
profile. If the primary Slurm partition requires GPUs, set
`infrastructure.runner.slurm.partition_cpu` to an eligible CPU partition; the
generated execution routes conversion, tokenization, solve-only MIP,
materialization, and other zero-GPU stages there. Setup-generated named MIP is
always solve-only and runs on CPU; any checkpoint materialization or validation
runs as an explicit later stage. Inspect the generated dry-run for the
authoritative resource and launcher choice.

Use the full flow to expose every section and nested setting:

```bash
python examples/puzzletron/puzzletron_setup_v2.py --full
```

Automation can use the same setup entry point without answering prompts. The
defaults file must provide every required value that has no resolved default:

```bash
python examples/puzzletron/puzzletron_setup_v2.py \
  --defaults /path/to/setup-v2-defaults.yaml \
  --campaign-dir /path/to/campaign \
  --profile smoke \
  --non-interactive
```

Non-interactive setup fails instead of guessing when a required answer has no
resolved default. It generates and validates the same smoke and production
bundles as the interactive wizard.

## Navigation and resume

Press **Esc** to return from any prompt. Selection prompts include a visible
**← Back** action, and text or numeric prompts accept `:back`.

The wizard saves accepted answers and navigation state in `answers_v2.yaml`.
Resume an interrupted setup with:

```bash
python examples/puzzletron/puzzletron_setup_v2.py --resume /path/to/campaign
```

## Generated files

The final review writes `resolved_defaults.yaml`, one campaign `README.md`, and
validated `smoke/` and `production/` bundles. The generated README presents
these as **Validate setup** and **Run campaign**. Each bundle contains
experiment, runner, and execution YAML plus a `dry-run-plan.txt`; users do not
need to construct a smoke configuration themselves. The wizard does not submit
either bundle, and the campaign is not automatically gated on validation.

For named width/depth MIP, setup reads the teacher hidden size and layer count
from the inspected model configuration. It always retains the teacher hidden
size as the full-width scenario, even when the selected search only changes
FFN dimensions. Missing or contradictory teacher geometry stops setup with the
model field that must be fixed. Do not add these values by hand to generated
YAML.

`dry-run-plan.txt` is a snapshot from the checkout that generated the bundle.
After updating ModelOpt, rerun the generated README's `--dry-run` command before
launching. Incompatible execution schemas and named-MIP bundles fail before
submission and report the setting to change or the setup-resume command that
regenerates both bundles. Named-MIP checks cover the teacher-width scenario,
depth domain, and realized-model validation. Bundles that omit an explicit
resource use the registered CPU-stage defaults.

The generated configuration can include multiple deployment measurements, independent optimization goals, stage resource profiles, and editable downstream flows. See [experiment overrides](configuration_overrides.md), [Slurm configuration](slurm_configuration.md), and [post-MIP pipelines](post_mip_pipeline.md) for those controls.

## Generated worker counts

The wizard writes `instances` for each stage in the generated execution YAML. An instance is an execution worker, not a requested student candidate. Each MIP run requests a solution pool through `solver.num_solutions`; the solver may return fewer solutions, and the post-MIP source and filters determine which distinct candidates continue through the flow.

Smoke bundles use one instance per stage. Production bundles default non-single stages to the configured `infrastructure.gpus_per_node`, unless a model profile or an advanced stage setting supplies another value. Setup does not generally reduce `instances` to match `num_solutions` or an upstream `top_k`. At runtime, post-MIP `evaluation` and `downstream_evaluation` stages use fewer workers when fewer candidate artifacts actually exist; other GPU candidate stages retain their configured worker count. Lower the advanced stage setting when an upstream filter bounds the input to a smaller set.

Generated bundles currently leave `execution.mode` unset and therefore use the default per-attempt scheduler path. To opt into a reusable single-node Slurm allocation, set `execution.mode: reusable_allocation` in the generated execution YAML and run a fresh dry-run. Compilation rejects any stage whose instances and parallel mesh cannot fit within `execution.defaults.gpus_per_node`. See [reusable single-node allocations](slurm_configuration.md#reusable-single-node-allocations) and [stage instances](slurm_configuration.md#stage-instances) for the execution and recovery behavior.
