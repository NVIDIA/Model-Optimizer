# Setup wizard

The Puzzletron setup wizard inspects a local checkpoint configuration or a
Hugging Face model configuration and generates self-contained smoke and
production bundles. It reads configuration metadata, not model weights, and
does not submit jobs.

## Profiles

The guided flow offers three profiles:

- **Quick smoke** creates the smallest campaign for checking campaign shape.
- **Balanced pruning** provides the recommended defaults for a first campaign.
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
generated execution routes conversion, tokenization, MIP, materialization, and
other zero-GPU stages there.

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

The generated configuration can include reusable execution profiles, multiple
deployment measurements, independent optimization goals, and editable
downstream flows. See [experiment overrides](configuration_overrides.md),
[Slurm configuration](slurm_configuration.md), and
[post-MIP pipelines](post_mip_pipeline.md) for those controls.
