# Puzzletron

Puzzletron searches for smaller and faster variants of a pretrained model by
combining width and depth importance, physical slicing, measured serving costs,
mixed-integer search, evaluation, and optional knowledge distillation. Each
stage writes resumable artifacts and contributes to one self-contained HTML
campaign report.

## Installation

Puzzletron uses ModelOpt together with compatible AutoModel, vLLM, and AIPerf
checkouts. Keep the four repositories as siblings so one environment can import
the exact source being tested:

```text
workspace/
├── Model-Optimizer/
├── Automodel/
├── vllm/
└── aiperf/
```

Create a virtual environment using a PyTorch/CUDA build supported by all four
repositories, then install them as one editable unit:

```bash
cd Model-Optimizer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ../Automodel
python -m pip install -e ../vllm
python -m pip install -e ../aiperf
python -m pip install -r examples/puzzletron/requirements.txt
```

GPU libraries must come from one mutually compatible environment. After
installing or rebuilding any compiled dependency, validate imports and one
unpruned forward pass before starting a campaign.

The checked-in HTML reports use Git LFS:

```bash
git lfs install
git lfs pull --include='examples/puzzletron/reports/*.html'
```

## Run with an agent

The canonical agent workflow is
[`running-puzzletron`](../../.agents/skills/running-puzzletron/SKILL.md). Ask an
agent to use that skill and provide the model, dataset, compute environment,
search space, resource constraints, and required downstream stages. For
example:

```text
Use .agents/skills/running-puzzletron/SKILL.md to run the Puzzletron campaign
at examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml.
Validate the smoke path first, execute the enabled DAG, resume compatible
artifacts, and regenerate and verify the report after every completed stage.
```

`.agents/` is the source of truth. Agent-specific paths such as
`.claude/skills/running-puzzletron` are compatibility symlinks and should not
be edited separately.

## Configuration

Configs use Hydra composition:

```text
examples/puzzletron/configs/
├── base.yaml                         # pipeline-wide defaults
└── families/
    └── <family>/
        ├── family.yaml               # descriptors, hooks, and family axes
        └── <model>/
            ├── model.yaml            # checkpoint metadata and legal domains
            └── runs/default.yaml     # exact end-to-end experiment
```

Site-specific paths can be overridden without editing the checked-in config:

```bash
export PUZZLETRON_RUN_ROOT=/shared/puzzle_runs/my_campaign
```

Named resource constraints and homogeneous/restricted search settings are
documented in [MIP profiles](docs/mip_profiles.md).

## Run the complete pipeline manually

Choose one tested entry config:

```bash
export CONFIG=examples/puzzletron/configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml
source .venv/bin/activate
python examples/puzzletron/main.py \
  --config "$CONFIG" \
  --stage full \
  --gpus-per-node 8
```

`--stage full` executes every enabled stage in dependency order. Completed
stages with valid manifests and acceptance markers are skipped; use `--force`
only when intentionally invalidating and rerunning the selected work.

On a scheduler, run the same command inside the site's container and launch
distributed stages with the topology declared by that stage's
`automodel.parallel` section. Do not assume one parallel recipe is valid for
all stages.

## Run step by step

Run one stage with the same entry config:

```bash
python examples/puzzletron/main.py \
  --config "$CONFIG" \
  --stage width_importance \
  --gpus-per-node 8
```

The authoritative dependencies and enabled-stage rules live in
[`stages/graph.py`](../../modelopt/torch/puzzletron/stages/graph.py).

| Stage | Purpose |
|---|---|
| `convert` | Convert the immutable Hugging Face teacher into the configured backend format. |
| `tokenize_data` | Build deterministic train and validation token caches. |
| `vllm_stats` | Measure exact runtime and memory costs for candidate subblocks. |
| `depth_importance` | Rank cumulative block or subblock removals. |
| `width_importance` | Collect activation-based rankings for every enabled width axis. |
| `sort` | Reorder the teacher so nested prefixes implement ranked width choices. |
| `sort_sanity` | Check that sorting preserves teacher outputs. |
| `width_sanity` | Compare ranked, random, and reverse slices on representative layers. |
| `slicing_sanity` | Verify dynamic slicing against physical materialization. |
| `bypass_sanity` | Overfit small local-distillation cases before production bypass. |
| `bypass` | Train nested replacement blocks across the configured search space. |
| `build_library` | Assemble sorted, bypassed, and no-op replacement candidates. |
| `replacement_scoring` | Score replacing one block or subblock at a time. |
| `mip` | Solve heterogeneous and homogeneous architecture searches under named constraints. |
| `zero_shot_evaluation` | Evaluate selected MIP recipes online without materializing every checkpoint. |
| `aiperf` | Materialize selected finalists and benchmark serving performance. |
| `global_distillation_sanity` | Overfit the selected global student as a correctness check. |
| `global_distillation` | Distill the selected architecture at the configured production scale. |
| `post_distillation_evaluation` | Evaluate the final distilled checkpoint. |

Independent DAG branches may run concurrently when they have disjoint writers.
Long-running stages should resume their durable checkpoints or immutable shards
rather than restarting completed work.

## Reports

The orchestrator refreshes the report around each stage. It can also be
regenerated without rerunning model work:

```bash
python examples/puzzletron/generate_campaign_progress_report.py \
  --puzzle-dir /shared/puzzle_runs/my_campaign \
  --model-name 'My model'
```

The output is
`<puzzle-dir>/artifacts/campaign_report/campaign_report.html`. It is a
self-contained file. Section fingerprints are cached under the campaign so
unchanged sections can be reused while new or partial results are incorporated.

## End-to-end tested models

The configs below are the exact current-code entry points for the completed
campaigns. Reports are stored through Git LFS.

| Model | Hugging Face model | Full experiment config | Verified report |
|---|---|---|---|
| Nemotron-3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | [default.yaml](configs/families/nemotron3/nano_30b_a3b_bf16/runs/default.yaml) | [HTML report](reports/nemotron3_nano_30b_a3b.html) |
| Qwen3p5_9B | `Qwen/Qwen3.5-9B` | [default.yaml](configs/families/qwen3_5/qwen3p5_9b/runs/default.yaml) | [HTML report](reports/qwen3p5_9b.html) |
