# Qwen 3.5 4B VLM FFN-width 10%-to-20% KD search

This campaign compares three smaller versions of `Qwen/Qwen3.5-4B`. Each version
reduces the language model's FFN width while leaving the vision path unchanged.

This page lists the campaign runs and the commands shared by all runs. Each run
summary owns its results, runtime, and limitations so that those details are not
repeated here.

## Runs

| Run | Status | What finished |
|---|---|---|
| [2026-09-01-r2](runs/2026-09-01-r2/summary.md) | Early comparison | Three students, a serving check, 64 KD steps, and two short quality benchmarks |

## Reproduce

Use the model and dataset versions recorded in the run's structured record.
Prepare the dataset where workers can read it, copy the maintained recipe, and
configure the reusable site file before launching.

```bash
python examples/puzzletron/materialize_dataset.py nemotron_vlm_v2 \
  --output /path/to/qwen3p5-vlm-campaign-data \
  --revision 51f4f4d219315c3283950994d4eb3d7fc30aa87b \
  --subsets sparsetables plotqa_cot wiki_en \
  --num-samples 64 \
  --max-shards-per-subset 1

cp examples/puzzletron/configs/recipes/qwen3p5_4b_vlm_campaign.yaml campaign.recipe.yaml
cp examples/puzzletron/configs/site.example.yaml puzzletron.site.yaml
# Set recipe data.path and run_root, then fill in the site placeholders.

python examples/puzzletron/puzzletron.py dry-run campaign.recipe.yaml \
  --site puzzletron.site.yaml
```

Run the dry run first and check the commands, paths, and requested GPUs. The
site file is only a template until its placeholders are replaced. The full
recipe also contains a fresh 256-step KD run and a final teacher comparison;
run `2026-09-01-r2` stopped before those steps.

See [maintained recipes](../../../../docs/maintained_recipes.md)
for environment preparation and lifecycle details.
