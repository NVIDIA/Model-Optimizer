# Evaluation with NeMo Evaluator (Alternative)

> **Recommended approach:** Use lm-eval for direct evaluation without a
> deployment server. See the main [README](../README.md#evaluation) for details.

This document describes an alternative evaluation flow using NeMo Evaluator
via `eval-factory`. It deploys the checkpoint as a local OpenAI-style completions
endpoint and runs evaluation against it.

## Prerequisites

- NeMo container (e.g. `nemo:25.11`) NeMo Evaluator and NeMo Export-Deploy
- The AnyModel deploy patch: `examples/puzzletron/evaluation/hf_deployable_anymodel.py`

## Deploy the Model Locally (example: interactive node, 2 GPUs)

```bash
# Repo root (not puzzle_dir)
export MODELOPT_WORKDIR=/path/to/Model-Optimizer
export NEMO_EXPORT_DEPLOY_DIR=/opt/Export-Deploy  # NeMo container default; adjust if needed

# Choose a checkpoint
export CHECKPOINT_PATH=/path/to/ckpts/teacher
# or a pruned checkpoint:
# export CHECKPOINT_PATH=/path/to/ckpts/ffn_8704_attn_no_op

# First time only: back up the original deployable
cp $NEMO_EXPORT_DEPLOY_DIR/nemo_deploy/llm/hf_deployable.py \
   $NEMO_EXPORT_DEPLOY_DIR/nemo_deploy/llm/hf_deployable.py.bak

# Patch the deployable for AnyModel support
cp examples/puzzletron/evaluation/hf_deployable_anymodel.py \
   $NEMO_EXPORT_DEPLOY_DIR/nemo_deploy/llm/hf_deployable.py

ray start --head --num-gpus 2 --port 6379 --disable-usage-stats

# Run in a separate terminal (blocks while server is up)
python $NEMO_EXPORT_DEPLOY_DIR/scripts/deploy/nlp/deploy_ray_hf.py \
  --model_path $CHECKPOINT_PATH \
  --model_id anymodel-hf \
  --num_replicas 1 \
  --num_gpus 2 \
  --num_gpus_per_replica 2 \
  --num_cpus_per_replica 16 \
  --trust_remote_code \
  --port 8083 \
  --device_map "auto" \
  --cuda_visible_devices "0,1"
```

`deploy_ray_hf.py` runs a long-lived server. Keep it running in another terminal
or background it (e.g., tmux) while you run eval. Adjust GPU counts and
`cuda_visible_devices` to match your node.

## Run MMLU

```bash
eval-factory run_eval \
  --eval_type mmlu \
  --model_id anymodel-hf \
  --model_type completions \
  --model_url http://0.0.0.0:8083/v1/completions/ \
  --output_dir $PUZZLE_DIR/evals/mmlu_anymodel \
  --overrides "config.params.parallelism=2,config.params.task=mmlu,config.params.extra.tokenizer=$CHECKPOINT_PATH,config.params.extra.tokenizer_backend=huggingface,config.params.request_timeout=6000"
```

For a quick debug run, add `,config.params.limit_samples=5` to the `--overrides` list.

Results can be viewed in the generated `results.yml` file.
