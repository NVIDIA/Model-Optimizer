# Evaluation with NeMo Evaluator (Alternative)

> **Recommended approach:** Use lm-eval for direct evaluation without a
> deployment server. See the main [README](../README.md#evaluation) for details.

Evaluate AnyModel checkpoints by deploying a local OpenAI-compatible completions endpoint and running benchmarks against it.

This flow requires Ray for serving the model and NeMo Export-Deploy (included in NeMo containers):

```bash
pip install ray
```

**1. Deploy the model (2 GPUs example):**

```bash
# Install the AnyModel-patched deployable (first time only: backs up the original)
# /opt/Export-Deploy is the default path in NeMo containers — adjust if needed
cp /opt/Export-Deploy/nemo_deploy/llm/hf_deployable.py /opt/Export-Deploy/nemo_deploy/llm/hf_deployable.py.bak
cp examples/puzzletron/evaluation/hf_deployable_anymodel.py /opt/Export-Deploy/nemo_deploy/llm/hf_deployable.py

# Start the server (blocks while running — use a separate terminal)
ray start --head --num-gpus 2 --port 6379 --disable-usage-stats
python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py \
    --model_path path/to/checkpoint \
    --model_id anymodel-hf \
    --num_gpus 2 --num_gpus_per_replica 2 --num_cpus_per_replica 16 \
    --trust_remote_code --port 8083 --device_map "auto" --cuda_visible_devices "0,1"
```

**2. Run MMLU:**

```bash
eval-factory run_eval \
    --eval_type mmlu \
    --model_id anymodel-hf \
    --model_type completions \
    --model_url http://0.0.0.0:8083/v1/completions/ \
    --output_dir examples/puzzletron/evals/mmlu_anymodel
```

For a quick debug run, add `--overrides "config.params.limit_samples=5"`.
