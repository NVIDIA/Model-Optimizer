# Qwen3.6-35B-A3B evaluation

The configs behind the [tutorial's results](../README.md#results), one per benchmark so each can be launched independently with its own walltime and sampling:

| Benchmark | Config | Temp | Runs | Walltime | Metric name |
| --- | --- | --- | --- | --- | --- |
| MMMU-Pro | [mmmu_pro.yaml](mmmu_pro.yaml) | 1.0 | 8 | 2:30 | `mmmu-pro_pass_at_1_symbolic_correct` |
| GPQA Diamond | [gpqa.yaml](gpqa.yaml) | 1.0 | 1 (avg-of-16) | 4:00 | `gpqa_pass_at_1_avg-of-16_symbolic_correct` |
| SciCode (Subtask) | [scicode.yaml](scicode.yaml) | **0.6** | 8 | 2:00 | `scicode_pass_at_1_subtask_accuracy` |
| AA-LCR | [aa_lcr.yaml](aa_lcr.yaml) | 1.0 | 8 | 2:00 | `aalcr_pass_at_1_judge_correct` |
| IFBench | [ifbench.yaml](ifbench.yaml) | 1.0 | 8 | 1:30 | `ifbench_pass_at_1_average_score` |
| tau2-bench Telecom | [tau2_telecom.yaml](tau2_telecom.yaml) | 1.0 | 3 | 1:30 | `tau2_bench_telecom_pass_at_1_pass_at_1` |
| AA-Omniscience<sup>*</sup> | [omniscience.yaml](omniscience.yaml) | 1.0 | 1 (avg-of-10) | 4:00 | `omniscience_pass_at_1_avg-of-10_judge_omni_index` |

<sup>*</sup> Reported separately in [Knowledge recall and calibration](../README.md#knowledge-recall-and-calibration), not in the six-benchmark mean.

Sampling follows the [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) card (`T=1.0, top_p=0.95, max_new_tokens=131072`), except SciCode, which the card specifies at `T=0.6`. `parallelism` varies (32 default, 16 for AA-LCR's long contexts and for tau2-bench's multi-turn tool round-trips, 8 for SciCode's long generations) as a throughput knob only. The same configs serve BF16 and NVFP4 unchanged — vLLM reads the FP8 KV-cache setting from the checkpoint's own `quantization_config`.

> [!IMPORTANT]
> **Three benchmarks need a second model endpoint besides the one under test.** AA-LCR and
> AA-Omniscience score answers with a judge (`NS_JUDGE_URL` / `LCR_JUDGE_MODEL_ID` or
> `OMNI_JUDGE_MODEL_ID`); tau2-bench drives a simulated user (`TAU2_ENDPOINT_URL` /
> `TAU2_USER_MODEL_ID`). All read `INFERENCE_API_KEY`, and none can produce its metric without that
> endpoint. Point them at `https://` URLs for the final endpoint:
> the key rides every request, and a redirect can downgrade it to `http` or send it to another
> origin. tau2-bench additionally needs its own **deployment** —
> `--enable-auto-tool-choice --tool-call-parser qwen3_coder` and `--max-num-seqs 16` — which is why
> it cannot share a config with the others (`deployment.command` is global). `qwen3_coder` is what
> the model card specifies; the template's `<tool_call>` markers resemble `hermes`, which would
> mis-parse tool calls and silently invalidate the benchmark. Its `top_k` / `presence_penalty` go
> through tau2's `agent_args` passthrough, which NEL's top-level `params` block does not accept.
> On an `ns_*` task they go through the adapter's `params_to_add` instead — the other five configs
> still omit them, deliberately, so they reproduce the results table; see [Output length](../README.md#output-length)
> for when `presence_penalty` helps and when it is pure cost.

> [!IMPORTANT]
> Every task sets `num_repeats: 1`; the repeat counts above come from **launching a config that many times**. N launches give N independent `pass@1` values, which is what `mean ± sem` and the paired tests need — `num_repeats: N` instead yields a single `pass@1[avg-of-N]`. GPQA and AA-Omniscience are the deliberate exceptions; AA-Omniscience's `±` is the sem across its 10 in-run repeats, which differ because each repeat shares one sampling seed across all 600 questions.

> [!IMPORTANT]
> **Run enough repeats.** These benchmarks are noisy, and 3 was not enough to tell signal from noise in either direction. At 3 repeats, AA-LCR (only 100 questions) showed a 3 pp swing that looked real and **vanished completely** at 8; IFBench's genuine +2.3 pp gain was **invisible** at 3 and only appeared at 8. Three runs also *understates* how noisy a benchmark is, so the measured spread looks tighter than it is.

<details>
<summary>Evaluation launch steps (click to expand)</summary>

Set `execution.hostname`, `execution.account` and `deployment.checkpoint_path` in the config, or override with `-o <option>=<value>`. Run the commands from the tutorial directory.

```bash
pip install "nemo-evaluator-launcher[all]==0.2.6"

# Required by every config:
export HF_TOKEN=<your_huggingface_token>
export SLURM_JOB_DIR=<path_to_slurm_job_output_dir>

# Required only by AA-LCR (judge), tau2-bench (user simulator) and AA-Omniscience (judge):
export INFERENCE_API_KEY=<key_for_those_endpoints>
export NS_JUDGE_URL=<judge_endpoint_url>
export LCR_JUDGE_MODEL_ID=<judge_model_id>
export TAU2_ENDPOINT_URL=<user_simulator_url>
export TAU2_USER_MODEL_ID=<user_simulator_model>
# Required only by AA-Omniscience (judge), also via NS_JUDGE_URL and INFERENCE_API_KEY:
export OMNI_JUDGE_MODEL_ID=<judge_model_id>

# One benchmark. To verify the pipeline first, add
# `-o ++evaluation.nemo_evaluator_config.config.params.limit_samples=8`
nemo-evaluator-launcher run --config evaluation/mmmu_pro.yaml

# The 8 independent runs behind the results table: launch the same config 8 times and pool the
# per-run pass@1 values. Each launch returns its own invocation id; record them so you can select
# exactly those runs when pooling (each eval directory also holds a small canary run).
for i in $(seq 1 8); do
    nemo-evaluator-launcher run --config evaluation/mmmu_pro.yaml
done

# All six benchmarks at their reported repeat counts
for cfg_runs in mmmu_pro:8 gpqa:1 scicode:8 aa_lcr:8 ifbench:8 tau2_telecom:3; do
    cfg=${cfg_runs%%:*}; runs=${cfg_runs##*:}
    for i in $(seq 1 "$runs"); do
        nemo-evaluator-launcher run --config "evaluation/${cfg}.yaml"
    done
done
```

</details>

For more details on NeMo Evaluator, see the [GitHub repo](https://github.com/NVIDIA-NeMo/evaluator) and [documentation](https://docs.nvidia.com/nemo/evaluator/latest/).
