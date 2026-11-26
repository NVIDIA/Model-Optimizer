# QAKTO reproducibility

[Back to the QAKTO tutorial](../README.md).

## KTO training

We fine-tuned the QAD baseline in NeMo RL on one four-GPU GB300 node. The parent tutorial saves the checkpoint with eight-way expert parallelism (EP=8), and Megatron redistributes its expert weights when loading at EP=4. Export the trained QAKTO model using the parent’s [NVFP4 export procedure](../../README.md#4-export).

### Collect KTO training responses

The experiment reused saved QAD-generated responses. To collect a new training set:

1. Use the [44/11 subtask split](data/subtask_split.yaml). Collect responses only for the 44 training subtasks and reserve the other 11 for evaluation.
2. Generate repeated responses with the QAD model and run the SciCode tests. Keep responses that pass as correct examples.
3. Inspect unfinished responses and keep those that repeat without making progress as looping examples. A token-cap hit alone does not confirm a loop.
4. Preserve each response’s original prompt, remove duplicates and keep examples within the 32,768-token training limit. Select 104 correct and 104 looping responses.

Follow [training.example.jsonl](data/training.example.jsonl): each row holds correct and looping responses with their own prompts. Prompts end in `<think>\n`; the processor adds EOS only to correct responses.

### Run KTO training

See [experiment_environment.yaml](../config/experiment_environment.yaml) for the software versions used in training and evaluation. Use [kto.yaml](../config/kto.yaml) to configure KTO training.

Set the paths in [run_kto.sh](run_kto.sh). For `PRETRAINED_CHECKPOINT`, use the Megatron QAD iteration directory, such as `iter_0000500`. Keep the model weights, saved ModelOpt quantization state, and original QAD `run_config.yaml` together in that directory. The [Qwen checkpoint helper](../src/qwen36_checkpoint.py) loads the language model and checks that quantization is active.

Run the launcher from the parent tutorial directory.

```bash
bash kto/reproducibility/run_kto.sh
```

On GB300 with NeMo 26.08.00, `run_kto.sh` starts a worker that enables FlashAttention 2 to reduce memory use and splits large FusedAdam groups to avoid running out of tensor handles. The worker leaves both fixes off in NeMo 26.10 RC8. In that version, Transformer Engine can choose fused attention automatically, and FusedAdam splits large updates itself. [Attention fix](../src/sm103_attention.py) · [Optimizer fix](../src/te_optimizer_groups.py)

## Evaluation

We run benchmarks and collect results using the [QAD tutorial’s evaluation setup and configs](../../evaluation/README.md), with presence penalty zero for both models. We compare the models using the paired analysis described in [Evaluating Models Efficiently During Optimization Experiments](https://github.com/NVIDIA/Model-Optimizer/blob/c8559d2b4e547960db1713640d3d54bb9b7c8280/docs/source/announcements/evaluating_models_efficiently_during_optimization_experiments/index.rst).

For tau2-bench Telecom, override `agent_args.presence_penalty` from 1.5 to 0 for both models.
