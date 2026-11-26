# QAKTO reproducibility

[Back to the QAKTO tutorial](../README.md).

## KTO training

We fine-tuned the QAD model with KTO to reduce reasoning loops. We ran training in NeMo RL with expert parallelism of four (EP=4) so it fit on one GB300 node with four GPUs. Export the trained QAKTO model using the parent’s [NVFP4 export procedure](../../README.md#4-export).

### Collect KTO training responses

The experiment reused saved QAD-generated responses. To collect a new training set:

1. Use the [44/11 subtask split](data/subtask_split.yaml). Collect responses only for the 44 training subtasks and reserve the other 11 for evaluation.
2. Generate repeated responses with the QAD model and run the SciCode tests. Keep responses that pass as correct examples.
3. Inspect unfinished responses and keep those that repeat without making progress as looping examples. A token-cap hit alone does not confirm a loop.
4. Preserve each response’s original prompt, remove duplicates and keep examples within the 32,768-token training limit. Select 104 correct and 104 looping responses.

Follow [training.example.jsonl](data/training.example.jsonl): each row holds correct and looping responses with their own prompts. Prompts end in `<think>\n`; the processor adds EOS only to correct responses.

### Run KTO training

See [experiment_environment.yaml](../config/experiment_environment.yaml) for the software versions used in training and evaluation. Use [kto.yaml](../config/kto.yaml) to configure KTO training. Set the paths in [run_kto.sh](run_kto.sh), then run from the parent tutorial directory:

```bash
bash kto/reproducibility/run_kto.sh
```

Set `PRETRAINED_CHECKPOINT` to the Megatron QAD checkpoint directory for the starting step (for example, `iter_0000500`). It must contain the model weights and saved ModelOpt quantization state. The launcher uses the [Qwen checkpoint helper](../src/qwen36_checkpoint.py) to load the language model and check that quantization is active.

If Megatron Bridge needs a separate architecture config, place a matching `run_config.yaml` in the checkpoint directory alongside the weights.

The original GB300 run on NeMo 26.08 needed the [SM103 attention patch](../src/sm103_attention.py). Use automatic attention selection with NeMo 26.10 RC8 or later. The RC8 QAD retest ran without the patch.

## Evaluation

We run benchmarks and collect results using the [QAD tutorial’s evaluation setup and configs](../../evaluation/README.md), with presence penalty zero for both models. We compare the models using the paired analysis described in [Evaluating Models Efficiently During Optimization Experiments](https://github.com/NVIDIA/Model-Optimizer/blob/c8559d2b4e547960db1713640d3d54bb9b7c8280/docs/source/announcements/evaluating_models_efficiently_during_optimization_experiments/index.rst).

For tau2-bench Telecom, override `agent_args.presence_penalty` from 1.5 to 0 for both models.
