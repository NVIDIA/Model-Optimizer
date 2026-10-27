# Reasoning-loop recovery experiments

Experiments on reducing repetitive reasoning and termination failures after quantization-aware
distillation (QAD). Each experiment keeps its own settings, data provenance, and results.

- [Prompt-7 recovery QAD](experiments/prompt7_recovery_qad/README.md): the original experiment using
  334 completed, filtered responses from the QAD model with an anti-repetition instruction.

- [DPO recovery](experiments/dpo_recovery/README.md): preference-pair preparation, six environment
  paths for training after iterative recovery QAD, and matched SciCode evaluation.

Large datasets, generated responses, checkpoints, and containers are not included. Runtime paths
are supplied separately; manifests use portable filenames and checksums. A missing public download
URL means that an artifact has not been published with this example.
