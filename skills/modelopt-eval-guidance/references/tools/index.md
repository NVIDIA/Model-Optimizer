# Evaluation Tools

The tool this guidance covers is NeMo Gym.

| Tool | Reference | Skills that run it |
|---|---|---|
| NeMo Gym | [gym.md](gym.md) | the Gym evaluation skills (`run-and-babysit-eval`, `add-model-eval-config`) |

The Gym skills come from the NVIDIA-NeMo/Gym repository (`.agents/skills`), not from this repository: the task needs
them installed. If they are missing, say so instead of improvising a run.

Runs of the default tool do not use this skill: they run through `modelopt-evaluation` and
`modelopt-launching-evals`, which are self-contained.
