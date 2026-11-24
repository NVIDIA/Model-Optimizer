# Evaluation Tools

One file per evaluation tool that needs tool-specific details. The generic guidance never names a tool; it points
here. Use the file of the tool the task names, and do not load another tool's skills.

| Tool | Reference | Skills that run it |
|---|---|---|
| NEL (NeMo Evaluator Launcher) | [nel.md](nel.md) | `modelopt-evaluation`, `modelopt-launching-evals` |
| NeMo Gym | [gym.md](gym.md) | the Gym evaluation skills (`run-and-babysit-eval`, `add-model-eval-config`) |

The Gym skills come from the NVIDIA-NeMo/Gym repository (`.agents/skills`), not from this repository: the task needs
them installed. If they are missing, say so instead of improvising a run.

**Default tool** (used when the task names none): NEL.
