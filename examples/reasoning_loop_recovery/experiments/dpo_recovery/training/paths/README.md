# Seven environment paths

This is a snapshot of saved validation artifacts on **2026-10-27**. Completion means the indicated
check passed, not that all environments are equivalent or that a workflow is upstream supported.
All GPU trials used four GB300 GPUs on one node. Paths share the preference dataset and objective,
except explicitly smaller compatibility smoke tests. No final path has been selected.

| Path | Approach | Validated milestone | Remaining work |
| --- | --- | --- | --- |
| [1](01_release_compat/README.md) | Release NeMo RL + loader fixes and QAD ModelOpt | 13 DPO steps; reload; NVFP4 export | Matched SciCode comparison submitted |
| [2](02_qad_stack_in_rl/README.md) | QAD Bridge/Core/ModelOpt inside NeMo RL image | 13 DPO steps; reload; NVFP4 export | Matched SciCode comparison submitted |
| [3](03_qad_then_dpo_in_rl/README.md) | QAD then DPO inside NeMo RL image | One QAD step → one DPO step and save | Full QAD→DPO/reload job running; export/evaluation follow |
| [4](04_rl_in_qad_container/README.md) | NeMo RL source inside original QAD image | 13 DPO steps; reload; NVFP4 export; SciCode 550/550 | Larger evaluation and workflow comparison |
| [5](05_vanilla_hf/README.md) | BF16 HF control with release bundled Bridge/Core | 13 DPO steps; policy/optimizer reload | BF16 export submitted; control evaluation follows |
| [6](06_current_main/README.md) | Pinned current main and its own dependencies | Dependency/import provenance; no successful DPO step | Resolve vanilla and quantized smoke failures |
| [7](07_main_in_qad_container/README.md) | Latest main source inside original QAD image | Distributed common-state read passed without reader patches | Import/API compatibility retry submitted; smoke/full training/reload/export/evaluation follow |

Each path has a README for the procedure, failure history, and next validation gate, plus
`path.yaml` for its configuration differences and source artifact names. File names in those
records refer to an external experiment workspace, not to missing files in this repository.
[Compatibility details](../compatibility/README.md) explain shared fixes; [versions.yaml](../versions.yaml)
pins the stacks. A smoke-passed path is not counted as full-training-passed.

## Main-based continuation

Paths 6 and 7 now also test the same newer main commit,
`89d8f0c5d217e0270a5718880d811e51f3951e39`. Earlier path-6 failures remain documented as a separate
pinned snapshot. Path 6 uses main's own dependency environment; path 7 first preserves the QAD
container stack. Neither path is assumed to eliminate all checkpoint compatibility changes.
