# Algorithm sheets

One file per draft-model algorithm. A sheet holds only what differs between
algorithms; the procedure lives in `../stages/`. `eagle3.md` is the worked example.

Keep sheets short. If something is true for every algorithm, it belongs in the
stage file instead.

## Required sections

Stage files reference these by heading, so use the headings verbatim.

| Heading | Contents |
| --- | --- |
| `## Pipeline tasks` | Table of task → script → purpose → output path. Task *count* varies per config — EAGLE3 offline is 4 tasks, DFlash offline is 2 — so describe the tasks this algorithm's examples actually use. |
| `## Recipe and training knobs` | The `modelopt_recipes/general/speculative_decoding/<algo>.yaml` path, plus the per-model overrides that usually need tuning. |
| `## Per-model adjustments` | The non-obvious knobs that vary by target model (attention type, MoE dims, tokenizer, `trust_remote_code`). |
| `## Success markers` | Per task, the log line that proves it worked, and the artifact it should leave behind. Consumed by review-logs and validate. |
| `## Quality gate` | The metric, where it appears in the log, and the pass threshold. |
| `## Known failures` | Error pattern → root cause → fix, for failures specific to this algorithm. Generic failures (OOM, NCCL, time limit) live in `../stages/triage.md`. |

## Adding a sheet

Source the facts from the repo rather than from memory:

- Launcher examples: `tools/launcher/examples/*/*/hf_*_<algo>.yaml` — task layout,
  scripts, container images, GPU sizing.
- Scripts: `tools/launcher/common/` — `eagle3/` and `specdec/` hold the training and
  hidden-state-dump entry points.
- Recipe: `modelopt_recipes/general/speculative_decoding/<algo>.yaml` — defaults.
- Implementation: `modelopt/torch/speculative/plugins/` — `hf_<algo>.py` and
  `modeling_<algo>.py`.

Then add a row to the algorithm table in `../../SKILL.md`.

Algorithms with launcher examples but no sheet yet: DFlash, DSpark, Domino.
