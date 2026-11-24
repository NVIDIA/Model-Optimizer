# Find an experiment and read its task logs

Use the supplied experiment path. Otherwise, look under `experiments/` relative
to the launcher root, or the configured `--job-dir`. For the default layout,
list recent runs with `ls -td experiments/cicd/cicd_* | head -10`. If no run
matches or the intended run is unclear, ask for its path or the model/YAML used.

Each experiment has one subdirectory per task. Match `*.out` (Slurm) and
`*.log` (local Docker). Set `exp_dir` to the selected run directory and
`lines` to the stage's tail length:

```bash
find "$exp_dir" -type f \( -name '*.out' -o -name '*.log' \) |
  sort | while IFS= read -r f; do
    printf '\n=== %s ===\n' "$f"
    tail -n "$lines" "$f"
  done
```
