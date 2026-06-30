# Testing

## Running Tests Locally

From the launcher directory:

```bash
cd Model-Optimizer/tools/launcher
uv pip install -e . pytest
uv run pytest -v
```

64 unit tests covering:

| File | Tests | Coverage |
|------|-------|---------|
| `test_core.py` | 16 | Dataclasses, factory registry, global_vars, env, versions |
| `test_core_extended.py` | 12 | Error cases, env merging, test_level, skip, detach |
| `test_slurm_config.py` | 9 | SlurmConfig defaults, env var overrides, factory |
| `test_docker_execution.py` | 10 | Docker executor mounts, run_jobs path selection |
| `test_slurm_executor.py` | 5 | Slurm executor mounts, tunnel params (mocked) |
| `test_yaml_formats.py` | 7 | YAML parsing, task_configs, overrides |
| `test_docker_launch.py` | 2 | End-to-end Docker launch via subprocess |

## CI

The GitHub Actions workflow (`.github/workflows/unit_tests.yml`) runs launcher tests on every PR:

```yaml
launcher:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v6
      with:
        submodules: recursive
    - name: Run launcher tests
      working-directory: tools/launcher
      run: |
        curl -LsSf https://astral.sh/uv/install.sh | sh
        uv venv .venv && uv pip install -e . pytest
        uv run pytest -v
```

The launcher job is a required check — PRs cannot merge if tests fail.

## Not Covered (by design)

These require live infrastructure and are tested manually:

- Actual SSH tunnel and sbatch submission
- Docker container launch with GPU workloads
- PatternPackager tar.gz and rsync
- nemo experiment status/logs polling
