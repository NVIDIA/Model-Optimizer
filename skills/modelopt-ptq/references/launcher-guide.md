# Using the ModelOpt Launcher for PTQ

Read the common skill's `launcher.md` first: how to run the launcher on SLURM or local Docker, size the job, and find its outputs. This page adds the PTQ config.

## HF Transformers PTQ Config

The launcher provides `common/hf/ptq.sh` which wraps `hf_ptq.py`. Configure via environment variables:

```yaml
job_name: <Model>_<Format>
pipeline:
  task_0:
    script: common/hf/ptq.sh
    environment:
      - HF_MODEL: <HuggingFace model ID, e.g. Qwen/Qwen3-0.6B>
      - QFORMAT: <format, e.g. nvfp4, fp8, int4_awq>
      - CALIB_SIZE: "512"
      - EXPORT_PATH: /scratchspace/exported_model
    slurm_config:
      _factory_: "slurm_factory"
      nodes: 1
      ntasks_per_node: 1
      gpus_per_node: <num_gpus>
```

Set `gpus_per_node` as the common `launcher.md` describes: many clusters accept only full nodes.

> **`EXTRA_PIP_DEPS` must avoid shell metacharacters.** It is written into an
> unquoted `export` in the generated sbatch script, so a value like
> `transformers>=4.57,<4.58` is mangled by shell redirection (`>`/`<`) and
> silently dropped — the deps never install. Use exact `==` pins (no `>`/`<`).
> The right version is **model-specific** — a brand-new architecture may need a
> newer transformers than the repo's library pin (e.g. Qwen3.5's `qwen3_5` needs
> `EXTRA_PIP_DEPS: "transformers==5.5.0"`); pick what the target model requires.

Extra `hf_ptq.py` flags can be passed via `args`:

```yaml
    args:
      - --batch_size 2
      - --trust_remote_code
```

## Output Location

`EXPORT_PATH` controls the path inside the container (default: `/scratchspace/exported_model`).

**Local Docker** — find the checkpoint on the local host:

```bash
find tools/launcher/local_experiments -name "config.json" -path "*/exported_model/*" 2>/dev/null
```

**Remote SLURM** — the checkpoint is on the remote machine, under the experiment's folder (see the common `launcher.md`).

## Known Issues

- **Megatron-LM submodule**: Only needed for `MegatronLMQuantizeTask` (Megatron models). HF PTQ via `common/hf/ptq.sh` does not require it.
