# SLURM Setup for PTQ

PTQ-specific SLURM details. For generic SLURM patterns (account discovery, job template,
monitoring), see the common skill's `slurm-setup.md`.

---

## 1. Container

Use `nvcr.io/nvidia/pytorch:26.09-py3` for Hugging Face PTQ (AMD64/ARM64).
Keep inference frameworks in the downstream serving environment. Look for a
cached `.sqsh` of this image and verify its provenance before reuse:

```bash
ls *.sqsh ../*.sqsh ~/containers/*.sqsh 2>/dev/null
```

**If a `.sqsh` exists**, use it directly with `--container-image=<path>`. Skip import.

**If no `.sqsh` exists**, import with enroot (caches for subsequent smoke tests and reruns):

```bash
export ENROOT_CACHE_PATH=/path/to/writable/enroot-cache
export ENROOT_DATA_PATH=/path/to/writable/enroot-data
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
enroot import --output /path/to/container.sqsh docker://nvcr.io#nvidia/pytorch:26.09-py3
```

If enroot import fails (e.g., permission errors on lustre), use pyxis inline pull as fallback — pass `--container-image="nvcr.io/nvidia/pytorch:26.09-py3"`. Note this re-pulls on every job.

### Resolve and verify the Python environment before GPU submission

NGC26.09 is the default native development stack, not a guarantee that its
Python packages resolve with current ModelOpt. Inspect inherited pip constraints,
installed distributions, and actual import paths. A venv with
`--system-site-packages` retains vendor conflicts and stale ModelOpt packages.
Unsetting `PIP_CONSTRAINT` alone does not reconcile those packages.
Torch runtime and distribution versions can differ; obtain pip constraints from
`importlib.metadata.version("torch")`, not `torch.__version__`.

Prefer an immutable CUDA development image with released, hardware-compatible
Torch wheels in a venv **without** system-site-packages. The PTQ skill's
`scripts/install_environment.sh` installs the exact source checkout and resolves
its HF dependencies with explicit Torch, Transformers, and Torchvision pins:

```bash
bash <ptq-skill>/scripts/install_environment.sh \
    <Model-Optimizer-source> <new-venv> <torch-version> <transformers-version> <torchvision-version>
<new-venv>/bin/python <ptq-skill>/scripts/verify_environment.py \
    --source <Model-Optimizer-source> --ref <exact-source-commit> \
    --model-class <required-transformers-class>
<new-venv>/bin/python <Model-Optimizer-source>/examples/hf_ptq/hf_ptq.py --help
```

This installer selects Torch SDPA; pass `--attn_implementation sdpa` to PTQ.
It omits optional `flash-attn` and the unused `transformers_stream_generator`
requirement, which imports APIs removed in Transformers 5. Models requiring
FlashAttention or other custom kernels need a separately resolved and tested
installation. Do not use this SDPA setup for those models.

For Qwen3.5/3.8 (`qwen3_5`), use Transformers `5.14.1` while it remains within
source ModelOpt's supported range; `5.5.4` misclassifies VLM language-model
weights during export. Recheck source metadata before choosing versions. Record
the exact source/model revisions, image digest, dependency freeze, interpreter,
import paths, and CUDA libraries. A reference clean-image Dockerfile is provided
in `scripts/`; its build arguments require explicit base-image and source pins.
Keep serving frameworks in a separate image. Run Python probes outside the
source checkout so its package directory and build metadata cannot shadow the
installed wheel. The reference image defaults to `/opt/ptq`.

Before launching calibration, require `pip check`, unambiguous ModelOpt
provenance, `hf_checkpoint_utils` import, model-class imports, `hf_ptq.py --help`,
and native CUDA library loads in the actual PTQ environment. Run the verifier
with `--require-cuda` on target hardware to exercise BF16 and NVFP4 kernels.
CPU checks cannot establish GPU compatibility. Preparation GPU submissions
consume the workflow's submission budget; record them before launching.

The released ARM64 `nvidia-cusparselt-cu13==0.8.1` wheel names `aarch64`
in its filename but declares the unsupported `manylinux2014_sbsa` WHEEL tag.
Its library is ARM64 ELF. For that exact packaging defect, explicitly pass
`--allow-cusparselt-sbsa` to the installer and verifier (Docker build argument
`PACKAGE_CHECK_FLAGS=--allow-cusparselt-sbsa`). The verifier retains pip's nonzero
return code and raw finding, checks the exact version/tag and ELF architecture,
and still requires native library loads and target GPU checks. Every other
`pip check` error remains blocking; no wheel metadata is rewritten.

If vendor Torch is necessary, build a deliberately reconciled derived image.
Remove stale ModelOpt distributions, account for every vendor constraint and
required dependency, and verify actual imports. Do not blindly upgrade vendor
Torch/CUDA, use a global `--no-deps` fallback, or mask packaging problems with
`PYTHONPATH`. Explain any vendor-only `pip check` exceptions; task-relevant
incompatibilities remain blocking.

---

## 2. GPU Sizing

Estimate GPU count from model size and available GPU memory. `hf_ptq.py` uses `device_map="auto"` so it fills GPUs automatically — request only as many as needed.

For multi-node PTQ (200B+ params), use `hf_ptq.py --use_fsdp2`. For the launch commands (`sbatch`
and manual `torchrun`) and the `--recipe` format, see the *Multi-Node Post-Training Quantization with
FSDP2* section of `examples/hf_ptq/README.md`.

Sizing guidance specific to this path: when the per-rank decoder shard approaches GPU capacity (200B+ at low rank count), either add more nodes (more ranks → smaller shard per rank) or add `--cpu_offload`. Layer detection is automatic; no YAML config needed.

Use the multi-node template from the common skill's `slurm-setup.md` section 4 as the job script wrapper.

---

## 3. Smoke Test

Before the full calibration run, submit a smoke test with `--calib_size 4` and `--time=00:30:00`.
This catches script errors cheaply before using GPU quota on a real run.

See the common skill's `slurm-setup.md` section 2 for the smoke test partition pattern.

Only submit the full calibration job after the smoke test exits cleanly.
