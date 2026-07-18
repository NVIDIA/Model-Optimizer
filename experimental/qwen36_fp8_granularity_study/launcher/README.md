# aws-cmh launcher

This directory contains one five-task Nemo-Run pipeline for each requested Qwen3.6
model:

- `qwen3.6-35b-a3b_aws-cmh.yaml`
- `qwen3.6-27b_aws-cmh.yaml`

Each pipeline stages the public Hugging Face model, tokenizer, and evaluation-data
prefix on `cpu_datamover`, then runs four sequential one-node jobs on `batch_long` with
all four GPUs on the node. The four candidate slots are per-tensor FP8, static
block-128 weight-only FP8, research-only dynamic block-128 W8A8 FP8, and MXFP8.
`allow_to_fail` is enabled so a candidate failure does not prevent later independent
candidates from running.

The per-tensor, dynamic-block, and MXFP8 task slots each run their requested W8A8
candidate followed by a weight-only control. Together with the shipped static-block
weight-only diagnostic, this produces seven candidate result directories without
adding Nemo tasks.

The configs deliberately leave the login host and SSH identity to the `aws_cmh`
cluster factory. They explicitly pin the Slurm account to `coreai_numerics_edge` and
use the cluster-local TensorRT-LLM `release:1.3.0rc11` ARM64 SquashFS, already proven
for Qwen3.6 FP8 and KL-based PTQ on aws-cmh. A read-only mount supplies the matching
Qwen3.6 Transformers checkout (5.7.0.dev0 at commit `74a2a4d0c790`) while the packaged
ModelOpt source remains authoritative. Every task validates the exact model class and
packaged ModelOpt source; GPU tasks additionally validate the C++/CUDA build toolchain
and compile both required extensions before loading model weights. The configs mount
the persistent study root at `/study`:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_numerics_edge/users/weimingc/qwen36_fp8_granularity_study
```

The staging task resolves and records exact Hugging Face repository revisions before
any GPU job runs. It also writes the first 1,056 source rows of
`abisee/cnn_dailymail` config `3.0.0`, train split, to an atomic model-specific JSONL,
recording its row count, SHA-256, dataset fingerprint, and source revision in the
staging manifest. The 1,056-row prefix is exact: packed calibration requests a
conservative pool of the first `128 * 8 = 1,024` raw documents (packing can fill
its rows before all contribute), while unpacked evaluation selects rows `[1024, 1056)`.
GPU tasks validate every JSONL row and its SHA-256 before setting
Hugging Face, Transformers, and Datasets offline modes. They load the staged model
revision with `--local-files-only` and pass the same local JSONL path to calibration
and evaluation, so no batch job needs dataset or model network access.

ModelOpt's FP8 and MXFP8 fake-quantization extensions are compiled before model
loading. Torch, CUDA, and Triton caches live under `/study/cache`, split by model, so
the sequential jobs reuse compiled binaries despite Pyxis disabling the home mount.

All candidates for a model reuse the same persisted reference-logit cache. The
initial screen uses 128 packed calibration samples and 32 held-out evaluation samples
at a fixed sequence length and batch shape of `512 x 1`; 32 packed calibration rows
are reused for activation-quantizer MSE. A per-invocation research hook handles varying
fused-MoE routed-token shapes. GPU tasks request 24 hours so paired candidates have
headroom, while each candidate records its actual phase and total wall time.

`run_study.sh run --candidates` also accepts a comma-separated list. This permits a
future controlled weight-only submatrix to execute inside the existing four GPU task
slots rather than exceeding the launcher's five-task limit.

The launcher packages exactly
`experimental/qwen36_fp8_granularity_study` through the existing
`tools/launcher/modules/Model-Optimizer` symlink. Therefore submission must use a
branch or commit containing this subtree; there is no mirrored copy of `study.py`.

Live submission and monitoring should use the Pensieve/modelopt MCP flow: resolve
`aws_cmh`, perform a dry run, obtain approval, then submit with the same cluster
configuration and source ref.
