Run `examples/Qwen/Qwen3-8B/hf_compare_v2.yaml` with a launcher v2 execution
profile. Override `baseline`, `candidate`, `dataset`, or `samples` through the
workflow parameters. Public defaults use immutable Hugging Face revisions;
an internal profile can resolve `candidate` from an MLflow `models:/name/version`
reference. Cluster and artifact credentials belong in the execution profile.

The single `validate` task consumes both checkpoints. It starts one vLLM server
at a time, runs lm-evaluation-harness's canonical eight-shot GSM8K CoT task on
64 test examples, then runs AIPerf at concurrency 1, 4, and 8. Each performance
cell uses one shared 32768-token prefix, 2048 new input tokens, 1024 output
tokens, three warmup requests, and 8/16/32 measured requests. Both arms use
temperature zero, prefix caching, BF16 KV cache, and one GPU with TP=PP=1.
If the cluster requires a larger allocation, set `allocated_gpus` accordingly;
the comparison selects the first assigned GPU by UUID and records the allocated
and active counts in `gpu.json`. All allocated GPUs count toward resource usage.
These small samples demonstrate checkpoint compatibility and the workflow;
they do not establish a full accuracy or performance acceptance gate.

On H100, set `candidate_linear_backend=marlin`. This measures NVFP4 weights
with BF16 activations (W4A16), rather than native NVFP4 activation computation.
The default `auto` lets vLLM select a supported kernel on other hardware.
Inspect the saved server logs for the actual selected implementation.

The comparison explicitly sets `--kv-cache-dtype bfloat16` for both arms so
checkpoint metadata cannot select a different KV precision. For a bounded cold
start, set `flashinfer_autotune=off` to disable FlashInfer autotuning in both
servers. This changes kernel tuning and can affect performance; compare only
curves with matching tuning settings. The default `auto` preserves vLLM's
choice. Recipes record both KV precision and the tuning setting.

For startup failures in TorchInductor, `compilation_mode=none` passes
`-cc.mode=0` to both servers, disabling `torch.compile`. vLLM can also disable
CUDA graphs when the selected graph mode requires compilation; inspect the
actual configuration in `server.log`. The default leaves vLLM's compilation
settings unchanged. Results record the chosen mode and full commands; compare
curves with matching modes.
See the [vLLM compilation debugging guide](https://docs.vllm.ai/en/v0.25.0/design/debug_vllm_compile/).

For CUDA startup failures, run `examples/evaluation/gpu_runtime_v2.yaml` with
a one-GPU profile first. It records the driver, Torch and CUDA versions, then
checks FP32 and BF16 linear operations without downloading models or installing
benchmark clients. An internal tracking profile saves `runtime/runtime.json`
even if a check fails. Passing these checks does not validate the full server.
The `suite=serving` parameter additionally checks Qwen3-sized projection
matrices, vLLM RMS normalization, and a real TorchInductor-compiled RMS
normalization. Use `launch_blocking=1` to report CUDA failures synchronously.
These operations use synthetic tensors and produce no benchmark measurements.
For clusters with a minimum GPU allocation, set `allocated_gpus` to that count.
The diagnostic uses one process on the first visible GPU and records both the
visible GPU count and `active_gpu_count=1`. All allocated GPUs count toward the
job's resource footprint, even though only one performs these checks.

The script installs pinned benchmark clients into a temporary virtual
environment, keeping the serving image's dependencies separate. For a prepared
environment, pass `--client-python /absolute/venv/bin/python` when calling the
script directly. The actual AIPerf tokenizer loader must pass an offline
encode/decode check before either server starts. The local cache mapping uses
the exact checkpoint tokenizer files and records their checksums.

Results include raw lm-eval predictions, AIPerf exports, exact commands,
tokenizer provenance, metrics, and a CSV curve per arm. With launcher tracking
enabled, results and metrics are logged to the existing task run, whose model
input records are supplied by the v2 artifact wrapper. The script does not
register another checkpoint. AIPerf exports can be imported into the existing
BTK/Speed results adapter after collection; it does not require Docker inside
the GPU allocation. The benchmark client here is AIPerf 0.13.0; it intentionally
differs from older BTK client images that pin 0.8.0.
