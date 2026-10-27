# Path 4: NeMo RL inside the original QAD container

The numbered items include environment setup and compatibility fixes; they are not a count of
independent bugs. Validation results and unresolved work are listed separately below.

## Setup and fixes

1. **Keep the QAD runtime.** Preserve NeMo 26.08.00's Bridge/Core, Torch, Transformers, and TE;
   overlay the actual QAD ModelOpt source. This path tests Python 3.12 source integration, not the
   release NeMo RL image's officially supported Python environment.
2. **Add NeMo RL and isolated dependencies.** Add the release container's NeMo RL source and missing
   import dependencies through `PYTHONPATH`; disable logging backends. Use the Qwen-specific source
   copy that omits the unrelated DeepSeek tokenizer patch intended for older Transformers.
3. **Use the same Python in Ray workers.** Select `/opt/venv/bin/python` and
   `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`; verify the effective module paths in workers as well as the driver.
4. **Use QAD's existing common-state reader.** Its Core already supports the checkpoint's
   `.distcp` `common_state` object despite absent `modelopt_state/common.pt`. Unlike path 1, this
   path does not need the older-reader patch. See [common state](../../compatibility/README.md#distributed-common-state).
5. **Construct and restore the native LM.** Use the language-only hybrid provider/spec, architecture
   parity audit, and strict LM prefix mapping where needed. Do not request absent vision weights
   or accept missing language-model shards.
6. **Handle text MRoPE and checkpoint detection.** Expand text positions to three axes and check the
   actual supplied checkpoint iteration rather than assuming `iter_0000000`.
7. **Audit quantizer restoration.** Check policy/reference quantizers on every rank after restoration.
   Keep policy quantization active for optimization; the frozen-reference forward temporarily
   disables quantization in this implementation.
8. **Enable SM103 attention.** Correct TE 2.17.1's eligibility check before model construction so
   head dimension 256 uses the validated FlashAttention path.
9. **Chunk log probabilities.** Use `use_linear_ce_fusion_loss` with
   `linear_ce_fusion_chunk_size: 128` to avoid the full-vocabulary FP32 allocation.
   See [attention and memory](../../compatibility/README.md#attention-and-log-probability-memory).
10. **Bound FusedAdam groups.** TE 2.17.1 exhausted its fixed tensor-handle pool despite adequate GPU
    memory. The experiment-local wrapper splits groups at 512 tensors, preserving parameter order
    and optimizer hyperparameters. Use the same grouping for optimizer reload. This is separate
    from attention OOM. See [optimizer details](../../compatibility/README.md#optimizer-and-optional-dependencies).

## Validation and remaining work

Validated milestones:

- 13-step DPO training: completed in 20m51s.
- Separate-process policy/optimizer reload: completed in 9m16s; validation loss 0.00014002833511441256.
- NVFP4 export: completed in 6m16s, with quantization metadata and safetensors checked.
- SciCode evaluation: 550/550 target responses, zero cap hits and zero missing final answers.

The top-level [results](../../../README.md#results) report training and held-out subtasks separately.
This is the most extensively validated path in the current snapshot, but it remains one candidate.
Use `nemo_rl_on_nemo/run_full.sh`, stack mode `qad-container`, and `aligned_stack/run_dpo.py`; then
rerun that driver with `--reload`. Its initial `run_inside.sh --smoke` result alone is not the full run.

See [path.yaml](path.yaml) for the configuration record and [shared fixes](../../compatibility/README.md).
