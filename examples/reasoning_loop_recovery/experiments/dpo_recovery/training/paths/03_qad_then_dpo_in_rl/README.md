# Path 3: QAD followed by DPO in NeMo RL

The numbered items include environment setup and compatibility fixes; they are not a count of
independent bugs. Validation results and unresolved work are listed separately below.

## Setup and fixes

1. **Prepare the path-2 environment.** Use the QAD Bridge/Core/ModelOpt overlay inside the release
   NeMo RL image, including its isolated dependencies, architecture audit, and worker provenance
   checks. See [path 2](../02_qad_stack_in_rl/README.md#setup-and-fixes). Its Core already reads
   distributed common state; no path-1 common-reader patch is needed.
2. **Use the real QAD starting point and data.** Start from the round-2 native iterative checkpoint
   and its original BF16 teacher. The smoke uses the eight shortest real examples from the frozen
   round-3 dataset, one QAD step, input-masked top-4096 KL, LR 1e-6, sequence 32,768, GBS=8,
   MBS=1, and EP=4. The full dataset has 954 examples and a 120-step schedule.
3. **Enter QAD through four-rank `torchrun`.** Importing the script during `--help` initialized
   `env://` without `RANK`; use a distributed entrypoint rather than a bare import probe.
4. **Apply SM103 correction once.** Install it in `distill.py` before model construction. Applying
   it twice broke source introspection. Retain the recognized chunked-logprob settings for DPO.
5. **Disable gradient accumulation fusion on both models.** Teacher and student otherwise require
   the absent `fused_weight_gradient_mlp_cuda` extension.
6. **Save and explicitly allow the new checkpoint.** Save QAD synchronously, then enable the native
   language-only provider and extend the strict prefix mapper's allowlist to that exact checkpoint.
   It still has a `decoder.*` root. Preserve missing-key checks, text MRoPE handling, pretrained
   iteration detection, and quantizer audits; saving in this environment alone does not remove them.

## Validation and remaining work

The one-step QAD stage and subsequent one-step DPO/save passed. Full round-3 QAD reproduction is
not completed in this documented snapshot. This is an environment-handoff test, not stock NeMo RL
QAD support or a new PTQ run from an HF deployment checkpoint.

Use `qad_then_dpo/run_inside.sh` and `run_qad.py`. The next gates are full round-3 QAD reproduction,
full DPO from that checkpoint, separate-process reload, NVFP4 export, and matched SciCode evaluation.
The one-step smoke does not supply evidence that a full recovery round is equivalent to the original.

See [path.yaml](path.yaml) for the configuration record and [shared fixes](../../compatibility/README.md).
