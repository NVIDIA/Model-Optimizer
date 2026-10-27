# Path 5: Vanilla BF16 Hugging Face control

The numbered items include environment setup and compatibility fixes; they are not a count of
independent bugs. Validation results and unresolved work are listed separately below.

## Setup and fixes

1. **Select the BF16 control.** Start from the original Qwen3.6-35B-A3B HF teacher with the release
   Bridge/Core. Disable `policy.quant_cfg` and remove `checkpointing.pretrained_checkpoint`.
   This path has no native QAD checkpoint or ModelOpt common-state handoff to repair.
2. **Enable chunked Qwen log probabilities.** The Qwen subclass overrides generic GPT forward and
   rejects its chunked-loss keyword. The experiment-local adapter preserves native preprocessing
   and decoder execution, then uses NeMo RL's official `from_parallel_hidden_states_to_logprobs`
   for the output stage. Temporarily bypass auxiliary MTP only there, restore its flag in `finally`,
   and restrict this verified adapter to TP=1.
3. **Check numerical agreement.** Compare the first eight positions against explicit full
   projection/log-softmax before optimization. Maximum observed error was 1.65e-6, below tolerance
   2e-3. See [the adapter details](../../compatibility/README.md#attention-and-log-probability-memory).
4. **Enable SM103 attention in BF16 workers too.** After chunking was fixed, backward requested
   28.57 GiB with roughly 262 GiB already used. Installing the same validated attention correction
   before model construction resolved this OOM; it must not be limited to quantized workers.

## Validation and remaining work

The initial eight-pair, 8,192-token, one-step smoke passed but did not establish full-run memory fit.
After the fixes above, full 104-pair, 32,768-token, 13-step training and separate-process
policy/optimizer reload passed, with diagnostic loss 0.00007148674870879127. The successful retry
segment took approximately 22m04s. This control separates generic model/DPO compatibility from
fake-quantized checkpoint restoration; it is not a replacement for the recovery-policy experiment.

Use `vanilla_hf_full/run_inside.sh`, `prepare_and_run.py`, `vl_chunked_adapter.py`, and its worker
import hook. Keep BF16 outputs separate from all quantized paths. No SciCode outcome for this BF16
DPO checkpoint is recorded here, and passing it does not test preservation of fake quantizers.

See [path.yaml](path.yaml) for the configuration record and [shared fixes](../../compatibility/README.md).
