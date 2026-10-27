# Path 7: Current NeMo RL main in the original QAD container

Keep path 4 as the working baseline and test newer NeMo RL source in the same NeMo 26.08.00
container. Preserve its QAD-compatible Bridge/Core and actual QAD ModelOpt source. Use the same
NeMo RL commit as the latest path-6 continuation: `89d8f0c5d217e0270a5718880d811e51f3951e39`.

## Runtime boundary

This main snapshot declares Python >=3.13.14, while the QAD image provides Python 3.12. The initial
source-integration probe deliberately preserves the QAD environment to establish which imports and
APIs work; it is an experimental compatibility test, not an officially supported installation.
Keep baseline results separate from diagnostic attempts that override version guards. If a supported
Python 3.13 environment is needed, its Torch/TE binaries must match that interpreter; copying the
QAD image's Python 3.12 native packages into it is not a valid solution.

The numbered items include environment setup and compatibility fixes; they are not a count of
independent bugs. Validation results and unresolved work are listed separately below.

## Setup and fixes

1. **Keep the QAD stack and pin newer NeMo RL.** Use the original QAD Bridge/Core and actual
   ModelOpt source with NeMo RL `89d8f0c5d217e0270a5718880d811e51f3951e39`, also used by the
   latest path-6 continuation. Record driver/worker module provenance; retain the Python runtime
   boundary above rather than treating a version-guard override as supported compatibility.
2. **Use the existing common-state reader.** Probe the round-3 checkpoint without loader hooks.
   Distributed ModelOpt common-state reading passed with the QAD Core; no path-1 reader patch
   or fabricated `common.pt` was required. Full weight/quantizer loading is a separate gate.
3. **Add the missing NeMo Lens source.** The initial DPO import failed on `nemo.lens`. Adding the
   existing Lens source resolved this import failure without replacing QAD's native packages.
4. **Adapt the missing Bridge API — diagnostic retry.** The next import required `LocalHFParamSpec`,
   absent from the older QAD Bridge. The retry adds the exact pure tensor-view descriptor class
   from Bridge `7e9dcd609edf2bb9478d3fa4d9b9f8ee3875c16e` through an experiment-local import hook,
   with a tensor-selection/shape check. This changes an API surface, not checkpoint serialization
   or quantization state; it has not established a successful DPO step.

## Validation and remaining work

After imports pass, validate native LM construction, complete weight/quantizer restoration, and
worker behavior. Run one DPO step and separate-process reload before the full 104-pair, 13-step
pilot. Then reload, export NVFP4, and run the shared 55-subtask evaluation. These are remaining
validation gates, not completed fixes.

The initial metadata read and the Lens import repair passed. The Bridge API adaptation remains a
diagnostic retry in this documented snapshot; no DPO step has passed. A successful metadata read
does not establish full quantizer/weight loading.

Keep any architecture, worker, or quantization-state fixes explicit. Newer NeMo RL alone does not
establish compatibility with every version of Bridge, Core, TE, and ModelOpt.

See [path.yaml](path.yaml), the [common configuration](../../dpo.yaml), and
[compatibility findings](../../compatibility/README.md).
