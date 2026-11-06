# Validation and limits

[Return to the API guide](README.md).

## Current correctness checks

| Area | Regression coverage |
|---|---|
| Core local repair | Independent update reference, valid-token accounting, transactional refusal, scale-only/joint optimizer resume |
| Native formats | IQ decoder levels and scale gradients, NF4 CUDA rounding, grouped FP8/NVFP4 versus scalar paths, frozen codes and exact reload |
| Transformers packing | Independent-document equivalence, recurrent resets, MRoPE, sliding windows, custom attention masks and caller cache rejection |
| Distributed repair | Unique data lanes, TP/EP/FSDP/HSDP ownership, unused expert owners, CP forward/backward, physical PP and no-partial-mutation refusal |
| Model checkpoints | Rank-owned physical fields, tied quantizers, PP key unions, relocation, corruption and incomplete-publication refusal |
| Text extensions | Independent architecture math, native BF16 reload values, callable FSDP owners and cache transforms versus fresh causal outputs |
| Diagnostics/data | Full-vocabulary KL, within-document targets, unique corpus split/cursor and packed-document isolation |

Relevant tests are under
[`tests/unit/torch/distill`](../../tests/unit/torch/distill),
[`tests/unit/torch/models`](../../tests/unit/torch/models),
[`tests/gpu/torch/distill/plugins`](../../tests/gpu/torch/distill/plugins) and
[`test_repair_formats.py`](../../tests/gpu/torch/quantization/test_repair_formats.py).

Latest focused hardening validation passed **222 distinct CPU cases** and
**40 GPU cases**: 10 native distributed, 2 PP/CP+PP repair, 3 DCP, 9 codec and 16 selected family
cases. The final core/adapter rerun passed 55 cases; the two native NVFP4 routes
were also rerun against the finalized output-head path. The expanded tests cover
biased, quantized and hooked heads, Muse/Mistral post-logit math, selected versus
skipped heads, collective diagnostic refusal, padding/MRoPE, native dtype and
source checkpoint reconstruction. Ruff/format passes across all 93 Python files
in the complete feature diff. The README flow executed on a tiny CPU model.

The distributed checks use four GB300 ranks with pinned HF 5.23/Torch 2.14 and
cover TP/EP/FSDP/CP, HSDP, physical PP/DCP and functional FSDP replay/backward.
The full 58-case release matrix below was not rerun for this increment. The
>2^31-element codec case remains manual and was not run. Ordinary single-GPU
pytest skips cases requiring an explicit four-rank launch.

BF16 ring versus serial NVFP4 attention retains its 0.3% aggregate weight bound;
exceeding it requires every element to be within an adjacent BF16 value. Other
elementwise, code/field, history and ownership checks remain. A worst 0.308%
weight-relative-L2 difference previously met that stricter rounding condition.

For an existing compatible four-GPU allocation, the focused distributed suites
are `test_transformers_native_parallel.py` and `test_transformers_checkpoint.py`
under `tests/gpu/torch/distill/plugins`; launch through `torch.distributed.run`
with four processes using the cluster's approved container/environment settings.
The validation workspace retains exact commands, source hashes and JSON reports
under `runs/bypass-hardening-20261106`. No model checkpoint is retained by the
profiling runs. These checks make no full pretrained-model quality claim.

## Completed release matrix

The November 6 development ledger is
`runs/transformers-aa-compatibility-20261106/COMPATIBILITY.md` with per-case JSON,
source review and numerical diagnostics in that validation workspace. The raw
operational artifacts are not bundled in the package.

All 58 model/format cases completed on pinned Transformers 5.23/Torch 2.14 using
native BF16 precision policies, max PTQ, small random two-layer text networks
and three scale-only updates. Combined eight-GPU placement uses dense FSDP4,
CP2 and EP2/expert FSDP2 for routed models. All cases use the reported
`replicated_full_context` CP fallback. These runs establish short numerical and
ownership compatibility, not pretrained quality or long-context efficiency.

Muse Spark NVFP4 is an explicit numerical exception: 1.20% worst relative
scalar-scale variation, accepted within 2.5%, versus 0.3% for other scalar and
all vector relative-L2 bounds. Codes, block scales, ownership and history checks
remain enforced. The FP32 eager oracle is close, but differs from BF16 grouped
execution in both dtype and expert implementation. Do not describe all 58 as
passing the original uniform 0.3% bound.

## Interpreting measurements

Repair minimizes local hidden-output MSE. Held-out whole-model
KL(teacher || candidate), over the full vocabulary, selects PTQ initializers and
assesses fidelity; PPL is secondary on a surrogate corpus. Neither tiny-model
MSE nor historical prefix speed establishes downstream benchmark quality.

Report actual execution modes, unique token counts, source/runtime revisions,
physical layouts, synchronized timing and PyTorch allocated/reserved memory.
Rank-local timers do not measure scheduler-allocated GPU-hours. DCP restore needs
a matching topology; a native checkpoint round trip does not validate deployment
conversion or realistic full-model peak memory.
