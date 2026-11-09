# Transformers text-model coverage

[Return to the API guide](README.md).

Coverage means small random text models executing actual model forwards,
quantization and repair. It does not imply pretrained quality, uniform raw
checkpoint import, multimodal training or universal format/topology support.
The broad baseline is Transformers 5.23.0.dev0 at
[`084b1fd1b9983dca2ce12441aec4478371b9fe8f`](https://github.com/huggingface/transformers/tree/084b1fd1b9983dca2ce12441aec4478371b9fe8f).

## Family adapters

The generic adapter discovers HF text roots; small family descriptors identify
semantic boundaries, packing and side state. The offline fixture includes Llama,
Qwen3/dense/MoE/hybrid/VL/Omni, Qwen 3.5+, DeepSeek V3/V3.2/V4, GLM DSA/Next,
Kimi K2.5/Linear/K3, MiniMax M3, MiMo, Nemotron H/Omni, Mistral Large 4 and
Muse Spark. Kimi K3 uses pinned official remote modeling code and compatible
optional FLA dependencies. VL/Omni entries exercise the text path only.

[Family tests](../../tests/gpu/torch/distill/plugins/test_transformers_families.py)
check real semantic units, packed documents, gradients, held-out MSE and frozen
integer codes. Broad descriptors can work for additional models, but discovery
alone is not format or distributed acceptance.

## Differentiable text extensions

| Model | Package | Important contract |
|---|---|---|
| DeepSeek V4.1 | `modelopt.torch.models.deepseek_v41` | Compressed/shared sparse attention, Engram, mHC, router and pre-down routing weights |
| Kimi K3.1 | `modelopt.torch.models.kimi_k31` | Bounded KDA, NoPE sparse MLA, latent SiTU experts and attention residuals |
| Step 5 | `modelopt.torch.models.step5` | CSA/SWA, zero-centered norms, shared/routed experts and clamped SwiGLU |
| MiMo V2.6 | `modelopt.torch.models.mimo_v2` | Thin upstream bridge preserving release KV geometry and FP32 expert accumulation |

Explicitly import an extension package before using its registered config/model
classes. These ports include the complete text decoder and omit vision/audio,
MTP and draft execution. Native dtype-aware loading preserves required FP32
parameters. A blanket `.to(bfloat16)` does not preserve that contract.

Checkpoint import contracts are explicit:

| Port | Accepted source conversion | Remaining limit |
|---|---|---|
| V4.1 | Published FP4/FP8 reconstruction with UE8M0 scales and expert fusion | Bounded synthetic acceptance, not a complete pretrained import |
| Kimi K3.1 | Reconstructed floating source names, padded A_log and split latent experts | Raw compressed-tensors MXFP4 remains refused |
| Step 5 | Floating/BF16 text tensors, split QKV and stacked experts | Raw FP8 requires source activation rounding as well as weight decoding |
| MiMo V2.6 | Reconstructed floating fused QKV, sink names and numbered experts | Raw MXFP4/FP8 tensors are refused; quantized-kernel parity is unverified |

Loaders validate dtypes, aliases and shapes before loading. Source packed-format
metadata is retained as `checkpoint_quantization_config`, preventing native
floating reload from reactivating the original HF quantizer. MiMo additionally
preserves its FP32 routing-correction bias through native BF16 construction and
reload; tests compare imported attention to source equations with nonzero
partial RoPE and the release's asymmetric Q/K/V geometry. Cached release headers
and code establish these layouts, not pretrained tensor values or quality.

Prepare V4.1 through `adapter.parallelize`; its model-owned EP dispatcher keeps
routing weights before the down projection and preserves the FP32 routed sum.
The generic dispatcher has different rounding semantics and is rejected.
V4.1 uses the faithful `full_context_teacher_local_unit_replay` CP fallback:
context ranks retain full model inputs and reduce disjoint-token objectives.
It does not shard sparse/recurrent attention or save its activation memory.
V4.1 has no validated native PP plan; shared compressed/index/candidate state
requires explicit stage transport before that combination can be supported.

The standalone `DeepseekV41MoE` accepts captured `[batch, sequence, hidden]`
inputs and explicit component checkpoint extraction. It is useful for local MSE
studies; it has no vocabulary logits and does not establish whole-model KLD.
Its ModuleList expert layout does not supply automatic fused-expert EP conversion.

## Release compatibility evidence

The November 6, 2026 matrix completed **58 model/format cases**: 29 reduced
release/config models, each with IQ2_XXS and NVFP4 W4A4. It includes 18 Qwen
3.5/3.6/3.8 releases and DeepSeek V4.1, Step 5, Kimi K3.1, GLM-5.3/Flash,
Ling-3.1, MiniMax-M3.1, Mistral Large 4, Muse Spark and MiMo V2.6 Pro/Flash.
Muse uses the native default config because the release config is gated.

The fixture retains representative architecture-specific blocks while reducing
width, vocabulary and expert count. Three updates use four unique 128-token
sequences each. Native combined placement uses eight GPUs with CP2, dense FSDP4,
and EP2/expert FSDP2 where experts exist. **CP is replicated full context** in
this matrix; no CP compute or activation-memory saving is established.

Fifty-seven cases meet the original 0.3% relative-L2 field bound. Muse NVFP4
uses an explicit 2.5% scalar bound; its worst BF16 scalar variation is 1.20%,
with identical codes/block scales. The FP32 eager oracle agrees within 0.00021%
on these scalars, but BF16 uses grouped experts, so this is not a dtype-only
comparison. [Validation](VALIDATION.md) and [Results](RESULTS.md) describe limits.
