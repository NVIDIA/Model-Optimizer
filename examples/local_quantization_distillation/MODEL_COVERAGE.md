# Native model compatibility

This is a compatibility check with **small random native models**, not a pretrained
quality result. The original configs were retrieved on November 3, 2026; the additional native
families below were checked on November 5, 2026. We ranked open-weight
models by the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models),
excluded deprecated entries, and counted reasoning-effort variants of one
checkpoint only once. Architecture support refers to the native AutoModel registry
in the checkout used for validation, not to possible Transformers fallback support.

All **133 cases passed** across jobs 1569709 and 1569798: 32 native model/config
prefixes across four published formats, four GDN W4A4 checks, and the selective
Kimi first-MoE check.
Each supported checkpoint runs its real native decoder implementations, the
tutorial's layerwise GPTQ helper, two scale-only Bypass updates, and an
exact-logit ModelOpt save/reload. Only language MLP/routed and shared expert
weights are quantized in this matrix; routers, attention and vision towers
remain unchanged. A separate native GDN W4A4 check covers activation calibration,
participation of every weight scale in backward, unchanged activation ranges,
and exact reload.

## Top 15 distinct checkpoints

| Rank | Checkpoint | Native implementation | Retained prefix |
|---|---|---|---:|
| 1 | [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `DeepseekV41ForCausalLM` | 25 |
| 2 | [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | `GlmMoeDsaForCausalLM` | 7 |
| 3 | [MiMo-V2.6-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | `MiMoV2ForCausalLM` | 8 |
| 4 | [Ling-3.1-Flash-1019](https://huggingface.co/inclusionAI/Ling-3.1-flash-1019) | Not registered (`BailingMoeV3ForCausalLM`) | — |
| 5 | [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) | `MiniMaxM3SparseForConditionalGeneration` | 4 |
| 6 | [Step 5 Preview](https://huggingface.co/stepfun-ai/Step-5-Preview) | Not registered (`Step5ForConditionalGeneration`) | — |
| 7 | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `Glm5NextForConditionalGeneration` | 5 |
| 8 | [MiMo-V2.6-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | `MiMoV2ForCausalLM` | 6 |
| 9 | [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) | `MistralLarge4ForCausalLM` (text decoder) | 7 representative blocks |
| 10 | Muse Spark Open | `MuseSpark1xOmniForCausalLM`; public native default config (Hub config gated) | 4 |
| 11 | [DeepSeek V4 Pro 0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) | `DeepseekV4ForCausalLM` | 5 |
| 12 | [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) | `LlamaForCausalLM` | 1 |
| 13 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `Qwen3_5ForConditionalGeneration` | 4 |
| 14 | [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | `MiniMaxM3SparseForConditionalGeneration` | 4 |
| 15 | [K2-Horizon-375B-A23B](https://huggingface.co/IFM/K2-Horizon-375B-A23B) | Not registered (`K2HorizonForCausalLM`) | — |

Prefixes preserve the first occurrence of each distinct joint layer pattern,
including dense versus MoE blocks, linear versus full attention, routing and
compression changes, and shared KV/index ownership. DeepSeek V4.1 needs 25
layers to retain Engram, candidate reuse and index-only reindexing; GLM-5.3
needs seven to include a full indexer on an MoE layer. Widths, vocabulary,
expert counts, vision depth and embedding tables are reduced. Global layer
identities, discriminating config arrays, native hash functions and native
forward kernels are retained. No checkpoint weights are downloaded.

## Qwen 3.5 and newer

The official Qwen Hub catalog was checked for 3.5, 3.6, 3.7, 3.8 and 4 releases.
The fixture covers all 18 identified unquantized generative checkpoints; SAE
artifacts and quantized repackages are outside this inventory. Every listed
checkpoint retains four language layers: three GDN blocks and one full
attention block. Flash-Next also retains its PLE block and QSA implementation.

| Release | Checkpoints | Native family |
|---|---|---|
| 3.5 dense | 0.8B, 2B, 4B, 9B, 27B; Base 0.8B, 2B, 4B, 9B | `qwen3_5` |
| 3.5 MoE | 35B-A3B, 122B-A10B, 397B-A17B; Base 35B-A3B | `qwen3_5_moe` |
| 3.6 | 27B; 35B-A3B | `qwen3_5`; `qwen3_5_moe` |
| 3.8 | 27B; 2.4T-A95B; Flash-Next | `qwen3_5`; `qwen3_5_moe`; `qwen3_8_flash_next` |

Flash-Next's published `Qwen4ExpForConditionalGeneration` is an AutoModel
registry alias of the Flash-Next implementation. Its CUDA attention requires
FlexAttention or Cute; the probe uses real FlexAttention, compact document
boundaries and microbatch size one. This AutoModel implementation is language-only
and declares no TP or PP support. Its model-owned PLE table uses the native
EP parallelizer for distributed execution; a generic EP=1 FSDP bootstrap does
not initialize that table. Bypass does not broaden those native limits.
The other checks use native Torch attention/expert backends and real FLA/causal
convolution kernels where required. Most Torch helper wrappers run eagerly to
avoid compiling duplicates; this is not a compiler performance measurement.

## Reproduce

The offline fixture in `tests/_test_utils/data/automodel_bypass_models.json`
contains full public configs, resolved Hub revisions, source URLs, SHA-256
digests, architecture matches and the retained-pattern rationale. Unsupported
architectures have no runnable prefix. The native builder lives in
`tests/_test_utils/automodel_families.py`; no remote model code is substituted.

With ModelOpt, matching AutoModel and its native kernels installed on a GPU:

```bash
pytest tests/gpu/torch/distill/plugins/test_automodel_families.py -q
```

These independent probes run separately from the multi-rank ownership suite
`test_automodel_local_distillation.py`, which checks native EP, CP, TP/FSDP2,
physical PP, diagnostic token accounting, and checkpoint/resume. Passing one
layout does not establish every family/format/topology combination. Deployment
exporters and inference runtimes retain their own format/architecture limits.

Original validated AutoModel revision: `bf6cae1d32d3a34a88aa6352f62f0cf1a9bb156a`.
The additional families require the newer native implementation described below.

## Published-format coverage

Each native prefix is checked with IQ2_S, IQ1_M, IQ2_XS and NVFP4 W4A4.
The first three retain the source activation precision; NVFP4 keeps activation
quantization enabled. Optional external format providers maintain their own tests
and release policy outside the ModelOpt checkout.

Additional revision-pinned prefixes cover DeepSeek-V4-Flash and Kimi-K3's
language model. Kimi retains its three KDA blocks, one MLA block, latent
expert projections and SiTU. DeepSeek's hash-gate checks compare exact teacher
and replay expert IDs, including temporary token-ID routing state.

Every registered family uses the shared native fast runner by default,
including Qwen and Kimi. It preserves native routing, shared experts and
dispatch while batching compatible IQ expert scale fields and using native
GPU level decoding. Mixed representations retain each quantizer's ordinary
path. Family-specific routing or packed-input state belongs in that family's
descriptor; new families need no copy of a shared router implementation.
This default does not establish a measured speedup for every model or format.

## Additional native families

Muse Spark, Mistral Large 4 and NVIDIA Nemotron 3.5 Super VL are checked against
AutoModel revision `752a71ebf731f84b19ca9af5e7c242be003082bd`. Each passes
IQ2_S, IQ1_M, IQ2_XS and NVFP4 W4A4 GPTQ → scale-only repair → exact native
reload, including packed document boundaries and padding. Muse and Nemotron
also receive real image tensors through their native vision encoders in these
small random-model probes. Default repair remains within the text decoder;
vision encoders/projectors and audio towers are not selected or tuned.

| Native family | Default units | Distinct state exercised |
|---|---|---|
| `muse_spark` | `feed_forward` MoE | Compressed routed experts, native pre/post expert normalization, attention-window schedule |
| `mistral_large4` | Dense MLP / routed MoE | RoPE/NoPE, full/shared indexer, non-adjacent shared K/V sources |
| `nemotron_omni` | Hybrid text MLP / MoE | Eight-block Mamba/MoE/attention prefix, native RADIO image path |

Mistral's native implementation at this revision exposes a text-only forward;
this check does not establish image support for that implementation. Its layerwise
PTQ replays the real prefix to regenerate cross-layer K/V and index state,
keeping cached calibration inputs limited to the current block. This costs extra
PTQ forwards. Resume restores the calibrated prefix before regenerating the
boundary's native layout and K/V state. It requires
`get_qdq_activations_from_prev_layer=True`; layerwise export is refused, so export
after calibration instead. Bypass retains the shared
fast MoE runner and does not cache the full corpus of teacher hidden states.

The Nemotron wrapper loads the checkpoint's own pinned RADIO Python/config code.
For its offline compatibility probe, prefetch only those small files (no weights):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    revision="fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe",
    allow_patterns=["*.py", "config.json"],
)
```

Muse uses AutoModel's public default config because the Hub config is gated;
no access restriction is bypassed. The family matrix checks one-GPU native
execution; the separate ownership tests exercise EP, CP, TP/FSDP2 and physical PP
on existing representative families. Native topology limits still apply, and
these results do not claim every new-family/topology combination was tested.
