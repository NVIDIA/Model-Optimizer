# Bypass model coverage

The release suite exercises the following 29 reduced text configurations with the same IQ2_XXS recipe, packed documents, two real Bypass updates, immutable code/teacher/router checks, and exact finalized ModelOpt reload. These are architecture/configuration checks with random small weights, not full released-checkpoint quality or throughput claims. Separate independent math/source-loading tests protect the four local ports; the eight-format codec/kernel tests protect format coverage.

| Release/configuration | Model family |
| --- | --- |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `deepseek_v41` |
| `zai-org/GLM-5.3` | `glm_moe_dsa` |
| `XiaomiMiMo/MiMo-V2.6-Pro-RL` | `mimo_v2` |
| `inclusionAI/Ling-3.1-flash-1019` | `bailing_hybrid` |
| `MiniMaxAI/MiniMax-M3.1-Flash` | `minimax_m3_vl` |
| `stepfun-ai/Step-5-Preview` | `step5` |
| `zai-org/GLM-5.3-Flash` | `glm5_next` |
| `XiaomiMiMo/MiMo-V2.6-Flash-RL` | `mimo_v2` |
| `mistralai/Mistral-Large-4-1T-A52B` | `mistral_large4` |
| `Qwen/Qwen3.8-27B` | `qwen3_5` |
| `Qwen/Qwen3.5-9B` | `qwen3_5` |
| `Qwen/Qwen3.5-4B` | `qwen3_5` |
| `Qwen/Qwen3.5-0.8B` | `qwen3_5` |
| `Qwen/Qwen3.5-2B` | `qwen3_5` |
| `Qwen/Qwen3.5-27B` | `qwen3_5` |
| `Qwen/Qwen3.5-122B-A10B` | `qwen3_5_moe` |
| `Qwen/Qwen3.5-9B-Base` | `qwen3_5` |
| `Qwen/Qwen3.5-2B-Base` | `qwen3_5` |
| `Qwen/Qwen3.5-397B-A17B` | `qwen3_5_moe` |
| `Qwen/Qwen3.5-35B-A3B` | `qwen3_5_moe` |
| `Qwen/Qwen3.5-4B-Base` | `qwen3_5` |
| `Qwen/Qwen3.5-0.8B-Base` | `qwen3_5` |
| `Qwen/Qwen3.5-35B-A3B-Base` | `qwen3_5_moe` |
| `Qwen/Qwen3.6-35B-A3B` | `qwen3_5_moe` |
| `Qwen/Qwen3.6-27B` | `qwen3_5` |
| `Qwen/Qwen3.8-Flash-Next` | `qwen4_exp` |
| `Qwen/Qwen3.8-2.4T-A95B` | `qwen3_5_moe_text` |
| `moonshotai/Kimi-K3.1-Pro` | `kimi_k31` |
| `meta-models/Muse-Spark-open-816B-A42B` | `muse_spark` |

Additional stock-model fixtures are available for ordinary Llama/Qwen3, Qwen3-MoE, Qwen3-Next, Qwen3-VL/Omni, linear-attention and recurrent model paths. Focused executed cases include Llama, mixed dense/MoE Qwen3-MoE, Qwen3-Next, Qwen3-VL and Nemotron-H; the PR validation lists the exact runs. Text-only execution keeps the native task/position/mask behavior; it does not qualify vision, audio, or generation training.

## Source and topology qualifications

| Model/area | Contract |
| --- | --- |
| DeepSeek V4.1 | Differentiable text/mHC/compressed-attention/Engram implementation with a published-text importer and independently checked mixed-precision rules. Engram requires its tokenizer map. Vision/draft auxiliary loading is explicit. Model-owned EP preserves routing weights before down projection. |
| Kimi K3.1 | Latent SiTU, bounded KDA, sparse MLA/indexer, and floating checkpoint conversion. FLA is optional for the fast CUDA KDA path; the Torch recurrence remains available. Raw compressed-tensors MXFP4 is refused. |
| Step 5 | Floating/BF16 text checkpoint contract, zero-centered FP32 norms, compressed/SWA attention, signed index scores and clamped experts. Published sibling shared-expert keys map into the MoE owner. Raw FP8 weight decoding alone is not claimed to reproduce source activation semantics. |
| MiMo V2.6 | Thin native MiMo bridge preserving nonzero partial RoPE, asymmetric QKV/KV geometry, attention sinks and FP32 routing accumulation. Floating text weights have explicit source-name conversion. Raw MXFP4/FP8 source tensors are refused. |
| Native ownership | Real four-GPU TP2/FSDP2/EP4 and EP2/replica2 oracles protect mixed dense/MoE IQ with frozen FP8/NVFP4, normalized gradients and clipping, including empty owners. The four local ports have full-model FSDP4/EP4 owner/reference cases. All four local-port owner/reference cases passed in the 75-case GB300 hardware suite, alongside all 29 release cases and the mixed shared-IQ2_S/routed-IQ1_M/frozen-NVFP4 HF roundtrip. |
| CP | Faithful native causal SDPA for declared paths when supported by Torch; otherwise actual replicated full-context execution is reported. |
| PP and export | PP greater than one is refused. Exact owner checkpoints restore on the same topology and original PTQ skeleton. Ordinary finalized ModelOpt save/reload is tested unsharded; consolidated deployment export of native EP owner shards is not claimed. |

Use stable Transformers 5.23.x for this integration. Portable core/math/import compatibility is also checked on supported Torch 2.11 and the base HF 5.5 lane; native GPU validation uses Torch 2.14/CUDA 13.4 on GB300. No full-weight models need downloading to run the reduced suites.
