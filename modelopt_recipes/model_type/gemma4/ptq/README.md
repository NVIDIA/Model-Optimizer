# Gemma 4 PTQ recipes

Recipes for the **`gemma4`** model type (multimodal, e.g.
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)). This is
a distinct architecture from the text-only `gemma` model type — see
[`../../gemma/ptq/`](../../gemma/ptq/) for that one. These recipes override the
algorithm defaults that ship in the general PTQ presets because Gemma needs
different settings to converge / stay accurate.

| Recipe | What's model-specific |
|--------|-----------------------|
| `w4a8_awq-kv_fp8_cast.yaml` | Uses `awq_lite` with `alpha_step: 1` instead of the default AWQ search (the default search overflows in TRT-LLM kernels on Gemma; the coarser sweep avoids it without measurably hurting accuracy). Numerics: INT4 block weights + FP8 inputs + FP8 KV-cache cast (constant amax, no KV calibration). |

The base numerics units and the standard disabled-quantizer list are inherited
from the shared `configs/`; only the algorithm fields are model-specific. The
multimodal vision branch (`*vision_tower*` / `*visual*` / `*embed_vision*`) is
kept in BF16 by the shared `default_disabled_quantizers` unit — quantizing it to
INT4 crashes export (`pack_int4_in_uint8` index-out-of-bounds, NVBug 6294017) and
is accuracy-harmful.
