# Gemma PTQ recipes

Recipes here override the algorithm defaults that ship in the general PTQ
presets because Gemma needs different settings to converge / stay accurate.

| Recipe | What's model-specific |
|--------|-----------------------|
| `w4a8_awq-kv_fp8_cast.yaml` | Uses `awq_lite` with `alpha_step: 1` instead of the default AWQ search. The default search overflows in TRT-LLM kernels on Gemma; the coarser sweep avoids it without measurably hurting accuracy. Numerics: INT4 block weights + FP8 inputs + FP8 KV-cache cast (constant amax, no KV calibration). |
| `int8_sq-kv_fp8_cast.yaml` | Sets SmoothQuant `alpha: 0.5` instead of the default `1.0`. Gemma 7B regresses with `alpha=1`; `0.5` recovers it. Numerics: INT8 per-channel weights + INT8 inputs + FP8 KV-cache cast. |

The base numerics units and the standard disabled-quantizer list are inherited
from the shared `configs/`; only the algorithm fields are model-specific.
