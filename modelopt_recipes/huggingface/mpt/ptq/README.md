# MPT PTQ recipes

| Recipe | What's model-specific |
|--------|-----------------------|
| `w4a8_awq-kv_fp8_cast.yaml` | Uses `awq_lite` with `alpha_step: 1` instead of the default AWQ search. The default search overflows in TRT-LLM kernels on MPT; the coarser sweep avoids it. Numerics: INT4 block weights + FP8 inputs + FP8 KV-cache cast (constant amax, no KV calibration). Same algorithm override applied to Gemma — see `huggingface/gemma/ptq/`. |

The base numerics units and the standard disabled-quantizer list are inherited
from the shared `configs/`; only the AWQ algorithm fields are model-specific.
