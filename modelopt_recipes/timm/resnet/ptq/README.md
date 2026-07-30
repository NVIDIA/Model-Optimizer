# timm ResNet PTQ recipes

These recipes support timm ResNet models built from the standard `BasicBlock`
or `Bottleneck` and keep their quantizer placement and numeric choices out of
the torch ONNX example:

- The three-channel stem convolution remains unquantized.
- Every block input is quantized once and shared by the main and identity
  shortcut paths.
- Projection shortcuts are quantized immediately before the residual add.
- INT8 quantizes the final block output before global pooling.
- MXFP8 and NVFP4 recipes use FP8 for convolution and residual inputs, matching
  TensorRT convolution support.

| Recipe | Numerics |
|--------|----------|
| `fp8.yaml` | FP8 convolution and residual inputs; classifier unquantized. |
| `int8.yaml` | INT8 convolution and residual inputs; classifier unquantized. |
| `mxfp8.yaml` | MXFP8 with FP8 convolution and residual inputs. |
| `nvfp4.yaml` | NVFP4 with FP8 convolution and residual inputs. |
| `nvfp4_awq_lite.yaml` | AWQ-lite NVFP4 AutoQuantize candidate with FP8 convolution and residual inputs. |

`static_fp8.quant_cfg.yaml` and `static_int8.quant_cfg.yaml` are shared recipe
snippets, not standalone recipes.
