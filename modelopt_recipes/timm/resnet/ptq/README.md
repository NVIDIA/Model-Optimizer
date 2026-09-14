# ResNet PTQ recipes

These recipes use the shared FP8 and INT8 numerics and standard disabled-quantizer units. Their
ResNet-specific change is the `*residual_quantizer` entry, which quantizes the shortcut immediately
before each residual addition.

The residual quantizer modules are inserted by
`examples/torch_onnx/torch_quant_to_onnx.py`; these recipes require that integration and do not add
the modules themselves.

| Recipe | ResNet-specific behavior |
|--------|--------------------------|
| `fp8.yaml` | Enables per-tensor FP8 shortcut quantization. |
| `int8.yaml` | Enables per-tensor INT8 shortcut quantization. |

Only FP8 and INT8 are supported for convolutional architectures because TensorRT has limited
convolution kernel support.
