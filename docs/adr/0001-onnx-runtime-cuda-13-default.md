# Use ONNX Runtime 1.30 with CUDA 13 on validated Linux x86-64 hosts

Status: Accepted (2026-10-14)

ONNX Runtime 1.30.x GPU wheels on PyPI use CUDA 13. ModelOpt's ONNX extra previously selected CUDA 12 CuPy and an older ONNX Runtime GPU wheel, so a Python 3.11+ Linux x86-64 install needed manual package replacement to use the CUDA 13 host stack. The ONNX PTQ modelopt container already uses a CUDA 13 PyTorch image.

For Python 3.11+ on Linux x86-64, the ONNX extra selects `cupy-cuda13x` and allows `onnxruntime-gpu>=1.29,<1.31`; the newest stable PyPI release in that range, currently 1.30.0, is the default. The lower bound permits the separately documented CUDA 12.8 path to use the stable 1.29.x wheel from Microsoft's CUDA 12 feed while no stable 1.30.x Python wheel is published there. CUDA 12 users must select `cupy-cuda12x` as well. Python 3.10 keeps the existing ONNX Runtime 1.22.x and CUDA 12 dependencies.

Other platform selectors remain at their existing versions until platform-specific tests verify ONNX Runtime 1.30 and the relevant execution provider. In particular, the native Windows ARM64 TensorRT-RTX plugin must be exercised with the new host wheel before its pin changes. These dependencies select the quantization host environment; exported ONNX models do not require CUDA 13 merely because they were produced by this environment.

Validation for the Linux x86-64 change covers dependency resolution, ONNX unit tests, representative INT8/FP8/INT4 quantization, and an ONNX Runtime session that executes through `CUDAExecutionProvider` with CPU fallback disabled.
