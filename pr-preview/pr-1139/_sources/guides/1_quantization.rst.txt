Quantization
############

ModelOpt quantization toolkit supports quantization for NVIDIA's hardware and software stack.
Currently ModelOpt supports quantization in PyTorch and ONNX frameworks.

ModelOpt is based on simulated quantization in the original precision to simulate, test, and
optimize for the best trade-off between the accuracy of the model and different low-precision
formats. To achieve actual speedups and memory savings, the model with simulated quantization can be
exported to deployment frameworks, like TensorRT or TensorRT-LLM. Please refer to the
`Model-Optimizer GitHub repository <https://github.com/NVIDIA/Model-Optimizer>`_
for more details and examples.

Below, you can find the documentation for the quantization toolkit in ModelOpt:

.. toctree::
    :maxdepth: 1

    ./_basic_quantization.rst
    ./_choosing_quant_methods.rst
    ./_pytorch_quantization.rst
    ./_customized_model_quantization.rst
    ./_compress_quantized_models.rst
    ./_onnx_quantization.rst
    ./windows_guides/_ONNX_PTQ_guide.rst
