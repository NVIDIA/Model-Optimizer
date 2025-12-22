======================
Installation for Linux
======================

System requirements
===================

Latest Model Optimizer (``nvidia-modelopt``) currently has the following system requirements:

+-------------------------+-----------------------------+
| OS                      |  Linux                      |
+-------------------------+-----------------------------+
| Architecture            |  x86_64, aarch64 (SBSA)     |
+-------------------------+-----------------------------+
| Python                  |  >=3.10,<3.13               |
+-------------------------+-----------------------------+
| CUDA                    |  >=12.0                     |
+-------------------------+-----------------------------+
| PyTorch                 |  >=2.6                      |
+-------------------------+-----------------------------+
| TensorRT-LLM (Optional) |  1.2.0rc4                   |
+-------------------------+-----------------------------+
| ONNX Runtime (Optional) |  1.22                       |
+-------------------------+-----------------------------+
| TensorRT (Optional)     |  >=10.0                     |
+-------------------------+-----------------------------+

Environment setup
=================

.. tab:: Docker image (Recommended)

    To use Model Optimizer with full dependencies (e.g. TensorRT/TensorRT-LLM deployment), we recommend using the
    `TensorRT-LLM docker image <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags>`_,
    e.g., ``nvcr.io/nvidia/tensorrt-llm/release:<version>``.

    Make sure to upgrade Model Optimizer to the latest version using ``pip`` as described in the next section.

    You would also need to setup appropriate environment variables for the TensorRT binaries as follows:

    .. code-block:: shell

        export PIP_CONSTRAINT=""
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/include:/usr/lib/x86_64-linux-gnu"

    You may need to install additional dependencies from the respective examples's `requirements.txt` file.

    **Alternative NVIDIA docker images**

    For PyTorch, you can also use `NVIDIA NGC PyTorch container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags>`_
    and for NVIDIA NeMo framework, you can use the `NeMo container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags>`_.
    Both of these containers come with Model Optimizer pre-installed. Make sure to update the Model Optimizer to the latest version if not already.

    For ONNX / TensorRT use cases, you can also use the `TensorRT container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags>`_
    which provides superior performance to the PyTorch container.

.. tab:: Local environment (PIP / Conda)

    **Setting up a virtual environment**

    We recommend setting up a virtual environment if you don't have one already. Run the following
    command to set up and activate a ``conda`` virtual environment named ``modelopt`` with Python 3.12:

    .. code-block:: shell

        conda create -n modelopt python=3.12 pip
        conda activate modelopt

    (Optional) **Install desired PyTorch version**

    By default, the latest PyTorch version available on ``pip`` will
    be installed. If you want to install a specific PyTorch version for a specific CUDA version, please first
    follow the instructions to `install your desired PyTorch version <https://pytorch.org/get-started/locally/>`_.

    (Optional) **Install other NVIDIA dependencies**

    If you wish to use ModelOpt in conjunction with other NVIDIA libraries (e.g. TensorRT, TensorRT-LLM, NeMo, Triton, etc.),
    please make sure to check the ease of installation of these libraries in a local environment. If you face any
    issues, we recommend using a docker image for a seamless experience. You may still choose to use other ModelOpt's
    features locally for example, quantizing a HuggingFace model and then use a docker image for deployment.

Install Model Optimizer
=======================

ModelOpt including its dependencies can be installed via ``pip``. Please review the license terms of ModelOpt and any
dependencies before use.

If you build and use ModelOpt's docker image, you can skip this step as the image already contains ModelOpt and all
optional dependencies pre-installed.
If you use other suggested docker images, ModelOpt is pre-installed with some of the below optional dependencies.
Make sure to upgrade to the latest version of ModelOpt (with appropriate optional dependencies you need) using pip as shown below.

.. code-block:: bash

    pip install -U "nvidia-modelopt[all]"

If you want to install only partial dependencies, please replace ``[all]`` with the desired
optional dependencies as described below.

**Identify correct partial dependencies**

Note that when installing ``nvidia-modelopt`` without any optional dependencies, only the ``modelopt.torch`` package
requirements are installed and other modules may not work without the appropriate optional
dependencies or ``[all]`` optional dependencies. Below is a list of optional dependencies that
need to be installed to correctly use the corresponding modules:

.. list-table::
    :widths: 30 30
    :header-rows: 1

    *   - Module
        - Optional dependencies
    *   - ``modelopt.onnx``
        - ``[onnx]``
    *   - ``modelopt.torch._deploy``
        - ``[onnx]``

Additionally, we support installing dependencies for following 3rd-party packages:

.. list-table::
    :widths: 30 30
    :header-rows: 1

    *   - Third-party package
        - Optional dependencies
    *   - Huggingface (``transformers``, ``diffusers``, etc.)
        - ``[hf]``

**Accelerated Quantization with Triton Kernels**

ModelOpt includes optimized quantization kernels implemented with Triton language that accelerate quantization
operations by approximately 40% compared to the default implementation. These kernels are particularly
beneficial for :doc:`AWQ <../guides/_choosing_quant_methods>` and Quantization-aware Training (QAT) workflows.

The Triton-based kernels currently support the NVFP4 quantization format, with support for additional
formats coming in future releases. To use these accelerated kernels, you need:

* CUDA device with compute capability >= 8.9 (e.g. RTX 40 series, RTX 6000, NVIDIA L40 or later)
* Triton package installed: ``pip install triton``

No additional configuration is required - the optimized kernels are used automatically when available
for your hardware and quantization format.

Check installation
==================

.. tip::

    When you use ModelOpt's PyTorch quantization APIs for the first time, it will compile the fast quantization kernels
    using your installed torch and CUDA if available.
    This may take a few minutes but subsequent quantization calls will be much faster.
    To invoke the compilation and check if it is successful or pre-compile for docker builds, run the following command:

    .. code-block:: bash

        python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"
