.. _Install-Page-Standalone-Windows:

================================================
Install ModelOpt-Windows as a Standalone Toolkit
================================================

The Model Optimizer - Windows (ModelOpt-Windows) can be installed as a standalone toolkit for quantizing ONNX models. Below are the setup steps:

**1. Setup Prerequisites**

Before using ModelOpt-Windows, the following components must be installed:

      - NVIDIA GPU and Graphics Driver
      - Python version >= 3.10 and < 3.13
      - Visual Studio 2022 / MSVC / C/C++ Build Tools
      - CUDA Toolkit and matching CuDNN for using CUDA path during calibration (e.g. for calibration of ONNX models using `onnxruntime-gpu` or CUDA EP)

Update ``PATH`` environment variable as needed for above prerequisites.

**2. Setup Virtual Environment (Optional but Recommended)**

It is recommended to use a virtual environment for managing Python dependencies. Tools such as *conda* or Python's built-in *venv* module can help create and activate a virtual environment. Example steps for using Python's *venv* module:

.. code-block:: shell

      $ mkdir myEnv
      $ python -m venv .\myEnv
      $ .\myEnv\Scripts\activate

In the newly created virtual environment, none of the required packages (e.g., onnx, onnxruntime, onnxruntime-directml, onnxruntime-gpu, nvidia-modelopt etc.) will be pre-installed.

**3.  Install ModelOpt-Windows Wheel**

To install the ONNX module of ModelOpt-Windows, run the following command:

.. code-block:: bash

    pip install "nvidia-modelopt[onnx]"

If you install ModelOpt-Windows without the extra ``[onnx]`` option, only the minimal core dependencies and the PyTorch module (``torch``) will be installed. Support for ONNX model quantization requires installing with ``[onnx]``.

**4. ONNX Model Quantization: Setup ONNX Runtime Execution Provider for Calibration**

The Post-Training Quantization (PTQ) process for ONNX models usually involves running the base model with user-supplied inputs, a process called calibration. The user-supplied model inputs are referred to as calibration data. To perform calibration, the base model must be run using a suitable ONNX Execution Provider (EP), such as *DmlExecutionProvider* (DirectML EP) or *CUDAExecutionProvider* (CUDA EP). There are different ONNX Runtime packages for each EP:

- *onnxruntime-directml* provides the DirectML EP.
- *onnxruntime-trt-rtx* provides TensorRT-RTX EP.
- *onnxruntime-gpu* provides the CUDA EP.
- *onnxruntime* provides the CPU EP.

By default, ModelOpt-Windows installs *onnxruntime-gpu*. The default CUDA version needed for *onnxruntime-gpu* since v1.19.0 is 12.x. The *onnxruntime-gpu* package (i.e. CUDA EP) has CUDA and cuDNN dependencies:

- Install CUDA and cuDNN:
    - For the ONNX Runtime GPU package, you need to install the appropriate version of CUDA and cuDNN. Refer to the `CUDA Execution Provider requirements <https://onnxruntime.ai/docs/install/#cuda-and-cudnn/>`_ for compatible versions of CUDA and cuDNN.

If you need to use any other EP for calibration, you can uninstall the existing *onnxruntime-gpu* package and install the corresponding package. For example, to use the DirectML EP, you can uninstall the existing *onnxruntime-gpu* package and install the *onnxruntime-directml* package:

  .. code-block:: bash

      pip uninstall onnxruntime-gpu
      pip install onnxruntime-directml

**5. Setup GPU Acceleration Tool for Quantization**

ModelOpt uses `CuPy <https://docs.cupy.dev/en/stable/install.html>`_ to accelerate
INT4 ONNX quantization. The ``nvidia-modelopt[onnx]`` extra installs ``cupy-cuda12x`` and
a CUDA 12-compatible *onnxruntime-gpu* version by default.

**CUDA 13.x Setup**

The steps below assume a CUDA 13.x Toolkit, compatible cudnn, and a compatible driver are already installed on the host.

Replace the CUDA-dependent packages installed by the default ONNX extra:

.. code-block:: bat

    python -m pip uninstall -y cupy-cuda12x onnxruntime-gpu
    python -m pip install cupy-cuda13x "onnxruntime-gpu>=1.27"

ONNX Runtime 1.27 and later GPU packages published on PyPI use CUDA 13.x by default.
Refer to the `ONNX Runtime CUDA Execution Provider requirements
<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_
before selecting or pinning an ONNX Runtime version.

.. note::

    Make sure a cuDNN 9 build for CUDA 13 is available on the host (for example, via the
    NVIDIA Windows installer/zip package, or the ``nvidia-cudnn-cu13`` Python wheel), and
    that the relevant environment variables like ``CUDA_PATH``, ``CUDA_HOME``, and ``PATH`` point to the CUDA 13.x installation).

**6. Verify Installation**

Ensure the following steps are verified:
      - **Task Manager**: Check that the GPU appears in the Task Manager, indicating that the graphics driver is installed and functioning.
      - **Python Interpreter**: Open the command line and type python. The Python interpreter should start, displaying the Python version.
      - **Onnxruntime Package**: Ensure that exactly one of the following is installed:
            - *onnxruntime-directml* (DirectML EP)
            - *onnxruntime-trt-rtx* (TensorRT-RTX EP)
            - *onnxruntime-gpu* (CUDA EP)
            - *onnxruntime* (CPU EP)
      - **CUDA Toolkit**: For CUDA workflows, verify that the selected Toolkit is found first and that ``nvcc`` reports the expected major version:

            .. code-block:: bat

                where nvcc
                nvcc --version

      - **ONNX and ONNX Runtime**: Ensure that imports succeed and that CUDA EP is available for CUDA workflows:

            .. code-block:: python

                python -c "import onnx; import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

      - **CuPy**: Verify that CuPy can allocate and execute on the GPU. For CUDA 13.x,
        ``runtimeGetVersion()`` should report a value beginning with ``13``:

            .. code-block:: python

                python -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.runtimeGetVersion(), cupy.arange(3).sum())"

      - **Environment Variables**: For workflows using CUDA dependencies (e.g., CUDA EP-based calibration), ensure environment variables such as ``CUDA_PATH``, ``CUDA_PATH_V12_x``, or ``CUDA_PATH_V13_x`` point to the intended Toolkit. Reopen the command prompt after changing persistent environment variables.
      - **ModelOpt-Windows Import Check**: Run the following command to ensure the installation is successful:

            .. code-block:: python

                python -c "import modelopt.onnx.quantization"

- If you encounter any difficulties during the installation process, please refer :ref:`FAQ_ModelOpt_Windows` FAQs for potential solutions and additional guidance.
