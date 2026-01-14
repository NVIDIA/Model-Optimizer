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

Update ``PATH`` environment variable as needed for above prerequisites.

**2. Setup Virtual Environment (Optional but Recommended)**

It is recommended to use a virtual environment for managing Python dependencies. Tools such as *conda* or Python's built-in *venv* module can help create and activate a virtual environment. Example steps for using Python's *venv* module:

.. code-block:: shell

      $ mkdir myEnv
      $ python -m venv .\myEnv
      $ .\myEnv\Scripts\activate

In the newly created virtual environment, none of the required packages (e.g., onnx, onnxruntime, onnxruntime-directml, onnxruntime-gpu, nvidia-modelopt) will be pre-installed.

**3.  Install ModelOpt-Windows Wheel**

To install the ModelOpt-Windows wheel, run the following command:

.. code-block:: bash

    pip install "nvidia-modelopt[onnx]"

This command installs ModelOpt-Windows and its ONNX module, along with the *onnxruntime-directml* (v1.20.0) package. If ModelOpt-Windows is installed without the additional parameter, only the bare minimum dependencies will be installed, without the relevant module and dependencies.

**4. Setup ONNX Runtime (ORT) for Calibration**

The ONNX Post-Training Quantization (PTQ) process involves running the base model with user-supplied inputs, a process called calibration. The user-supplied model inputs are referred to as calibration data. To perform calibration, the base model must be run using a suitable ONNX Execution Provider (EP), such as *DmlExecutionProvider* (DirectML EP) or *CUDAExecutionProvider* (CUDA EP). There are different ONNX Runtime packages for each EP:

- *onnxruntime-directml* provides the DirectML EP.
- *onnxruntime-gpu* provides the CUDA EP.
- *onnxruntime* provides the CPU EP.

By default, ModelOpt-Windows installs *onnxruntime-directml* and uses the DirectML EP (v1.20.0) for calibration. No additional dependencies are required.
If you prefer to use the CUDA EP for calibration, uninstall the existing *onnxruntime-directml* package and install the *onnxruntime-gpu* package, which requires CUDA and cuDNN dependencies:

- Uninstall *onnxruntime-directml*:

  .. code-block:: bash

      pip uninstall onnxruntime-directml

- Install CUDA and cuDNN:
    - For the ONNX Runtime GPU package, you need to install the appropriate version of CUDA and cuDNN. Refer to the `CUDA Execution Provider requirements <https://onnxruntime.ai/docs/install/#cuda-and-cudnn/>`_ for compatible versions of CUDA and cuDNN.

- Install ONNX Runtime GPU (CUDA 12.x):

  .. code-block:: bash

      pip install onnxruntime-gpu

  - The default CUDA version for *onnxruntime-gpu* since v1.19.0 is 12.x.

**5. Setup GPU Acceleration Tool for Quantization**

By default, ModelOpt-Windows utilizes the `cupy-cuda12x <https://cupy.dev//>`_ tool for GPU acceleration during the INT4 ONNX quantization process. This is compatible with CUDA 12.x.

**6. Verify Installation**

Ensure the following steps are verified:
      - **Task Manager**: Check that the GPU appears in the Task Manager, indicating that the graphics driver is installed and functioning.
      - **Python Interpreter**: Open the command line and type python. The Python interpreter should start, displaying the Python version.
      - **Onnxruntime Package**: Ensure that one of the following is installed:
            - *onnxruntime-directml* (DirectML EP)
            - *onnxruntime-gpu* (CUDA EP)
            - *onnxruntime* (CPU EP)
      - **Onnx and Onnxruntime Import**: Ensure that following python command runs successfully.
            .. code-block:: python

                python -c "import onnx; import onnxruntime"
      - **Environment Variables**: For workflows using CUDA dependencies (e.g., CUDA EP-based calibration), ensure environment variables like *CUDA_PATH*, *CUDA_V12_4*, or *CUDA_V11_8* etc. are set correctly. Reopen the command-prompt if any environment variable is updated or newly created.
      - **ModelOpt-Windows Import Check**: Run the following command to ensure the installation is successful:

            .. code-block:: python

                python -c "import modelopt.onnx.quantization"

- If you encounter any difficulties during the installation process, please refer :ref:`FAQ_ModelOpt_Windows` FAQs for potential solutions and additional guidance.
