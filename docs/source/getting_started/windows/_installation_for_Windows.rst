.. _Install-Page-Windows:

========================
Installation for Windows
========================

**System Requirements**

The following system requirements are necessary to install and use Model Optimizer - Windows:

+-------------------------+-----------------------------+
| OS                      |  Windows                    |
+-------------------------+-----------------------------+
| Architecture            |  amd64 (x86_64)             |
+-------------------------+-----------------------------+
| Python                  |  >=3.10,<3.13               |
+-------------------------+-----------------------------+
| CUDA                    |  >=12.0                     |
+-------------------------+-----------------------------+
| ONNX Runtime            |  1.20.0                     |
+-------------------------+-----------------------------+
| Nvidia Driver           |  565.90 or newer            |
+-------------------------+-----------------------------+
| Nvidia GPU              |  RTX 40 and 50 series       |
+-------------------------+-----------------------------+

.. note::
   - Make sure to use GPU-compatible driver and other dependencies (e.g. torch etc.). For instance, support for Blackwell GPU might be present in Nvidia 570+ driver, and CUDA-12.8+.
   - We currently support *Single-GPU* configuration.

The Model Optimizer - Windows can be used in following ways:

.. toctree::
   :glob:
   :maxdepth: 1

   ./_installation_standalone.rst
   ./_installation_with_olive.rst
