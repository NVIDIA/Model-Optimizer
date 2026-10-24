.. _Install-Page-Windows-ARM64:

===================================
Install ModelOpt on Windows on Arm
===================================

ModelOpt supports ONNX quantization workflows on native Windows ARM64 environments. The
quantization APIs, formats, and generated ONNX models are the same as on Windows x64; only the
environment setup differs.

Windows ARM64 support is experimental because some third-party packages do not yet publish
Windows ARM64 wheels. Install dependencies that provide compatible wheels normally. Build
PyArrow locally only when a selected ModelOpt workflow or one of its dependencies requires it.
ModelOpt core and the ``onnx`` extra do not directly depend on PyArrow.

Tested configuration
====================

The following configuration has been tested. These versions are not minimum requirements.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Component
     - Tested configuration
   * - Platform
     - Windows ARM64
   * - Python
     - 3.13.14 ARM64
   * - Visual Studio
     - 2026 Community with ARM64 C++ tools
   * - LLVM
     - 22.1.8 Windows ARM64
   * - CUDA and CuPy
     - CUDA 13.4 and ``cupy-cuda13x`` 14.2.0
   * - PyArrow
     - 26.0.0 development source build
   * - ONNX Runtime and TensorRT RTX ABI EP
     - 1.24.4 and 0.4.0
   * - ModelOpt
     - 0.47.0 development tree

Prerequisites
=============

Install the following native ARM64 components:

* Python 3.13
* A compatible NVIDIA GPU driver and CUDA 13 Toolkit
* Visual Studio with the ARM64 C++ build tools and a Windows SDK
* Git
* LLVM for Windows ARM64 from the `LLVM releases page
  <https://github.com/llvm/llvm-project/releases>`_

Apache Arrow documents the native toolchain in its `Windows ARM64 C++ build guide
<https://arrow.apache.org/docs/dev/developers/cpp/windows.html#building-on-windows-arm64-using-ninja-and-clang>`_.
Apache Arrow classifies this build configuration as experimental.

Configure the environment
=========================

Open PowerShell. Set ``$ModelOptSource`` to an existing checkout, or let the following commands
clone ModelOpt into the default location. This check ensures that ``Push-Location`` later in the
guide always receives a valid path.

.. code-block:: powershell

   $Workspace = "$env:USERPROFILE\Desktop"
   $ModelOptSource = "$Workspace\Model-Optimizer"
   if (-not (Test-Path "$ModelOptSource\.git")) {
       git clone https://github.com/NVIDIA/Model-Optimizer.git $ModelOptSource
       if ($LASTEXITCODE) { throw "ModelOpt clone failed" }
   }

Create one virtual environment for ModelOpt and any local dependency builds:

.. code-block:: powershell

   $Python = "$env:LOCALAPPDATA\Programs\Python\Python313-arm64\python.exe"
   $Venv = "$Workspace\py313-onnx-modelopt"

   if (-not (Test-Path $Python)) { throw "Set `$Python to the native ARM64 Python executable" }
   if (-not (Test-Path "$Venv\Scripts\python.exe")) {
       & $Python -m venv $Venv
   }
   $PythonExe = "$Venv\Scripts\python.exe"
   & $PythonExe -m pip install --upgrade pip

Install ModelOpt core from the checkout, followed by ARM64-compatible ONNX dependencies. This
explicit dependency installation avoids selecting the Windows x64 ``onnxruntime-gpu`` package.

.. code-block:: powershell

   Push-Location $ModelOptSource
   try {
       & $PythonExe -m pip install -e .
   } finally {
       Pop-Location
   }

   & $PythonExe -m pip install `
       cppimport cupy-cuda13x lief ml_dtypes `
       "onnx~=1.21.0" "onnx-graphsurgeon>=0.6.1" `
       "onnxconverter-common~=1.16.0" onnxscript `
       "onnxslim>=0.1.76" "polygraphy>=0.49.22" `
       "onnxruntime>=1.24.2" onnxruntime-ep-nv-tensorrt-rtx-cu13

If a different workflow requires another ModelOpt extra, install its compatible dependencies in
the same environment. If dependency resolution stops because no Windows ARM64 PyArrow wheel is
available, complete the following source build and then rerun the original installation command.

Build PyArrow when required
===========================

Some optional workflows use packages such as ``datasets`` that depend on PyArrow. Skip this
section when ``python -m pip check`` reports no missing PyArrow dependency.

The steps below follow Apache Arrow's `Windows ARM64 C++ build
<https://arrow.apache.org/docs/dev/developers/cpp/windows.html#building-on-windows-arm64-using-ninja-and-clang>`_
and `self-contained wheel
<https://arrow.apache.org/docs/dev/developers/python/building.html#self-contained-wheel>`_
instructions. They include two temporary compatibility adjustments needed by the tested LLVM and
Arrow development versions. Re-evaluate those adjustments when using a newer Arrow release.

Prepare the source and toolchain
--------------------------------

Preserve LF endings when creating the Arrow checkout:

.. code-block:: powershell

   $ArrowSource = "$Workspace\arrow"
   if (-not (Test-Path "$ArrowSource\.git")) {
       git -c core.autocrlf=false clone https://github.com/apache/arrow.git $ArrowSource
       if ($LASTEXITCODE) { throw "Arrow clone failed" }
   }

Set the build paths. Change ``$LlvmRoot`` if LLVM is installed elsewhere. ``vswhere`` locates a
Visual Studio installation that contains the ARM64 C++ toolchain instead of assuming a particular
Visual Studio edition or installation directory.

.. code-block:: powershell

   $ArrowBuild = "$ArrowSource\cpp\build-py313-arm64"
   $ArrowHome = "$Venv\arrow-dist"
   $LlvmRoot = "C:\Program Files\LLVM"
   $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

   if (-not (Test-Path "$LlvmRoot\bin\clang-cl.exe")) {
       throw "Set `$LlvmRoot to a native Windows ARM64 LLVM installation"
   }
   if (-not (Test-Path $VsWhere)) { throw "Visual Studio Installer was not found" }
   $VsRoot = & $VsWhere -latest -products * `
       -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 `
       -property installationPath
   if (-not $VsRoot) { throw "Visual Studio ARM64 C++ tools were not found" }

   & "$VsRoot\Common7\Tools\Launch-VsDevShell.ps1" `
       -Arch arm64 -HostArch arm64 -SkipAutomaticLocation
   $env:PATH = "$LlvmRoot\bin;$Venv\Scripts;$env:PATH"

   & $PythonExe -m pip install --upgrade `
       build cmake ninja "cython>=3.1" "numpy>=2.0" `
       "scikit-build-core>=1.0" "setuptools_scm[toml]>=8" "libcst>=1.8.6"

Confirm that Python reports ``ARM64`` and Clang reports an ARM64 target:

.. code-block:: powershell

   & $PythonExe -c "import platform; print(platform.machine())"
   & "$LlvmRoot\bin\clang-cl.exe" --version

Build Arrow C++
---------------

Create the Arrow Dataset compatibility header required by the tested toolchain:

.. code-block:: powershell

   $DatasetShim = "$Venv\arrow-dataset-fileinfo.h"
   @'
   #pragma once
   #ifdef ARROW_DS_EXPORTING
   #include "arrow/filesystem/filesystem.h"
   #endif
   '@ | Set-Content -LiteralPath $DatasetShim -Encoding ascii
   $DatasetShim = $DatasetShim.Replace('\', '/')

Configure the local-file features commonly used by ModelOpt workflows. This configuration disables
bzip2 to avoid Arrow's Makefile-based bzip2 recipe with ``clang-cl``.

.. code-block:: powershell

   & "$Venv\Scripts\cmake.exe" `
       -S "$ArrowSource\cpp" -B $ArrowBuild -G Ninja `
       "-DCMAKE_MAKE_PROGRAM=$Venv\Scripts\ninja.exe" `
       "-DCMAKE_C_COMPILER=$LlvmRoot\bin\clang-cl.exe" `
       "-DCMAKE_CXX_COMPILER=$LlvmRoot\bin\clang-cl.exe" `
       "-DCMAKE_INSTALL_PREFIX=$ArrowHome" `
       -DCMAKE_BUILD_TYPE=Release `
       -DARROW_DEPENDENCY_SOURCE=BUNDLED `
       -DARROW_BUILD_SHARED=ON -DARROW_BUILD_STATIC=OFF `
       -DARROW_BUILD_TESTS=OFF -DARROW_BUILD_EXAMPLES=OFF `
       -DARROW_ACERO=ON -DARROW_COMPUTE=ON `
       -DARROW_CSV=ON -DARROW_DATASET=ON -DARROW_FILESYSTEM=ON `
       -DARROW_JSON=ON -DARROW_PARQUET=ON `
       -DARROW_WITH_BROTLI=ON -DARROW_WITH_BZ2=OFF `
       -DARROW_WITH_LZ4=ON -DARROW_WITH_SNAPPY=ON `
       -DARROW_WITH_ZLIB=ON -DARROW_WITH_ZSTD=ON `
       -DARROW_SIMD_LEVEL=NONE -DARROW_RUNTIME_SIMD_LEVEL=NONE `
       -DPARQUET_REQUIRE_ENCRYPTION=OFF `
       "-DARROW_CXXFLAGS=/FI$DatasetShim"
   if ($LASTEXITCODE) { throw "Arrow configuration failed" }

For the tested Arrow development source, make generated xsimd use LLVM's standard
``arm_neon.h`` definitions:

.. code-block:: powershell

   $XsimdHeader = "$ArrowBuild\_deps\xsimd-src\include\xsimd\types\xsimd_neon_register.hpp"
   $XsimdText = [IO.File]::ReadAllText($XsimdHeader)
   $Old = "#if defined(_WIN32) && XSIMD_WITH_NEON64"
   $New = "#if defined(_WIN32) && XSIMD_WITH_NEON64 && !defined(__clang__)"
   if ($XsimdText.Contains($Old)) {
       [IO.File]::WriteAllText($XsimdHeader, $XsimdText.Replace($Old, $New))
   }

Build and install Arrow C++:

.. code-block:: powershell

   & "$Venv\Scripts\cmake.exe" --build $ArrowBuild --parallel 4
   if ($LASTEXITCODE) { throw "Arrow build failed" }
   & "$Venv\Scripts\cmake.exe" --install $ArrowBuild
   if ($LASTEXITCODE) { throw "Arrow install failed" }

Build and install the PyArrow wheel
-----------------------------------

Build a self-contained wheel containing the Arrow and Parquet C++ libraries:

.. code-block:: powershell

   $env:ARROW_HOME = $ArrowHome
   $env:CMAKE_PREFIX_PATH = $ArrowHome
   $env:CC = "$LlvmRoot\bin\clang-cl.exe"
   $env:CXX = "$LlvmRoot\bin\clang-cl.exe"
   $env:CMAKE_GENERATOR = "Ninja"
   $env:CMAKE_BUILD_PARALLEL_LEVEL = "4"
   $env:PYARROW_BUNDLE_ARROW_CPP = "ON"

   Push-Location "$ArrowSource\python"
   try {
       & $PythonExe -m build --wheel --no-isolation .
       if ($LASTEXITCODE) { throw "PyArrow wheel build failed" }
   } finally {
       Pop-Location
   }

   $Wheel = Get-ChildItem "$ArrowSource\python\dist\pyarrow-*-cp313-cp313-win_arm64.whl" |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
   if (-not $Wheel) { throw "The Windows ARM64 PyArrow wheel was not produced" }
   & $PythonExe -m pip install --force-reinstall $Wheel.FullName

Rerun the installation command that originally requested PyArrow. The local wheel can be reused in
other CPython 3.13 ARM64 environments. Rebuild it when the Python ABI, Arrow source version, or
native toolchain changes.

Verify the installation
=======================

This smoke test verifies architecture, GPU execution, the optional PyArrow build, and TensorRT RTX
ABI EP registration. Remove the PyArrow imports when that optional dependency was not installed.

.. code-block:: powershell

   @'
   import platform

   import cupy as cp
   import onnxruntime as ort
   import onnxruntime_ep_nv_tensorrt_rtx as trt_rtx_ep
   import pyarrow as pa
   import pyarrow.compute
   import pyarrow.dataset
   import pyarrow.parquet

   from modelopt.onnx.quantization import int4

   assert platform.machine().lower() in {"arm64", "aarch64"}
   assert int4.has_cupy
   assert cp.arange(4).sum().item() == 6

   ep_name = trt_rtx_ep.get_ep_name()
   if ep_name not in ort.get_available_providers():
       ort.register_execution_provider_library(ep_name, trt_rtx_ep.get_library_path())
   assert ep_name in ort.get_available_providers(), f"No compatible {ep_name} provider found"

   print("Python:", platform.python_version(), platform.machine())
   print("CuPy/CUDA:", cp.__version__, cp.cuda.runtime.runtimeGetVersion())
   print("PyArrow:", pa.__version__, pa.runtime_info())
   print("TensorRT RTX ABI EP:", ep_name)
   '@ | & $PythonExe -

   & $PythonExe -m pip check

After this environment check passes, use the :doc:`standard Windows ONNX PTQ guide
</guides/windows_guides/_ONNX_PTQ_guide>` and examples.

Known limitations and troubleshooting
=====================================

* The locally built wheel is specific to CPython 3.13 and Windows ARM64.
* Arrow SIMD is disabled in the tested configuration.
* bzip2 and Parquet encryption are not included.
* If bundled Thrift reports a corrupt patch, create the Arrow checkout with LF endings as shown.
* If CMake or PyArrow finds stale files, use a new Arrow build directory.
