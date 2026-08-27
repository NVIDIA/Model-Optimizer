# Configure ModelOpt and build PyArrow on N1x

N1x uses Python 3.13 on Windows ARM64. Install ModelOpt and all dependencies
that publish compatible wheels with `pip`; build only PyArrow from source.

Only environment setup is platform-specific. After it is complete, use the
[standard ONNX PTQ examples](../../onnx_ptq/README.md); quantization APIs,
formats, and generated models are the same as on regular Windows.

The PyArrow steps follow Apache Arrow's official
[Windows ARM64 C++ build](https://arrow.apache.org/docs/dev/developers/cpp/windows.html#building-on-windows-arm64-using-ninja-and-clang)
and [self-contained wheel](https://arrow.apache.org/docs/dev/developers/python/building.html#self-contained-wheel)
instructions, with two fixes needed by the tested toolchain.

> [!WARNING]
> Apache Arrow considers Windows ARM64 support experimental.

The following configuration has been tested; these versions are not minimum
requirements.

| Component | Tested configuration |
|---|---|
| Platform | N1x, Windows ARM64 |
| Python | 3.13.14 ARM64 |
| Visual Studio | 2026 Community, ARM64 C++ tools |
| LLVM | 22.1.8 Windows ARM64 |
| CUDA and CuPy | CUDA 13.4, `cupy-cuda13x` 14.2.0 |
| PyArrow | 26.0.0 development source build |
| ONNX Runtime and TensorRT RTX ABI EP | 1.24.4 and 0.4.0 |
| ModelOpt | 0.47.0 development tree |

## Configure the N1x environment

Create one virtual environment for both ModelOpt and the local PyArrow build:

```powershell
$Python = "$env:LOCALAPPDATA\Programs\Python\Python313-arm64\python.exe"
$Venv = "$env:USERPROFILE\Desktop\py313-onnx-modelopt"
$ModelOptSource = "$env:USERPROFILE\Desktop\Model-Optimizer"

if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    & $Python -m venv $Venv
}
$PythonExe = "$Venv\Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip
```

Use `pip` for ModelOpt and every dependency that has an ARM64 wheel:

```powershell
Push-Location $ModelOptSource
try {
    & $PythonExe -m pip install -e ".[onnx]"
} finally {
    Pop-Location
}
```

If another ModelOpt workflow, extra, or requirements file requests PyArrow,
build and install the missing wheel below, then rerun that original `pip install`
command. Do not build the other dependencies from source.

## Prepare the PyArrow build

Install Visual Studio ARM64 C++ tools, the Windows SDK, Git, and LLVM for
Windows ARM64. Preserve LF endings when creating a new Arrow checkout:

```powershell
git -c core.autocrlf=false clone https://github.com/apache/arrow.git `
    "$env:USERPROFILE\Desktop\arrow"
```

Set the build paths and load the Visual Studio ARM64 environment:

```powershell
$ArrowSource = "$env:USERPROFILE\Desktop\arrow"
$ArrowBuild = "$ArrowSource\cpp\build-py313-arm64"
$ArrowHome = "$Venv\arrow-dist"
$LlvmRoot = "$Venv\llvm-arm64"
$VsRoot = "C:\Program Files\Microsoft Visual Studio\18\Community"

& "$VsRoot\Common7\Tools\Launch-VsDevShell.ps1" `
    -Arch arm64 -HostArch arm64 -SkipAutomaticLocation
$env:PATH = "$LlvmRoot\bin;$Venv\Scripts;$env:PATH"

& $PythonExe -m pip install --upgrade `
    build cmake ninja "cython>=3.1" "numpy>=2.0" `
    "scikit-build-core>=1.0" "setuptools_scm[toml]>=8" "libcst>=1.8.6"
```

Confirm that Python reports `ARM64` and LLVM reports an ARM64 target:

```powershell
& $PythonExe -c "import platform; print(platform.machine())"
& "$LlvmRoot\bin\clang-cl.exe" --version
```

## Build Arrow C++

Create the small Arrow Dataset compatibility header required by the tested
Visual Studio and LLVM combination:

```powershell
$DatasetShim = "$Venv\arrow-dataset-fileinfo.h"
@'
#pragma once
#ifdef ARROW_DS_EXPORTING
#include "arrow/filesystem/filesystem.h"
#endif
'@ | Set-Content -LiteralPath $DatasetShim -Encoding ascii
$DatasetShim = $DatasetShim.Replace('\', '/')
```

Configure the local-file features commonly used by ModelOpt. bzip2 is disabled
to avoid Arrow's Makefile-based bzip2 recipe on `clang-cl`.

```powershell
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
```

Make generated xsimd use LLVM's standard `arm_neon.h` definitions:

```powershell
$XsimdHeader = "$ArrowBuild\_deps\xsimd-src\include\xsimd\types\xsimd_neon_register.hpp"
$XsimdText = [IO.File]::ReadAllText($XsimdHeader)
$Old = "#if defined(_WIN32) && XSIMD_WITH_NEON64"
$New = "#if defined(_WIN32) && XSIMD_WITH_NEON64 && !defined(__clang__)"
if ($XsimdText.Contains($Old)) {
    [IO.File]::WriteAllText($XsimdHeader, $XsimdText.Replace($Old, $New))
}
```

Build and install Arrow C++:

```powershell
& "$Venv\Scripts\cmake.exe" --build $ArrowBuild --parallel 4
if ($LASTEXITCODE) { throw "Arrow build failed" }
& "$Venv\Scripts\cmake.exe" --install $ArrowBuild
if ($LASTEXITCODE) { throw "Arrow install failed" }
```

## Build and install the PyArrow wheel

```powershell
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
& $PythonExe -m pip install --force-reinstall $Wheel.FullName
```

If the original N1x setup command stopped because PyArrow had no compatible
wheel, rerun it now. `pip` will use the installed local PyArrow wheel and resolve
the remaining published dependencies normally.

The wheel can be reused in other N1x environments running CPython 3.13 ARM64.
Rebuild it when the Python ABI, Arrow source version, or native toolchain changes.

## Validate the environment

This smoke test checks the WoA environment, not quantization behavior:

```powershell
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
devices = [device for device in ort.get_ep_devices() if device.ep_name == ep_name]
if not devices:
    ort.register_execution_provider_library(ep_name, trt_rtx_ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == ep_name]
assert devices, f"No compatible {ep_name} device found"

print("Python:", platform.python_version(), platform.machine())
print("CuPy/CUDA:", cp.__version__, cp.cuda.runtime.runtimeGetVersion())
print("PyArrow:", pa.__version__, pa.runtime_info())
print("TensorRT RTX ABI EP:", ep_name)
'@ | & $PythonExe -

& $PythonExe -m pip check
```

## Known limitations

- The built wheel is specific to CPython 3.13 and Windows ARM64.
- Arrow SIMD is disabled in this configuration.
- bzip2 and Parquet encryption are not included.
- Quantization APIs, formats, and examples are unchanged from standard Windows.

## Troubleshooting

- If bundled Thrift reports a corrupt patch, use an Arrow checkout with LF line
  endings as shown above.
- If CMake or PyArrow finds stale files, use a new Arrow build directory.
