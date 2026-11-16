# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check actual CUDA index expressions on the host without large allocations."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestFP8KernelIndexing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler unavailable")
        source = (
            Path(__file__).resolve().parents[4]
            / "modelopt/torch/kernels/quantization/gemm/tensor_quant_gpu_fp8.cu"
        ).read_text()
        cls.expressions = re.findall(r"((?:size_t|int) tid = [^;]+;)\s*(for \([^\n]+\)) \{", source)
        if len(cls.expressions) != 2:
            raise AssertionError("expected both FP8 kernel index loops")
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        functions = []
        for index, (declaration, loop) in enumerate(cls.expressions):
            functions.append(f"""
void kernel{index}(size_t n, unsigned block, unsigned thread) {{
    struct Dim {{ unsigned x; }} blockIdx{{block}}, blockDim{{128}}, threadIdx{{thread}};
    {declaration}
    size_t count = 0, first = 0, last = 0;
    {loop} {{
        if (count == 0) first = idx;
        last = idx;
        ++count;
    }}
    std::printf("%zu %zu %zu\\n", count, first, last);
}}
""")
        driver = root / "index.cpp"
        driver.write_text(
            "#include <cstddef>\n#include <cstdio>\n#include <cstdlib>\n"
            + "\n".join(functions)
            + """
int main(int argc, char** argv) {
    const size_t n = std::strtoull(argv[1], nullptr, 10);
    const unsigned block = std::strtoul(argv[2], nullptr, 10);
    const unsigned thread = std::strtoul(argv[3], nullptr, 10);
    kernel0(n, block, thread);
    kernel1(n, block, thread);
}
"""
        )
        cls.binary = root / "index"
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-fsanitize=signed-integer-overflow",
                "-fno-sanitize-recover=all",
                str(driver),
                "-o",
                str(cls.binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def check_indices(self, n, thread):
        block, local = divmod(thread, 128)
        run = subprocess.run(
            [str(self.binary), str(n), str(block), str(local)],
            check=True,
            capture_output=True,
            text=True,
        )
        indices = list(range(4 * thread, min(4 * (thread + 1), n)))
        expected = (len(indices), indices[0] if indices else 0, indices[-1] if indices else 0)
        for row in run.stdout.splitlines():
            assert tuple(map(int, row.split())) == expected

    def test_regular_and_partial_tiles(self):
        for n in (0, 1, 3, 4, 5, 511, 512, 513, 4097):
            for thread in (0, 1, 127, 128, n // 4, n // 4 + 1):
                with self.subTest(n=n, thread=thread):
                    self.check_indices(n, thread)

    def test_signed_boundary_and_large_tensor_tail(self):
        for n in (2**31 - 4, 2**31 - 1, 2**31, 2**31 + 4, 2_284_199_424, 2**33 + 3):
            for thread in (2**29 - 2, 2**29 - 1, 2**29, 2**29 + 1, n // 4, n // 4 + 1):
                with self.subTest(n=n, thread=thread):
                    self.check_indices(n, thread)


if __name__ == "__main__":
    unittest.main()
