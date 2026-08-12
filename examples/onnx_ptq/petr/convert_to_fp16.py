# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from modelopt.onnx import utils as onnx_utils
from modelopt.onnx.autocast import convert_to_mixed_precision


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a PETR ONNX model to FP16")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("calibration_data")
    return parser.parse_args()


def main():
    args = parse_args()
    model = convert_to_mixed_precision(
        onnx_path=args.input,
        low_precision_type="fp16",
        calibration_data=args.calibration_data,
        keep_io_types=True,
        providers=["cuda:0", "cpu"],
    )
    onnx_utils.save_onnx(model, args.output)


if __name__ == "__main__":
    main()
