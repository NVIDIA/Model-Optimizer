/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

constexpr int8_t INT4_MIN = -8;
constexpr int8_t INT4_MAX = 7;
constexpr int8_t UINT4_MAX = 15;
constexpr int8_t UINT4_MIN = 0;

namespace {
int8_t ClipAndRoundOneNumber(bool is_signed, const double num) {

  const int8_t upper = is_signed ? INT4_MAX : UINT4_MAX;
  const int8_t lower = is_signed ? INT4_MIN : UINT4_MIN;

  // clip the number to range [lower, upper]
  const double min_num = num < upper ? num : upper;
  const double max_num = min_num >= lower ? min_num : lower;

  // round the number to nearest integer
  return static_cast<int8_t>(round(max_num));
}
} // namespace

//
// Arguments:
//             is_signed:    bool, whether to convert to signed int4 or uint4.
//              inputs[]:    array containing the input elements
//      input_array_size:    number of elements in input array
//             outputs[]:    int8 array for storing the packed output
//     output_array_size:    size of the output array,
//                                  - should be large enough to accommodate all final entries
//                                  - should be at least input_array_size/2
// Return value:
//          the number of elements written in output array
//
// Note:
//       (1) When input array is of odd length
//              - 0 is used as 2nd element for packing, and it goes to MSB.
//              - "(input_array_size/2)+1" is the required output array length.
//

template <typename T>
size_t RoundAndPack(const bool is_signed, const py::array_t<T> &inputs_arg,
                    const py::size_t input_array_size, py::array_t<int8_t> &outputs_arg,
                    const py::size_t output_array_size) {

  if (!input_array_size || !output_array_size) {
    return 0;
  }

  if (output_array_size <
      ceil(input_array_size / 2.0f)) // should be "==" for most/practical purposes
  {
    return 0;
  }

  const auto buf = inputs_arg.request();
  const T *inputs = static_cast<const T *>(buf.ptr);

  auto buf2 = outputs_arg.request();
  int8_t *outputs = static_cast<int8_t *>(buf2.ptr);

  if (!inputs || !outputs || (buf.ndim != 1) || (buf2.ndim != 1)) {
    return 0;
  }

  size_t i, j;
  for (i = 0, j = 0; (i < (input_array_size - 1)) && (j < output_array_size); i += 2, j++) {
    int8_t v1 = ClipAndRoundOneNumber(is_signed, static_cast<double>(inputs[i]));
    int8_t v2 = ClipAndRoundOneNumber(is_signed, static_cast<double>(inputs[i + 1]));
    outputs[j] =
        ((v2 << 4) | (v1 & 0x0F)); // index i (v1) goes to LSB and index i+1 (v2) goes to MSB
  }

  if (input_array_size % 2) {
    assert(i == (input_array_size - 1));
    assert(j <= (output_array_size - 1));
    // 4-bit number goes to LSB, MSB remains 0
    outputs[j] = 0x0F & ClipAndRoundOneNumber(is_signed, inputs[i]);
    j++;
    i++;
  }

  assert(i == input_array_size);
  assert(j == ceil(input_array_size / 2.0f));

  return j;
}

PYBIND11_MODULE(modelopt_round_and_pack_ext, m) {
  m.doc() = "C++ implementation of a basic round_and_pack logic";
  m.def("round_and_pack", &RoundAndPack<float>, "Round-And-Pack C++ Implementation");
  m.def("round_and_pack", &RoundAndPack<double>, "Round-And-Pack C++ Implementation");
  m.def("round_and_pack", &RoundAndPack<int8_t>, "Round-And-Pack C++ Implementation");
  m.def("round_and_pack", &RoundAndPack<uint8_t>, "Round-And-Pack C++ Implementation");
}

/*
<%
setup_pybind11(cfg)
%>
*/
