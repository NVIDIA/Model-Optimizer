#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Download RULER calibration data for attention sparsity.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
ESSAYS_DIR="${DATA_DIR}/essays"
URLS_FILE="${DATA_DIR}/PaulGrahamEssays_URLs.txt"
URLS_URL="https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"

mkdir -p "${ESSAYS_DIR}"

# Download URL list if not exists
if [ ! -f "${URLS_FILE}" ]; then
    echo "Downloading URL list..."
    curl -fsSL "${URLS_URL}" -o "${URLS_FILE}"
fi

# Download essays from GitHub URLs
echo -n "Downloading essays"
count=0
while IFS= read -r url || [ -n "$url" ]; do
    if [[ "${url}" == https://github.com*.txt ]]; then
        filename=$(basename "${url}")
        filepath="${ESSAYS_DIR}/${filename}"
        if [ ! -f "${filepath}" ]; then
            raw_url="${url/github.com/raw.githubusercontent.com}"
            raw_url="${raw_url/\/raw\//\/}"
            curl -fsSL "${raw_url}" -o "${filepath}" 2>/dev/null && echo -n "."
            count=$((count + 1))
        fi
    fi
done < "${URLS_FILE}"
echo " done"

echo "Downloaded ${count} essays to ${ESSAYS_DIR}"
