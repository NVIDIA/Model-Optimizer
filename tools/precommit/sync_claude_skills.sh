#!/bin/bash
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
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
skills_src="$repo_root/.agents/skills"
skills_dst="$repo_root/.claude/skills"

added=()
for d in "$skills_src"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  link="$skills_dst/$name"
  if [ ! -e "$link" ] && [ ! -L "$link" ]; then
    ln -s "../../.agents/skills/$name" "$link"
    added+=("$link")
  fi
done

if [ "${#added[@]}" -gt 0 ]; then
  git add "${added[@]}"
  echo "claude-skills-sync: staged ${#added[@]} new symlink(s) in .claude/skills/"
fi
