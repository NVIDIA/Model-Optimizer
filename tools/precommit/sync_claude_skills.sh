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
skills_src="$repo_root/skills"
skills_dst="$repo_root/.claude/skills"

changed=()
for d in "$skills_src"/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  link="$skills_dst/$name"
  target="../../skills/$name"
  if [ -L "$link" ]; then
    [ "$(readlink "$link")" != "$target" ] || continue
    ln -sfn "$target" "$link"
  elif [ ! -e "$link" ]; then
    ln -s "$target" "$link"
  else
    continue
  fi
  changed+=("$link")
done

if [ "${#changed[@]}" -gt 0 ]; then
  git add "${changed[@]}"
  echo "claude-skills-sync: staged ${#changed[@]} symlink update(s) in .claude/skills/"
fi
