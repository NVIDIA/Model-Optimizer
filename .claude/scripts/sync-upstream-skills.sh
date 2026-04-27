#!/usr/bin/env bash
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

# Re-vendor upstream Claude skills from NVIDIA-NeMo/Evaluator at a pinned SHA.
#
# Scope: only skills we vendor verbatim (launching-evals, accessing-mlflow).
# The `evaluation` skill is a *modified* fork of upstream nel-assistant and is
# NOT managed by this script — update it manually when pulling upstream changes.
#
# Usage:
#   .claude/scripts/sync-upstream-skills.sh            # re-vendor at the pinned SHA
#   UPSTREAM_SHA=<sha> .claude/scripts/sync-upstream-skills.sh   # bump to a new SHA
#
# Requires: gh, base64, awk. Run from the repo root.
#
# The script overwrites .claude/skills/<skill>/ with upstream contents and
# re-applies our provenance lines into each SKILL.md frontmatter. If you have
# local changes to a vendored skill, they will be lost — that is expected,
# since vendored-verbatim skills should not be modified locally.

set -euo pipefail

# Pinned upstream commit. Bump this (or pass UPSTREAM_SHA=...) when syncing.
DEFAULT_SHA="8fa16b237d11e213ea665d5bad6b44d393762317"
SHA="${UPSTREAM_SHA:-$DEFAULT_SHA}"
SHORT_SHA="${SHA:0:7}"

UPSTREAM_REPO="NVIDIA-NeMo/Evaluator"
UPSTREAM_BASE="packages/nemo-evaluator-launcher/.claude/skills"
DEST_BASE=".claude/skills"

if [[ ! -d "$DEST_BASE" ]]; then
    echo "error: run from the repo root (expected $DEST_BASE/ to exist)" >&2
    exit 1
fi

echo "Syncing upstream skills from $UPSTREAM_REPO @ $SHORT_SHA"

fetch_tree() {
    local skill="$1"
    local path="$2"
    gh api "repos/$UPSTREAM_REPO/contents/$UPSTREAM_BASE/$skill/$path?ref=$SHA" \
        -q '.[] | "\(.type)\t\(.name)"'
}

fetch_file() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    gh api "repos/$UPSTREAM_REPO/contents/$src?ref=$SHA" -q '.content' | base64 -d > "$dst"
}

fetch_skill_recursive() {
    local skill="$1"
    local subpath="${2:-}"
    local remote="$UPSTREAM_BASE/$skill"
    [[ -n "$subpath" ]] && remote="$remote/$subpath"

    local entries
    entries=$(gh api "repos/$UPSTREAM_REPO/contents/$remote?ref=$SHA" -q '.[] | "\(.type)\t\(.name)"')

    while IFS=$'\t' read -r type name; do
        local rel_path
        if [[ -n "$subpath" ]]; then
            rel_path="$subpath/$name"
        else
            rel_path="$name"
        fi

        if [[ "$type" == "file" ]]; then
            local dst="$DEST_BASE/$skill/$rel_path"
            echo "  fetch: $dst"
            fetch_file "$UPSTREAM_BASE/$skill/$rel_path" "$dst"
        elif [[ "$type" == "dir" ]]; then
            fetch_skill_recursive "$skill" "$rel_path"
        fi
    done <<< "$entries"
}

# Inject our provenance lines into a SKILL.md frontmatter, right after the
# `description:` line. Idempotent — removes any existing provenance block first.
inject_provenance() {
    local skill="$1"
    local extra_note="${2:-}"
    local path="$DEST_BASE/$skill/SKILL.md"

    awk -v sha="$SHA" -v short="$SHORT_SHA" -v skill="$skill" -v extra="$extra_note" '
        BEGIN { in_fm = 0; injected = 0; fm_end_seen = 0 }
        # Skip any pre-existing provenance or license lines we own
        /^license: Apache-2\.0$/ && in_fm && !fm_end_seen { next }
        /^# Vendored verbatim/ && in_fm && !fm_end_seen { next }
        /^# https:\/\/github\.com\/NVIDIA-NeMo\/Evaluator\/tree\// && in_fm && !fm_end_seen { next }
        /^# To re-sync:/ && in_fm && !fm_end_seen { next }
        /^# Note: this skill depends on the mlflow-mcp/ && in_fm && !fm_end_seen { next }
        /^# configured in the user/ && in_fm && !fm_end_seen { next }
        {
            print
            if ($0 == "---") {
                if (in_fm == 0) { in_fm = 1 }
                else { in_fm = 0; fm_end_seen = 1 }
            }
            if (in_fm && !injected && $0 ~ /^description: /) {
                print "license: Apache-2.0"
                print "# Vendored verbatim from NVIDIA NeMo Evaluator (commit " short ")"
                print "# https://github.com/NVIDIA-NeMo/Evaluator/tree/" sha "/packages/nemo-evaluator-launcher/.claude/skills/" skill
                print "# To re-sync: .claude/scripts/sync-upstream-skills.sh"
                if (extra != "") {
                    n = split(extra, lines, "\\|")
                    for (i = 1; i <= n; i++) print "# " lines[i]
                }
                injected = 1
            }
        }
    ' "$path" > "$path.tmp"
    mv "$path.tmp" "$path"
}

for skill in launching-evals accessing-mlflow; do
    echo ""
    echo "== $skill =="
    rm -rf "${DEST_BASE:?}/$skill"
    fetch_skill_recursive "$skill"

    case "$skill" in
        accessing-mlflow)
            inject_provenance "$skill" \
                "Note: this skill depends on the mlflow-mcp MCP server (https://github.com/kkruglik/mlflow-mcp)|configured in the user's Claude Code setup."
            ;;
        *)
            inject_provenance "$skill"
            ;;
    esac
done

echo ""
echo "Done. Review with: git diff $DEST_BASE/launching-evals $DEST_BASE/accessing-mlflow"
echo "If the SHA changed, update DEFAULT_SHA at the top of this script before committing."
