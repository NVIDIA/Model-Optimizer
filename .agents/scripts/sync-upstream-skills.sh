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
# Scope: skills we vendor from upstream (launching-evals, accessing-mlflow). Their bodies are
# verbatim; launching-evals gets a local `description:` override (see DESC_LAUNCHING_EVALS).
# The `evaluation` skill is a *modified* fork of upstream nel-assistant and is
# NOT managed by this script — update it manually when pulling upstream changes.
#
# Usage:
#   .agents/scripts/sync-upstream-skills.sh            # re-vendor at the pinned SHA
#   UPSTREAM_SHA=<sha> .agents/scripts/sync-upstream-skills.sh   # bump to a new SHA
#
# Requires: gh, base64, awk. Run from the repo root.
#
# The script overwrites skills/<skill>/ through the
# .agents/skills compatibility symlink and
# re-applies our provenance lines into each SKILL.md frontmatter. If you have
# local changes to a vendored skill, they will be lost — that is expected,
# since vendored skills should not be modified locally. The only intended local
# difference is the description override below, which is re-applied on every sync.
#
# Note: .claude/skills/ (and other agent-specific skill dirs) are symlinks to
# root skills/ — see .agents/README.md.

set -euo pipefail

# Pinned upstream commit. Bump this (or pass UPSTREAM_SHA=...) when syncing.
DEFAULT_SHA="8fa16b237d11e213ea665d5bad6b44d393762317"
SHA="${UPSTREAM_SHA:-$DEFAULT_SHA}"
SHORT_SHA="${SHA:0:7}"

# Local description override for launching-evals: scope the upstream "ALWAYS triggers" text to
# NEL runs so the skill does not claim evaluations run with other tools. Keep on one line, with no
# double quotes or backslashes: it is written as a double-quoted YAML string (the text contains ": ").
DESC_LAUNCHING_EVALS="NEL backend - run, monitor, analyze, and debug LLM evaluations via nemo-evaluator-launcher. Use for NEL runs: the task or user names NEL or names no evaluation tool (NEL is the default), or an invocation ID or run directory comes from NEL; evaluations run with another named tool use that tool's own skills. Covers checking status and live progress, debugging failed runs, exporting artifacts and logs, and analyzing results. Do NOT use for creating or modifying evaluation configs."

UPSTREAM_REPO="NVIDIA-NeMo/Evaluator"
UPSTREAM_BASE="packages/nemo-evaluator-launcher/.claude/skills"
DEST_BASE=".agents/skills"

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
            local local_path="$rel_path"
            [[ "$local_path" == "tests.json" ]] && local_path="evals/evals.json"
            local dst="$DEST_BASE/modelopt-$skill/$local_path"
            echo "  fetch: $dst"
            fetch_file "$UPSTREAM_BASE/$skill/$rel_path" "$dst"
        elif [[ "$type" == "dir" ]]; then
            fetch_skill_recursive "$skill" "$rel_path"
        fi
    done <<< "$entries"
}

# Inject our provenance lines into a SKILL.md frontmatter, right after the
# `description:` line. Idempotent — removes any existing provenance block first.
# Optional 3rd argument replaces the upstream `description:` text and changes the provenance
# wording to "vendored, description overridden".
inject_provenance() {
    local skill="$1"
    local extra_note="${2:-}"
    local desc_override="${3:-}"
    local path="$DEST_BASE/modelopt-$skill/SKILL.md"

    # The override goes through ENVIRON, not -v, so awk does not process backslash escapes.
    DESC_OVERRIDE="$desc_override" awk -v sha="$SHA" -v short="$SHORT_SHA" -v skill="$skill" -v extra="$extra_note" '
        BEGIN { in_fm = 0; injected = 0; fm_end_seen = 0; skip_desc = 0; desc = ENVIRON["DESC_OVERRIDE"] }
        # Skip any pre-existing provenance or license lines we own
        /^license: Apache-2\.0$/ && in_fm && !fm_end_seen { next }
        /^# Vendored / && in_fm && !fm_end_seen { next }
        /^# https:\/\/github\.com\/NVIDIA-NeMo\/Evaluator\/tree\// && in_fm && !fm_end_seen { next }
        /^# To re-sync:/ && in_fm && !fm_end_seen { next }
        /^# Note: this skill depends on the mlflow-mcp/ && in_fm && !fm_end_seen { next }
        /^# configured in the user/ && in_fm && !fm_end_seen { next }
        {
            # With a description override, drop the upstream description continuation lines (every
            # line up to the next frontmatter key or the closing ---).
            if (in_fm && !fm_end_seen && skip_desc) {
                if ($0 ~ /^[A-Za-z_][A-Za-z0-9_-]*:/ || $0 == "---") skip_desc = 0
                else next
            }
            if (in_fm && !fm_end_seen && $0 ~ /^name: /) $0 = "name: modelopt-" skill
            if (in_fm && !fm_end_seen && desc != "" && $0 ~ /^description: /) { $0 = "description: \"" desc "\""; skip_desc = 1 }
            print
            if ($0 == "---") {
                if (in_fm == 0) { in_fm = 1 }
                else { in_fm = 0; fm_end_seen = 1 }
            }
            if (in_fm && !injected && $0 ~ /^description: /) {
                print "license: Apache-2.0"
                if (desc != "") print "# Vendored from NVIDIA NeMo Evaluator (commit " short "), description overridden (see sync-upstream-skills.sh)"
                else print "# Vendored verbatim from NVIDIA NeMo Evaluator (commit " short ")"
                print "# https://github.com/NVIDIA-NeMo/Evaluator/tree/" sha "/packages/nemo-evaluator-launcher/.claude/skills/" skill
                print "# To re-sync: .agents/scripts/sync-upstream-skills.sh"
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
    rm -rf "${DEST_BASE:?}/modelopt-$skill"
    fetch_skill_recursive "$skill"
    if [[ -f "$DEST_BASE/modelopt-$skill/evals/evals.json" ]]; then
        ln -s evals/evals.json "$DEST_BASE/modelopt-$skill/tests.json"
    fi

    case "$skill" in
        accessing-mlflow)
            inject_provenance "$skill" \
                "Note: this skill depends on the mlflow-mcp MCP server (https://github.com/kkruglik/mlflow-mcp)|configured in the user's Claude Code setup."
            ;;
        launching-evals)
            inject_provenance "$skill" "" "$DESC_LAUNCHING_EVALS"
            ;;
        *)
            inject_provenance "$skill"
            ;;
    esac
done

echo ""
echo "Done. Review with: git diff $DEST_BASE/modelopt-launching-evals $DEST_BASE/modelopt-accessing-mlflow"
echo "If the SHA changed, update DEFAULT_SHA at the top of this script before committing."
