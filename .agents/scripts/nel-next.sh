#!/usr/bin/env bash
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

# nel-next.sh — run NeMo Evaluator "next" (nel 0.3.x) from an ISOLATED venv.
#
# A few AA benchmarks (Terminal-Bench 2.x, SWE-bench) run on `nemo-evaluator`
# 0.3.x + `harbor` extra — a different package/CLI/config-schema from the eval
# skill's default `nemo-evaluator-launcher` 0.2.6 (CLI `nel eval run X.yaml`,
# overrides `-O a.b.c=v`, schema services/benchmarks/cluster/output). Installing
# it into the 0.2.6 env would clobber `nel`, so this keeps 0.3.x in its own venv
# (default ~/.local/share/nel/venvs/nel-next) and forwards all args to its `nel`.
#
# Usage (source .env FIRST so the config's ${VAR}s resolve; this never reads secrets):
#   .agents/scripts/nel-next.sh --setup-only|--which|--version
#   .agents/scripts/nel-next.sh eval run <config.yaml> [--dry-run|--submit] [-O k=v ...]
#   .agents/scripts/nel-next.sh eval {status|logs|report|merge|resume|stop} -r <run_id>
#
# Install source (env overrides): NEL_NEXT_SPEC (PyPI default, "nemo-evaluator[harbor]==0.3.*"),
# or NEL_NEXT_ORIGIN [+ NEL_NEXT_REF] for the internal git build; NEL_NEXT_VENV for the path.
# The venv rebuilds automatically when the resolved spec changes (tracked via a sentinel).
set -euo pipefail

NEL_NEXT_SPEC="${NEL_NEXT_SPEC:-nemo-evaluator[harbor]==0.3.*}"
NEL_NEXT_ORIGIN="${NEL_NEXT_ORIGIN:-}"
NEL_NEXT_REF="${NEL_NEXT_REF:-}"
VENV_DIR="${NEL_NEXT_VENV:-$HOME/.local/share/nel/venvs/nel-next}"
NEL_BIN="$VENV_DIR/bin/nel"
SENTINEL="$VENV_DIR/.installed"
LOCK_FILE="${VENV_DIR}.lock"

if [[ -n "$NEL_NEXT_ORIGIN" ]]; then
  INSTALL_SPEC="nemo-evaluator[harbor] @ ${NEL_NEXT_ORIGIN}${NEL_NEXT_REF:+@${NEL_NEXT_REF}}"
else
  INSTALL_SPEC="$NEL_NEXT_SPEC"
fi

_log() { printf '\033[2m  %s\033[0m\n' "$*" >&2; }

_do_install() {
  if [[ -f "$SENTINEL" ]] && [[ "$(cat "$SENTINEL" 2>/dev/null)" == "$INSTALL_SPEC" ]] \
     && "$VENV_DIR/bin/python" -c 'import nemo_evaluator' 2>/dev/null; then
    return 0
  fi
  command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2; return 1; }
  _log "Setting up nel-next venv (~1-2 min)…  spec: ${INSTALL_SPEC}  venv: ${VENV_DIR}"
  rm -rf "$VENV_DIR"
  uv venv --python 3.12 "$VENV_DIR" >&2
  uv pip install --python "$VENV_DIR/bin/python" "$INSTALL_SPEC" >&2
  printf '%s' "$INSTALL_SPEC" > "$SENTINEL"
  _log "nel-next $("$VENV_DIR/bin/python" -c 'import nemo_evaluator; print(nemo_evaluator.__version__)' 2>/dev/null || echo '?') installed."
}

_ensure_venv() {
  mkdir -p "$(dirname "$VENV_DIR")"
  if command -v flock >/dev/null 2>&1; then
    ( flock -x -w 300 200 || { echo "ERROR: timed out on venv lock" >&2; exit 1; }
      _do_install ) 200>"$LOCK_FILE"
  else
    _do_install
  fi
}

case "${1:-}" in
  --setup-only|--which) _ensure_venv; echo "$NEL_BIN"; exit 0 ;;
  --version)            _ensure_venv; "$VENV_DIR/bin/python" -c 'import nemo_evaluator; print(nemo_evaluator.__version__)'; exit 0 ;;
  -h|--help)            awk '/^# nel-next\.sh/{p=1} /^set /{p=0} p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "")                   echo "ERROR: no args. Try: nel-next.sh eval run <config.yaml> [--dry-run]  (or --help)" >&2; exit 2 ;;
esac

_ensure_venv
exec "$NEL_BIN" "$@"
