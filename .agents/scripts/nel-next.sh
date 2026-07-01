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
#   .agents/scripts/nel-next.sh mlflow-push -r <run_id> -c <config.yaml> [-- -o k=v ...]
#     Post-run: SLURM doesn't auto-export. Pulls the merged bundle(s) + pushes to MLflow
#     using the config's export_config.mlflow (resolves ${MLFLOW_TRACKING_URI}, forces
#     emit_traces=false to avoid the per-sample hang). Run after `source .env`.
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

# Post-run MLflow push. SLURM runs don't auto-export; this stages each merged
# benchmark bundle's eval-*.json from the cluster (the dev box doesn't mount the
# run dir) and exports it to MLflow with traces off (avoids the per-sample hang).
_mlflow_push() {
  local rid="" cfg=""; local -a extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -r|--run-id) rid="$2"; shift 2 ;;
      -c|--config) cfg="$2"; shift 2 ;;
      --)          shift; extra+=("$@"); break ;;
      *)           extra+=("$1"); shift ;;
    esac
  done
  [[ -n "$rid" && -n "$cfg" ]] || { echo "usage: nel-next.sh mlflow-push -r <run_id> -c <config.yaml> [-- -o k=v ...]" >&2; return 2; }
  [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg" >&2; return 2; }
  _ensure_venv

  # Derive cluster + mlflow settings from the config (literals; no env interpolation needed).
  local cfgvars
  cfgvars="$("$VENV_DIR/bin/python" - "$cfg" <<'PY'
import os, sys, yaml, json, shlex
c = yaml.safe_load(open(sys.argv[1])) or {}
cl = c.get("cluster", {}) or {}
out = c.get("output", {}) or {}
ml = ((out.get("export_config", {}) or {}).get("mlflow", {})) or {}
# Resolve ${MLFLOW_TRACKING_URI} (set by modelopttools:eval-config, sourced from .env).
turi = os.path.expandvars(ml.get("tracking_uri") or "")
# Fall back to the canonical host if unset/unresolved or the broken mlflow-nemo-evaluator
# alias (its 308 redirect strips /api/... -> REST 405).
if (not turi) or turi.startswith("${") or ("mlflow-nemo-evaluator" in turi):
    turi = os.getenv("MLFLOW_TRACKING_URI") or "https://mlflow.frontier-evals.nvidia.com/"
def emit(k, v): print(f"{k}={shlex.quote(str(v))}")
emit("MLP_HOST", cl.get("hostname", "")); emit("MLP_USER", cl.get("username", ""))
emit("MLP_OUTDIR", out.get("dir", "")); emit("MLP_TURI", turi)
emit("MLP_EXP", ml.get("experiment_name", "") or ""); emit("MLP_DESC", ml.get("description", "") or "")
emit("MLP_TAGS", json.dumps(ml.get("tags") or {}))
PY
)" || { echo "ERROR: failed to parse $cfg" >&2; return 1; }
  local MLP_HOST MLP_USER MLP_OUTDIR MLP_TURI MLP_EXP MLP_DESC MLP_TAGS
  eval "$cfgvars"
  [[ -n "$MLP_HOST" && -n "$MLP_OUTDIR" ]] || { echo "ERROR: cluster.hostname / output.dir missing in $cfg" >&2; return 1; }

  local sshdest="${MLP_USER:+$MLP_USER@}$MLP_HOST" run="$MLP_OUTDIR/$rid"
  _log "Locating merged bundles under $run on $sshdest …"
  local -a benchdirs
  mapfile -t benchdirs < <(ssh -o BatchMode=yes "$sshdest" \
    "find '$run' -mindepth 2 -maxdepth 2 -name 'eval-*.json' -not -path '*/shard_*' -printf '%h\n' 2>/dev/null | sort -u")
  [[ ${#benchdirs[@]} -gt 0 ]] || { echo "ERROR: no merged eval-*.json under $run — run 'nel-next.sh eval merge -r $rid' first" >&2; return 1; }

  local staged; staged="$(mktemp -d "${TMPDIR:-/tmp}/nel-mlflow-push.XXXXXX")"
  trap 'rm -rf "$staged"' RETURN
  local -a localbundles=(); local b name
  for b in "${benchdirs[@]}"; do
    name="$(basename "$b")"; mkdir -p "$staged/$name"
    rsync -rt --no-perms --no-group -e "ssh -o BatchMode=yes" "$sshdest:$b/eval-*.json" "$staged/$name/" >&2
    localbundles+=("$staged/$name"); _log "staged bundle: $name"
  done

  local -a oargs=(-o "tracking_uri=$MLP_TURI" -o emit_traces=false)
  [[ -n "$MLP_EXP"  ]] && oargs+=(-o "experiment_name=$MLP_EXP")
  [[ -n "$MLP_DESC" ]] && oargs+=(-o "description=$MLP_DESC")
  [[ "$MLP_TAGS" != "{}" ]] && oargs+=(-o "tags=$MLP_TAGS")
  _log "Exporting ${#localbundles[@]} bundle(s) to MLflow: $MLP_TURI"
  MLFLOW_TRACKING_URI="$MLP_TURI" "$NEL_BIN" export "${localbundles[@]}" --dest mlflow "${oargs[@]}" "${extra[@]}"
}

case "${1:-}" in
  --setup-only|--which) _ensure_venv; echo "$NEL_BIN"; exit 0 ;;
  --version)            _ensure_venv; "$VENV_DIR/bin/python" -c 'import nemo_evaluator; print(nemo_evaluator.__version__)'; exit 0 ;;
  -h|--help)            awk '/^# nel-next\.sh/{p=1} /^set /{p=0} p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  mlflow-push)          _mlflow_push "${@:2}"; exit $? ;;
  "")                   echo "ERROR: no args. Try: nel-next.sh eval run <config.yaml> [--dry-run]  (or --help)" >&2; exit 2 ;;
esac

_ensure_venv
exec "$NEL_BIN" "$@"
