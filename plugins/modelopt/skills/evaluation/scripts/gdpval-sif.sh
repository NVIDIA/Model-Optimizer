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

# gdpval-sif.sh — ensure the GDPVal Stirrup Apptainer SIF exists on THIS cluster.
#
# Build-if-absent, reuse-if-present. Self-contained: the SIF is built and reused
# on the TARGET cluster's own filesystem — this NEVER copies a SIF from another
# cluster. Idempotent, so it's safe to run before every `nel run`; a subsequent
# run reuses the built SIF instantly.
#
# Usage:
#   "$SKILL_DIR/scripts/gdpval-sif.sh" [<sif-dir-or-file>] [--commit <sha>] [--force|--check]
#     <sif-dir-or-file>  Persistent path on the target cluster's shared FS.
#                        DEFAULTS to $GDPVAL_SIF_DIR (from .env) when omitted. A
#                        directory -> <dir>/$GDPVAL_SIF_NAME (default python-3.13.gdpval.sif,
#                        matching the example config); a *.sif path
#                        is used verbatim. Bind-mount this SAME dir into the eval
#                        container at /gdpval/sif (see recipes/examples/gym/).
#     --commit <sha>     NeMo Gym commit whose gdpval.def to build. Keep in sync
#                        with the config's install_on_the_fly.commit.
#     --force            Rebuild even if the SIF already exists.
#     --check            Verify-only preflight: exit 0 if the expected SIF exists,
#                        is non-trivial in size, and is a readable SIF. NOTE: it
#                        inspects the filesystem it RUNS ON — run it on the cluster
#                        (srun/ssh), not the submitting box, or you validate the
#                        wrong filesystem.
#                        Never builds.
#                        Use before `nel run` — NEL's mount validation is `test -d`
#                        and cannot see a missing/misnamed SIF file.
#
# Requires `apptainer` (or `singularity`) on PATH with unprivileged/fakeroot
# build support, plus network egress to GitHub/base image. Run on a node that has
# it — a login node, or (preferred for the ~30-min build) the CPU partition:
#   srun -p cpu -t 01:00:00 --pty \
#     "$SKILL_DIR/scripts/gdpval-sif.sh" /lustre/<...>/gdpval/sif
#
# Env overrides: GDPVAL_GYM_COMMIT, GDPVAL_SIF_NAME, APPTAINER_BIN.
set -euo pipefail

# Keep GDPVAL_GYM_COMMIT in sync with install_on_the_fly.commit in the config.
GDPVAL_GYM_COMMIT="${GDPVAL_GYM_COMMIT:-dd41196f620f2af99947d776cbe5da9439d2a08d}"  # pragma: allowlist secret
GDPVAL_SIF_NAME="${GDPVAL_SIF_NAME:-python-3.13.gdpval.sif}"
APPTAINER_BIN="${APPTAINER_BIN:-}"

_log() { printf '\033[2m  %s\033[0m\n' "$*" >&2; }
_die() { printf '\033[31mgdpval-sif: %s\033[0m\n' "$*" >&2; exit 1; }
_usage() { sed -n '/^# gdpval-sif\.sh/,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'; }

# --- parse args ---
target=""; force=0; check=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit) GDPVAL_GYM_COMMIT="${2:?--commit needs a value}"; shift 2 ;;
    --force)  force=1; shift ;;
    --check)  check=1; shift ;;
    -h|--help) _usage; exit 0 ;;
    -*) _die "unknown flag: $1 (see --help)" ;;
    *) [[ -z "$target" ]] || _die "unexpected extra arg: $1"; target="$1"; shift ;;
  esac
done
# Default to $GDPVAL_SIF_DIR (.env) when no path is given, so agents run it hands-free.
target="${target:-${GDPVAL_SIF_DIR:-}}"
[[ -n "$target" ]] || { _usage; _die "no path given and GDPVAL_SIF_DIR is unset — pass a dir or set GDPVAL_SIF_DIR (see recipes/env.example)"; }

# --- resolve dir vs *.sif ---
if [[ "$target" == *.sif ]]; then
  sif="$target"; sif_dir="$(dirname "$target")"
else
  sif_dir="$target"; sif="$sif_dir/$GDPVAL_SIF_NAME"
fi
# --- verify-only mode (preflight) ---
# NEL's submit-time mount validation runs `test -d`, so it only proves the SIF *dir*
# exists — a dir holding the WRONG sif name (e.g. python-3.12 after a gym bump to a
# 3.13 def) passes validation, and the Stirrup agent then SILENTLY falls back to
# non-sandboxed exec. Run this before submitting to fail loudly instead.
if [[ "$check" -eq 1 ]]; then
  if [[ -f "$sif" ]]; then
    # -f alone would pass on a truncated or 0-byte file (e.g. an interrupted copy).
    # A real GDPVal SIF is ~1-4 GB; anything under 100 MB is not one.
    _sz=$(stat -c %s "$sif" 2>/dev/null || echo 0)
    if [[ "$_sz" -lt 104857600 ]]; then
      printf '\033[31mgdpval-sif: %s exists but is only %s bytes — truncated/incomplete\033[0m\n' "$sif" "$_sz" >&2
      echo "  Rebuild with: $0 --force ${sif_dir}" >&2
      exit 1
    fi
    if command -v apptainer >/dev/null 2>&1 && ! apptainer inspect "$sif" >/dev/null 2>&1; then
      printf '\033[31mgdpval-sif: %s is not a readable SIF (apptainer inspect failed)\033[0m\n' "$sif" >&2
      exit 1
    fi
    _log "SIF present: $sif ($(du -h "$sif" 2>/dev/null | cut -f1))"
    echo "$sif"; exit 0
  fi
  printf '\033[31mgdpval-sif: MISSING expected SIF: %s\033[0m\n' "$sif" >&2
  if [[ -d "$sif_dir" ]]; then
    echo "  dir exists but does not contain it; found:" >&2
    if ls -1 "$sif_dir"/*.sif >/dev/null 2>&1; then ls -1 "$sif_dir"/*.sif | sed 's/^/    /' >&2
    else echo "    (no .sif files)" >&2; fi
  fi
  echo "  Build it with: $0 ${sif_dir}   (or --commit <gym-sha> for a different def)" >&2
  exit 1
fi

mkdir -p "$sif_dir" || _die "cannot create SIF dir: $sif_dir"

# --- reuse if present ---
if [[ -f "$sif" && "$force" -eq 0 ]]; then
  _log "reusing existing SIF (no rebuild): $sif"
  echo "$sif"; exit 0
fi

# --- locate apptainer/singularity ---
if [[ -z "$APPTAINER_BIN" ]]; then
  APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
fi
[[ -n "$APPTAINER_BIN" ]] || _die "apptainer/singularity not found on PATH. Run on a node that has it \
(e.g. 'module load apptainer', or inside the eval image). This script does NOT copy a SIF from another cluster."

def_url="https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/${GDPVAL_GYM_COMMIT}/responses_api_agents/stirrup_agent/containers/gdpval.def"
tmp="${sif_dir}/.build.$$.${GDPVAL_SIF_NAME}"
def_local="${sif_dir}/.gdpval.$$.def"
lock="${sif_dir}/.gdpval-sif.lock"

# --- build under a flock (double-checked) so concurrent runs don't double-build ---
exec 9>"$lock" || _die "cannot open lock file: $lock"
_log "acquiring build lock ($lock) ..."
flock 9
# Re-check inside the lock: another builder may have finished while we waited.
if [[ -f "$sif" && "$force" -eq 0 ]]; then
  _log "another builder produced it: $sif"
  echo "$sif"; exit 0
fi

_log "building GDPVal SIF (this can take ~20-40 min)"
_log "  gym commit: ${GDPVAL_GYM_COMMIT}"
_log "  def:        ${def_url}"
_log "  dest:       ${sif}"
# Leave no temp artefacts if we are killed or exit early. $tmp is renamed on success,
# so this only ever removes leftovers.
trap 'rm -f "$tmp" "$def_local"' EXIT
rm -f "$tmp" "$def_local"
# apptainer build cannot take a remote def URL as its source — fetch the def to a
# local file first, then build from it.
if command -v curl >/dev/null 2>&1; then curl -fsSL "$def_url" -o "$def_local"
else wget -qO "$def_local" "$def_url"; fi
[ -s "$def_local" ] || { rm -f "$def_local"; _die "failed to download def from $def_url"; }
# Prefer --fakeroot (needs an /etc/subuid entry for the build user); fall back to an
# unprivileged build where fakeroot is unavailable.
if "$APPTAINER_BIN" build --fakeroot "$tmp" "$def_local"; then
  :
# A failed --fakeroot attempt can leave a partial $tmp behind, and apptainer refuses an
# existing destination — clear it or the unprivileged fallback can never succeed.
elif rm -f "$tmp" && "$APPTAINER_BIN" build "$tmp" "$def_local"; then
  _log "built without --fakeroot (unprivileged mode)"
else
  rm -f "$tmp" "$def_local"
  _die "apptainer build failed (see output above)."
fi
rm -f "$def_local"

# Atomic publish: a partial build never looks complete.
mv -f "$tmp" "$sif" || { rm -f "$tmp"; _die "failed to move built SIF into place: $sif"; }
_log "done: $sif"
echo "$sif"
