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

# nel-check.sh — assert the `nel` on PATH is the validated launcher (Step 1).
#
# Presence is not enough: a base environment can already carry an older
# `nemo-evaluator-launcher` (directly, or via `nemo-evaluator-launcher-internal`,
# which ships its own launcher version). Running a baseline and a candidate on
# different launchers folds a harness change into the measured model delta, so
# this fails loudly instead of silently scoring on whatever is installed.
#
# Usage:
#   nel-check.sh              # assert PATH `nel` == validated version
#   nel-check.sh --version    # print the validated version
#   nel-check.sh --spec       # print the pip spec to install
#
# Set NEL_ALLOW_UNVALIDATED=1 to downgrade the mismatch to a warning (dev/canary
# only — never for scored runs). GDPVal has no such escape hatch: it goes through
# nel-gdpval.sh, which hard-pins the launcher. See references/launcher-version.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./nel-validated-version.sh
source "$SCRIPT_DIR/nel-validated-version.sh"

case "${1:-}" in
  -h|--help) awk '/^# nel-check\.sh/{p=1} /^set /{p=0} p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  --version) echo "$NEL_VALIDATED_VERSION"; exit 0 ;;
  --spec)    echo "$NEL_VALIDATED_SPEC"; exit 0 ;;
esac

_install_hint() {
  echo "  pip install '${NEL_VALIDATED_SPEC}'" >&2
  echo "Then re-run this check. If a stale launcher keeps winning, it is likely" >&2
  echo "pulled in by 'nemo-evaluator-launcher-internal' — uninstall or pin that too." >&2
}

command -v nel >/dev/null 2>&1 || {
  echo "ERROR: 'nel' not found on PATH. Install the validated launcher:" >&2
  _install_hint
  exit 1
}

# `nel --version` prints the version table on stdout and log lines on stderr.
# The table lists several packages; `nemo_evaluator_launcher` is the one that
# determines generated-Slurm and schema behavior.
version_table="$(nel --version 2>/dev/null || true)"
found="$(awk -F': ' '$1 == "nemo_evaluator_launcher" { print $2; exit }' <<<"$version_table")"

if [[ -z "$found" ]]; then
  echo "ERROR: could not read 'nemo_evaluator_launcher' from 'nel --version'." >&2
  echo "Got:" >&2
  echo "$version_table" >&2
  _install_hint
  exit 1
fi

if [[ "$found" != "$NEL_VALIDATED_VERSION" ]]; then
  if [[ "${NEL_ALLOW_UNVALIDATED:-}" == "1" ]]; then
    echo "WARNING: nel ${found} is NOT the validated ${NEL_VALIDATED_VERSION}." >&2
    echo "WARNING: NEL_ALLOW_UNVALIDATED=1 — dev/canary only. Do not report these" >&2
    echo "WARNING: scores, and never compare them against a ${NEL_VALIDATED_VERSION} baseline." >&2
    echo "nemo_evaluator_launcher: ${found} (UNVALIDATED)"
    exit 0
  fi
  echo "ERROR: 'nel' on PATH is ${found}, but the validated launcher is ${NEL_VALIDATED_VERSION}." >&2
  echo "Scoring on a different launcher makes the run non-comparable with runs on" >&2
  echo "${NEL_VALIDATED_VERSION}. Install the validated launcher:" >&2
  _install_hint
  exit 1
fi

# Record this line with the scores (Step 9) — it is the harness half of the delta.
echo "nemo_evaluator_launcher: ${found} (validated)"
