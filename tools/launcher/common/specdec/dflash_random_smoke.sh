#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

if [ -z "${DRAFT_CKPT_DIR:-}" ]; then
  echo "ERROR: DRAFT_CKPT_DIR is required"
  exit 1
fi

# Build random checkpoint in exported-checkpoint-0 so vllm_smoke_test.sh can auto-detect.
python3 "${SCRIPT_DIR}/build_dflash_random_checkpoint.py" \
  --output "${DRAFT_CKPT_DIR}/exported-checkpoint-0"

# Ensure smoke script is executable in the container
chmod +x "${SCRIPT_DIR}/vllm_smoke_test.sh"

# Delegate to the standard smoke script (do not reimplement its logic).
"${SCRIPT_DIR}/vllm_smoke_test.sh"
