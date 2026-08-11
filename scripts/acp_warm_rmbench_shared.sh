#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WARM_RMBENCH_STAGE=shared
unset WARM_RMBENCH_SPECIALIST_TASK
exec bash "${SCRIPT_DIR}/acp_warm_rmbench_specialist.sh" "$@"
