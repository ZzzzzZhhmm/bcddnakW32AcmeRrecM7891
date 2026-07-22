#!/usr/bin/env bash
set -euo pipefail

# Build the one-task, one-checkpoint online contract used by an RMBench SOTA
# specialist. The generic builder still performs every artifact/provenance
# check; this wrapper only selects the seed-3407 matrix and task profile.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${WARM_RMBENCH_TASK:-}" ]]; then
  printf 'ERROR: WARM_RMBENCH_TASK is required\n' >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export WARM_RMBENCH_MATRIX="${WARM_RMBENCH_MATRIX:-${PROJECT_ROOT}/configs/ablation/rmbench_sota_matrix.json}"
export WARM_EVALUATION_NAMESPACE="${WARM_EVALUATION_NAMESPACE:-warm-rmbench-sota-v1}"
exec bash "${SCRIPT_DIR}/build_warm_rmbench_contract_bundle_server.sh"
