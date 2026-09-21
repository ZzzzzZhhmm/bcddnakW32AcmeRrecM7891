#!/usr/bin/env bash
set -euo pipefail
# One web-console Bash command. Uses local files only, no Git or package install.
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${WARM_PYTHON:-/mnt/afs/task3_2/L202500276_lwz/envs/warm-rmbench-eval/bin/python}"
SPEC="${1:-${CODE_DIR}/configs/nonreal/verify.json}"
OUTPUT_ROOT="${WARM_NONREAL_OUTPUT:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/nonreal72h_jobs}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WARM_REQUIRE_TORCH_TESTS=1
exec "${PYTHON_BIN}" "${CODE_DIR}/scripts/nonreal_job.py" \
  --spec "${SPEC}" --output-root "${OUTPUT_ROOT}" "${@:2}"
