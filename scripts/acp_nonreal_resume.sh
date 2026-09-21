#!/usr/bin/env bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="${1:?Usage: bash scripts/acp_nonreal_resume.sh /absolute/plan.json}"
PY="${WARM_PYTHON:-/mnt/afs/task3_2/L202500276_lwz/envs/warm-rmbench-eval/bin/python}"
export PATH="$(dirname -- "$PY"):$PATH"
export PYTHONPATH="$CODE/src:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM/checkpoints}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Respect ACP-assigned visibility; never remap to GPUs outside the allocation.
source "$CODE/scripts/warm_server_common.sh"
warm_configure_job_local_caches "/tmp/warm-nonreal-resume-${USER:-user}-${HOSTNAME:-node}-$$"
cd "$CODE"
set -o pipefail
"$PY" "$CODE/scripts/nonreal_resume.py" run --plan "$PLAN" --port "${MASTER_PORT:-29622}" \
  2>&1 | tee -a "$(dirname -- "$PLAN")/launcher.log"
