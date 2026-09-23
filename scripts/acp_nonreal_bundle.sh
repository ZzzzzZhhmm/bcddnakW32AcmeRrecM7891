#!/usr/bin/env bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="${1:-$CODE/../bundle_plan.json}"
PY="${WARM_PYTHON:-/mnt/afs/task3_2/L202500276_lwz/envs/warm-rmbench-eval/bin/python}"
[[ -f "$PLAN" ]] || { echo "Missing bundle plan: $PLAN" >&2; exit 2; }
export PATH="$(dirname -- "$PY"):$PATH"
export PYTHONPATH="$CODE/src:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export DIFFSYNTH_SKIP_DOWNLOAD=true TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Preserve the allocation's visible GPU list. Post-training probes use its cuda:0.
source "$CODE/scripts/warm_server_common.sh"
warm_configure_job_local_caches "/tmp/warm-nonreal-bundle-${USER:-user}-${HOSTNAME:-node}-$$"
cd "$CODE"
"$PY" "$CODE/scripts/nonreal_bundle.py" --plan "$PLAN" 2>&1 | tee -a "$(dirname -- "$PLAN")/bundle.launcher.log"
