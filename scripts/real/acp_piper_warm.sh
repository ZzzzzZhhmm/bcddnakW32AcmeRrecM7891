#!/bin/bash
# ACP: Piper full-WARM smoke on the Phase-1 7D FastWAM base and Piper bank.
# WARM heads start random (context_dim=770). LIBERO 40-task WARM weights are
# not loaded. This is not an attested formal experiment.
#
# Complete WARM must use ZeRO-1. A raw `python` launch (same as Phase-1
# FastWAM) loads the 5B model and then dies on the first video/layer-9
# backward before writing metrics. Official LIBERO full-WARM uses
# scripts/train_zero1.sh for the same reason.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
# shellcheck source=scripts/real/_gpu_env.sh
source "${PROJECT_DIR}/scripts/real/_gpu_env.sh"
resolve_piper_gpus
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROCESSED="${PROCESSED:-${PROJECT_DIR}/real/piper/processed/pilot_v1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROCESSED}/finetune_from_libero/checkpoints/weights/step_000200.pt}"
RUN_STEPS="${RUN_STEPS:-80}"
SAVE_EVERY="${SAVE_EVERY:-${RUN_STEPS}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROCESSED}/warm_from_piper_fastwam/run_${RUN_ID}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_DIR}/scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

cd "${PROJECT_DIR}"
if [[ -x "${CONDA_ENV_DIR}/bin/python" ]]; then
  PYTHON="${CONDA_ENV_DIR}/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi
if [[ -x "${CONDA_ENV_DIR}/bin/accelerate" ]]; then
  ACCELERATE="${CONDA_ENV_DIR}/bin/accelerate"
else
  ACCELERATE="${ACCELERATE:-accelerate}"
fi

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "ERROR: Phase-1 Piper FastWAM base not found: ${BASE_CHECKPOINT}"
  exit 2
fi
if [[ ! -f "${PROCESSED}/memory/COMPLETE.json" ]]; then
  echo "ERROR: Piper memory COMPLETE missing: ${PROCESSED}/memory/COMPLETE.json"
  exit 2
fi
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: Accelerate ZeRO-1 config missing: ${ACCELERATE_CONFIG}"
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/console.log"
echo "Piper WARM ACP start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${LOG}"
echo "output=${OUTPUT_DIR} run_steps=${RUN_STEPS} save_every=${SAVE_EVERY} NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} zero=1" | tee -a "${LOG}"

"${ACCELERATE}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  scripts/real/train_piper_warm.py \
  --run-steps "${RUN_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG}"
