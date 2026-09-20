#!/bin/bash
# ACP: Piper complete-WARM holdout eval (loss-only) on the 400-step smoke ckpt.
# Not a real-robot success test. Formal WARM eval does not run closed-loop infer.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
# shellcheck source=scripts/real/_gpu_env.sh
source "${PROJECT_DIR}/scripts/real/_gpu_env.sh"
resolve_piper_gpus
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROCESSED="${PROCESSED:-${PROJECT_DIR}/real/piper/processed/pilot_v1}"
CHECKPOINT="${CHECKPOINT:-${PROCESSED}/warm_from_piper_fastwam/run_20260920_040507/checkpoints/weights/step_000400.pt}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MASTER_PORT="${MASTER_PORT:-29501}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROCESSED}/warm_from_piper_fastwam/eval_step000400_${RUN_ID}}"
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

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: WARM checkpoint not found: ${CHECKPOINT}"
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
echo "Piper WARM holdout eval start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${LOG}"
echo "checkpoint=${CHECKPOINT}" | tee -a "${LOG}"
echo "output=${OUTPUT_DIR} eval_num_samples=${EVAL_NUM_SAMPLES} NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} zero=1" | tee -a "${LOG}"

"${ACCELERATE}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  scripts/real/eval_piper_warm.py \
  --checkpoint "${CHECKPOINT}" \
  --eval-num-samples "${EVAL_NUM_SAMPLES}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG}"
