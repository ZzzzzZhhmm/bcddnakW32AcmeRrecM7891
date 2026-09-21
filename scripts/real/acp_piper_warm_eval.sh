#!/bin/bash
# ACP: Piper complete-WARM holdout eval (loss-only) on the 400-step smoke ckpt.
# Not a real-robot success test. Formal WARM eval does not run closed-loop infer.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
# shellcheck source=scripts/real/_gpu_env.sh
source "${PROJECT_DIR}/scripts/real/_gpu_env.sh"
# shellcheck source=scripts/real/_acp_log.sh
source "${PROJECT_DIR}/scripts/real/_acp_log.sh"
resolve_piper_gpus
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROCESSED="${PROCESSED:-${PROJECT_DIR}/real/piper/processed/pilot_v1}"
CHECKPOINT="${CHECKPOINT:-${PROCESSED}/warm_from_piper_fastwam/run_20260920_040507/checkpoints/weights/step_000400.pt}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROCESSED}/warm_from_piper_fastwam/eval_step000400_${RUN_ID}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_DIR}/scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"
piper_acp_begin_logs "${OUTPUT_DIR}/console.log"
trap 'rc=$?; piper_acp_finish "${rc}"; exit "${rc}"' EXIT
piper_resolve_conda_bins
piper_ensure_master_port

echo "Piper WARM holdout eval start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "checkpoint=${CHECKPOINT}"
echo "output=${OUTPUT_DIR} eval_num_samples=${EVAL_NUM_SAMPLES} NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_PORT=${MASTER_PORT} zero=1"
echo "PYTHON=${PYTHON} ACCELERATE=${ACCELERATE}"

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
piper_gpu_preflight

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
  --output-dir "${OUTPUT_DIR}"
