#!/bin/bash
# ACP: migrate official LIBERO FastWAM to Piper 7D, then fine-tune on the
# processed Piper pilot. This is FastWAM fine-tune, not formal WARM
# retrospection. LIBERO bank/stats/40-task context are not reused.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
# shellcheck source=scripts/real/_gpu_env.sh
source "${PROJECT_DIR}/scripts/real/_gpu_env.sh"
resolve_piper_gpus
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_DIR}/scripts/accelerate_configs/accelerate_zero1_ds.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

LIBERO_FASTWAM="${LIBERO_FASTWAM:-${PROJECT_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
PROCESSED="${PROCESSED:-${PROJECT_DIR}/real/piper/processed/pilot_v1}"
MIGRATED="${MIGRATED:-${PROCESSED}/libero_uncond_piper7d.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROCESSED}/finetune_from_libero}"
RUN_STEPS="${RUN_STEPS:-200}"

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

if [[ ! -f "${MIGRATED}" ]]; then
  "${PYTHON}" scripts/real/migrate_libero_fastwam_to_piper.py \
    --source "${LIBERO_FASTWAM}" \
    --output "${MIGRATED}"
fi

echo "Piper FastWAM finetune NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

"${ACCELERATE}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  scripts/real/smoke_train_piper.py \
  --skip-text-embeds \
  --init-checkpoint "${MIGRATED}" \
  --run-steps "${RUN_STEPS}" \
  --output-dir "${OUTPUT_DIR}"
