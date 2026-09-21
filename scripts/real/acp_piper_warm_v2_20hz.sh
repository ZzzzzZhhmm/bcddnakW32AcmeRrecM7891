#!/bin/bash
# ACP: full-epoch 20Hz pilot_v2 FastWAM, then full-epoch complete-WARM.
# Uses every train window (2465 / epoch). Does not touch pilot_v1 or 16.67Hz.
# Both stages use ZeRO-1. Set NUM_GPUS or CUDA_VISIBLE_DEVICES.
# WARM heads start random (context_dim=770). This is not an attested formal experiment.
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

PROCESSED="${PROCESSED:-${PROJECT_DIR}/real/piper/processed/pilot_v2}"
FASTWAM_INIT="${FASTWAM_INIT:-${PROJECT_DIR}/real/piper/processed/pilot_v1/finetune_from_libero/checkpoints/weights/step_000200.pt}"
FASTWAM_EPOCHS="${FASTWAM_EPOCHS:-20}"
WARM_EPOCHS="${WARM_EPOCHS:-20}"
TRAIN_WINDOWS="${TRAIN_WINDOWS:-2465}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-$(piper_epoch_steps "${TRAIN_WINDOWS}")}"
SAVE_EVERY="${SAVE_EVERY:-${STEPS_PER_EPOCH}}"
EVAL_EVERY="${EVAL_EVERY:-${STEPS_PER_EPOCH}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FASTWAM_DIR="${FASTWAM_DIR:-${PROCESSED}/finetune_from_v1/run_${RUN_ID}}"
WARM_DIR="${WARM_DIR:-${PROCESSED}/warm_from_piper_fastwam/run_${RUN_ID}}"
FASTWAM_CONFIG="${FASTWAM_CONFIG:-${PROJECT_DIR}/configs/real/piper_fastwam_v2_20hz.local.yaml}"
WARM_CONFIG="${WARM_CONFIG:-${PROJECT_DIR}/configs/real/piper_warm_v2_20hz.local.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_DIR}/scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

cd "${PROJECT_DIR}"
mkdir -p "${FASTWAM_DIR}" "${WARM_DIR}"
piper_acp_begin_logs "${WARM_DIR}/console.log"
trap 'rc=$?; piper_acp_finish "${rc}"; exit "${rc}"' EXIT
piper_resolve_conda_bins
piper_ensure_master_port

echo "Piper 20Hz full-epoch FastWAM+WARM ACP start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "processed=${PROCESSED}"
echo "NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_PORT=${MASTER_PORT} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "PYTHON=${PYTHON} ACCELERATE=${ACCELERATE}"
echo "fastwam_dir=${FASTWAM_DIR} warm_dir=${WARM_DIR}"
echo "fastwam_epochs=${FASTWAM_EPOCHS} warm_epochs=${WARM_EPOCHS} steps_per_epoch=${STEPS_PER_EPOCH} train_windows=${TRAIN_WINDOWS}"
echo "save_every=${SAVE_EVERY} eval_every=${EVAL_EVERY} eval_num_samples=0 (full DEV) zero=1"
if [[ ! -f "${PROCESSED}/memory/COMPLETE.json" ]]; then
  echo "ERROR: 20Hz memory COMPLETE missing: ${PROCESSED}/memory/COMPLETE.json"
  exit 2
fi
if [[ ! -f "${FASTWAM_INIT}" ]]; then
  echo "ERROR: Piper 7D FastWAM init not found: ${FASTWAM_INIT}"
  exit 2
fi
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: Accelerate ZeRO-1 config missing: ${ACCELERATE_CONFIG}"
  exit 2
fi
piper_gpu_preflight

launch() {
  "${ACCELERATE}" launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    --main_process_port "${MASTER_PORT}" \
    "$@"
}

echo "=== Stage A: full-epoch FastWAM on all 20Hz train windows ==="
launch \
  scripts/real/smoke_train_piper.py \
  --config "${FASTWAM_CONFIG}" \
  --init-checkpoint "${FASTWAM_INIT}" \
  --num-epochs "${FASTWAM_EPOCHS}" \
  --save-every "${SAVE_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --output-dir "${FASTWAM_DIR}"

FASTWAM_CKPT="$(find "${FASTWAM_DIR}/checkpoints/weights" -name 'step_*.pt' | sort | tail -n 1 || true)"
if [[ -z "${FASTWAM_CKPT}" || ! -f "${FASTWAM_CKPT}" ]]; then
  echo "ERROR: Stage A FastWAM checkpoint missing under ${FASTWAM_DIR}/checkpoints/weights"
  exit 2
fi
echo "Stage A FastWAM checkpoint: ${FASTWAM_CKPT}"

echo "=== preparing source-run contracts (single process, like LIBERO) ==="
"${PYTHON}" "${PROJECT_DIR}/scripts/real/train_piper_warm.py" \
  --prepare-contracts \
  --config "${WARM_CONFIG}" \
  --base-checkpoint "${FASTWAM_CKPT}" \
  --output-dir "${WARM_DIR}" \
  --overwrite-contracts

echo "=== Stage B: full-epoch complete WARM on the 20Hz bank ==="
launch \
  "${PROJECT_DIR}/scripts/real/train_piper_warm.py" \
  --config "${WARM_CONFIG}" \
  --base-checkpoint "${FASTWAM_CKPT}" \
  --num-epochs "${WARM_EPOCHS}" \
  --save-every "${SAVE_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${WARM_DIR}"
