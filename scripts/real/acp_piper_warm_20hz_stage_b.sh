#!/bin/bash
# ACP: Stage B only. Full-epoch complete-WARM on packed 20Hz pilot_20hz
# (25 episodes, 4 tasks, context_dim=772) using finished Stage A FastWAM
# weights. Does not rerun Stage A, touch pilot_v1, the 10-episode v2 20Hz
# set, or 16.67Hz. ZeRO-1. WARM heads start random.
# This is not an attested formal experiment.
#
# GPU knobs (must match if both are set):
#   NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3   # default
#   NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1
#   NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
# shellcheck source=scripts/real/_gpu_env.sh
source "${PROJECT_DIR}/scripts/real/_gpu_env.sh"
# shellcheck source=scripts/real/_acp_log.sh
source "${PROJECT_DIR}/scripts/real/_acp_log.sh"
if [[ -z "${NUM_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=4
fi
resolve_piper_gpus
piper_clear_distributed_env
piper_export_offline_model_env
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROCESSED="${PROCESSED:-${PROJECT_DIR}/real/piper/processed/pilot_20hz}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_DIR}/real/piper/processed/pilot_v2/finetune_from_v1/run_20260920_194900/checkpoints/weights/step_012340.pt}"
WARM_EPOCHS="${WARM_EPOCHS:-20}"
TRAIN_WINDOWS="${TRAIN_WINDOWS:-9976}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-$(piper_epoch_steps "${TRAIN_WINDOWS}")}"
SAVE_EVERY="${SAVE_EVERY:-${STEPS_PER_EPOCH}}"
EVAL_EVERY="${EVAL_EVERY:-${STEPS_PER_EPOCH}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WARM_DIR="${WARM_DIR:-${PROCESSED}/warm_from_stage_a/run_${RUN_ID}}"
WARM_CONFIG="${WARM_CONFIG:-${PROJECT_DIR}/configs/real/piper_warm_20hz.local.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_DIR}/scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

cd "${PROJECT_DIR}"
mkdir -p "${WARM_DIR}"
piper_acp_begin_logs "${WARM_DIR}/console.log"
trap 'rc=$?; piper_acp_finish "${rc}"; exit "${rc}"' EXIT

piper_resolve_conda_bins
piper_ensure_master_port

echo "Piper 20Hz full-set Stage B complete-WARM ACP start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "processed=${PROCESSED}"
echo "NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_PORT=${MASTER_PORT} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "PYTHON=${PYTHON} ACCELERATE=${ACCELERATE}"
echo "base_checkpoint=${BASE_CHECKPOINT}"
echo "warm_dir=${WARM_DIR}"
echo "warm_epochs=${WARM_EPOCHS} steps_per_epoch=${STEPS_PER_EPOCH} train_windows=${TRAIN_WINDOWS}"
echo "save_every=${SAVE_EVERY} eval_every=${EVAL_EVERY} eval_num_samples=0 (full DEV) zero=1"

if [[ ! -f "${PROCESSED}/memory/COMPLETE.json" ]]; then
  echo "ERROR: 20Hz memory COMPLETE missing: ${PROCESSED}/memory/COMPLETE.json"
  exit 2
fi
if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "ERROR: Stage A FastWAM checkpoint not found: ${BASE_CHECKPOINT}"
  exit 2
fi
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: Accelerate ZeRO-1 config missing: ${ACCELERATE_CONFIG}"
  exit 2
fi
if [[ ! -f "${WARM_CONFIG}" ]]; then
  echo "ERROR: WARM config missing: ${WARM_CONFIG}"
  exit 2
fi

piper_gpu_preflight

CONTRACT_DIR="${CONTRACT_DIR:-${PROCESSED}/warm_contracts}"

echo "=== preparing source-run contracts (single process, like LIBERO) ==="
"${PYTHON}" "${PROJECT_DIR}/scripts/real/train_piper_warm.py" \
  --prepare-contracts \
  --config "${WARM_CONFIG}" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --output-dir "${WARM_DIR}"

if [[ ! -f "${CONTRACT_DIR}/.source_contracts.ready" ]]; then
  echo "ERROR: source-run contracts ready marker missing: ${CONTRACT_DIR}/.source_contracts.ready"
  exit 2
fi
if [[ ! -f "${CONTRACT_DIR}/train_source.json" || ! -f "${CONTRACT_DIR}/dev_source.json" ]]; then
  echo "ERROR: train/dev source-run contracts missing under ${CONTRACT_DIR}"
  exit 2
fi

echo "=== preflight Stage B (no 5B load) ==="
"${PYTHON}" "${PROJECT_DIR}/scripts/real/train_piper_warm.py" \
  --preflight \
  --config "${WARM_CONFIG}" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --num-epochs "${WARM_EPOCHS}" \
  --save-every "${SAVE_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${WARM_DIR}"

echo "=== launching Stage B ==="
echo "${ACCELERATE} launch --config_file ${ACCELERATE_CONFIG} --num_processes ${NUM_GPUS} --main_process_port ${MASTER_PORT} ${PROJECT_DIR}/scripts/real/train_piper_warm.py"

"${ACCELERATE}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  "${PROJECT_DIR}/scripts/real/train_piper_warm.py" \
  --config "${WARM_CONFIG}" \
  --base-checkpoint "${BASE_CHECKPOINT}" \
  --num-epochs "${WARM_EPOCHS}" \
  --save-every "${SAVE_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${WARM_DIR}"
