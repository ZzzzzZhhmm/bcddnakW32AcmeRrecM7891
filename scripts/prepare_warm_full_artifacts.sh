#!/usr/bin/env bash
set -euo pipefail

# Build the immutable LIBERO artifacts consumed by complete WARM. This script
# runs on the Linux GPU server, never on the local Windows development machine.
# Large artifacts remain outside Git under WARM_ARTIFACT_ROOT.

required_env=(
  WARM_ARTIFACT_ROOT
  LIBERO_DATA_ROOT
  FASTWAM_BASE_CHECKPOINT
  WARM_DINO_CHECKPOINT
  WARM_DINO_REVISION
  WARM_VAE_CHECKPOINT
)
for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "error: ${name} must be set" >&2
    exit 2
  fi
done

if [[ ! "${WARM_DINO_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: WARM_DINO_REVISION must be the pinned 40-character Hub commit" >&2
  exit 2
fi
if [[ ! -f "${FASTWAM_BASE_CHECKPOINT}" ]]; then
  echo "error: base checkpoint not found: ${FASTWAM_BASE_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -e "${WARM_DINO_CHECKPOINT}" ]]; then
  echo "error: DINO checkpoint not found: ${WARM_DINO_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -e "${WARM_VAE_CHECKPOINT}" ]]; then
  echo "error: VAE checkpoint not found: ${WARM_VAE_CHECKPOINT}" >&2
  exit 2
fi

DATA_ROOTS=(
  "${LIBERO_DATA_ROOT}/libero_spatial_no_noops_lerobot"
  "${LIBERO_DATA_ROOT}/libero_object_no_noops_lerobot"
  "${LIBERO_DATA_ROOT}/libero_goal_no_noops_lerobot"
  "${LIBERO_DATA_ROOT}/libero_10_no_noops_lerobot"
)
DATASET_ARGS=()
for root in "${DATA_ROOTS[@]}"; do
  if [[ ! -d "${root}" ]]; then
    echo "error: LIBERO dataset root not found: ${root}" >&2
    exit 2
  fi
  DATASET_ARGS+=(--dataset-root "${root}")
done

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
CATALOG="${M1}/libero_catalog.json"
AUDIT="${M1}/libero_audit.json"
FEATURES="${M1}/features"
BANK="${M1}/banks/hybrid_h32"
TRAIN_CANDIDATES="${M2}/candidates/hybrid_h32_train_k32"
DEV_CANDIDATES="${M2}/candidates/hybrid_h32_dev_k32"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"

for output in \
  "${CATALOG}" "${AUDIT}" "${M1}/train_stats" "${FEATURES}" "${BANK}" \
  "${M1}/oracle/hybrid_h32.json" \
  "${TRAIN_CANDIDATES}" "${DEV_CANDIDATES}" \
  "${TRAIN_CONTRACT}" "${DEV_CONTRACT}"; do
  if [[ -e "${output}" ]]; then
    echo "error: immutable output already exists: ${output}" >&2
    exit 2
  fi
done

mkdir -p "${M1}/banks" "${M1}/oracle" "${M2}/candidates" "${M2}/contracts"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

python scripts/build_warm_episode_catalog.py \
  "${DATASET_ARGS[@]}" \
  --dev-per-task 5 \
  --seed 20260713 \
  --output "${CATALOG}"

python scripts/audit_warm_lerobot.py \
  --catalog "${CATALOG}" \
  "${DATASET_ARGS[@]}" \
  --hash-episode-tables \
  --output "${AUDIT}"

python scripts/compute_warm_train_stats.py \
  --catalog "${CATALOG}" \
  --audit-report "${AUDIT}" \
  "${DATASET_ARGS[@]}" \
  --data-config configs/data/libero_2cam.yaml \
  --output "${M1}/train_stats"

python scripts/precompute_warm_features.py \
  --data-config configs/data/libero_2cam.yaml \
  --catalog "${CATALOG}" \
  --audit-report "${AUDIT}" \
  "${DATASET_ARGS[@]}" \
  --dataset-stats "${M1}/train_stats/dataset_stats.json" \
  --dataset-stats-manifest "${M1}/train_stats/train_stats_manifest.json" \
  --dino-checkpoint "${WARM_DINO_CHECKPOINT}" \
  --dino-revision "${WARM_DINO_REVISION}" \
  --include-vae \
  --vae-checkpoint "${WARM_VAE_CHECKPOINT}" \
  --device "${WARM_PRECOMPUTE_DEVICE:-cuda}" \
  --dtype "${WARM_PRECOMPUTE_DTYPE:-bfloat16}" \
  --dino-batch-size "${WARM_DINO_BATCH_SIZE:-64}" \
  --vae-batch-size "${WARM_VAE_BATCH_SIZE:-4}" \
  --output "${FEATURES}"

python scripts/build_warm_event_bank.py \
  --feature-list "${FEATURES}/train_features.list" \
  --catalog "${CATALOG}" \
  --audit-report "${AUDIT}" \
  --output "${BANK}" \
  --summary "${M1}/banks/hybrid_h32.summary.json" \
  --normalizer-contract "${FEATURES}/contracts/normalizer_contract.json" \
  --encoder-contract "${FEATURES}/contracts/encoder_contract.json" \
  --camera-contract "${FEATURES}/contracts/camera_contract.json" \
  --action-horizon 32 \
  --start-mode hybrid

python scripts/evaluate_warm_oracle.py \
  --bank "${BANK}" \
  --feature-list "${FEATURES}/dev_features.list" \
  --catalog "${CATALOG}" \
  --audit-report "${AUDIT}" \
  --output "${M1}/oracle/hybrid_h32.json" \
  --query-stride 4 \
  --top-k 1,4,8,16,32 \
  --arm-loss mse

python scripts/build_warm_candidate_cache.py \
  --bank "${BANK}" \
  --catalog "${CATALOG}" \
  --audit-report "${AUDIT}" \
  --feature-list "${FEATURES}/train_features.list" \
  --output "${TRAIN_CANDIDATES}" \
  --query-split train \
  --query-stride 1 \
  --top-k 32 \
  --summary "${M2}/candidates/hybrid_h32_train_k32.summary.json"

python scripts/build_warm_candidate_cache.py \
  --bank "${BANK}" \
  --catalog "${CATALOG}" \
  --audit-report "${AUDIT}" \
  --feature-list "${FEATURES}/dev_features.list" \
  --output "${DEV_CANDIDATES}" \
  --query-split dev \
  --query-stride 1 \
  --top-k 32 \
  --summary "${M2}/candidates/hybrid_h32_dev_k32.summary.json"

python scripts/build_warm_source_run_contract.py \
  --bank "${BANK}" \
  --candidate-cache "${TRAIN_CANDIDATES}" \
  --base-checkpoint "${FASTWAM_BASE_CHECKPOINT}" \
  --output "${TRAIN_CONTRACT}" \
  --query-split train \
  --expected-action-horizon 32 \
  --expected-action-dim 7

python scripts/build_warm_source_run_contract.py \
  --bank "${BANK}" \
  --candidate-cache "${DEV_CANDIDATES}" \
  --base-checkpoint "${FASTWAM_BASE_CHECKPOINT}" \
  --output "${DEV_CONTRACT}" \
  --query-split dev \
  --expected-action-horizon 32 \
  --expected-action-dim 7

printf '%s\n' \
  "WARM full artifacts published without overwrite:" \
  "  train contract: ${TRAIN_CONTRACT}" \
  "  dev contract:   ${DEV_CONTRACT}" \
  "  train features: ${FEATURES}/train_features.list" \
  "  dev features:   ${FEATURES}/dev_features.list"
