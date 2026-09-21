#!/usr/bin/env bash
set -euo pipefail

# Launch complete WARM training from one immutable artifact tree produced by
# prepare_warm_full_artifacts.sh. This requires a Linux CUDA server.

: "${WARM_ARTIFACT_ROOT:?Set WARM_ARTIFACT_ROOT}"
: "${FASTWAM_BASE_CHECKPOINT:?Set FASTWAM_BASE_CHECKPOINT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
TRAIN_CACHE="${M2}/candidates/hybrid_h32_train_k32"
DEV_CACHE="${M2}/candidates/hybrid_h32_dev_k32"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"

required_paths=(
  "${FASTWAM_BASE_CHECKPOINT}"
  "${M1}/banks/hybrid_h32"
  "${TRAIN_CACHE}"
  "${DEV_CACHE}"
  "${TRAIN_CONTRACT}"
  "${DEV_CONTRACT}"
  "${M1}/features/train_features.list"
  "${M1}/features/dev_features.list"
  "${M1}/libero_catalog.json"
  "${M1}/libero_audit.json"
  "${TRAIN_STATS}"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "error: required immutable artifact not found: ${path}" >&2
    exit 2
  fi
done
exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=libero_warm_2cam224_1e-4 \
  "model.run_contract_path=${TRAIN_CONTRACT}" \
  "model.validation_run_contract_path=${DEV_CONTRACT}" \
  "model.base_checkpoint_path=${FASTWAM_BASE_CHECKPOINT}" \
  "data.warm_candidates.train.bank_directory=${M1}/banks/hybrid_h32" \
  "data.warm_candidates.train.candidate_directory=${TRAIN_CACHE}" \
  "data.warm_candidates.train.catalog_path=${M1}/libero_catalog.json" \
  "data.warm_candidates.train.normalization_stats_path=${TRAIN_STATS}" \
  "data.warm_candidates.train.audit_report_path=${M1}/libero_audit.json" \
  "data.warm_candidates.train.retrospective_feature_list=${M1}/features/train_features.list" \
  "data.warm_candidates.val.bank_directory=${M1}/banks/hybrid_h32" \
  "data.warm_candidates.val.candidate_directory=${DEV_CACHE}" \
  "data.warm_candidates.val.catalog_path=${M1}/libero_catalog.json" \
  "data.warm_candidates.val.normalization_stats_path=${TRAIN_STATS}" \
  "data.warm_candidates.val.audit_report_path=${M1}/libero_audit.json" \
  "data.warm_candidates.val.retrospective_feature_list=${M1}/features/dev_features.list" \
  "$@"
