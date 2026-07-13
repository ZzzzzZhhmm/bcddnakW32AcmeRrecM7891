#!/usr/bin/env bash
set -euo pipefail

# Launch complete three-camera RMBench WARM training from immutable artifacts.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/warm_server_common.sh
source "${SCRIPT_DIR}/warm_server_common.sh"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
RMBENCH_DATASET_REVISION="${RMBENCH_DATASET_REVISION:-855e90e1213d150bf4889130e83398f107314681}"
warm_require_env \
  WARM_ARTIFACT_ROOT \
  RMBENCH_LEROBOT_ROOT \
  RMBENCH_TEXT_CACHE \
  FASTWAM_BASE_CHECKPOINT \
  WARM_TRAIN_OUTPUT
warm_require_sha40 "${RMBENCH_DATASET_REVISION}" "RMBENCH_DATASET_REVISION"
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_require_private_checkout
warm_configure_offline_logging
warm_refuse_existing_output "${WARM_TRAIN_OUTPUT}"

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
TRAIN_CACHE="${M2}/candidates/hybrid_h32_train_k32"
DEV_CACHE="${M2}/candidates/hybrid_h32_dev_k32"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"
CATALOG="${M1}/rmbench_catalog.json"
AUDIT="${M1}/rmbench_audit.json"

warm_require_file_or_directory \
  "${FASTWAM_BASE_CHECKPOINT}" \
  "${RMBENCH_LEROBOT_ROOT}" \
  "${RMBENCH_TEXT_CACHE}" \
  "${M1}/rmbench_conversion_manifest.json" \
  "${CATALOG}" \
  "${AUDIT}" \
  "${M1}/train_stats/train_stats_manifest.json" \
  "${TRAIN_STATS}" \
  "${M1}/features/train_features.list" \
  "${M1}/features/dev_features.list" \
  "${M1}/banks/hybrid_h32" \
  "${TRAIN_CACHE}" \
  "${DEV_CACHE}" \
  "${TRAIN_CONTRACT}" \
  "${DEV_CONTRACT}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export RMBENCH_LEROBOT_ROOT
export RMBENCH_DATASET_STATS="${TRAIN_STATS}"
export RMBENCH_TEXT_CACHE

# Revalidate revision/catalog identity.  Full MP4 hashes were already bound by
# the production audit; this fast preflight still checks every inventory path
# and size before allocating GPUs.
python scripts/validate_rmbench_conversion.py \
  --dataset-root "${RMBENCH_LEROBOT_ROOT}" \
  --source-revision "${RMBENCH_DATASET_REVISION}" \
  --rmbench-code-revision "${RMBENCH_CODE_REVISION}" \
  --skip-artifact-byte-hashes

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
WANDB_ENABLED="${WARM_WANDB_ENABLED:-false}"
if [[ "${WANDB_ENABLED}" != "true" && "${WANDB_ENABLED}" != "false" ]]; then
  warm_die "WARM_WANDB_ENABLED must be true or false"
fi

# Extra arguments are limited to optimizer/runtime knobs.  Because Hydra uses
# last-write-wins semantics, allowing a second data/model/output override here
# would bypass the immutable artifact checks above.
for override in "$@"; do
  key="${override%%=*}"
  key="${key#+}"
  key="${key#~}"
  case "${key}" in
    task|data|data.*|model|model.*|output_dir|resume|wandb.enabled|wandb.mode|--config-name|--config-name*)
      warm_die "protected formal-training override is not allowed: ${override}"
      ;;
  esac
done

exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=rmbench_warm_3cam384_1e-4 \
  "output_dir=${WARM_TRAIN_OUTPUT}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.mode=${WANDB_MODE}" \
  "model.run_contract_path=${TRAIN_CONTRACT}" \
  "model.validation_run_contract_path=${DEV_CONTRACT}" \
  "model.base_checkpoint_path=${FASTWAM_BASE_CHECKPOINT}" \
  "data.train.dataset_dirs=[${RMBENCH_LEROBOT_ROOT}]" \
  "data.train.text_embedding_cache_dir=${RMBENCH_TEXT_CACHE}" \
  "data.warm_candidates.train.bank_directory=${M1}/banks/hybrid_h32" \
  "data.warm_candidates.train.candidate_directory=${TRAIN_CACHE}" \
  "data.warm_candidates.train.catalog_path=${CATALOG}" \
  "data.warm_candidates.train.normalization_stats_path=${TRAIN_STATS}" \
  "data.warm_candidates.train.audit_report_path=${AUDIT}" \
  "data.warm_candidates.train.retrospective_feature_list=${M1}/features/train_features.list" \
  "data.val.dataset_dirs=[${RMBENCH_LEROBOT_ROOT}]" \
  "data.val.text_embedding_cache_dir=${RMBENCH_TEXT_CACHE}" \
  "data.warm_candidates.val.bank_directory=${M1}/banks/hybrid_h32" \
  "data.warm_candidates.val.candidate_directory=${DEV_CACHE}" \
  "data.warm_candidates.val.catalog_path=${CATALOG}" \
  "data.warm_candidates.val.normalization_stats_path=${TRAIN_STATS}" \
  "data.warm_candidates.val.audit_report_path=${AUDIT}" \
  "data.warm_candidates.val.retrospective_feature_list=${M1}/features/dev_features.list" \
  "$@"
