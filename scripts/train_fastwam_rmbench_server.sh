#!/usr/bin/env bash
set -euo pipefail

# Train the exact no-memory FastWAM comparison on the same converted RMBench
# demonstrations, episode split, action statistics, and processor as WARM.

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
  FASTWAM_RMBENCH_TRAIN_OUTPUT
warm_require_sha40 "${RMBENCH_DATASET_REVISION}" "RMBENCH_DATASET_REVISION"
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_require_private_checkout
warm_configure_offline_logging
warm_refuse_existing_output "${FASTWAM_RMBENCH_TRAIN_OUTPUT}"

M1="${WARM_ARTIFACT_ROOT}/m1"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"
CATALOG="${M1}/rmbench_catalog.json"
AUDIT="${M1}/rmbench_audit.json"
warm_require_file_or_directory \
  "${FASTWAM_BASE_CHECKPOINT}" \
  "${RMBENCH_LEROBOT_ROOT}" \
  "${RMBENCH_TEXT_CACHE}" \
  "${M1}/rmbench_conversion_manifest.json" \
  "${M1}/train_stats/train_stats_manifest.json" \
  "${TRAIN_STATS}" \
  "${CATALOG}" \
  "${AUDIT}"
if [[ ! -f "${FASTWAM_BASE_CHECKPOINT}" ]]; then
  warm_die "FASTWAM_BASE_CHECKPOINT must be a weight file, not a trainer-state directory"
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export RMBENCH_LEROBOT_ROOT
export RMBENCH_DATASET_STATS="${TRAIN_STATS}"
export RMBENCH_EPISODE_CATALOG="${CATALOG}"
export RMBENCH_TEXT_CACHE

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
for override in "$@"; do
  key="${override%%=*}"
  key="${key#+}"
  key="${key#~}"
  case "${key}" in
    task|data|data.*|model|model.*|output_dir|resume|wandb.enabled|wandb.mode|--config-name|--config-name*)
      warm_die "protected formal-baseline override is not allowed: ${override}"
      ;;
  esac
done

exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  task=rmbench_fastwam_3cam384_1e-4 \
  "output_dir=${FASTWAM_RMBENCH_TRAIN_OUTPUT}" \
  "resume=${FASTWAM_BASE_CHECKPOINT}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.mode=${WANDB_MODE}" \
  "data.train.dataset_dirs=[${RMBENCH_LEROBOT_ROOT}]" \
  "data.train.pretrained_norm_stats=${TRAIN_STATS}" \
  "data.train.episode_catalog_path=${CATALOG}" \
  "data.val.dataset_dirs=[${RMBENCH_LEROBOT_ROOT}]" \
  "data.val.pretrained_norm_stats=${TRAIN_STATS}" \
  "data.val.episode_catalog_path=${CATALOG}" \
  "$@"
