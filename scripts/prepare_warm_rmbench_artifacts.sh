#!/usr/bin/env bash
set -euo pipefail

# Convert the pinned official RMBench demonstrations and publish the immutable
# three-camera H=32 artifacts consumed by WARM.  This is a Linux GPU-server job.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/warm_server_common.sh
source "${SCRIPT_DIR}/warm_server_common.sh"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
RMBENCH_DATASET_REVISION="${RMBENCH_DATASET_REVISION:-855e90e1213d150bf4889130e83398f107314681}"
RMBENCH_SOURCE_REVISION="${RMBENCH_SOURCE_REVISION:-${RMBENCH_DATASET_REVISION}}"
WARM_RMBENCH_DATA_PROFILE="${WARM_RMBENCH_DATA_PROFILE:-official50-dev45}"
RMBENCH_SOURCE_DATASET="${RMBENCH_SOURCE_DATASET:-TianxingChen/RMBench}"
RMBENCH_DATASET_ID="${RMBENCH_DATASET_ID:-rmbench_demo_clean_v1}"
# One bank anchor per closed-loop replan interval.
# Change-point/gripper events are unioned with these dense anchors by hybrid
# mining, so event semantics remain factual and no online suffix shifting is
# needed to manufacture intermediate phases.
WARM_RMBENCH_REPLAN_STRIDE="${WARM_RMBENCH_REPLAN_STRIDE:-4}"
if [[ ! "${WARM_RMBENCH_REPLAN_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  warm_die "WARM_RMBENCH_REPLAN_STRIDE must be a positive integer"
fi

warm_require_env \
  WARM_ARTIFACT_ROOT \
  RMBENCH_ROOT \
  RMBENCH_SOURCE_ROOT \
  RMBENCH_LEROBOT_ROOT \
  RMBENCH_HF_REVISION_MARKER \
  FASTWAM_BASE_CHECKPOINT \
  WARM_DINO_CHECKPOINT \
  WARM_DINO_REVISION \
  WARM_VAE_CHECKPOINT
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_require_sha40 "${RMBENCH_DATASET_REVISION}" "RMBENCH_DATASET_REVISION"
warm_require_sha40 "${RMBENCH_SOURCE_REVISION}" "RMBENCH_SOURCE_REVISION"
warm_require_sha40 "${WARM_DINO_REVISION}" "WARM_DINO_REVISION"
warm_register_safe_directory "${PROJECT_ROOT}"
warm_require_private_checkout
warm_require_read_only_external_checkout "${RMBENCH_ROOT}" "${RMBENCH_CODE_REVISION}"
warm_configure_offline_logging
warm_require_file_or_directory \
  "${RMBENCH_SOURCE_ROOT}" \
  "${RMBENCH_HF_REVISION_MARKER}" \
  "${FASTWAM_BASE_CHECKPOINT}" \
  "${WARM_DINO_CHECKPOINT}" \
  "${WARM_VAE_CHECKPOINT}"
warm_refuse_existing_output "${WARM_ARTIFACT_ROOT}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
python -c \
  'import sys; from fastwam.benchmarks.rmbench import validate_hf_revision_marker; validate_hf_revision_marker(sys.argv[1], expected_revision=sys.argv[2])' \
  "${RMBENCH_HF_REVISION_MARKER}" "${RMBENCH_DATASET_REVISION}"

if [[ ! -e "${RMBENCH_LEROBOT_ROOT}" && ! -L "${RMBENCH_LEROBOT_ROOT}" ]]; then
  python scripts/convert_rmbench_to_lerobot.py \
    --source-root "${RMBENCH_SOURCE_ROOT}" \
    --output-root "${RMBENCH_LEROBOT_ROOT}" \
    --source-revision "${RMBENCH_SOURCE_REVISION}" \
    --source-dataset "${RMBENCH_SOURCE_DATASET}" \
    --rmbench-code-revision "${RMBENCH_CODE_REVISION}" \
    --data-revision "${WARM_CODE_REVISION}" \
    --dataset-id "${RMBENCH_DATASET_ID}" \
    --profile "${WARM_RMBENCH_DATA_PROFILE}" \
    --split-seed 3407 \
    --workers "${WARM_CONVERSION_WORKERS:-4}"
fi

python scripts/validate_rmbench_conversion.py \
  --dataset-root "${RMBENCH_LEROBOT_ROOT}" \
  --source-revision "${RMBENCH_SOURCE_REVISION}" \
  --rmbench-code-revision "${RMBENCH_CODE_REVISION}"

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
CATALOG="${RMBENCH_LEROBOT_ROOT}/meta/warm_episode_catalog.json"
AUDIT="${M1}/rmbench_audit.json"
FEATURES="${M1}/features"
BANK="${M1}/banks/hybrid_h32"
TRAIN_CANDIDATES="${M2}/candidates/hybrid_h32_train_k32"
DEV_CANDIDATES="${M2}/candidates/hybrid_h32_dev_k32"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"

mkdir -p "${M1}/banks" "${M1}/oracle" "${M2}/candidates" "${M2}/contracts"
cp -- "${CATALOG}" "${M1}/rmbench_catalog.json"
cp -- \
  "${RMBENCH_LEROBOT_ROOT}/meta/rmbench_conversion_manifest.json" \
  "${M1}/rmbench_conversion_manifest.json"

python scripts/audit_warm_lerobot.py \
  --catalog "${M1}/rmbench_catalog.json" \
  --dataset-root "${RMBENCH_LEROBOT_ROOT}" \
  --hash-episode-tables \
  --camera-key observation.images.cam_high \
  --camera-key observation.images.cam_left_wrist \
  --camera-key observation.images.cam_right_wrist \
  --output "${AUDIT}"

python scripts/compute_warm_robotwin_train_stats.py \
  --catalog "${M1}/rmbench_catalog.json" \
  --audit-report "${AUDIT}" \
  --dataset-root "${RMBENCH_LEROBOT_ROOT}" \
  --data-config configs/data/rmbench_3cam.yaml \
  --dataset-revision "${RMBENCH_SOURCE_REVISION}" \
  --output "${M1}/train_stats"

python scripts/precompute_warm_features.py \
  --benchmark-profile robotwin \
  --data-config configs/data/rmbench_3cam.yaml \
  --catalog "${M1}/rmbench_catalog.json" \
  --audit-report "${AUDIT}" \
  --dataset-root "${RMBENCH_LEROBOT_ROOT}" \
  --dataset-stats "${M1}/train_stats/dataset_stats.json" \
  --dataset-stats-manifest "${M1}/train_stats/train_stats_manifest.json" \
  --dino-checkpoint "${WARM_DINO_CHECKPOINT}" \
  --dino-revision "${WARM_DINO_REVISION}" \
  --camera observation.images.cam_high \
  --camera observation.images.cam_left_wrist \
  --camera observation.images.cam_right_wrist \
  --semantic-camera observation.images.cam_high \
  --concat-mode robotwin \
  --include-vae \
  --vae-checkpoint "${WARM_VAE_CHECKPOINT}" \
  --device "${WARM_PRECOMPUTE_DEVICE:-cuda}" \
  --dtype "${WARM_PRECOMPUTE_DTYPE:-bfloat16}" \
  --dino-batch-size "${WARM_DINO_BATCH_SIZE:-64}" \
  --vae-batch-size "${WARM_VAE_BATCH_SIZE:-4}" \
  --output "${FEATURES}"

python scripts/build_warm_event_bank.py \
  --feature-list "${FEATURES}/train_features.list" \
  --catalog "${M1}/rmbench_catalog.json" \
  --audit-report "${AUDIT}" \
  --output "${BANK}" \
  --summary "${M1}/banks/hybrid_h32.summary.json" \
  --normalizer-contract "${FEATURES}/contracts/normalizer_contract.json" \
  --encoder-contract "${FEATURES}/contracts/encoder_contract.json" \
  --camera-contract "${FEATURES}/contracts/camera_contract.json" \
  --action-horizon 32 \
  --uniform-stride "${WARM_RMBENCH_REPLAN_STRIDE}" \
  --start-mode hybrid

python scripts/evaluate_warm_oracle.py \
  --bank "${BANK}" \
  --feature-list "${FEATURES}/dev_features.list" \
  --catalog "${M1}/rmbench_catalog.json" \
  --audit-report "${AUDIT}" \
  --output "${M1}/oracle/hybrid_h32.json" \
  --query-stride "${WARM_RMBENCH_REPLAN_STRIDE}" \
  --top-k 1,4,8,16,32 \
  --arm-loss mse

for split in train dev; do
  if [[ "${split}" == train ]]; then
    feature_list="${FEATURES}/train_features.list"
    cache="${TRAIN_CANDIDATES}"
  else
    feature_list="${FEATURES}/dev_features.list"
    cache="${DEV_CANDIDATES}"
  fi
  # Candidate caches remain frame-complete because the runtime dataset
  # binds samples to exact QueryIds; phase-aware sampling is a training
  # policy, not a lossy cache-build shortcut.
  python scripts/build_warm_candidate_cache.py \
    --bank "${BANK}" \
    --catalog "${M1}/rmbench_catalog.json" \
    --audit-report "${AUDIT}" \
    --feature-list "${feature_list}" \
    --output "${cache}" \
    --query-split "${split}" \
    --query-stride 1 \
    --include-partial-action-queries \
    --top-k 32 \
    --summary "${M2}/candidates/hybrid_h32_${split}_k32.summary.json"
done

# No expensive WARM optimization is allowed to consume a merely
# schema-valid bank.  Prove dense temporal chains, phase coverage in every
# trajectory third, and exact train/dev teacher-forced retrieval parity first.
python scripts/qualify_warm_rmbench_artifacts.py \
  --bank "${BANK}" \
  --train-feature-list "${FEATURES}/train_features.list" \
  --train-candidate-cache "${TRAIN_CANDIDATES}" \
  --dev-feature-list "${FEATURES}/dev_features.list" \
  --dev-candidate-cache "${DEV_CANDIDATES}" \
  --output "${M1}/qualification/rmbench_h32.json" \
  --action-horizon 32 \
  --query-stride "${WARM_RMBENCH_REPLAN_STRIDE:-4}" \
  --max-event-stride "${WARM_RMBENCH_REPLAN_STRIDE:-4}" \
  --phase-tolerance "${WARM_RMBENCH_PHASE_TOLERANCE:-0.10}" \
  --phase-recall-threshold \
    "32=${WARM_RMBENCH_MIN_PHASE_RECALL_K32:-0.85}" \
  --phase-recall-threshold \
    "128=${WARM_RMBENCH_MIN_PHASE_RECALL_K128:-0.95}" \
  --parity-max-queries "${WARM_RMBENCH_PARITY_MAX_QUERIES:-2048}" \
  --require-partial-action-queries

python scripts/build_warm_source_run_contract.py \
  --bank "${BANK}" \
  --candidate-cache "${TRAIN_CANDIDATES}" \
  --base-checkpoint "${FASTWAM_BASE_CHECKPOINT}" \
  --output "${TRAIN_CONTRACT}" \
  --query-split train \
  --expected-action-horizon 32 \
  --expected-action-dim 14

python scripts/build_warm_source_run_contract.py \
  --bank "${BANK}" \
  --candidate-cache "${DEV_CANDIDATES}" \
  --base-checkpoint "${FASTWAM_BASE_CHECKPOINT}" \
  --output "${DEV_CONTRACT}" \
  --query-split dev \
  --expected-action-horizon 32 \
  --expected-action-dim 14

printf '%s\n' \
  "RMBench WARM artifacts published without overwrite:" \
  "  dataset:       ${RMBENCH_LEROBOT_ROOT}" \
  "  train contract: ${TRAIN_CONTRACT}" \
  "  dev contract:   ${DEV_CONTRACT}" \
  "  oracle report:  ${M1}/oracle/hybrid_h32.json" \
  "  qualification:  ${M1}/qualification/rmbench_h32.json"
