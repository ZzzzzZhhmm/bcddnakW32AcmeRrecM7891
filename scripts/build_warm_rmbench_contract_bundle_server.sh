#!/usr/bin/env bash
set -euo pipefail

# Build every experiment/task online contract after a WARM checkpoint has been
# trained and attested.  The output root is immutable: choose a new path for a
# new checkpoint, matrix or runtime recipe.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/warm_server_common.sh
source "${SCRIPT_DIR}/warm_server_common.sh"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
warm_require_env \
  WARM_ARTIFACT_ROOT \
  RMBENCH_ROOT \
  FASTWAM_BASE_CHECKPOINT \
  WARM_CHECKPOINT \
  WARM_TRAINING_ATTESTATION \
  WARM_RMBENCH_ONLINE_CONTRACT \
  WARM_DINO_CHECKPOINT \
  WARM_VAE_CHECKPOINT \
  WARM_TEXT_ENCODER \
  WARM_TOKENIZER
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_register_safe_directory "${PROJECT_ROOT}"
warm_require_private_checkout
warm_require_read_only_external_checkout "${RMBENCH_ROOT}" "${RMBENCH_CODE_REVISION}"
warm_configure_offline_logging
warm_refuse_existing_output "${WARM_RMBENCH_ONLINE_CONTRACT}"

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"
CATALOG="${M1}/rmbench_catalog.json"
AUDIT="${M1}/rmbench_audit.json"

warm_require_file_or_directory \
  "${FASTWAM_BASE_CHECKPOINT}" \
  "${WARM_CHECKPOINT}" \
  "${WARM_TRAINING_ATTESTATION}" \
  "${WARM_DINO_CHECKPOINT}" \
  "${WARM_VAE_CHECKPOINT}" \
  "${WARM_TEXT_ENCODER}" \
  "${WARM_TOKENIZER}" \
  "${M1}/banks/hybrid_h32" \
  "${M1}/features/contracts/normalizer_contract.json" \
  "${M1}/features/contracts/encoder_contract.json" \
  "${M1}/features/contracts/camera_contract.json" \
  "${TRAIN_STATS}" \
  "${CATALOG}" \
  "${AUDIT}" \
  "${TRAIN_CONTRACT}" \
  "${DEV_CONTRACT}" \
  "$(pwd)/configs/data/rmbench_3cam.yaml" \
  "${WARM_RMBENCH_MATRIX:-$(pwd)/configs/ablation/rmbench_reproducible_matrix.json}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
BUNDLE_ARGS=()
REPLAN_STEPS="${WARM_REPLAN_STEPS:-10}"
RECENT_EVENT_CAPACITY="${WARM_RECENT_EVENT_CAPACITY:-6}"
ACTION_SUMMARY_CAPACITY="${WARM_ACTION_SUMMARY_CAPACITY:-2}"
TOP_K="${WARM_TOP_K:-32}"
if [[ -n "${WARM_RMBENCH_TASK:-}" ]]; then
  IFS=$'\t' read -r VALIDATED_TASK MEMORY_REGIME TRAIN_STEPS \
    PROFILE_RECENT PROFILE_SUMMARIES PROFILE_REPLAN INFERENCE_STEPS PROFILE_TOP_K < <(
      python scripts/plan_warm_rmbench_sota.py \
        --registry "${WARM_RMBENCH_SOTA_REGISTRY:-configs/rmbench/sota_v1.json}" \
        --task "${WARM_RMBENCH_TASK}" \
        --format tsv
    )
  [[ "${VALIDATED_TASK}" == "${WARM_RMBENCH_TASK}" ]] || \
    warm_die "task registry returned an inconsistent task"
  REPLAN_STEPS="${WARM_REPLAN_STEPS:-${PROFILE_REPLAN}}"
  RECENT_EVENT_CAPACITY="${WARM_RECENT_EVENT_CAPACITY:-${PROFILE_RECENT}}"
  ACTION_SUMMARY_CAPACITY="${WARM_ACTION_SUMMARY_CAPACITY:-${PROFILE_SUMMARIES}}"
  TOP_K="${WARM_TOP_K:-${PROFILE_TOP_K}}"
  case "${INFERENCE_STEPS}" in
    8) PROFILE_EXPERIMENT="full_warm_ode08" ;;
    10) PROFILE_EXPERIMENT="full_warm" ;;
    *) warm_die "no formal SOTA experiment id for ${INFERENCE_STEPS} ODE steps" ;;
  esac
  BUNDLE_ARGS+=(--task "${WARM_RMBENCH_TASK}")
  BUNDLE_ARGS+=(--experiment "${WARM_EXPERIMENT_ID:-${PROFILE_EXPERIMENT}}")
fi
exec python scripts/build_warm_rmbench_contract_bundle.py \
  --matrix "${WARM_RMBENCH_MATRIX:-$(pwd)/configs/ablation/rmbench_reproducible_matrix.json}" \
  --suite "${WARM_RMBENCH_SUITE:-official9}" \
  --rmbench-root "${RMBENCH_ROOT}" \
  --output-root "${WARM_RMBENCH_ONLINE_CONTRACT}" \
  --training-run-contract "${TRAIN_CONTRACT}" \
  --validation-run-contract "${DEV_CONTRACT}" \
  --warm-checkpoint "${WARM_CHECKPOINT}" \
  --training-attestation "${WARM_TRAINING_ATTESTATION}" \
  --base-checkpoint "${FASTWAM_BASE_CHECKPOINT}" \
  --bank "${M1}/banks/hybrid_h32" \
  --normalizer-contract "${M1}/features/contracts/normalizer_contract.json" \
  --encoder-contract "${M1}/features/contracts/encoder_contract.json" \
  --camera-contract "${M1}/features/contracts/camera_contract.json" \
  --data-config "$(pwd)/configs/data/rmbench_3cam.yaml" \
  --dino-checkpoint "${WARM_DINO_CHECKPOINT}" \
  --normalization-stats "${TRAIN_STATS}" \
  --catalog "${CATALOG}" \
  --audit-report "${AUDIT}" \
  --vae-checkpoint "${WARM_VAE_CHECKPOINT}" \
  --text-encoder "${WARM_TEXT_ENCODER}" \
  --tokenizer "${WARM_TOKENIZER}" \
  --evaluation-namespace "${WARM_EVALUATION_NAMESPACE:-warm-rmbench-full-v1}" \
  --top-k "${TOP_K}" \
  --replan-steps "${REPLAN_STEPS}" \
  --recent-event-capacity "${RECENT_EVENT_CAPACITY}" \
  --action-summary-capacity "${ACTION_SUMMARY_CAPACITY}" \
  "${BUNDLE_ARGS[@]}" \
  --dino-device "${WARM_DINO_DEVICE:-cuda}" \
  --dino-batch-size "${WARM_DINO_BATCH_SIZE:-1}" \
  --device "${WARM_EVAL_DEVICE:-cuda}" \
  --mixed-precision "${WARM_MIXED_PRECISION:-bf16}"
