#!/usr/bin/env bash
set -euo pipefail

# Evaluate exactly one task/checkpoint pair.  This is the primitive used by
# one-GPU ACP jobs and by the specialist batch orchestrator; it deliberately
# bypasses the shared-checkpoint suite manager.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/warm_server_common.sh
source "${SCRIPT_DIR}/warm_server_common.sh"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
RMBENCH_DATASET_REVISION="${RMBENCH_DATASET_REVISION:-855e90e1213d150bf4889130e83398f107314681}"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOTA_REGISTRY="${WARM_RMBENCH_SOTA_REGISTRY:-${PROJECT_ROOT}/configs/rmbench/sota_v1.json}"
TASK_NAME="${WARM_RMBENCH_TASK:-}"
EXPERIMENT_ID="${WARM_EXPERIMENT_ID:-}"

warm_require_env \
  WARM_ARTIFACT_ROOT \
  RMBENCH_ROOT \
  RMBENCH_HF_REVISION_MARKER \
  FASTWAM_BASE_CHECKPOINT \
  WARM_CHECKPOINT \
  WARM_TRAINING_ATTESTATION \
  WARM_RMBENCH_ONLINE_CONTRACT \
  WARM_DINO_CHECKPOINT \
  WARM_VAE_CHECKPOINT \
  WARM_TEXT_ENCODER \
  WARM_TOKENIZER \
  WARM_EVAL_ROOT \
  WARM_RMBENCH_TASK
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_require_sha40 "${RMBENCH_DATASET_REVISION}" "RMBENCH_DATASET_REVISION"
warm_register_safe_directory "${PROJECT_ROOT}"
warm_require_private_checkout
warm_require_read_only_external_checkout "${RMBENCH_ROOT}" "${RMBENCH_CODE_REVISION}"
warm_configure_offline_logging
warm_refuse_existing_output "${WARM_EVAL_ROOT}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
IFS=$'\t' read -r VALIDATED_TASK MEMORY_REGIME TRAIN_STEPS \
  RECENT_EVENT_CAPACITY ACTION_SUMMARY_CAPACITY REPLAN_STEPS \
  INFERENCE_STEPS TOP_K < <(
    python scripts/plan_warm_rmbench_sota.py \
      --registry "${SOTA_REGISTRY}" \
      --task "${TASK_NAME}" \
      --format tsv
  )
if [[ "${VALIDATED_TASK}" != "${TASK_NAME}" ]]; then
  warm_die "task registry returned an inconsistent task"
fi
if [[ -z "${EXPERIMENT_ID}" ]]; then
  case "${INFERENCE_STEPS}" in
    8) EXPERIMENT_ID="full_warm_ode08" ;;
    10) EXPERIMENT_ID="full_warm" ;;
    *) warm_die "no formal SOTA experiment id for ${INFERENCE_STEPS} ODE steps" ;;
  esac
fi

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"
CATALOG="${M1}/rmbench_catalog.json"
AUDIT="${M1}/rmbench_audit.json"
TASK_CONTRACT="${WARM_RMBENCH_ONLINE_CONTRACT}/${EXPERIMENT_ID}/${TASK_NAME}.json"
TASK_SEEDS="${WARM_RMBENCH_ONLINE_CONTRACT}/seeds/${TASK_NAME}.seed_protocol.npy"

warm_require_file_or_directory \
  "${RMBENCH_HF_REVISION_MARKER}" \
  "${FASTWAM_BASE_CHECKPOINT}" \
  "${WARM_CHECKPOINT}" \
  "${WARM_TRAINING_ATTESTATION}" \
  "${WARM_RMBENCH_ONLINE_CONTRACT}" \
  "${TASK_CONTRACT}" \
  "${TASK_SEEDS}" \
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
  "${DEV_CONTRACT}"

python -c \
  'import sys; from scripts.build_warm_rmbench_contract_bundle import validate_contract_bundle; validate_contract_bundle(sys.argv[1], experiment_id=sys.argv[2], task_names=[sys.argv[3]])' \
  "${WARM_RMBENCH_ONLINE_CONTRACT}" "${EXPERIMENT_ID}" "${TASK_NAME}"
python -c \
  'import sys; from fastwam.benchmarks.rmbench import validate_hf_revision_marker; validate_hf_revision_marker(sys.argv[1], expected_revision=sys.argv[2])' \
  "${RMBENCH_HF_REVISION_MARKER}" "${RMBENCH_DATASET_REVISION}"

exec python experiments/rmbench/eval_rmbench_single.py \
  "ckpt=${WARM_CHECKPOINT}" \
  "gpu_id=${WARM_GPU_ID:-0}" \
  "seed=3407" \
  "EVALUATION.rmbench_root=${RMBENCH_ROOT}" \
  "EVALUATION.hf_revision_marker=${RMBENCH_HF_REVISION_MARKER}" \
  "EVALUATION.task_name=${TASK_NAME}" \
  "EVALUATION.output_dir=${WARM_EVAL_ROOT}" \
  "EVALUATION.policy_overrides.sim_task=rmbench_warm_online_3cam384_full" \
  "EVALUATION.policy_overrides.device=${WARM_EVAL_DEVICE:-cuda}" \
  "EVALUATION.policy_overrides.dataset_stats_path=${TRAIN_STATS}" \
  "EVALUATION.policy_overrides.action_horizon=32" \
  "EVALUATION.policy_overrides.replan_steps=${REPLAN_STEPS}" \
  "EVALUATION.policy_overrides.num_inference_steps=${INFERENCE_STEPS}" \
  "EVALUATION.policy_overrides.warm_online_contract_path=${WARM_RMBENCH_ONLINE_CONTRACT}" \
  "EVALUATION.policy_overrides.warm_training_attestation_path=${WARM_TRAINING_ATTESTATION}" \
  "EVALUATION.policy_overrides.warm_training_run_contract_path=${TRAIN_CONTRACT}" \
  "EVALUATION.policy_overrides.warm_validation_run_contract_path=${DEV_CONTRACT}" \
  "EVALUATION.policy_overrides.warm_base_checkpoint_path=${FASTWAM_BASE_CHECKPOINT}" \
  "EVALUATION.policy_overrides.warm_bank_directory=${M1}/banks/hybrid_h32" \
  "EVALUATION.policy_overrides.warm_normalizer_contract_path=${M1}/features/contracts/normalizer_contract.json" \
  "EVALUATION.policy_overrides.warm_encoder_contract_path=${M1}/features/contracts/encoder_contract.json" \
  "EVALUATION.policy_overrides.warm_camera_contract_path=${M1}/features/contracts/camera_contract.json" \
  "EVALUATION.policy_overrides.warm_m1_data_config_path=${PROJECT_ROOT}/configs/data/rmbench_3cam.yaml" \
  "EVALUATION.policy_overrides.warm_dino_checkpoint_path=${WARM_DINO_CHECKPOINT}" \
  "EVALUATION.policy_overrides.warm_catalog_path=${CATALOG}" \
  "EVALUATION.policy_overrides.warm_audit_report_path=${AUDIT}" \
  "EVALUATION.policy_overrides.warm_initial_states_path=${WARM_RMBENCH_ONLINE_CONTRACT}" \
  "++EVALUATION.policy_overrides.warm_vae_checkpoint_path=${WARM_VAE_CHECKPOINT}" \
  "++EVALUATION.policy_overrides.warm_text_encoder_path=${WARM_TEXT_ENCODER}" \
  "++EVALUATION.policy_overrides.warm_tokenizer_path=${WARM_TOKENIZER}" \
  "++EVALUATION.policy_overrides.warm_ablation_mode=full" \
  "++EVALUATION.policy_overrides.warm_memory_corruption=clean" \
  "++EVALUATION.policy_overrides.warm_experiment_id=${EXPERIMENT_ID}" \
  "++EVALUATION.policy_overrides.warm_evaluation_namespace=${WARM_EVALUATION_NAMESPACE:-warm-rmbench-sota-v1}" \
  "++EVALUATION.policy_overrides.warm_top_k=${TOP_K}" \
  "++EVALUATION.policy_overrides.warm_recent_event_capacity=${RECENT_EVENT_CAPACITY}" \
  "++EVALUATION.policy_overrides.warm_action_summary_capacity=${ACTION_SUMMARY_CAPACITY}" \
  "++EVALUATION.policy_overrides.warm_dino_device=${WARM_DINO_DEVICE:-cuda}" \
  "++EVALUATION.policy_overrides.warm_dino_batch_size=${WARM_DINO_BATCH_SIZE:-1}" \
  "mixed_precision=${WARM_MIXED_PRECISION:-bf16}"
