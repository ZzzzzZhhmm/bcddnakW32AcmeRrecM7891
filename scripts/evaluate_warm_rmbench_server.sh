#!/usr/bin/env bash
set -euo pipefail

# Run one immutable official RMBench suite for a trained WARM checkpoint.
# Ablation/corruption controls are closed enums used by the matrix runner.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/warm_server_common.sh
source "${SCRIPT_DIR}/warm_server_common.sh"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
RMBENCH_DATASET_REVISION="${RMBENCH_DATASET_REVISION:-855e90e1213d150bf4889130e83398f107314681}"
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
  WARM_EVAL_ROOT
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_require_sha40 "${RMBENCH_DATASET_REVISION}" "RMBENCH_DATASET_REVISION"
warm_register_safe_directory "${PROJECT_ROOT}"
warm_require_private_checkout
warm_require_read_only_external_checkout "${RMBENCH_ROOT}" "${RMBENCH_CODE_REVISION}"
warm_configure_offline_logging
warm_refuse_existing_output "${WARM_EVAL_ROOT}"

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"
CATALOG="${M1}/rmbench_catalog.json"
AUDIT="${M1}/rmbench_audit.json"

warm_require_file_or_directory \
  "${RMBENCH_HF_REVISION_MARKER}" \
  "${FASTWAM_BASE_CHECKPOINT}" \
  "${WARM_CHECKPOINT}" \
  "${WARM_TRAINING_ATTESTATION}" \
  "${WARM_RMBENCH_ONLINE_CONTRACT}" \
  "${WARM_DINO_CHECKPOINT}" \
  "${WARM_VAE_CHECKPOINT}" \
  "${WARM_TEXT_ENCODER}" \
  "${WARM_TOKENIZER}" \
  "${M1}/banks/hybrid_h32" \
  "${M1}/features/contracts/normalizer_contract.json" \
  "${M1}/features/contracts/encoder_contract.json" \
  "${M1}/features/contracts/camera_contract.json" \
  "${TRAIN_STATS}" \
  "${M1}/train_stats/train_stats_manifest.json" \
  "${CATALOG}" \
  "${AUDIT}" \
  "${TRAIN_CONTRACT}" \
  "${DEV_CONTRACT}"

ABLATION="${WARM_ABLATION_MODE:-full}"
CORRUPTION="${WARM_MEMORY_CORRUPTION:-clean}"
INFERENCE_STEPS="${WARM_NUM_INFERENCE_STEPS:-10}"
SUITE="${WARM_RMBENCH_SUITE:-official9}"
EXPERIMENT_ID="${WARM_EXPERIMENT_ID:-full_warm}"
case "${ABLATION}" in
  full|context_only|source_only_no_consequence) ;;
  *) warm_die "unsupported WARM_ABLATION_MODE=${ABLATION}" ;;
esac
case "${CORRUPTION}" in
  clean|wrong_event|reversed_action|phase_shift|effect_mismatch) ;;
  *) warm_die "unsupported WARM_MEMORY_CORRUPTION=${CORRUPTION}" ;;
esac
case "${INFERENCE_STEPS}" in
  2|4|8|10) ;;
  *) warm_die "WARM_NUM_INFERENCE_STEPS must be one of 2,4,8,10" ;;
esac
case "${SUITE}" in
  official9|pilot3) ;;
  *) warm_die "WARM_RMBENCH_SUITE must be official9 or pilot3" ;;
esac
if [[ ! "${EXPERIMENT_ID}" =~ ^[a-z][a-z0-9_]{1,63}$ ]]; then
  warm_die "WARM_EXPERIMENT_ID must match ^[a-z][a-z0-9_]{1,63}$"
fi
if [[ ! -d "${WARM_RMBENCH_ONLINE_CONTRACT}" ]]; then
  warm_die "WARM_RMBENCH_ONLINE_CONTRACT must be the contract-bundle root"
fi
if [[ -e "${WARM_RMBENCH_ONLINE_CONTRACT}/.incomplete.json" ]]; then
  warm_die "WARM_RMBENCH_ONLINE_CONTRACT is an incomplete bundle"
fi
if [[ "${SUITE}" == official9 ]]; then
  RMBENCH_TASKS=(
    observe_and_pickup rearrange_blocks put_back_block swap_blocks swap_T
    blocks_ranking_try press_button cover_blocks battery_try
  )
else
  RMBENCH_TASKS=(put_back_block rearrange_blocks battery_try)
fi
for task_name in "${RMBENCH_TASKS[@]}"; do
  warm_require_file_or_directory \
    "${WARM_RMBENCH_ONLINE_CONTRACT}/${EXPERIMENT_ID}/${task_name}.json" \
    "${WARM_RMBENCH_ONLINE_CONTRACT}/seeds/${task_name}.seed_protocol.npy"
done

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
python -c \
  'import sys; from scripts.build_warm_rmbench_contract_bundle import validate_contract_bundle; validate_contract_bundle(sys.argv[1], experiment_id=sys.argv[2], task_names=sys.argv[3:])' \
  "${WARM_RMBENCH_ONLINE_CONTRACT}" "${EXPERIMENT_ID}" "${RMBENCH_TASKS[@]}"
python -c \
  'import sys; from fastwam.benchmarks.rmbench import validate_hf_revision_marker; validate_hf_revision_marker(sys.argv[1], expected_revision=sys.argv[2])' \
  "${RMBENCH_HF_REVISION_MARKER}" "${RMBENCH_DATASET_REVISION}"

exec python experiments/rmbench/run_rmbench_manager.py \
  "ckpt=${WARM_CHECKPOINT}" \
  "EVALUATION.rmbench_root=${RMBENCH_ROOT}" \
  "EVALUATION.hf_revision_marker=${RMBENCH_HF_REVISION_MARKER}" \
  "EVALUATION.suite=${SUITE}" \
  "EVALUATION.output_dir=${WARM_EVAL_ROOT}" \
  "EVALUATION.policy_overrides.sim_task=rmbench_warm_online_3cam384_full" \
  "EVALUATION.policy_overrides.device=${WARM_EVAL_DEVICE:-cuda}" \
  "EVALUATION.policy_overrides.dataset_stats_path=${TRAIN_STATS}" \
  "EVALUATION.policy_overrides.action_horizon=32" \
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
  "EVALUATION.policy_overrides.warm_m1_data_config_path=$(pwd)/configs/data/rmbench_3cam.yaml" \
  "EVALUATION.policy_overrides.warm_dino_checkpoint_path=${WARM_DINO_CHECKPOINT}" \
  "EVALUATION.policy_overrides.warm_catalog_path=${CATALOG}" \
  "EVALUATION.policy_overrides.warm_audit_report_path=${AUDIT}" \
  "EVALUATION.policy_overrides.warm_initial_states_path=${WARM_RMBENCH_ONLINE_CONTRACT}" \
  "++EVALUATION.policy_overrides.warm_vae_checkpoint_path=${WARM_VAE_CHECKPOINT}" \
  "++EVALUATION.policy_overrides.warm_text_encoder_path=${WARM_TEXT_ENCODER}" \
  "++EVALUATION.policy_overrides.warm_tokenizer_path=${WARM_TOKENIZER}" \
  "++EVALUATION.policy_overrides.warm_ablation_mode=${ABLATION}" \
  "++EVALUATION.policy_overrides.warm_memory_corruption=${CORRUPTION}" \
  "++EVALUATION.policy_overrides.warm_experiment_id=${EXPERIMENT_ID}" \
  "++EVALUATION.policy_overrides.warm_evaluation_namespace=${WARM_EVALUATION_NAMESPACE:-warm-rmbench-full-v1}" \
  "++EVALUATION.policy_overrides.warm_top_k=${WARM_TOP_K:-32}" \
  "++EVALUATION.policy_overrides.warm_recent_event_capacity=${WARM_RECENT_EVENT_CAPACITY:-6}" \
  "++EVALUATION.policy_overrides.warm_action_summary_capacity=${WARM_ACTION_SUMMARY_CAPACITY:-2}" \
  "++EVALUATION.policy_overrides.warm_dino_device=${WARM_DINO_DEVICE:-cuda}" \
  "++EVALUATION.policy_overrides.warm_dino_batch_size=${WARM_DINO_BATCH_SIZE:-1}" \
  "MULTIRUN.num_gpus=${WARM_NUM_GPUS:-1}" \
  "MULTIRUN.max_tasks_per_gpu=${WARM_MAX_TASKS_PER_GPU:-1}" \
  "mixed_precision=${WARM_MIXED_PRECISION:-bf16}" \
  "seed=${WARM_ROOT_SEED:-17}"
