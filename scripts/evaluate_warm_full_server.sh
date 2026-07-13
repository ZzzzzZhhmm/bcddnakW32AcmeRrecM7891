#!/usr/bin/env bash
set -euo pipefail

# Resolve, contract-bind, and execute one complete-WARM LIBERO task. Repeat for
# every task id/seed; use a distinct WARM_EVAL_ROOT for each immutable job.

required_env=(
  WARM_ARTIFACT_ROOT
  FASTWAM_BASE_CHECKPOINT
  WARM_CHECKPOINT
  WARM_TRAINING_ATTESTATION
  WARM_DINO_CHECKPOINT
  WARM_VAE_CHECKPOINT
  WARM_TEXT_ENCODER
  WARM_TOKENIZER
  WARM_EVAL_ROOT
  WARM_TASK_SUITE
  WARM_TASK_ID
  WARM_TASK_DESCRIPTION
  WARM_INITIAL_STATES
  WARM_BDDL
)
for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "error: ${name} must be set" >&2
    exit 2
  fi
done
if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: formal evaluation requires a clean Git checkout" >&2
  exit 2
fi

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
ROOT_SEED="${WARM_ROOT_SEED:-17}"
NAMESPACE="${WARM_EVALUATION_NAMESPACE:-warm-libero-full-v1}"
CONTRACT="${WARM_EVAL_ROOT}/online_contract.json"
RESOLVED_CONFIG="${WARM_EVAL_ROOT}/resolved_config.yaml"
RESULT_DIR="${WARM_EVAL_ROOT}/results"

required_paths=(
  "${FASTWAM_BASE_CHECKPOINT}"
  "${WARM_CHECKPOINT}"
  "${WARM_TRAINING_ATTESTATION}"
  "${WARM_DINO_CHECKPOINT}"
  "${WARM_VAE_CHECKPOINT}"
  "${WARM_TEXT_ENCODER}"
  "${WARM_TOKENIZER}"
  "${WARM_INITIAL_STATES}"
  "${WARM_BDDL}"
  "${M1}/banks/hybrid_h32"
  "${M1}/features/contracts/normalizer_contract.json"
  "${M1}/features/contracts/encoder_contract.json"
  "${M1}/features/contracts/camera_contract.json"
  "${M1}/train_stats/dataset_stats.json"
  "${M1}/libero_catalog.json"
  "${M1}/libero_audit.json"
  "${M2}/contracts/hybrid_h32_train_source.json"
  "${M2}/contracts/hybrid_h32_dev_source.json"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "error: required evaluation input not found: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${WARM_EVAL_ROOT}" ]]; then
  echo "error: immutable evaluation root already exists: ${WARM_EVAL_ROOT}" >&2
  exit 2
fi
mkdir -p "${WARM_EVAL_ROOT}"

HYDRA_OVERRIDES=(
  task=libero_warm_online_2cam224_full
  "ckpt=${WARM_CHECKPOINT}"
  "EVALUATION.task_suite_name=${WARM_TASK_SUITE}"
  "EVALUATION.task_id=${WARM_TASK_ID}"
  "EVALUATION.output_dir=${RESULT_DIR}"
  "EVALUATION.dataset_stats_path=${M1}/train_stats/dataset_stats.json"
  "EVALUATION.device=${WARM_EVAL_DEVICE:-cuda}"
  "EVALUATION.warm_online.mode=full_retrospection"
  "EVALUATION.warm_online.contract_path=${CONTRACT}"
  "EVALUATION.warm_online.training_attestation_path=${WARM_TRAINING_ATTESTATION}"
  "EVALUATION.warm_online.training_run_contract_path=${M2}/contracts/hybrid_h32_train_source.json"
  "EVALUATION.warm_online.validation_run_contract_path=${M2}/contracts/hybrid_h32_dev_source.json"
  "EVALUATION.warm_online.base_checkpoint_path=${FASTWAM_BASE_CHECKPOINT}"
  "EVALUATION.warm_online.bank_directory=${M1}/banks/hybrid_h32"
  "EVALUATION.warm_online.normalizer_contract_path=${M1}/features/contracts/normalizer_contract.json"
  "EVALUATION.warm_online.encoder_contract_path=${M1}/features/contracts/encoder_contract.json"
  "EVALUATION.warm_online.camera_contract_path=${M1}/features/contracts/camera_contract.json"
  "EVALUATION.warm_online.m1_data_config_path=configs/data/libero_2cam.yaml"
  "EVALUATION.warm_online.dino_checkpoint_path=${WARM_DINO_CHECKPOINT}"
  "EVALUATION.warm_online.catalog_path=${M1}/libero_catalog.json"
  "EVALUATION.warm_online.audit_report_path=${M1}/libero_audit.json"
  "EVALUATION.warm_online.evaluation_namespace=${NAMESPACE}"
  "EVALUATION.warm_online.top_k=32"
  "seed=${ROOT_SEED}"
)

python experiments/libero/eval_libero_single.py \
  "${HYDRA_OVERRIDES[@]}" \
  --cfg job --resolve > "${RESOLVED_CONFIG}"

python scripts/build_warm_online_contract.py \
  --training-run-contract "${M2}/contracts/hybrid_h32_train_source.json" \
  --validation-run-contract "${M2}/contracts/hybrid_h32_dev_source.json" \
  --warm-checkpoint "${WARM_CHECKPOINT}" \
  --training-attestation "${WARM_TRAINING_ATTESTATION}" \
  --bank "${M1}/banks/hybrid_h32" \
  --normalizer-contract "${M1}/features/contracts/normalizer_contract.json" \
  --encoder-contract "${M1}/features/contracts/encoder_contract.json" \
  --camera-contract "${M1}/features/contracts/camera_contract.json" \
  --data-config configs/data/libero_2cam.yaml \
  --dino-checkpoint "${WARM_DINO_CHECKPOINT}" \
  --normalization-stats "${M1}/train_stats/dataset_stats.json" \
  --catalog "${M1}/libero_catalog.json" \
  --audit-report "${M1}/libero_audit.json" \
  --resolved-eval-config "${RESOLVED_CONFIG}" \
  --vae-checkpoint "${WARM_VAE_CHECKPOINT}" \
  --text-encoder "${WARM_TEXT_ENCODER}" \
  --tokenizer "${WARM_TOKENIZER}" \
  --evaluation-namespace "${NAMESPACE}" \
  --task-suite "${WARM_TASK_SUITE}" \
  --task-id "${WARM_TASK_ID}" \
  --task-description "${WARM_TASK_DESCRIPTION}" \
  --initial-states "${WARM_INITIAL_STATES}" \
  --bddl "${WARM_BDDL}" \
  --root-seed "${ROOT_SEED}" \
  --top-k 32 \
  --source-policy fixed_context_top1 \
  --memory-sigma 0.2 \
  --action-horizon 32 \
  --action-dim 7 \
  --output "${CONTRACT}"

python experiments/libero/eval_libero_single.py "${HYDRA_OVERRIDES[@]}"
