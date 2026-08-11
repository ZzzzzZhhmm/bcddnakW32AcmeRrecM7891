#!/usr/bin/env bash
set -euo pipefail

# Launch complete three-camera RMBench WARM training from immutable artifacts.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/warm_server_common.sh
source "${SCRIPT_DIR}/warm_server_common.sh"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
RMBENCH_DATASET_REVISION="${RMBENCH_DATASET_REVISION:-855e90e1213d150bf4889130e83398f107314681}"
RMBENCH_SOURCE_REVISION="${RMBENCH_SOURCE_REVISION:-${RMBENCH_DATASET_REVISION}}"
warm_require_env \
  WARM_ARTIFACT_ROOT \
  RMBENCH_LEROBOT_ROOT \
  RMBENCH_TEXT_CACHE \
  FASTWAM_BASE_CHECKPOINT \
  WARM_TRAIN_OUTPUT
warm_require_sha40 "${RMBENCH_DATASET_REVISION}" "RMBENCH_DATASET_REVISION"
warm_require_sha40 "${RMBENCH_SOURCE_REVISION}" "RMBENCH_SOURCE_REVISION"
warm_require_sha40 "${RMBENCH_CODE_REVISION}" "RMBENCH_CODE_REVISION"
warm_register_safe_directory "${PROJECT_ROOT}"
warm_require_private_checkout
warm_configure_offline_logging

WARM_RESUME_STATE="${WARM_RESUME_STATE:-}"
if [[ -z "${WARM_RESUME_STATE}" ]]; then
  warm_refuse_existing_output "${WARM_TRAIN_OUTPUT}"
else
  if [[ ! -d "${WARM_TRAIN_OUTPUT}" ]]; then
    warm_die "resume requires an existing WARM_TRAIN_OUTPUT directory"
  fi
  if [[ ! -d "${WARM_RESUME_STATE}" ]]; then
    warm_die "WARM_RESUME_STATE is not a directory: ${WARM_RESUME_STATE}"
  fi
  WARM_RESUME_STATE="$(
    python - "${WARM_TRAIN_OUTPUT}" "${WARM_RESUME_STATE}" <<'PY'
from pathlib import Path
import re
import sys

output = Path(sys.argv[1]).expanduser().resolve(strict=True)
state = Path(sys.argv[2]).expanduser().resolve(strict=True)
expected_parent = output / "checkpoints" / "state"
if state.parent != expected_parent:
    raise SystemExit(
        "WARM_RESUME_STATE must be an immediate child of "
        f"{expected_parent}, got {state}"
    )
if re.fullmatch(r"step_[0-9]{6,}", state.name) is None:
    raise SystemExit("WARM_RESUME_STATE must be named step_<global_step>")
if not (state / "trainer_state.json").is_file():
    raise SystemExit("WARM_RESUME_STATE is missing trainer_state.json")
print(state)
PY
  )"
  printf 'RMBench formal resume: output=%s state=%s\n' \
    "${WARM_TRAIN_OUTPUT}" "${WARM_RESUME_STATE}"
fi

SOTA_REGISTRY="${WARM_RMBENCH_SOTA_REGISTRY:-configs/rmbench/sota_v1.json}"
STAGE="${WARM_RMBENCH_STAGE:-shared}"
SPECIALIST_TASK="${WARM_RMBENCH_SPECIALIST_TASK:-}"
case "${STAGE}" in
  shared|specialist) ;;
  *) warm_die "WARM_RMBENCH_STAGE must be shared or specialist" ;;
esac
INITIALIZATION_CHECKPOINT="${WARM_INITIALIZATION_CHECKPOINT:-}"
INITIALIZATION_PARENT_CONFIG="${WARM_INITIALIZATION_PARENT_CONFIG:-}"
INITIALIZATION_FORK_MANIFEST="${WARM_INITIALIZATION_FORK_MANIFEST:-}"
INITIALIZATION_FORK_REASON="${WARM_INITIALIZATION_FORK_REASON:-}"
ALLOW_DIRECT_BASE_SPECIALIST="${WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST:-false}"
if [[ "${ALLOW_DIRECT_BASE_SPECIALIST}" != "true" && \
      "${ALLOW_DIRECT_BASE_SPECIALIST}" != "false" ]]; then
  warm_die "WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST must be true or false"
fi
if [[ "${STAGE}" == "shared" ]]; then
  if [[ -n "${INITIALIZATION_CHECKPOINT}${INITIALIZATION_PARENT_CONFIG}${INITIALIZATION_FORK_MANIFEST}${INITIALIZATION_FORK_REASON}" ]]; then
    warm_die "shared training cannot consume specialist-fork initialization"
  fi
elif [[ -n "${INITIALIZATION_CHECKPOINT}" ]]; then
  warm_require_file_or_directory \
    "${INITIALIZATION_CHECKPOINT}" \
    "${INITIALIZATION_CHECKPOINT%.pt}.training.json" \
    "${INITIALIZATION_PARENT_CONFIG}"
  [[ -n "${INITIALIZATION_FORK_MANIFEST}" ]] \
    || warm_die "WARM_INITIALIZATION_FORK_MANIFEST is required"
  [[ -n "${INITIALIZATION_FORK_REASON}" ]] \
    || warm_die "WARM_INITIALIZATION_FORK_REASON is required"
  [[ -z "${WARM_RESUME_STATE}" ]] \
    || warm_die "specialist initialization and WARM_RESUME_STATE are mutually exclusive"
elif [[ "${ALLOW_DIRECT_BASE_SPECIALIST}" != "true" ]]; then
  warm_die "specialist training requires a shared WARM checkpoint; direct base is opt-in only"
fi

M1="${WARM_ARTIFACT_ROOT}/m1"
M2="${WARM_ARTIFACT_ROOT}/m2"
TRAIN_CACHE="${M2}/candidates/hybrid_h32_train_k32"
DEV_CACHE="${M2}/candidates/hybrid_h32_dev_k32"
TRAIN_CONTRACT="${M2}/contracts/hybrid_h32_train_source.json"
DEV_CONTRACT="${M2}/contracts/hybrid_h32_dev_source.json"
TRAIN_STATS="${M1}/train_stats/dataset_stats.json"
CATALOG="${M1}/rmbench_catalog.json"
AUDIT="${M1}/rmbench_audit.json"
QUALIFICATION="${M1}/qualification/rmbench_h32.json"

warm_require_file_or_directory \
  "${FASTWAM_BASE_CHECKPOINT}" \
  "${RMBENCH_LEROBOT_ROOT}" \
  "${RMBENCH_LEROBOT_ROOT}/meta/warm_instruction_variants.jsonl" \
  "${RMBENCH_TEXT_CACHE}" \
  "${M1}/rmbench_conversion_manifest.json" \
  "${CATALOG}" \
  "${AUDIT}" \
  "${M1}/train_stats/train_stats_manifest.json" \
  "${TRAIN_STATS}" \
  "${M1}/features/train_features.list" \
  "${M1}/features/dev_features.list" \
  "${M1}/banks/hybrid_h32" \
  "${QUALIFICATION}" \
  "${TRAIN_CACHE}" \
  "${DEV_CACHE}" \
  "${TRAIN_CONTRACT}" \
  "${DEV_CONTRACT}"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
PROFILE_ARGS=()
if [[ -n "${WARM_RMBENCH_DATA_PROFILE:-}" ]]; then
  PROFILE_ARGS+=(--data-profile "${WARM_RMBENCH_DATA_PROFILE}")
fi
IFS=$'\t' read -r ROOT_SEED DATA_PROFILE SHARED_STEPS SAMPLER_MODE EVENT_BOOST \
  RECENT_EVENT_CAPACITY ACTION_SUMMARY_CAPACITY REPLAN_STEPS < <(
  python scripts/plan_warm_rmbench_sota.py \
    --registry "${SOTA_REGISTRY}" \
    "${PROFILE_ARGS[@]}" \
    --format tsv \
    --validate-manifest "${M1}/rmbench_conversion_manifest.json"
)
TRAIN_STEPS="${SHARED_STEPS}"
TASK_FILTER=null
if [[ "${STAGE}" == specialist ]]; then
  if [[ -z "${SPECIALIST_TASK}" ]]; then
    warm_die "specialist stage requires WARM_RMBENCH_SPECIALIST_TASK"
  fi
  IFS=$'\t' read -r TASK_NAME MEMORY_REGIME TRAIN_STEPS \
    RECENT_EVENT_CAPACITY ACTION_SUMMARY_CAPACITY REPLAN_STEPS \
    INFERENCE_STEPS TOP_K < <(
      python scripts/plan_warm_rmbench_sota.py \
        --registry "${SOTA_REGISTRY}" \
        --task "${SPECIALIST_TASK}" \
        --format tsv
    )
  TASK_FILTER="[${TASK_NAME}]"
fi
if [[ -n "${WARM_MAX_STEPS:-}" ]]; then
  if [[ ! "${WARM_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    warm_die "WARM_MAX_STEPS must be a positive integer"
  fi
  TRAIN_STEPS="${WARM_MAX_STEPS}"
fi
if [[ -n "${WARM_RUN_STEPS:-}" && ! "${WARM_RUN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  warm_die "WARM_RUN_STEPS must be a positive integer"
fi
printf 'RMBench training profile: stage=%s task=%s data=%s max_steps=%s recent_events=%s action_summaries=%s replan_steps=%s\n' \
  "${STAGE}" \
  "${SPECIALIST_TASK:-all}" \
  "${DATA_PROFILE}" \
  "${TRAIN_STEPS}" \
  "${RECENT_EVENT_CAPACITY}" \
  "${ACTION_SUMMARY_CAPACITY}" \
  "${REPLAN_STEPS}"

export RMBENCH_LEROBOT_ROOT
export RMBENCH_DATASET_STATS="${TRAIN_STATS}"
export RMBENCH_TEXT_CACHE

# Revalidate revision/catalog identity.  Full MP4 hashes were already bound by
# the production audit; this fast preflight still checks every inventory path
# and size before allocating GPUs.
python scripts/validate_rmbench_conversion.py \
  --dataset-root "${RMBENCH_LEROBOT_ROOT}" \
  --source-revision "${RMBENCH_SOURCE_REVISION}" \
  --rmbench-code-revision "${RMBENCH_CODE_REVISION}" \
  --skip-artifact-byte-hashes

# Recompute the no-training acceptance gate from immutable bytes.  Merely
# finding a stale JSON report is insufficient: any bank/cache rebuild or
# changed threshold must fail before a GPU process is launched.
python scripts/qualify_warm_rmbench_artifacts.py \
  --bank "${M1}/banks/hybrid_h32" \
  --train-feature-list "${M1}/features/train_features.list" \
  --train-candidate-cache "${TRAIN_CACHE}" \
  --dev-feature-list "${M1}/features/dev_features.list" \
  --dev-candidate-cache "${DEV_CACHE}" \
  --output "${QUALIFICATION}" \
  --action-horizon 32 \
  --query-stride 4 \
  --max-event-stride 4 \
  --phase-tolerance 0.10 \
  --phase-recall-threshold 32=0.85 \
  --phase-recall-threshold 128=0.95 \
  --parity-max-queries 2048 \
  --require-partial-action-queries \
  --verify-existing

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_MACHINES="${NUM_MACHINES:-${NNODES:-1}}"
export NUM_MACHINES
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-128}"
for entry in \
  "NPROC_PER_NODE:${NPROC_PER_NODE}" \
  "NUM_MACHINES:${NUM_MACHINES}" \
  "PER_DEVICE_BATCH_SIZE:${PER_DEVICE_BATCH_SIZE}" \
  "TARGET_GLOBAL_BATCH_SIZE:${TARGET_GLOBAL_BATCH_SIZE}"; do
  field="${entry%%:*}"
  value="${entry#*:}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    warm_die "${field} must be a positive integer"
  fi
done
MICRO_GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE * NUM_MACHINES))
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ ! "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    warm_die "GRADIENT_ACCUMULATION_STEPS must be a positive integer"
  fi
else
  if (( TARGET_GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
    warm_die "TARGET_GLOBAL_BATCH_SIZE must be divisible by batch_size * world_size"
  fi
  GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))
fi
EFFECTIVE_GLOBAL_BATCH=$((MICRO_GLOBAL_BATCH * GRADIENT_ACCUMULATION_STEPS))
if (( EFFECTIVE_GLOBAL_BATCH != TARGET_GLOBAL_BATCH_SIZE )); then
  warm_die "effective global batch ${EFFECTIVE_GLOBAL_BATCH} != target ${TARGET_GLOBAL_BATCH_SIZE}"
fi
printf 'RMBench batch contract: per_device=%s world=%s grad_accum=%s global=%s\n' \
  "${PER_DEVICE_BATCH_SIZE}" \
  "$((NPROC_PER_NODE * NUM_MACHINES))" \
  "${GRADIENT_ACCUMULATION_STEPS}" \
  "${EFFECTIVE_GLOBAL_BATCH}"
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
    task|data|data.*|model|model.*|output_dir|resume|initialization_checkpoint|initialization_fork_manifest|rmbench_training_stage|allow_direct_base_specialist|seed|max_steps|run_steps|batch_size|gradient_accumulation_steps|sampler|sampler.*|wandb.enabled|wandb.mode|--config-name|--config-name*)
      warm_die "protected formal-training override is not allowed: ${override}"
      ;;
  esac
done

TRAIN_OVERRIDES=(
  task=rmbench_warm_3cam384_1e-4
  "output_dir=${WARM_TRAIN_OUTPUT}"
  "wandb.enabled=${WANDB_ENABLED}"
  "wandb.mode=${WANDB_MODE}"
  "model.run_contract_path=${TRAIN_CONTRACT}"
  "model.validation_run_contract_path=${DEV_CONTRACT}"
  "model.base_checkpoint_path=${FASTWAM_BASE_CHECKPOINT}"
  "rmbench_training_stage=${STAGE}"
  "allow_direct_base_specialist=${ALLOW_DIRECT_BASE_SPECIALIST}"
  "seed=${ROOT_SEED}"
  "max_steps=${TRAIN_STEPS}"
  "batch_size=${PER_DEVICE_BATCH_SIZE}"
  "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "sampler.mode=${SAMPLER_MODE}"
  "sampler.event_boost=${EVENT_BOOST}"
  "data.train.dataset_dirs=[${RMBENCH_LEROBOT_ROOT}]"
  "data.train.episode_task_allowlist=${TASK_FILTER}"
  "data.train.text_embedding_cache_dir=${RMBENCH_TEXT_CACHE}"
  "data.warm_candidates.train.bank_directory=${M1}/banks/hybrid_h32"
  "data.warm_candidates.train.candidate_directory=${TRAIN_CACHE}"
  "data.warm_candidates.train.catalog_path=${CATALOG}"
  "data.warm_candidates.train.normalization_stats_path=${TRAIN_STATS}"
  "data.warm_candidates.train.audit_report_path=${AUDIT}"
  "data.warm_candidates.train.retrospective_feature_list=${M1}/features/train_features.list"
  "data.warm_candidates.train.retrospective_recent_event_capacity=${RECENT_EVENT_CAPACITY}"
  "data.warm_candidates.train.retrospective_action_summary_capacity=${ACTION_SUMMARY_CAPACITY}"
  "data.warm_candidates.train.retrospective_action_summary_chunk_size=${REPLAN_STEPS}"
  "data.val.dataset_dirs=[${RMBENCH_LEROBOT_ROOT}]"
  "data.val.episode_task_allowlist=${TASK_FILTER}"
  "data.val.text_embedding_cache_dir=${RMBENCH_TEXT_CACHE}"
  "data.warm_candidates.val.bank_directory=${M1}/banks/hybrid_h32"
  "data.warm_candidates.val.candidate_directory=${DEV_CACHE}"
  "data.warm_candidates.val.catalog_path=${CATALOG}"
  "data.warm_candidates.val.normalization_stats_path=${TRAIN_STATS}"
  "data.warm_candidates.val.audit_report_path=${AUDIT}"
  "data.warm_candidates.val.retrospective_feature_list=${M1}/features/dev_features.list"
  "data.warm_candidates.val.retrospective_recent_event_capacity=${RECENT_EVENT_CAPACITY}"
  "data.warm_candidates.val.retrospective_action_summary_capacity=${ACTION_SUMMARY_CAPACITY}"
  "data.warm_candidates.val.retrospective_action_summary_chunk_size=${REPLAN_STEPS}"
  "model.retrospection.episode_action_chunk_size=${REPLAN_STEPS}"
)
if [[ -n "${INITIALIZATION_CHECKPOINT}" ]]; then
  TRAIN_OVERRIDES+=(
    "initialization_checkpoint=${INITIALIZATION_CHECKPOINT}"
    "initialization_fork_manifest=${INITIALIZATION_FORK_MANIFEST}"
  )
fi
if [[ -n "${WARM_RESUME_STATE}" ]]; then
  TRAIN_OVERRIDES+=("resume=${WARM_RESUME_STATE}")
fi
if [[ -n "${WARM_RUN_STEPS:-}" ]]; then
  TRAIN_OVERRIDES+=("run_steps=${WARM_RUN_STEPS}")
fi
TRAIN_OVERRIDES+=("$@")

if [[ "${WARM_PREFLIGHT_RESOLVE:-true}" != "true" && \
      "${WARM_PREFLIGHT_RESOLVE:-true}" != "false" ]]; then
  warm_die "WARM_PREFLIGHT_RESOLVE must be true or false"
fi
if [[ "${WARM_PREFLIGHT_RESOLVE:-true}" == "true" ]]; then
  PREFLIGHT_OUTPUT="${WARM_PREFLIGHT_OUTPUT:-${WARM_TRAIN_OUTPUT}.resolved_config.preflight.yaml}"
  mkdir -p "$(dirname -- "${PREFLIGHT_OUTPUT}")"
  PREFLIGHT_TEMP="${PREFLIGHT_OUTPUT}.tmp.$$"
  trap 'rm -f -- "${PREFLIGHT_TEMP:-}"' EXIT
  python scripts/train.py \
    "${TRAIN_OVERRIDES[@]}" \
    --cfg job --resolve > "${PREFLIGHT_TEMP}"
  mv -f -- "${PREFLIGHT_TEMP}" "${PREFLIGHT_OUTPUT}"
  trap - EXIT
  printf 'RMBench Hydra preflight: %s\n' "${PREFLIGHT_OUTPUT}"
  if [[ -n "${INITIALIZATION_CHECKPOINT}" ]]; then
    python scripts/build_warm_training_fork_manifest.py \
      --parent-checkpoint "${INITIALIZATION_CHECKPOINT}" \
      --parent-config "${INITIALIZATION_PARENT_CONFIG}" \
      --child-config "${PREFLIGHT_OUTPUT}" \
      --output "${INITIALIZATION_FORK_MANIFEST}" \
      --fork-reason "${INITIALIZATION_FORK_REASON}"
  fi
elif [[ -n "${INITIALIZATION_CHECKPOINT}" ]]; then
  warm_die "formal specialist fork requires WARM_PREFLIGHT_RESOLVE=true"
fi

exec bash scripts/train_zero1.sh "${NPROC_PER_NODE}" "${TRAIN_OVERRIDES[@]}"
