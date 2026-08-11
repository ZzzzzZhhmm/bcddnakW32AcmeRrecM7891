#!/usr/bin/env bash
set -euo pipefail

# Four-H100 ACP entrypoint for official50 RMBench WARM shared/specialist runs.
#
# The primary checkout is an operator workspace: it may be updated while long
# jobs are running. Formal checkpoint attestations require the code tree to
# stay at one clean commit for the entire run. The first invocation therefore
# creates or reuses a detached worktree at the selected commit and re-executes
# this launcher from that immutable code tree. Outputs and model inputs
# continue to live under PROJECT_DIR and its sibling artifact roots.
#
# This entrypoint never fetches, pulls, commits, or checks out the primary
# worktree.

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
WARM_RMBENCH_SPECIALIST_TASK="${WARM_RMBENCH_SPECIALIST_TASK:-}"
WARM_RMBENCH_STAGE="${WARM_RMBENCH_STAGE:-specialist}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
WARM_TRAIN_CODE_DIR="${WARM_TRAIN_CODE_DIR:-}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

register_git_safe_directory() {
  local directory="$1"
  local canonical existing
  canonical="$(cd -- "${directory}" && pwd -P)" \
    || fail "cannot resolve Git checkout: ${directory}"
  existing="$(git config --global --get-all safe.directory 2>/dev/null || true)"
  if ! printf '%s\n' "${existing}" | grep -Fqx -- "${canonical}"; then
    git config --global --add safe.directory "${canonical}" \
      || fail "cannot register exact Git safe.directory: ${canonical}"
  fi
}

case "${WARM_RMBENCH_STAGE}" in
  shared)
    [[ -z "${WARM_RMBENCH_SPECIALIST_TASK}" ]] \
      || fail "shared stage must not set WARM_RMBENCH_SPECIALIST_TASK"
    ;;
  specialist)
    [[ -n "${WARM_RMBENCH_SPECIALIST_TASK}" ]] \
      || fail "specialist stage requires WARM_RMBENCH_SPECIALIST_TASK"
    case "${WARM_RMBENCH_SPECIALIST_TASK}" in
      observe_and_pickup|rearrange_blocks|put_back_block|swap_blocks|swap_T|\
      blocks_ranking_try|press_button|cover_blocks|battery_try) ;;
      *)
        fail "unsupported RMBench specialist task: ${WARM_RMBENCH_SPECIALIST_TASK}"
        ;;
    esac
    ;;
  *) fail "WARM_RMBENCH_STAGE must be shared or specialist" ;;
esac

if [[ ! -x "${CONDA_ENV_DIR}/bin/python" || \
      ! -x "${CONDA_ENV_DIR}/bin/accelerate" ]]; then
  fail "persistent warm environment is incomplete: ${CONDA_ENV_DIR}"
fi

export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Isolate the code before allocating GPUs or constructing an output directory.
if [[ -z "${WARM_TRAIN_CODE_DIR}" ]]; then
  [[ -e "${PROJECT_DIR}/.git" ]] \
    || fail "WARM Git checkout not found: ${PROJECT_DIR}"
  register_git_safe_directory "${PROJECT_DIR}"
  cd "${PROJECT_DIR}"

  # shellcheck source=scripts/warm_server_common.sh
  source "${PROJECT_DIR}/scripts/warm_server_common.sh"
  warm_require_private_checkout \
    || fail "primary WARM checkout failed formal provenance validation"
  TRAIN_COMMIT="${WARM_CODE_REVISION}"
  WARM_TRAIN_WORKTREE_ROOT="${WARM_TRAIN_WORKTREE_ROOT:-${PROJECT_DIR}_training/code}"
  WARM_TRAIN_LOCK_ROOT="${WARM_TRAIN_LOCK_ROOT:-${PROJECT_DIR}_training/locks}"
  TRAIN_CODE="${WARM_TRAIN_WORKTREE_ROOT}/${TRAIN_COMMIT}"
  mkdir -p "${WARM_TRAIN_WORKTREE_ROOT}" "${WARM_TRAIN_LOCK_ROOT}"
  command -v flock >/dev/null 2>&1 \
    || fail "flock is required for race-safe training worktree creation"

  exec 9>"${WARM_TRAIN_LOCK_ROOT}/formal-training-worktree.lock"
  flock -x 9
  if [[ ! -e "${TRAIN_CODE}/.git" ]]; then
    [[ ! -e "${TRAIN_CODE}" ]] \
      || fail "non-worktree path already exists: ${TRAIN_CODE}"
    git -C "${PROJECT_DIR}" worktree add --detach \
      "${TRAIN_CODE}" "${TRAIN_COMMIT}" \
      || fail "cannot create detached training worktree at ${TRAIN_COMMIT}"
  fi
  register_git_safe_directory "${TRAIN_CODE}"
  [[ "$(git -C "${TRAIN_CODE}" rev-parse HEAD)" == "${TRAIN_COMMIT}" ]] \
    || fail "training worktree does not match commit ${TRAIN_COMMIT}"
  [[ -z "$(git -C "${TRAIN_CODE}" status --porcelain --untracked-files=all)" ]] \
    || fail "training worktree is dirty: ${TRAIN_CODE}"
  flock -u 9
  exec 9>&-

  printf 'isolated_training_commit=%s\nisolated_training_code=%s\n' \
    "${TRAIN_COMMIT}" "${TRAIN_CODE}"
  exec env \
    PROJECT_DIR="${PROJECT_DIR}" \
    WARM_TRAIN_CODE_DIR="${TRAIN_CODE}" \
    WARM_CODE_REVISION="${TRAIN_COMMIT}" \
    WARM_RMBENCH_STAGE="${WARM_RMBENCH_STAGE}" \
    WARM_RMBENCH_SPECIALIST_TASK="${WARM_RMBENCH_SPECIALIST_TASK}" \
    bash "${TRAIN_CODE}/scripts/acp_warm_rmbench_specialist.sh" "$@"
fi

[[ -e "${WARM_TRAIN_CODE_DIR}/.git" ]] \
  || fail "isolated WARM training worktree not found: ${WARM_TRAIN_CODE_DIR}"
register_git_safe_directory "${WARM_TRAIN_CODE_DIR}"
[[ -n "${WARM_CODE_REVISION:-}" ]] \
  || fail "WARM_CODE_REVISION is missing after worktree isolation"
[[ "$(git -C "${WARM_TRAIN_CODE_DIR}" rev-parse HEAD)" == "${WARM_CODE_REVISION}" ]] \
  || fail "isolated training worktree commit changed"
[[ -z "$(git -C "${WARM_TRAIN_CODE_DIR}" status --porcelain --untracked-files=all)" ]] \
  || fail "isolated training worktree is dirty: ${WARM_TRAIN_CODE_DIR}"

export PYTHONPATH="${WARM_TRAIN_CODE_DIR}/src:${WARM_TRAIN_CODE_DIR}:${PYTHONPATH:-}"
cd "${WARM_TRAIN_CODE_DIR}"

# DeepSpeed is imported by Accelerate before scripts/train.py runs.  Configure
# and create its Triton cache now so a fresh ACP container cannot fail while
# probing a missing /root/.triton/autotune directory.  Use node-local storage,
# not AFS, for compiler caches and locks.
# shellcheck source=scripts/warm_server_common.sh
source "${WARM_TRAIN_CODE_DIR}/scripts/warm_server_common.sh"
WARM_JOB_LOCAL_CACHE_ROOT="${WARM_JOB_LOCAL_CACHE_ROOT:-/tmp/${USER:-warm}/warm-rmbench-cache/${HOSTNAME:-local}/${WARM_RMBENCH_SPECIALIST_TASK:-shared}}"
warm_configure_job_local_caches "${WARM_JOB_LOCAL_CACHE_ROOT}" \
  || fail "job-local compiler cache preflight failed"

IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#CUDA_DEVICE_LIST[@]} != 4 )); then
  fail "specialist training requires exactly four visible GPUs; got ${CUDA_VISIBLE_DEVICES}"
fi

python - "${WARM_TRAIN_CODE_DIR}" <<'PY'
import sys
from pathlib import Path

import fastwam
import torch

code_root = Path(sys.argv[1]).resolve()
module_path = Path(fastwam.__file__).resolve()
try:
    module_path.relative_to(code_root)
except ValueError as error:
    raise SystemExit(
        "fastwam import escaped the isolated training worktree: "
        f"module={module_path}, worktree={code_root}"
    ) from error
print(f"formal_code_import={module_path}")

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"expected exactly four visible CUDA devices, got {count}")
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    gib = props.total_memory / 1024**3
    if "H100" not in props.name or gib < 75:
        raise SystemExit(
            f"GPU {index} must be an 80GB H100, got {props.name} ({gib:.1f} GiB)"
        )
    print(f"gpu[{index}]={props.name} memory_gib={gib:.1f}")
PY

export WARM_ARTIFACT_ROOT="${WARM_ARTIFACT_ROOT:-${PROJECT_DIR}_artifacts/rmbench_official50_v1}"
export RMBENCH_LEROBOT_ROOT="${RMBENCH_LEROBOT_ROOT:-/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_official50_lerobot}"
export RMBENCH_TEXT_CACHE="${RMBENCH_TEXT_CACHE:-${PROJECT_DIR}_artifacts/text/rmbench_official50_v1}"
export FASTWAM_BASE_CHECKPOINT="${FASTWAM_BASE_CHECKPOINT:-${PROJECT_DIR}/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
export WARM_RMBENCH_DATA_PROFILE="${WARM_RMBENCH_DATA_PROFILE:-official50-dev45}"
export WARM_RMBENCH_STAGE
export WARM_RMBENCH_SPECIALIST_TASK
case "${WARM_RMBENCH_DATA_PROFILE}" in
  official50-dev45|scale200-dev190|scale500-dev480) ;;
  *) fail "unsupported RMBench data profile: ${WARM_RMBENCH_DATA_PROFILE}" ;;
esac

SPECIALIST_RUN_ROOT="${WARM_SPECIALIST_RUN_ROOT:-${PROJECT_DIR}/runs/rmbench_official50_specialists}"
SHARED_RUN_ROOT="${WARM_SHARED_RUN_ROOT:-${PROJECT_DIR}/runs/rmbench_official50_shared}"
# V5 uses dense factual events, explicit successor rows, same-domain typed
# episode memory, independent dual-gripper timing, and checkpoint schema v8.
# It is incompatible with earlier specialists whose sparse bank / phase
# cursor could loop one source forever or collapse both gripper channels.
if [[ "${WARM_RMBENCH_STAGE}" == "shared" ]]; then
  export WARM_TRAIN_OUTPUT="${WARM_TRAIN_OUTPUT:-${SHARED_RUN_ROOT}/shared-${WARM_RMBENCH_DATA_PROFILE}-s3407-v5}"
else
  export WARM_TRAIN_OUTPUT="${WARM_TRAIN_OUTPUT:-${SPECIALIST_RUN_ROOT}/${WARM_RMBENCH_SPECIALIST_TASK}-${WARM_RMBENCH_DATA_PROFILE}-s3407-v5}"
fi

# A formal specialist starts from the complete shared WARM model, but with a
# brand-new optimizer/scheduler/step trajectory.  Falling back to the generic
# FastWAM base is intentionally noisy and requires an explicit research-only
# acknowledgement.
if [[ "${WARM_RMBENCH_STAGE}" == "specialist" ]]; then
  WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST="${WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST:-false}"
  if [[ "${WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST}" != "true" && \
        "${WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST}" != "false" ]]; then
    fail "WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST must be true or false"
  fi
  if [[ -z "${WARM_RMBENCH_SHARED_CHECKPOINT:-}" ]]; then
    WARM_RMBENCH_SHARED_CHECKPOINT="${SHARED_RUN_ROOT}/shared-${WARM_RMBENCH_DATA_PROFILE}-s3407-v5/checkpoints/weights/step_030000.pt"
  fi
  if [[ -f "${WARM_RMBENCH_SHARED_CHECKPOINT}" ]]; then
    export WARM_INITIALIZATION_CHECKPOINT="$(cd -- "$(dirname -- "${WARM_RMBENCH_SHARED_CHECKPOINT}")" && pwd -P)/$(basename -- "${WARM_RMBENCH_SHARED_CHECKPOINT}")"
    export WARM_INITIALIZATION_PARENT_CONFIG="$(dirname -- "$(dirname -- "$(dirname -- "${WARM_INITIALIZATION_CHECKPOINT}")")")/config.yaml"
    export WARM_INITIALIZATION_FORK_MANIFEST="${WARM_TRAIN_OUTPUT}.fork.json"
    export WARM_INITIALIZATION_FORK_REASON="shared nine-task WARM to ${WARM_RMBENCH_SPECIALIST_TASK} specialist; fresh optimizer/scheduler/step"
    [[ -f "${WARM_INITIALIZATION_CHECKPOINT%.pt}.training.json" ]] \
      || fail "shared WARM checkpoint lacks formal training attestation: ${WARM_INITIALIZATION_CHECKPOINT}"
    [[ -f "${WARM_INITIALIZATION_PARENT_CONFIG}" ]] \
      || fail "shared WARM parent config is missing: ${WARM_INITIALIZATION_PARENT_CONFIG}"
  elif [[ "${WARM_RMBENCH_ALLOW_DIRECT_BASE_SPECIALIST}" == "true" ]]; then
    printf 'WARNING: explicit direct-base specialist fallback enabled; no shared WARM initialization\n' >&2
    unset WARM_INITIALIZATION_CHECKPOINT WARM_INITIALIZATION_PARENT_CONFIG \
      WARM_INITIALIZATION_FORK_MANIFEST WARM_INITIALIZATION_FORK_REASON
  else
    fail "shared WARM checkpoint is required before specialists: ${WARM_RMBENCH_SHARED_CHECKPOINT}"
  fi
fi
export WARM_PREFLIGHT_RESOLVE="${WARM_PREFLIGHT_RESOLVE:-true}"
export WARM_PREFLIGHT_OUTPUT="${WARM_PREFLIGHT_OUTPUT:-${WARM_TRAIN_OUTPUT}.resolved_config.preflight.yaml}"
export WARM_WANDB_ENABLED="${WARM_WANDB_ENABLED:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export NPROC_PER_NODE=4
export NUM_MACHINES=1
export PER_DEVICE_BATCH_SIZE=8
export TARGET_GLOBAL_BATCH_SIZE=128
export GRADIENT_ACCUMULATION_STEPS=4

mkdir -p -- "$(dirname -- "${WARM_TRAIN_OUTPUT}")"
CONSOLE_LOG="${WARM_CONSOLE_LOG:-${WARM_TRAIN_OUTPUT}.console.log}"
TEE_ARGS=()
if [[ -n "${WARM_RESUME_STATE:-}" ]]; then
  TEE_ARGS=(-a)
fi

printf 'stage=%s\ntask=%s\noutput=%s\nconsole_log=%s\ncommit=%s\n' \
  "${WARM_RMBENCH_STAGE}" \
  "${WARM_RMBENCH_SPECIALIST_TASK:-all}" \
  "${WARM_TRAIN_OUTPUT}" \
  "${CONSOLE_LOG}" \
  "${WARM_CODE_REVISION}"

set -o pipefail
bash scripts/train_warm_rmbench_server.sh \
  num_workers=4 \
  learning_rate=5.0e-5 \
  weight_decay=1.0e-2 \
  log_every=10 \
  save_every=1000 \
  eval_every=500 \
  eval_num_samples=32 \
  2>&1 | tee "${TEE_ARGS[@]}" "${CONSOLE_LOG}"
