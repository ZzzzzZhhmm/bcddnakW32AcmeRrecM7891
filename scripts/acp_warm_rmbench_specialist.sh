#!/usr/bin/env bash
set -euo pipefail

# Four-H100 ACP entrypoint for one official50 RMBench WARM specialist.
# It intentionally does not fetch, pull, commit, or modify repository files.

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
WARM_RMBENCH_SPECIALIST_TASK="${WARM_RMBENCH_SPECIALIST_TASK:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ -z "${WARM_RMBENCH_SPECIALIST_TASK}" ]]; then
  printf 'ERROR: WARM_RMBENCH_SPECIALIST_TASK is required\n' >&2
  exit 2
fi

case "${WARM_RMBENCH_SPECIALIST_TASK}" in
  observe_and_pickup|rearrange_blocks|put_back_block|swap_blocks|swap_T|\
  blocks_ranking_try|press_button|cover_blocks|battery_try) ;;
  *)
    printf 'ERROR: unsupported RMBench specialist task: %s\n' \
      "${WARM_RMBENCH_SPECIALIST_TASK}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${CONDA_ENV_DIR}/bin/python" || \
      ! -x "${CONDA_ENV_DIR}/bin/accelerate" ]]; then
  printf 'ERROR: persistent warm environment is incomplete: %s\n' \
    "${CONDA_ENV_DIR}" >&2
  exit 2
fi

export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_DIR}"
if ! git config --global --get-all safe.directory 2>/dev/null \
  | grep -Fqx -- "${PROJECT_DIR}"; then
  git config --global --add safe.directory "${PROJECT_DIR}"
fi

IFS=',' read -r -a CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#CUDA_DEVICE_LIST[@]} != 4 )); then
  printf 'ERROR: specialist training requires exactly four visible GPUs; got %s\n' \
    "${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

python - <<'PY'
import torch

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
export WARM_RMBENCH_DATA_PROFILE=official50-dev45
export WARM_RMBENCH_STAGE=specialist
export WARM_RMBENCH_SPECIALIST_TASK

SPECIALIST_RUN_ROOT="${WARM_SPECIALIST_RUN_ROOT:-${PROJECT_DIR}/runs/rmbench_official50_specialists}"
export WARM_TRAIN_OUTPUT="${WARM_TRAIN_OUTPUT:-${SPECIALIST_RUN_ROOT}/${WARM_RMBENCH_SPECIALIST_TASK}-s3407-v1}"
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

printf 'task=%s\noutput=%s\nconsole_log=%s\ncommit=%s\n' \
  "${WARM_RMBENCH_SPECIALIST_TASK}" \
  "${WARM_TRAIN_OUTPUT}" \
  "${CONSOLE_LOG}" \
  "$(git rev-parse HEAD)"

set -o pipefail
bash scripts/train_warm_rmbench_server.sh \
  num_workers=4 \
  learning_rate=1.0e-4 \
  weight_decay=1.0e-2 \
  log_every=10 \
  save_every=2000 \
  eval_every=0 \
  2>&1 | tee "${TEE_ARGS[@]}" "${CONSOLE_LOG}"
