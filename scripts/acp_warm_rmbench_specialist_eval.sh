#!/usr/bin/env bash
set -euo pipefail

# One-H100 ACP entrypoint for one formal RMBench specialist evaluation.
#
# The launcher itself may live at a newer operational commit than the trained
# checkpoint. Formal WARM contracts require the model/evaluator code to match
# the checkpoint attestation exactly, so this script creates or reuses a
# detached, read-only worktree at the attested training commit. It performs no
# fetch, pull, checkout of the primary worktree, or other network operation.

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
RMBENCH_ROOT="${RMBENCH_ROOT:-/mnt/afs/task3_2/L202500276_lwz/external/RMBench-official}"
RMBENCH_HF_REVISION_MARKER="${RMBENCH_HF_REVISION_MARKER:-/mnt/afs/task3_2/L202500276_lwz/datasets/rmbench_hf_revision.json}"
WARM_ARTIFACT_ROOT="${WARM_ARTIFACT_ROOT:-${PROJECT_DIR}_artifacts/rmbench_official50_v1}"

FASTWAM_BASE_CHECKPOINT="${FASTWAM_BASE_CHECKPOINT:-${PROJECT_DIR}/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
WARM_DINO_CHECKPOINT="${WARM_DINO_CHECKPOINT:-${PROJECT_DIR}/checkpoints/dinov2-base}"
WARM_VAE_CHECKPOINT="${WARM_VAE_CHECKPOINT:-${PROJECT_DIR}/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
WARM_TEXT_ENCODER="${WARM_TEXT_ENCODER:-${PROJECT_DIR}/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors}"
WARM_TOKENIZER="${WARM_TOKENIZER:-${PROJECT_DIR}/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl}"

WARM_RMBENCH_TASK="${WARM_RMBENCH_TASK:-}"
WARM_ROOT_SEED="${WARM_ROOT_SEED:-3407}"
WARM_EVAL_LABEL="${WARM_EVAL_LABEL:-formal100-s3407-v1}"
WARM_EVAL_BASE="${WARM_EVAL_BASE:-${PROJECT_DIR}_evaluations}"
WARM_EVAL_WORKTREE_ROOT="${WARM_EVAL_WORKTREE_ROOT:-${WARM_EVAL_BASE}/code}"
WARM_RMBENCH_EVAL_BASE="${WARM_RMBENCH_EVAL_BASE:-${WARM_EVAL_BASE}/rmbench_official50}"
WARM_SPECIALIST_RUN_ROOT="${WARM_SPECIALIST_RUN_ROOT:-${PROJECT_DIR}/runs/rmbench_official50_specialists}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

register_git_safe_directory() {
  local directory="$1"
  if ! git config --global --get-all safe.directory 2>/dev/null \
    | grep -Fqx -- "${directory}"; then
    git config --global --add safe.directory "${directory}" \
      || fail "cannot register exact Git safe.directory: ${directory}"
  fi
}

[[ -n "${WARM_RMBENCH_TASK}" ]] \
  || fail "WARM_RMBENCH_TASK is required"
case "${WARM_RMBENCH_TASK}" in
  observe_and_pickup) DEFAULT_FINAL_STEP=6000 ;;
  rearrange_blocks|put_back_block) DEFAULT_FINAL_STEP=8000 ;;
  swap_blocks|swap_T) DEFAULT_FINAL_STEP=10000 ;;
  blocks_ranking_try) DEFAULT_FINAL_STEP=14000 ;;
  press_button|cover_blocks|battery_try) DEFAULT_FINAL_STEP=12000 ;;
  *) fail "unsupported RMBench specialist task: ${WARM_RMBENCH_TASK}" ;;
esac
[[ "${WARM_ROOT_SEED}" == "3407" ]] \
  || fail "formal RMBench specialist evaluation requires root seed 3407"
[[ "${WARM_EVAL_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || fail "WARM_EVAL_LABEL contains unsafe characters"
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]] \
  || fail "one specialist evaluation uses exactly one visible GPU"
[[ -x "${CONDA_ENV_DIR}/bin/python" ]] \
  || fail "persistent warm Python environment is incomplete: ${CONDA_ENV_DIR}"
[[ -e "${PROJECT_DIR}/.git" ]] \
  || fail "WARM Git checkout not found: ${PROJECT_DIR}"
[[ -e "${RMBENCH_ROOT}/.git" ]] \
  || fail "pinned RMBench Git checkout not found: ${RMBENCH_ROOT}"

PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true

# A fresh ACP container usually runs as root while the persistent AFS trees
# belong to another uid. Admit only these exact paths; never use '*'.
register_git_safe_directory "${PROJECT_DIR}"
register_git_safe_directory "${RMBENCH_ROOT}"

[[ "$(git -C "${RMBENCH_ROOT}" rev-parse HEAD)" == "${RMBENCH_CODE_REVISION}" ]] \
  || fail "RMBench checkout revision does not match ${RMBENCH_CODE_REVISION}"

FINAL_STEP="${WARM_FINAL_STEP:-${DEFAULT_FINAL_STEP}}"
[[ "${FINAL_STEP}" =~ ^[1-9][0-9]*$ ]] \
  || fail "WARM_FINAL_STEP must be a positive integer"
STEP_TEXT="$(printf '%06d' "${FINAL_STEP}")"
WARM_TRAIN_RUN_DIR="${WARM_TRAIN_RUN_DIR:-${WARM_SPECIALIST_RUN_ROOT}/${WARM_RMBENCH_TASK}-s3407-v1}"
WARM_CHECKPOINT="${WARM_CHECKPOINT:-${WARM_TRAIN_RUN_DIR}/checkpoints/weights/step_${STEP_TEXT}.pt}"
WARM_TRAINING_ATTESTATION="${WARM_TRAINING_ATTESTATION:-${WARM_CHECKPOINT%.pt}.training.json}"
WARM_TRAIN_CONFIG="${WARM_TRAIN_CONFIG:-}"

required_paths=(
  "${WARM_CHECKPOINT}"
  "${WARM_TRAINING_ATTESTATION}"
  "${RMBENCH_HF_REVISION_MARKER}"
  "${FASTWAM_BASE_CHECKPOINT}"
  "${WARM_DINO_CHECKPOINT}"
  "${WARM_VAE_CHECKPOINT}"
  "${WARM_TEXT_ENCODER}"
  "${WARM_TOKENIZER}"
  "${WARM_ARTIFACT_ROOT}/m1/banks/hybrid_h32"
  "${WARM_ARTIFACT_ROOT}/m1/features/contracts/normalizer_contract.json"
  "${WARM_ARTIFACT_ROOT}/m1/features/contracts/encoder_contract.json"
  "${WARM_ARTIFACT_ROOT}/m1/features/contracts/camera_contract.json"
  "${WARM_ARTIFACT_ROOT}/m1/train_stats/dataset_stats.json"
  "${WARM_ARTIFACT_ROOT}/m1/rmbench_catalog.json"
  "${WARM_ARTIFACT_ROOT}/m1/rmbench_audit.json"
  "${WARM_ARTIFACT_ROOT}/m2/contracts/hybrid_h32_train_source.json"
  "${WARM_ARTIFACT_ROOT}/m2/contracts/hybrid_h32_dev_source.json"
)
for path in "${required_paths[@]}"; do
  [[ -e "${path}" ]] || fail "required evaluation input not found: ${path}"
done

# The formal contract builder re-hashes the full checkpoint. This lightweight
# preflight validates identity fields before creating a Git worktree or output.
ATTESTATION_FACTS="$("${PYTHON_BIN}" - \
  "${WARM_TRAINING_ATTESTATION}" "${WARM_CHECKPOINT}" \
  "${WARM_RMBENCH_TASK}" "${FINAL_STEP}" <<'PY'
import json
import re
import sys
from pathlib import Path

attestation_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
expected_task = sys.argv[3]
expected_step = int(sys.argv[4])
value = json.loads(attestation_path.read_text(encoding="utf-8"))
required = {
    "git_commit",
    "checkpoint_step",
    "actual_global_step",
    "actual_max_steps",
    "source_policy",
    "effective_batch_size",
    "root_seed",
}
missing = required - set(value)
if missing:
    raise SystemExit(f"training attestation is missing fields: {sorted(missing)}")
commit = str(value["git_commit"])
if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit("training attestation has an invalid Git commit")
match = re.fullmatch(r"step_(\d{6})\.pt", checkpoint_path.name)
if match is None or int(match.group(1)) != expected_step:
    raise SystemExit("checkpoint filename does not match the selected final step")
if not (
    int(value["checkpoint_step"])
    == int(value["actual_global_step"])
    == int(value["actual_max_steps"])
    == expected_step
):
    raise SystemExit("checkpoint is not the completed specialist training step")
if value["source_policy"] != "fixed_context_top1":
    raise SystemExit("formal WARM evaluation requires fixed_context_top1")
if int(value["effective_batch_size"]) != 128:
    raise SystemExit("specialist checkpoint does not use global batch 128")
if int(value["root_seed"]) != 3407:
    raise SystemExit("specialist checkpoint does not use root seed 3407")
print(commit, expected_step)
PY
)" || fail "cannot validate final checkpoint attestation facts"
read -r TRAIN_COMMIT CHECKPOINT_STEP <<< "${ATTESTATION_FACTS}"

git -C "${PROJECT_DIR}" cat-file -e "${TRAIN_COMMIT}^{commit}" 2>/dev/null \
  || fail "training commit ${TRAIN_COMMIT} is absent locally; synchronize Git once in CCI"

EVAL_CODE="${WARM_EVAL_WORKTREE_ROOT}/${TRAIN_COMMIT}"
WORKTREE_LOCK_ROOT="${WARM_EVAL_BASE}/locks"
mkdir -p "${WARM_EVAL_WORKTREE_ROOT}" "${WORKTREE_LOCK_ROOT}"
command -v flock >/dev/null 2>&1 \
  || fail "flock is required for race-safe ACP evaluation worktree creation"
exec 9>"${WORKTREE_LOCK_ROOT}/rmbench-eval-worktree.lock"
flock -x 9
if [[ ! -e "${EVAL_CODE}/.git" ]]; then
  [[ ! -e "${EVAL_CODE}" ]] \
    || fail "non-worktree path already exists: ${EVAL_CODE}"
  git -C "${PROJECT_DIR}" worktree add --detach "${EVAL_CODE}" "${TRAIN_COMMIT}"
fi
register_git_safe_directory "${EVAL_CODE}"
[[ "$(git -C "${EVAL_CODE}" rev-parse HEAD)" == "${TRAIN_COMMIT}" ]] \
  || fail "evaluation worktree does not match checkpoint commit ${TRAIN_COMMIT}"
[[ -z "$(git -C "${EVAL_CODE}" status --porcelain --untracked-files=normal)" ]] \
  || fail "evaluation worktree is dirty: ${EVAL_CODE}"
flock -u 9
exec 9>&-

[[ -f "${EVAL_CODE}/scripts/build_warm_rmbench_sota_task_contract_server.sh" ]] \
  || fail "training commit lacks the RMBench specialist contract builder"
[[ -f "${EVAL_CODE}/scripts/evaluate_warm_rmbench_task_server.sh" ]] \
  || fail "training commit lacks the RMBench specialist evaluator"

export PYTHONPATH="${EVAL_CODE}/src:${EVAL_CODE}"
"${PYTHON_BIN}" - \
  "${WARM_TRAIN_RUN_DIR}" "${WARM_TRAIN_CONFIG}" \
  "${WARM_TRAINING_ATTESTATION}" "${WARM_RMBENCH_TASK}" \
  "${FINAL_STEP}" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

from fastwam.models.warm.training_attestation import (
    load_training_attestation,
    training_config_hashes,
)

run_dir = Path(sys.argv[1])
explicit_config = sys.argv[2].strip()
attestation_path = Path(sys.argv[3])
expected_task = sys.argv[4]
expected_step = int(sys.argv[5])
attestation = load_training_attestation(attestation_path)
candidates = (
    [Path(explicit_config)]
    if explicit_config
    else sorted(run_dir.glob("config*.yaml"))
)
matches = []
for config_path in candidates:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        continue
    full_sha, _ = training_config_hashes(config)
    if full_sha == attestation.resolved_train_config_sha256:
        matches.append((config_path, config, full_sha))
if len(matches) != 1:
    raise SystemExit(
        "expected exactly one resolved training config matching the "
        f"checkpoint attestation, found {[str(item[0]) for item in matches]}"
    )
config_path, config, full_sha = matches[0]
if int(config.get("seed", -1)) != 3407:
    raise SystemExit("resolved training config seed is not 3407")
if int(config.get("max_steps", -1)) != expected_step:
    raise SystemExit("resolved training config max_steps differs from checkpoint")
data = config.get("data")
if not isinstance(data, dict):
    raise SystemExit("resolved training config has no data mapping")
for split in ("train", "val"):
    node = data.get(split)
    if not isinstance(node, dict):
        raise SystemExit(f"resolved training config has no data.{split} mapping")
    allowlist = node.get("episode_task_allowlist")
    if list(allowlist or []) != [expected_task]:
        raise SystemExit(
            f"data.{split}.episode_task_allowlist does not select "
            f"{expected_task!r}: {allowlist!r}"
        )
print(
    f"training_config_ok path={config_path} task={expected_task} "
    f"step={expected_step} sha256={full_sha}"
)
PY
export WARM_CODE_REVISION="${TRAIN_COMMIT}"
export WARM_RMBENCH_TASK
export WARM_ROOT_SEED
export WARM_CHECKPOINT
export WARM_TRAINING_ATTESTATION
export WARM_ARTIFACT_ROOT
export RMBENCH_ROOT
export RMBENCH_HF_REVISION_MARKER
export FASTWAM_BASE_CHECKPOINT
export WARM_DINO_CHECKPOINT
export WARM_VAE_CHECKPOINT
export WARM_TEXT_ENCODER
export WARM_TOKENIZER

export WARM_EXPERIMENT_ID="${WARM_EXPERIMENT_ID:-full_warm}"
export WARM_EVALUATION_NAMESPACE="${WARM_EVALUATION_NAMESPACE:-warm-rmbench-sota-v1}"
export WARM_GPU_ID=0
export WARM_EVAL_DEVICE="${WARM_EVAL_DEVICE:-cuda}"
export WARM_DINO_DEVICE="${WARM_DINO_DEVICE:-cuda}"
export WARM_DINO_BATCH_SIZE="${WARM_DINO_BATCH_SIZE:-1}"
export WARM_MIXED_PRECISION="${WARM_MIXED_PRECISION:-bf16}"

CHECKPOINT_NAME="$(basename "${WARM_CHECKPOINT}" .pt)"
WARM_RMBENCH_ONLINE_CONTRACT="${WARM_RMBENCH_ONLINE_CONTRACT:-${WARM_RMBENCH_EVAL_BASE}/contracts/${WARM_RMBENCH_TASK}-${CHECKPOINT_NAME}-${TRAIN_COMMIT:0:12}}"
WARM_EVAL_ROOT="${WARM_EVAL_ROOT:-${WARM_RMBENCH_EVAL_BASE}/results/${WARM_RMBENCH_TASK}/${CHECKPOINT_NAME}/${WARM_EVAL_LABEL}}"
export WARM_RMBENCH_ONLINE_CONTRACT WARM_EVAL_ROOT

[[ ! -e "${WARM_EVAL_ROOT}" && ! -L "${WARM_EVAL_ROOT}" ]] \
  || fail "immutable evaluation root already exists: ${WARM_EVAL_ROOT}; change WARM_EVAL_LABEL"
[[ ! -e "${WARM_EVAL_ROOT}.console.log" ]] \
  || fail "evaluation console log already exists: ${WARM_EVAL_ROOT}.console.log; change WARM_EVAL_LABEL"
[[ ! -e "${WARM_RMBENCH_ONLINE_CONTRACT}/.incomplete.json" ]] \
  || fail "incomplete online contract exists: ${WARM_RMBENCH_ONLINE_CONTRACT}"

mkdir -p \
  "$(dirname "${WARM_RMBENCH_ONLINE_CONTRACT}")" \
  "$(dirname "${WARM_EVAL_ROOT}")"

"${PYTHON_BIN}" - <<'PY'
import torch

count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(f"expected exactly one visible CUDA device, got {count}")
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 1024**3
if "H100" not in props.name or gib < 75:
    raise SystemExit(
        f"formal specialist evaluation requires an 80GB H100, "
        f"got {props.name} ({gib:.1f} GiB)"
    )
print(f"gpu[0]={props.name} memory_gib={gib:.1f}")
PY

printf '%s\n' \
  "launcher_commit=$(git -C "${PROJECT_DIR}" rev-parse HEAD)" \
  "training_commit=${TRAIN_COMMIT}" \
  "evaluation_code=${EVAL_CODE}" \
  "task=${WARM_RMBENCH_TASK}" \
  "checkpoint_step=${CHECKPOINT_STEP}" \
  "checkpoint=${WARM_CHECKPOINT}" \
  "contract=${WARM_RMBENCH_ONLINE_CONTRACT}" \
  "evaluation_root=${WARM_EVAL_ROOT}"

cd "${EVAL_CODE}"
set -o pipefail
{
  if [[ ! -e "${WARM_RMBENCH_ONLINE_CONTRACT}" ]]; then
    printf '%s\n' "===== BUILD RMBENCH SPECIALIST CONTRACT ====="
    bash scripts/build_warm_rmbench_sota_task_contract_server.sh
  else
    printf 'reusing complete online contract: %s\n' \
      "${WARM_RMBENCH_ONLINE_CONTRACT}"
  fi
  printf '%s\n' "===== RUN OFFICIAL 100-EPISODE RMBENCH EVALUATION ====="
  bash scripts/evaluate_warm_rmbench_task_server.sh
} 2>&1 | tee "${WARM_EVAL_ROOT}.console.log"
