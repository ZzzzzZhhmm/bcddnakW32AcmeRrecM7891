#!/usr/bin/env bash
set -euo pipefail

# Reusable ACP/CCI entrypoint for one complete-WARM LIBERO rollout job.
#
# EVAL_ACTION=prepare performs the persistent, one-time setup and full final
# checkpoint verification. EVAL_ACTION=run reuses that setup, regenerates or
# validates the exact task inputs, and executes the contract-bound evaluator.
# No network operation is performed by this script.

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
WARM_ARTIFACT_ROOT="${WARM_ARTIFACT_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM_artifacts/libero_v1}"
WARM_TRAIN_RUN_DIR="${WARM_TRAIN_RUN_DIR:-${PROJECT_DIR}/runs/libero_warm_2cam224_1e-4/warm-full-4xh100-zero1-numerics-20260718-110409}"
WARM_CHECKPOINT="${WARM_CHECKPOINT:-${WARM_TRAIN_RUN_DIR}/checkpoints/weights/step_019100.pt}"
WARM_TRAINING_ATTESTATION="${WARM_TRAINING_ATTESTATION:-${WARM_CHECKPOINT%.pt}.training.json}"

FASTWAM_BASE_CHECKPOINT="${FASTWAM_BASE_CHECKPOINT:-${PROJECT_DIR}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
WARM_DINO_CHECKPOINT="${WARM_DINO_CHECKPOINT:-${PROJECT_DIR}/checkpoints/dinov2-base}"
WARM_VAE_CHECKPOINT="${WARM_VAE_CHECKPOINT:-${PROJECT_DIR}/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
WARM_TEXT_ENCODER="${WARM_TEXT_ENCODER:-${PROJECT_DIR}/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors}"
WARM_TOKENIZER="${WARM_TOKENIZER:-${PROJECT_DIR}/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl}"

WARM_EVAL_BASE="${WARM_EVAL_BASE:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations}"
WARM_EVAL_WORKTREE_ROOT="${WARM_EVAL_WORKTREE_ROOT:-${WARM_EVAL_BASE}/code}"
WARM_EVAL_INPUT_ROOT="${WARM_EVAL_INPUT_ROOT:-${WARM_EVAL_BASE}/inputs}"

EVAL_ACTION="${EVAL_ACTION:-run}"
WARM_TASK_SUITE="${WARM_TASK_SUITE:-libero_10}"
WARM_TASK_ID="${WARM_TASK_ID:-0}"
WARM_ROOT_SEED="${WARM_ROOT_SEED:-17}"
WARM_EVAL_LABEL="${WARM_EVAL_LABEL:-formal}"
WARM_EVALUATION_NAMESPACE="${WARM_EVALUATION_NAMESPACE:-warm-libero-full-v1}"
WARM_EVAL_DEVICE="${WARM_EVAL_DEVICE:-cuda}"
WARM_REQUIRE_MUJOCO_VERSION="${WARM_REQUIRE_MUJOCO_VERSION:-3.3.2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

case "${EVAL_ACTION}" in
  prepare|run) ;;
  *) fail "EVAL_ACTION must be prepare or run, got ${EVAL_ACTION}" ;;
esac

case "${WARM_TASK_SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10) ;;
  *) fail "unsupported WARM_TASK_SUITE=${WARM_TASK_SUITE}" ;;
esac
[[ "${WARM_TASK_ID}" =~ ^[0-9]+$ ]] || fail "WARM_TASK_ID must be non-negative"
[[ "${WARM_ROOT_SEED}" =~ ^[0-9]+$ ]] || fail "WARM_ROOT_SEED must be non-negative"
[[ "${WARM_EVAL_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "WARM_EVAL_LABEL contains unsafe characters"
if [[ "${EVAL_ACTION}" == run && "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  fail "one LIBERO task uses one GPU; set CUDA_VISIBLE_DEVICES to one device"
fi

[[ -d "${PROJECT_DIR}/.git" ]] || fail "WARM repository not found: ${PROJECT_DIR}"
[[ -x "${CONDA_ENV_DIR}/bin/python" ]] || fail "Python environment not found: ${CONDA_ENV_DIR}"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

required_paths=(
  "${WARM_CHECKPOINT}"
  "${WARM_TRAINING_ATTESTATION}"
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
  "${WARM_ARTIFACT_ROOT}/m1/libero_catalog.json"
  "${WARM_ARTIFACT_ROOT}/m1/libero_audit.json"
  "${WARM_ARTIFACT_ROOT}/m2/contracts/hybrid_h32_train_source.json"
  "${WARM_ARTIFACT_ROOT}/m2/contracts/hybrid_h32_dev_source.json"
)
for path in "${required_paths[@]}"; do
  [[ -e "${path}" ]] || fail "required evaluation input not found: ${path}"
done

# Read only the canonical sidecar here. The full checkpoint hash is verified in
# prepare mode and, unconditionally, by build_warm_online_contract.py on every
# formal run before the model is allocated.
ATTESTATION_FACTS="$(${PYTHON_BIN} - "${WARM_TRAINING_ATTESTATION}" "${WARM_CHECKPOINT}" <<'PY'
import json
import re
import sys
from pathlib import Path

attestation_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
with attestation_path.open(encoding="utf-8") as handle:
    value = json.load(handle)
match = re.fullmatch(r"step_(\d{6})\.pt", checkpoint_path.name)
if match is None:
    raise SystemExit("checkpoint must use the canonical step_NNNNNN.pt name")
step = int(match.group(1))
required = {
    "git_commit",
    "checkpoint_step",
    "actual_global_step",
    "actual_max_steps",
    "source_policy",
    "effective_batch_size",
}
missing = required - set(value)
if missing:
    raise SystemExit(f"training attestation is missing fields: {sorted(missing)}")
if not re.fullmatch(r"[0-9a-f]{40}", str(value["git_commit"])):
    raise SystemExit("training attestation contains an invalid Git commit")
if not (
    int(value["checkpoint_step"])
    == int(value["actual_global_step"])
    == int(value["actual_max_steps"])
    == step
):
    raise SystemExit("checkpoint is not the completed final training step")
if value["source_policy"] != "fixed_context_top1":
    raise SystemExit("complete WARM evaluation requires fixed_context_top1")
if int(value["effective_batch_size"]) != 128:
    raise SystemExit("unexpected effective training batch size")
print(value["git_commit"], step)
PY
)" || fail "cannot validate final training-attestation facts"
read -r TRAIN_COMMIT CHECKPOINT_STEP <<< "${ATTESTATION_FACTS}"

git -C "${PROJECT_DIR}" cat-file -e "${TRAIN_COMMIT}^{commit}" 2>/dev/null \
  || fail "training commit ${TRAIN_COMMIT} is absent locally; synchronize Git once before ACP evaluation"

EVAL_CODE="${WARM_EVAL_WORKTREE_ROOT}/${TRAIN_COMMIT}"
mkdir -p "${WARM_EVAL_WORKTREE_ROOT}"
if [[ ! -e "${EVAL_CODE}/.git" ]]; then
  [[ ! -e "${EVAL_CODE}" ]] || fail "non-worktree path already exists: ${EVAL_CODE}"
  git -C "${PROJECT_DIR}" worktree add --detach "${EVAL_CODE}" "${TRAIN_COMMIT}"
fi
[[ "$(git -C "${EVAL_CODE}" rev-parse HEAD)" == "${TRAIN_COMMIT}" ]] \
  || fail "evaluation worktree does not match the checkpoint commit"
[[ -z "$(git -C "${EVAL_CODE}" status --porcelain)" ]] \
  || fail "evaluation worktree is dirty: ${EVAL_CODE}"
[[ -f "${EVAL_CODE}/scripts/evaluate_warm_full_server.sh" ]] \
  || fail "training commit has no complete-WARM LIBERO evaluator"

export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_DIR}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${EVAL_CODE}/src:${EVAL_CODE}"

export WARM_ARTIFACT_ROOT FASTWAM_BASE_CHECKPOINT WARM_CHECKPOINT
export WARM_TRAINING_ATTESTATION WARM_DINO_CHECKPOINT WARM_VAE_CHECKPOINT
export WARM_TEXT_ENCODER WARM_TOKENIZER WARM_EVALUATION_NAMESPACE
export WARM_EVAL_DEVICE WARM_TASK_SUITE WARM_TASK_ID WARM_ROOT_SEED

WARM_REQUIRE_CUDA="${WARM_REQUIRE_CUDA:-$([[ "${EVAL_ACTION}" == run ]] && echo true || echo false)}"
export WARM_REQUIRE_CUDA WARM_REQUIRE_MUJOCO_VERSION
"${PYTHON_BIN}" - <<'PY'
import os

import mujoco
import torch
from libero.libero import benchmark, get_libero_path

expected_mujoco = os.environ["WARM_REQUIRE_MUJOCO_VERSION"]
if mujoco.__version__ != expected_mujoco:
    raise SystemExit(
        f"MuJoCo version mismatch: {mujoco.__version__} != {expected_mujoco}"
    )
if os.environ["WARM_REQUIRE_CUDA"] == "true" and not torch.cuda.is_available():
    raise SystemExit("CUDA is required for WARM rollout but is unavailable")
suites = benchmark.get_benchmark_dict()
required = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
if not required.issubset(suites):
    raise SystemExit(f"LIBERO registry is incomplete: {sorted(suites)}")
print(f"runtime_ok torch={torch.__version__} mujoco={mujoco.__version__}")
print(f"bddl_root={get_libero_path('bddl_files')}")
print(f"init_states_root={get_libero_path('init_states')}")
PY

prepare_task_inputs() {
  local suite="$1" task_id="$2"
  "${PYTHON_BIN}" - "${WARM_EVAL_INPUT_ROOT}" "${suite}" "${task_id}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from libero.libero import benchmark, get_libero_path

root = Path(sys.argv[1]).expanduser().resolve()
suite_name = sys.argv[2]
task_id = int(sys.argv[3])
suite = benchmark.get_benchmark_dict()[suite_name]()
n_tasks = int(suite.n_tasks)
if task_id < 0 or task_id >= n_tasks:
    raise SystemExit(f"task id {task_id} is outside [0, {n_tasks}) for {suite_name}")
task = suite.get_task(task_id)
states = suite.get_task_init_states(task_id)
if hasattr(states, "detach"):
    states = states.detach().cpu().numpy()
states = np.ascontiguousarray(np.asarray(states))
if states.ndim < 2 or states.shape[0] <= 0 or not np.isfinite(states).all():
    raise SystemExit("LIBERO returned invalid initial states")

task_root = root / suite_name / f"task_{task_id:02d}"
task_root.mkdir(parents=True, exist_ok=True)
states_path = task_root / "initial_states.npy"
if states_path.exists():
    existing = np.load(states_path, allow_pickle=False)
    if (
        existing.dtype != states.dtype
        or existing.shape != states.shape
        or not np.array_equal(existing, states)
    ):
        raise SystemExit(f"persistent initial-state snapshot changed: {states_path}")
else:
    temporary = task_root / f".{states_path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as handle:
        np.save(handle, states, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, states_path)

bddl_path = (
    Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
).resolve()
if not bddl_path.is_file():
    raise SystemExit(f"LIBERO BDDL is missing: {bddl_path}")
metadata = {
    "task_suite": suite_name,
    "task_id": task_id,
    "task_description": str(task.language),
    "initial_states_path": str(states_path),
    "initial_states_array_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
    "initial_states_shape": list(states.shape),
    "initial_states_dtype": str(states.dtype),
    "bddl_path": str(bddl_path),
}
metadata_path = task_root / "metadata.json"
encoded = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8") + b"\n"
if metadata_path.exists():
    if metadata_path.read_bytes() != encoded:
        raise SystemExit(f"persistent LIBERO task metadata changed: {metadata_path}")
else:
    temporary = task_root / f".{metadata_path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, metadata_path)
print(metadata_path)
PY
}

if [[ "${EVAL_ACTION}" == prepare ]]; then
  export WARM_VERIFY_CHECKPOINT="${WARM_CHECKPOINT}"
  export WARM_VERIFY_ATTESTATION="${WARM_TRAINING_ATTESTATION}"
  export WARM_EXPECTED_STEP="${CHECKPOINT_STEP}"
  "${PYTHON_BIN}" - <<'PY'
import os
from fastwam.models.warm.training_attestation import verify_training_attestation

attestation = verify_training_attestation(
    os.environ["WARM_VERIFY_CHECKPOINT"],
    os.environ["WARM_VERIFY_ATTESTATION"],
)
expected = int(os.environ["WARM_EXPECTED_STEP"])
if not (
    attestation.checkpoint_step
    == attestation.actual_global_step
    == attestation.actual_max_steps
    == expected
):
    raise SystemExit("verified checkpoint is not the completed final step")
print(
    "checkpoint_ok "
    f"step={attestation.checkpoint_step} "
    f"sha256={attestation.checkpoint_sha256} "
    f"git_commit={attestation.git_commit}"
)
PY
  for suite in libero_spatial libero_object libero_goal libero_10; do
    for task_id in $(seq 0 9); do
      prepare_task_inputs "${suite}" "${task_id}" >/dev/null
    done
  done
  echo "PREPARE_OK"
  echo "evaluation_worktree=${EVAL_CODE}"
  echo "task_inputs=${WARM_EVAL_INPUT_ROOT}"
  echo "checkpoint_step=${CHECKPOINT_STEP}"
  exit 0
fi

TASK_METADATA="$(prepare_task_inputs "${WARM_TASK_SUITE}" "${WARM_TASK_ID}")"
WARM_TASK_DESCRIPTION="$(${PYTHON_BIN} - "${TASK_METADATA}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["task_description"])
PY
)"
WARM_INITIAL_STATES="$(${PYTHON_BIN} - "${TASK_METADATA}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["initial_states_path"])
PY
)"
WARM_BDDL="$(${PYTHON_BIN} - "${TASK_METADATA}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["bddl_path"])
PY
)"
export WARM_TASK_DESCRIPTION WARM_INITIAL_STATES WARM_BDDL

STEP_LABEL="step_$(printf '%06d' "${CHECKPOINT_STEP}")"
if [[ -z "${WARM_EVAL_ROOT:-}" ]]; then
  WARM_EVAL_ROOT="${WARM_EVAL_BASE}/results/${STEP_LABEL}/${WARM_TASK_SUITE}/task_$(printf '%02d' "${WARM_TASK_ID}")/seed_${WARM_ROOT_SEED}/${WARM_EVAL_LABEL}"
fi
export WARM_EVAL_ROOT
[[ ! -e "${WARM_EVAL_ROOT}" ]] \
  || fail "immutable evaluation root already exists: ${WARM_EVAL_ROOT}; use a new WARM_EVAL_LABEL"
mkdir -p "$(dirname "${WARM_EVAL_ROOT}")"

echo "EVAL_PREFLIGHT_OK"
echo "checkpoint=${WARM_CHECKPOINT}"
echo "training_commit=${TRAIN_COMMIT}"
echo "evaluation_code=${EVAL_CODE}"
echo "task=${WARM_TASK_SUITE}/${WARM_TASK_ID}"
echo "task_description=${WARM_TASK_DESCRIPTION}"
echo "root_seed=${WARM_ROOT_SEED}"
echo "evaluation_root=${WARM_EVAL_ROOT}"

cd "${EVAL_CODE}"
set -o pipefail
bash scripts/evaluate_warm_full_server.sh \
  2>&1 | tee "${WARM_EVAL_ROOT}.console.log"
