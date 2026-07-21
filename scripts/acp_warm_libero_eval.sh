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
WARM_LIBERO_CONFIG_ROOT="${WARM_LIBERO_CONFIG_ROOT:-${WARM_EVAL_BASE}/libero_config}"
WARM_LIBERO_SOURCE_DIR="${WARM_LIBERO_SOURCE_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM_external/LIBERO-8f1084e3132a}"

EVAL_ACTION="${EVAL_ACTION:-run}"
WARM_TASK_SUITE="${WARM_TASK_SUITE:-libero_10}"
WARM_TASK_ID="${WARM_TASK_ID:-0}"
WARM_ROOT_SEED="${WARM_ROOT_SEED:-17}"
WARM_EVAL_LABEL="${WARM_EVAL_LABEL:-formal}"
WARM_EVALUATION_NAMESPACE="${WARM_EVALUATION_NAMESPACE:-warm-libero-full-v1}"
WARM_EVALUATION_NAMESPACE_BASE="${WARM_EVALUATION_NAMESPACE}"
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
[[ -f "${WARM_LIBERO_SOURCE_DIR}/libero/libero/__init__.py" ]] \
  || fail "pinned LIBERO source not found: ${WARM_LIBERO_SOURCE_DIR}"
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
WARM_FORMAL_EVAL_LAUNCHER="${WARM_FORMAL_EVAL_LAUNCHER:-${PROJECT_DIR}/scripts/evaluate_warm_full_server.sh}"
mkdir -p "${WARM_EVAL_WORKTREE_ROOT}"
if [[ ! -e "${EVAL_CODE}/.git" ]]; then
  [[ ! -e "${EVAL_CODE}" ]] || fail "non-worktree path already exists: ${EVAL_CODE}"
  git -C "${PROJECT_DIR}" worktree add --detach "${EVAL_CODE}" "${TRAIN_COMMIT}"
fi
# AFS/NFS containers can map the persisted worktree to an owner different from
# the current ephemeral container user. Register only this attested, absolute
# worktree path before asking Git to inspect it; do not use the unsafe wildcard
# safe.directory setting.
if ! git config --global --get-all safe.directory 2>/dev/null \
  | grep -Fqx -- "${EVAL_CODE}"; then
  git config --global --add safe.directory "${EVAL_CODE}"
fi
[[ "$(git -C "${EVAL_CODE}" rev-parse HEAD)" == "${TRAIN_COMMIT}" ]] \
  || fail "evaluation worktree does not match the checkpoint commit"
[[ -z "$(git -C "${EVAL_CODE}" status --porcelain)" ]] \
  || fail "evaluation worktree is dirty: ${EVAL_CODE}"
[[ -f "${EVAL_CODE}/scripts/evaluate_warm_full_server.sh" ]] \
  || fail "training commit has no complete-WARM LIBERO evaluator"
[[ -f "${WARM_FORMAL_EVAL_LAUNCHER}" ]] \
  || fail "formal WARM evaluation launcher not found: ${WARM_FORMAL_EVAL_LAUNCHER}"

# The completed step-019100 checkpoint is bound to an evaluator commit that
# contains one rollout-only shape bug: after the first factual action summary,
# its compact 15-D signature is compared with a 21-D preview signature.  Keep
# the checkpoint's clean historical worktree intact and apply only a narrow
# startup repair whose source hash and effective namespace are recorded in the
# immutable evaluation evidence.  Any unexpected source revision fails closed.
WARM_EVAL_COMPAT_PYTHONPATH=""
KNOWN_ACTION_SIGNATURE_COMMIT="c4763a975298de6f00939360551616af7902d57a"
KNOWN_ACTION_SIGNATURE_SOURCE_SHA256="aeede91e8770706c03c4c25fd670cf8a12749956755020a2f674971d32ec829f"
ACTION_SIGNATURE_SOURCE="${EVAL_CODE}/src/fastwam/memory/online_episode_memory.py"
ACTION_SIGNATURE_SOURCE_SHA256="$(${PYTHON_BIN} - "${ACTION_SIGNATURE_SOURCE}" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
if [[ "${TRAIN_COMMIT}" == "${KNOWN_ACTION_SIGNATURE_COMMIT}" ]]; then
  [[ "${ACTION_SIGNATURE_SOURCE_SHA256}" == "${KNOWN_ACTION_SIGNATURE_SOURCE_SHA256}" ]] \
    || fail "known evaluation commit has an unexpected online-memory source hash"
  WARM_EVAL_COMPATIBILITY_ID="action-summary-signature-v1"
  WARM_EVAL_COMPATIBILITY_DIR="${PROJECT_DIR}/scripts/evaluation_compat/action_summary_signature_v1"
  WARM_EVAL_COMPATIBILITY_FILE="${WARM_EVAL_COMPATIBILITY_DIR}/sitecustomize.py"
  WARM_EVAL_COMPATIBILITY_RELATIVE="scripts/evaluation_compat/action_summary_signature_v1/sitecustomize.py"
  [[ -f "${WARM_EVAL_COMPATIBILITY_FILE}" ]] \
    || fail "required evaluation compatibility repair is missing"
  WARM_EVAL_COMPATIBILITY_SHA256="$(${PYTHON_BIN} - "${WARM_EVAL_COMPATIBILITY_FILE}" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
  TRACKED_COMPATIBILITY_SHA256="$(
    git -C "${PROJECT_DIR}" show "HEAD:${WARM_EVAL_COMPATIBILITY_RELATIVE}" \
      | "${PYTHON_BIN}" -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
  )" || fail "compatibility repair is not committed in the current repository"
  [[ "${WARM_EVAL_COMPATIBILITY_SHA256}" == "${TRACKED_COMPATIBILITY_SHA256}" ]] \
    || fail "compatibility repair differs from the current committed file"
  WARM_EVAL_COMPATIBILITY_LAUNCHER_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
  WARM_EVALUATION_NAMESPACE="${WARM_EVALUATION_NAMESPACE_BASE}-compat-${WARM_EVAL_COMPATIBILITY_SHA256:0:12}"
  WARM_EVAL_COMPAT_PYTHONPATH="${WARM_EVAL_COMPATIBILITY_DIR}:"
  export WARM_EVAL_COMPATIBILITY_ID WARM_EVAL_COMPATIBILITY_FILE
  export WARM_EVAL_COMPATIBILITY_SHA256 WARM_EVAL_COMPATIBILITY_LAUNCHER_COMMIT
  export WARM_EVAL_COMPATIBILITY_SOURCE_SHA256="${ACTION_SIGNATURE_SOURCE_SHA256}"
  export WARM_EVAL_COMPAT_ACTION_SIGNATURE="${WARM_EVAL_COMPATIBILITY_ID}"
  export WARM_EVAL_COMPAT_TRAIN_COMMIT="${TRAIN_COMMIT}"
  export WARM_EVAL_COMPAT_GRIPPER_INDICES="6"
elif [[ "${ACTION_SIGNATURE_SOURCE_SHA256}" == "${KNOWN_ACTION_SIGNATURE_SOURCE_SHA256}" ]]; then
  fail "known-buggy online-memory source appeared under an unexpected commit"
fi

export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_DIR}/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
# LIBERO otherwise prompts on its first import.  ACP jobs are non-interactive,
# so keep one explicit configuration on the persistent AFS evaluation volume.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${WARM_LIBERO_CONFIG_ROOT}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
# Official LIBERO init-state bundles predate PyTorch 2.6 and contain NumPy
# objects.  PyTorch 2.6+ defaults unspecified torch.load callsites to
# weights_only=True.  The init-state bundle is from our pinned, SHA-verified
# official LIBERO archive, and every WARM checkpoint loaded here is separately
# attested, so restore the legacy loader only for callsites that omit the flag.
# Remove the opposing global override in case the container image set it.
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${WARM_EVAL_COMPAT_PYTHONPATH}${EVAL_CODE}/src:${EVAL_CODE}:${WARM_LIBERO_SOURCE_DIR}"

export WARM_ARTIFACT_ROOT FASTWAM_BASE_CHECKPOINT WARM_CHECKPOINT
export WARM_TRAINING_ATTESTATION WARM_DINO_CHECKPOINT WARM_VAE_CHECKPOINT
export WARM_TEXT_ENCODER WARM_TOKENIZER WARM_EVALUATION_NAMESPACE
export WARM_EVALUATION_NAMESPACE_BASE
export WARM_EVAL_DEVICE WARM_TASK_SUITE WARM_TASK_ID WARM_ROOT_SEED

WARM_REQUIRE_CUDA="${WARM_REQUIRE_CUDA:-$([[ "${EVAL_ACTION}" == run ]] && echo true || echo false)}"
export WARM_REQUIRE_CUDA WARM_REQUIRE_MUJOCO_VERSION
# Discover the editable official LIBERO installation without importing
# libero.libero (that import is interactive when no config exists), then write
# the canonical config atomically.  Missing simulator packages are diagnosed
# together so users do not have to discover them one at a time.
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
from pathlib import Path

import yaml

required_modules = ("mujoco", "robosuite", "bddl", "libero")
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "missing LIBERO evaluation modules: "
        + ", ".join(missing)
        + "; run scripts/setup_warm_libero_eval_env.sh once in CCI"
    )

spec = importlib.util.find_spec("libero")
locations = list(spec.submodule_search_locations or ()) if spec is not None else []
if len(locations) != 1:
    raise SystemExit(f"cannot identify one official LIBERO package root: {locations}")
outer_package = Path(locations[0]).resolve()
benchmark_root = outer_package / "libero"
paths = {
    "benchmark_root": str(benchmark_root),
    "bddl_files": str(benchmark_root / "bddl_files"),
    "init_states": str(benchmark_root / "init_files"),
    "datasets": str(outer_package / "datasets"),
    "assets": str(benchmark_root / "assets"),
}
for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
    if not Path(paths[key]).exists():
        raise SystemExit(f"official LIBERO installation is incomplete: {key}={paths[key]}")

config_root = Path(os.environ["LIBERO_CONFIG_PATH"]).expanduser().resolve()
config_root.mkdir(parents=True, exist_ok=True)
config_path = config_root / "config.yaml"
encoded = yaml.safe_dump(paths, sort_keys=True).encode("utf-8")
if config_path.exists():
    current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if current != paths:
        raise SystemExit(
            f"persistent LIBERO config does not match the installed package: {config_path}"
        )
else:
    temporary = config_root / f".{config_path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, config_path)
print(f"libero_config={config_path}")
PY
"${PYTHON_BIN}" - <<'PY'
import os

import mujoco
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401

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
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

# LIBERO prints registry information and missing-demo-dataset warnings to
# stdout while importing and resolving paths.  This helper's stdout is a
# machine-readable protocol consumed by Bash and must contain exactly one
# metadata path, so contain third-party chatter locally.  Python exceptions
# and warnings still reach stderr and preserve actionable diagnostics.
with contextlib.redirect_stdout(io.StringIO()):
    from libero.libero import benchmark, get_libero_path

root = Path(sys.argv[1]).expanduser().resolve()
suite_name = sys.argv[2]
task_id = int(sys.argv[3])
with contextlib.redirect_stdout(io.StringIO()):
    suite = benchmark.get_benchmark_dict()[suite_name]()
n_tasks = int(suite.n_tasks)
if task_id < 0 or task_id >= n_tasks:
    raise SystemExit(f"task id {task_id} is outside [0, {n_tasks}) for {suite_name}")
with contextlib.redirect_stdout(io.StringIO()):
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

with contextlib.redirect_stdout(io.StringIO()):
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
  # Import checks alone cannot detect a broken EGL runtime or a MuJoCo /
  # robosuite ABI mismatch.  Create one real two-camera environment, restore a
  # canonical initial state, and validate the observations before accepting the
  # one-time preparation.  Bound the check so a simulator reset cannot leave an
  # ACP job hanging indefinitely.
  WARM_PREPARE_RENDER_TIMEOUT_SECONDS="${WARM_PREPARE_RENDER_TIMEOUT_SECONDS:-180}"
  [[ "${WARM_PREPARE_RENDER_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
    || fail "WARM_PREPARE_RENDER_TIMEOUT_SECONDS must be a positive integer"
  timeout "${WARM_PREPARE_RENDER_TIMEOUT_SECONDS}" "${PYTHON_BIN}" - <<'PY'
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
states = suite.get_task_init_states(0)
if hasattr(states, "detach"):
    states = states.detach().cpu().numpy()
states = np.asarray(states)
env = OffScreenRenderEnv(
    bddl_file_name=str(bddl),
    camera_heights=256,
    camera_widths=256,
)
try:
    env.seed(0)
    env.reset()
    observation = env.set_init_state(states[0])
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.asarray(observation[key])
        if image.shape != (256, 256, 3) or not np.isfinite(image).all():
            raise SystemExit(f"invalid LIBERO camera observation {key}: {image.shape}")
finally:
    env.close()
print("libero_render_smoke_ok")
PY
  echo "PREPARE_OK"
  echo "evaluation_worktree=${EVAL_CODE}"
  echo "task_inputs=${WARM_EVAL_INPUT_ROOT}"
  echo "checkpoint_step=${CHECKPOINT_STEP}"
  exit 0
fi

TASK_METADATA="$(prepare_task_inputs "${WARM_TASK_SUITE}" "${WARM_TASK_ID}")"
[[ -n "${TASK_METADATA}" && "${TASK_METADATA}" != *$'\n'* ]] \
  || fail "task-input helper returned a malformed metadata path"
[[ -f "${TASK_METADATA}" ]] \
  || fail "task metadata was not published: ${TASK_METADATA}"
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
echo "evaluation_launcher=${WARM_FORMAL_EVAL_LAUNCHER}"
if [[ -n "${WARM_EVAL_COMPATIBILITY_ID:-}" ]]; then
  echo "evaluation_compatibility=${WARM_EVAL_COMPATIBILITY_ID}"
  echo "evaluation_compatibility_sha256=${WARM_EVAL_COMPATIBILITY_SHA256}"
  echo "evaluation_namespace=${WARM_EVALUATION_NAMESPACE}"
fi
echo "task=${WARM_TASK_SUITE}/${WARM_TASK_ID}"
echo "task_description=${WARM_TASK_DESCRIPTION}"
echo "root_seed=${WARM_ROOT_SEED}"
echo "evaluation_root=${WARM_EVAL_ROOT}"

cd "${EVAL_CODE}"
set -o pipefail
bash "${WARM_FORMAL_EVAL_LAUNCHER}" \
  2>&1 | tee "${WARM_EVAL_ROOT}.console.log"
