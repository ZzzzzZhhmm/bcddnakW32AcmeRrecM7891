#!/usr/bin/env bash
set -euo pipefail

# One-time, persistent LIBERO runtime setup for ACP/CCI evaluation.  This keeps
# the official simulator outside the private WARM worktree and deliberately
# does not install LIBERO's legacy requirements.txt, whose old NumPy, Hydra,
# Transformers and WandB pins would damage the trained WARM environment.

PROJECTS_ROOT="${PROJECTS_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
LIBERO_COMMIT="${LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
LIBERO_SOURCE_DIR="${LIBERO_SOURCE_DIR:-${PROJECTS_ROOT}/WARM_external/LIBERO-${LIBERO_COMMIT:0:12}}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "${CONDA_ENV_DIR}/bin/python" ]] \
  || fail "Python environment not found: ${CONDA_ENV_DIR}"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"

echo "Installing the pinned simulator runtime into ${CONDA_ENV_DIR}"
# robosuite 1.4.0 declares opencv-python.  WARM intentionally uses the
# headless build that already supplies cv2, so install robosuite without its
# dependency resolver to avoid replacing NumPy/OpenCV in the training env.
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "mujoco==3.3.2" \
  "numba==0.60.0" \
  "scipy==1.14.1" \
  "bddl==1.0.1" \
  "gym==0.25.2" \
  "easydict==1.9" \
  "matplotlib==3.8.4" \
  "future==0.18.2"
"${PYTHON_BIN}" -m pip install --no-cache-dir --no-deps "robosuite==1.4.0"

mkdir -p "$(dirname "${LIBERO_SOURCE_DIR}")"
if [[ ! -e "${LIBERO_SOURCE_DIR}/.git" ]]; then
  [[ ! -e "${LIBERO_SOURCE_DIR}" ]] \
    || fail "non-Git path already exists: ${LIBERO_SOURCE_DIR}"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    "${LIBERO_SOURCE_DIR}"
fi
if ! git config --global --get-all safe.directory 2>/dev/null \
  | grep -Fqx -- "${LIBERO_SOURCE_DIR}"; then
  git config --global --add safe.directory "${LIBERO_SOURCE_DIR}"
fi
git -C "${LIBERO_SOURCE_DIR}" cat-file -e "${LIBERO_COMMIT}^{commit}" 2>/dev/null \
  || fail "pinned LIBERO commit is unavailable locally: ${LIBERO_COMMIT}"
git -C "${LIBERO_SOURCE_DIR}" checkout --detach "${LIBERO_COMMIT}"
[[ -z "$(git -C "${LIBERO_SOURCE_DIR}" status --porcelain)" ]] \
  || fail "official LIBERO checkout is dirty: ${LIBERO_SOURCE_DIR}"
"${PYTHON_BIN}" -m pip install --no-deps -e "${LIBERO_SOURCE_DIR}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import importlib.util

import cv2
import mujoco
import numpy
import torch

for module in ("robosuite", "bddl", "libero"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing module after setup: {module}")
if mujoco.__version__ != "3.3.2":
    raise SystemExit(f"unexpected MuJoCo version: {mujoco.__version__}")
if numpy.__version__ != "1.26.4":
    raise SystemExit(f"WARM NumPy pin changed unexpectedly: {numpy.__version__}")
if importlib.metadata.version("robosuite") != "1.4.0":
    raise SystemExit("unexpected robosuite version")
print(
    "LIBERO_RUNTIME_SETUP_OK "
    f"torch={torch.__version__} numpy={numpy.__version__} "
    f"mujoco={mujoco.__version__} cv2={cv2.__version__}"
)
PY

echo "libero_source=${LIBERO_SOURCE_DIR}"
echo "Next: EVAL_ACTION=prepare bash scripts/acp_warm_libero_eval.sh"
