#!/usr/bin/env bash
set -euo pipefail

# One-time, persistent LIBERO runtime setup for ACP/CCI evaluation.  This keeps
# the official simulator outside the private WARM worktree and deliberately
# does not install LIBERO's legacy requirements.txt, whose old NumPy, Hydra,
# Transformers and WandB pins would damage the trained WARM environment.

PROJECTS_ROOT="${PROJECTS_ROOT:-/mnt/afs/task3_2/L202500276_lwz/projects}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
LIBERO_COMMIT="${LIBERO_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
WARM_EXTERNAL_ROOT="${WARM_EXTERNAL_ROOT:-${PROJECTS_ROOT}/WARM_external}"
WARM_BOOTSTRAP_DIR="${WARM_BOOTSTRAP_DIR:-${WARM_EXTERNAL_ROOT}/bootstrap/libero_eval_v1}"
WARM_PIP_CACHE_DIR="${WARM_PIP_CACHE_DIR:-${WARM_EXTERNAL_ROOT}/pip_cache}"
LIBERO_SOURCE_DIR="${LIBERO_SOURCE_DIR:-${WARM_EXTERNAL_ROOT}/LIBERO-${LIBERO_COMMIT:0:12}}"
ROBOSUITE_WHEEL="${ROBOSUITE_WHEEL:-${WARM_BOOTSTRAP_DIR}/robosuite-1.4.0-py3-none-any.whl}"
LIBERO_SOURCE_ARCHIVE="${LIBERO_SOURCE_ARCHIVE:-${WARM_BOOTSTRAP_DIR}/LIBERO-8f1084e3132a.tar.gz}"
ROBOSUITE_WHEEL_SHA256="aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994"
LIBERO_SOURCE_ARCHIVE_SHA256="effdd60e8c6377a9583b0508d6e347193d609995a9afa8e00b15cf2bd7c9e8ba"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

TEMP_SOURCE_DIR=""
cleanup_temporary_source() {
  local external_real temp_real
  [[ -n "${TEMP_SOURCE_DIR}" && -d "${TEMP_SOURCE_DIR}" ]] || return 0
  external_real="$(realpath -m "${WARM_EXTERNAL_ROOT}")" || return 0
  temp_real="$(realpath -m "${TEMP_SOURCE_DIR}")" || return 0
  if [[ "${temp_real}" == "${external_real}"/* \
        && "${temp_real}" == *.extracting.* ]]; then
    rm -rf --one-file-system -- "${temp_real}"
  else
    echo "WARNING: refusing to clean unsafe temporary path: ${temp_real}" >&2
  fi
}
trap cleanup_temporary_source EXIT

verify_sha256() {
  local path="$1" expected="$2" label="$3" actual
  actual="$(sha256sum "${path}" | awk '{print $1}')" \
    || fail "cannot hash ${label}: ${path}"
  [[ "${actual}" == "${expected}" ]] \
    || fail "${label} SHA-256 mismatch: ${actual} != ${expected}"
}

[[ -x "${CONDA_ENV_DIR}/bin/python" ]] \
  || fail "Python environment not found: ${CONDA_ENV_DIR}"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
mkdir -p "${WARM_BOOTSTRAP_DIR}" "${WARM_PIP_CACHE_DIR}"
export PIP_CACHE_DIR="${WARM_PIP_CACHE_DIR}"

echo "Installing the pinned simulator runtime into ${CONDA_ENV_DIR}"
# robosuite 1.4.0 declares opencv-python.  WARM intentionally uses the
# headless build that already supplies cv2, so install robosuite without its
# dependency resolver to avoid replacing NumPy/OpenCV in the training env.
"${PYTHON_BIN}" -m pip install \
  "mujoco==3.3.2" \
  "numba==0.60.0" \
  "scipy==1.14.1" \
  "bddl==1.0.1" \
  "gym==0.25.2" \
  "easydict==1.9" \
  "matplotlib==3.8.4" \
  "future==0.18.2"
if [[ -f "${ROBOSUITE_WHEEL}" ]]; then
  verify_sha256 "${ROBOSUITE_WHEEL}" "${ROBOSUITE_WHEEL_SHA256}" "robosuite wheel"
  "${PYTHON_BIN}" -m pip install --no-deps "${ROBOSUITE_WHEEL}"
else
  echo "WARNING: offline robosuite wheel is absent: ${ROBOSUITE_WHEEL}"
  echo "         Falling back to the configured Python package index."
  "${PYTHON_BIN}" -m pip install --no-deps "robosuite==1.4.0"
fi

mkdir -p "$(dirname "${LIBERO_SOURCE_DIR}")"
if [[ ! -e "${LIBERO_SOURCE_DIR}" && -f "${LIBERO_SOURCE_ARCHIVE}" ]]; then
  verify_sha256 \
    "${LIBERO_SOURCE_ARCHIVE}" \
    "${LIBERO_SOURCE_ARCHIVE_SHA256}" \
    "LIBERO source archive"
  TEMP_SOURCE_DIR="${LIBERO_SOURCE_DIR}.extracting.$$"
  [[ "${TEMP_SOURCE_DIR}" == "${WARM_EXTERNAL_ROOT}"/* ]] \
    || fail "unsafe temporary LIBERO extraction path: ${TEMP_SOURCE_DIR}"
  mkdir "${TEMP_SOURCE_DIR}"
  # AFS commonly root-squashes ephemeral container users.  Preserve file
  # contents, not archive uid/gid/mode metadata, so extraction never attempts
  # a forbidden chown/chmod to uid=0,gid=0.
  tar \
    --extract \
    --gzip \
    --file "${LIBERO_SOURCE_ARCHIVE}" \
    --directory "${TEMP_SOURCE_DIR}" \
    --no-same-owner \
    --no-same-permissions
  "${PYTHON_BIN}" - "${TEMP_SOURCE_DIR}" "${LIBERO_COMMIT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
marker = root / ".warm_upstream.json"
payload = {
    "schema": "warm.external-libero-source.v1",
    "upstream": "https://github.com/Lifelong-Robot-Learning/LIBERO",
    "git_commit": sys.argv[2],
}
with marker.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  mv "${TEMP_SOURCE_DIR}" "${LIBERO_SOURCE_DIR}"
  TEMP_SOURCE_DIR=""
elif [[ ! -e "${LIBERO_SOURCE_DIR}/.git" && ! -e "${LIBERO_SOURCE_DIR}/.warm_upstream.json" ]]; then
  [[ ! -e "${LIBERO_SOURCE_DIR}" ]] \
    || fail "non-Git path already exists: ${LIBERO_SOURCE_DIR}"
  echo "WARNING: offline LIBERO archive is absent: ${LIBERO_SOURCE_ARCHIVE}"
  echo "         Falling back to the public HTTPS repository."
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    "${LIBERO_SOURCE_DIR}"
fi
if [[ -e "${LIBERO_SOURCE_DIR}/.git" ]]; then
  if ! git config --global --get-all safe.directory 2>/dev/null \
    | grep -Fqx -- "${LIBERO_SOURCE_DIR}"; then
    git config --global --add safe.directory "${LIBERO_SOURCE_DIR}"
  fi
  git -C "${LIBERO_SOURCE_DIR}" cat-file -e "${LIBERO_COMMIT}^{commit}" 2>/dev/null \
    || fail "pinned LIBERO commit is unavailable locally: ${LIBERO_COMMIT}"
  git -C "${LIBERO_SOURCE_DIR}" checkout --detach "${LIBERO_COMMIT}"
  [[ -z "$(git -C "${LIBERO_SOURCE_DIR}" status --porcelain)" ]] \
    || fail "official LIBERO checkout is dirty: ${LIBERO_SOURCE_DIR}"
else
  "${PYTHON_BIN}" - "${LIBERO_SOURCE_DIR}/.warm_upstream.json" "${LIBERO_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as handle:
    value = json.load(handle)
if value.get("schema") != "warm.external-libero-source.v1":
    raise SystemExit("invalid offline LIBERO source marker")
if value.get("git_commit") != sys.argv[2]:
    raise SystemExit("offline LIBERO source commit mismatch")
PY
fi
[[ -f "${LIBERO_SOURCE_DIR}/libero/libero/__init__.py" ]] \
  || fail "LIBERO source package is incomplete: ${LIBERO_SOURCE_DIR}"

# Official LIBERO uses a double namespace layout: the importable package is
# LIBERO/libero/libero while the outer LIBERO/libero directory has no
# __init__.py.  Some modern setuptools PEP 660 editable hooks install the
# distribution metadata but fail to expose the parent namespace.  A plain .pth
# pointing at the pinned source root is simpler, persistent, and keeps all task
# assets available directly from the verified source tree.
"${PYTHON_BIN}" - "${LIBERO_SOURCE_DIR}" "${CONDA_ENV_DIR}" <<'PY'
import os
import sys
import sysconfig
from pathlib import Path

source = Path(sys.argv[1]).resolve()
expected_prefix = Path(sys.argv[2]).resolve()
actual_prefix = Path(sys.prefix).resolve()
if actual_prefix != expected_prefix:
    raise SystemExit(
        f"wrong Python environment for LIBERO path install: {actual_prefix} "
        f"!= {expected_prefix}"
    )
purelib = Path(sysconfig.get_path("purelib")).resolve()
if not purelib.is_relative_to(expected_prefix):
    raise SystemExit(f"site-packages escaped the WARM environment: {purelib}")
pth_path = purelib / "warm_pinned_libero_source.pth"
encoded = (str(source) + "\n").encode("utf-8")
if pth_path.exists() and pth_path.read_bytes() != encoded:
    raise SystemExit(f"existing LIBERO path binding disagrees: {pth_path}")
if not pth_path.exists():
    temporary = purelib / f".{pth_path.name}.{os.getpid()}.tmp"
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, pth_path)
print(f"libero_path_binding={pth_path}")
PY

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
