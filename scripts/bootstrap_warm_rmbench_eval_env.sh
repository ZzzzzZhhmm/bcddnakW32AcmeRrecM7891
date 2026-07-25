#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

# One-time CCI bootstrap for the persistent RMBench evaluation environment.
#
# This venv reuses the already validated WARM training environment through
# --system-site-packages, then adds the pinned simulator stack without
# downgrading WARM's PyTorch.  The target lives on persistent AFS and is reused
# by every later ACP evaluation job.

PROJECT_DIR="${PROJECT_DIR:-/mnt/afs/task3_2/L202500276_lwz/projects/WARM}"
WARM_BASE_ENV_DIR="${WARM_BASE_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
WARM_RMBENCH_EVAL_ENV_DIR="${WARM_RMBENCH_EVAL_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm-rmbench-eval}"
RMBENCH_ROOT="${RMBENCH_ROOT:-/mnt/afs/task3_2/L202500276_lwz/external/RMBench-official}"
WARM_EXTERNAL_ROOT="${WARM_EXTERNAL_ROOT:-${PROJECT_DIR}_external}"
CUROBO_SOURCE="${CUROBO_SOURCE:-${WARM_EXTERNAL_ROOT}/curobo-d64c4b005459}"
RMBENCH_CODE_REVISION="${RMBENCH_CODE_REVISION:-57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c}"
CUROBO_REVISION="${CUROBO_REVISION:-d64c4b005459db10c5dd867d8b30a87d5bda9bdb}"
CUROBO_TAG="${CUROBO_TAG:-v0.7.8}"
CUROBO_GIT_URL="${CUROBO_GIT_URL:-https://github.com/NVlabs/curobo.git}"
CUROBO_FETCH_ATTEMPTS="${CUROBO_FETCH_ATTEMPTS:-6}"
CONSTRAINTS="${PROJECT_DIR}/scripts/constraints/warm_rmbench_eval.constraints"
WARM_RMBENCH_WHEELHOUSE="${WARM_RMBENCH_WHEELHOUSE:-${WARM_EXTERNAL_ROOT}/wheelhouse/rmbench-eval-v2}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WARM_EXTERNAL_ROOT}/pip-cache/rmbench-eval-v2}"
WARM_RMBENCH_PYPI_MIRROR="${WARM_RMBENCH_PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
WARM_RMBENCH_PYPI_FALLBACK="${WARM_RMBENCH_PYPI_FALLBACK:-https://pypi.org/simple}"
SAPIEN_WHEEL_NAME="sapien-3.0.0b1-cp310-cp310-manylinux2014_x86_64.whl"
SAPIEN_WHEEL_URL="https://files.pythonhosted.org/packages/26/e5/7f22f5be26009eceddba57a9e4280e5dc5cef8bc5f29ef3a453db2f7d723/${SAPIEN_WHEEL_NAME}"
SAPIEN_WHEEL_SHA256="9763215d52374e48db16d8c2e89fb385a5625690c0c88cf2c636c1abe99aef6c"
SAPIEN_WHEEL_SIZE="49596610"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

retry_git() {
  local attempt
  for ((attempt = 1; attempt <= CUROBO_FETCH_ATTEMPTS; attempt++)); do
    if GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 "$@"; then
      return 0
    fi
    if ((attempt < CUROBO_FETCH_ATTEMPTS)); then
      printf 'Git attempt %d/%d failed; retrying in %d seconds.\n' \
        "${attempt}" "${CUROBO_FETCH_ATTEMPTS}" "$((attempt * 2))" >&2
      sleep "$((attempt * 2))"
    fi
  done
  return 1
}

[[ -x "${WARM_BASE_ENV_DIR}/bin/python" ]] \
  || fail "base WARM Python is absent: ${WARM_BASE_ENV_DIR}/bin/python"
[[ -f "${PROJECT_DIR}/scripts/check_warm_rmbench_eval_runtime.py" ]] \
  || fail "runtime checker is absent from WARM checkout"
[[ -f "${CONSTRAINTS}" ]] \
  || fail "runtime constraints are absent: ${CONSTRAINTS}"
[[ -f "${PROJECT_DIR}/scripts/install_open3d_rgb_guard.py" ]] \
  || fail "Open3D RGB guard installer is absent"
[[ -f "${PROJECT_DIR}/scripts/download_verified_http_ranges.py" ]] \
  || fail "verified parallel downloader is absent"
[[ -e "${RMBENCH_ROOT}/.git" ]] \
  || fail "pinned RMBench checkout is absent: ${RMBENCH_ROOT}"
command -v git >/dev/null 2>&1 || fail "git is required"
command -v flock >/dev/null 2>&1 || fail "flock is required"
git config --global --add safe.directory "${RMBENCH_ROOT}" 2>/dev/null || true

mkdir -p \
  "$(dirname "${WARM_RMBENCH_EVAL_ENV_DIR}")" \
  "${WARM_EXTERNAL_ROOT}" \
  "${WARM_RMBENCH_WHEELHOUSE}" \
  "${PIP_CACHE_DIR}"
exec 9>"${WARM_RMBENCH_EVAL_ENV_DIR}.bootstrap.lock"
flock -x 9

"${WARM_BASE_ENV_DIR}/bin/python" - <<'PY'
import sys
import numpy
import torch

expected = {
    "python": (3, 10),
    "numpy": "1.26.4",
    "torch": "2.7.1+cu128",
}
if sys.version_info[:2] != expected["python"]:
    raise SystemExit(f"base Python is {sys.version_info[:2]}, expected (3, 10)")
if numpy.__version__ != expected["numpy"]:
    raise SystemExit(
        f"base numpy is {numpy.__version__}, expected {expected['numpy']}"
    )
if torch.__version__ != expected["torch"]:
    raise SystemExit(
        f"base torch is {torch.__version__}, expected {expected['torch']}"
    )
print(
    f"base_runtime_ok python={sys.version.split()[0]} "
    f"numpy={numpy.__version__} torch={torch.__version__}"
)
PY

if [[ ! -x "${WARM_RMBENCH_EVAL_ENV_DIR}/bin/python" ]]; then
  "${WARM_BASE_ENV_DIR}/bin/python" -m venv \
    --system-site-packages "${WARM_RMBENCH_EVAL_ENV_DIR}"
fi

PYTHON_BIN="${WARM_RMBENCH_EVAL_ENV_DIR}/bin/python"
PIP=("${PYTHON_BIN}" -m pip)

# Download every completed wheel into persistent AFS storage.  The mirror is
# fast from the target cluster; exact-package fallback preserves portability.
# --no-deps is deliberate: yourdfpy declares trimesh[easy], whose VHACD,
# Embree, XAtlas, and Open3D-adjacent extras are unused by CuRobo's
# load_meshes=False URDF path.  Build requirements are installed first because
# toppra 0.6.3 is source-only and needs Cython plus inherited NumPy at build
# time.
BUILD_PACKAGES=(
  "setuptools==80.9.0"
  "wheel==0.45.1"
  "ninja==1.11.1.4"
  "Cython==0.29.37"
  "pybind11==2.13.6"
  "setuptools_scm==8.2.0"
)
RUNTIME_PACKAGES=(
  "scipy==1.10.1"
  "transforms3d==0.4.2"
  "mplib==0.2.1"
  "gymnasium==0.29.1"
  "trimesh==4.4.3"
  "pyglet==1.5.31"
  "toppra==0.6.3"
  "warp-lang==1.11.1"
  "scikit-image==0.22.0"
  "yourdfpy==0.0.60"
  "lxml==5.3.0"
  "six==1.17.0"
  "pyperclip==1.9.0"
  "numpy-quaternion==2024.0.13"
  "networkx==3.4.2"
  "pyyaml==6.0.2"
  "tqdm==4.66.5"
  "importlib_resources==6.5.2"
)
DOWNLOAD_PACKAGES=("${BUILD_PACKAGES[@]}" "${RUNTIME_PACKAGES[@]}")
for package in "${DOWNLOAD_PACKAGES[@]}"; do
  if ! "${PIP[@]}" download \
    --no-deps \
    --dest "${WARM_RMBENCH_WHEELHOUSE}" \
    --index-url "${WARM_RMBENCH_PYPI_MIRROR}" \
    "${package}"
  then
    printf 'Mirror miss for %s; retrying the exact package from PyPI.\n' \
      "${package}" >&2
    "${PIP[@]}" download \
      --no-deps \
      --dest "${WARM_RMBENCH_WHEELHOUSE}" \
      --index-url "${WARM_RMBENCH_PYPI_FALLBACK}" \
      "${package}"
  fi
done

# The exact official SAPIEN beta is not retained by common PyPI mirrors.
# Eight verified range requests turn a multi-hour single-connection transfer
# into a resumable one-time download while retaining the upstream SHA-256.
if ! "${PYTHON_BIN}" \
  "${PROJECT_DIR}/scripts/download_verified_http_ranges.py" \
  --url "${SAPIEN_WHEEL_URL}" \
  --output "${WARM_RMBENCH_WHEELHOUSE}/${SAPIEN_WHEEL_NAME}" \
  --sha256 "${SAPIEN_WHEEL_SHA256}" \
  --size "${SAPIEN_WHEEL_SIZE}" \
  --workers "${WARM_RMBENCH_DOWNLOAD_WORKERS:-8}"
then
  printf 'Parallel SAPIEN download failed; falling back to resumable wheelhouse download.\n' >&2
  rm -f "${WARM_RMBENCH_WHEELHOUSE}/${SAPIEN_WHEEL_NAME}"
  "${PIP[@]}" download \
    --no-deps \
    --dest "${WARM_RMBENCH_WHEELHOUSE}" \
    --index-url "${WARM_RMBENCH_PYPI_FALLBACK}" \
    "sapien==3.0.0b1"
fi

# Make source builds deterministic and independent of pip's temporary build
# environment.  In particular, toppra's pyproject requires Cython and NumPy;
# NumPy is inherited from the checkpoint-attested WARM base runtime.
"${PIP[@]}" install \
  --no-index \
  --find-links "${WARM_RMBENCH_WHEELHOUSE}" \
  --no-deps \
  -c "${CONSTRAINTS}" \
  "${BUILD_PACKAGES[@]}"

# Persist the only source-only Python dependency as a platform wheel.  This is
# performed once per AFS wheelhouse and prevents every later ACP environment
# from repeating toppra's Cython compilation.
if ! find "${WARM_RMBENCH_WHEELHOUSE}" \
  -maxdepth 1 \
  -type f \
  -name 'toppra-0.6.3-*.whl' \
  -print -quit | grep -q .
then
  "${PIP[@]}" wheel \
    --no-index \
    --find-links "${WARM_RMBENCH_WHEELHOUSE}" \
    --no-deps \
    --no-build-isolation \
    -c "${CONSTRAINTS}" \
    --wheel-dir "${WARM_RMBENCH_WHEELHOUSE}" \
    "toppra==0.6.3"
fi

# This is intentionally not RMBench/script/requirements.txt: its torch==2.4.1
# pin is incompatible with the checkpoint-attested WARM runtime.
"${PIP[@]}" install \
  --no-index \
  --find-links "${WARM_RMBENCH_WHEELHOUSE}" \
  --no-deps \
  --no-build-isolation \
  -c "${CONSTRAINTS}" \
  "sapien==3.0.0b1" \
  "${RUNTIME_PACKAGES[@]}"

# Ensure patched binary packages live in the evaluation overlay rather than
# being inherited from (and accidentally modifying) the base WARM environment.
"${PIP[@]}" install \
  --force-reinstall \
  --no-index \
  --find-links "${WARM_RMBENCH_WHEELHOUSE}" \
  --no-deps \
  -c "${CONSTRAINTS}" \
  "sapien==3.0.0b1" \
  "mplib==0.2.1"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/install_open3d_rgb_guard.py" \
  --source "${PROJECT_DIR}/scripts/runtime_shims/open3d"

# CuRobo's full checkout is dominated by ~172 MB of example robot/scene
# assets. RMBench supplies its own embodiment assets and only needs CuRobo's
# code, CUDA extensions, and 160 KB runtime configs. A blobless sparse checkout
# therefore keeps the exact v0.7.8 commit while greatly reducing GitHub
# transfer size and TLS exposure.
git config --global --add safe.directory "${CUROBO_SOURCE}" 2>/dev/null || true
if [[ ! -e "${CUROBO_SOURCE}/.git" ]]; then
  [[ ! -e "${CUROBO_SOURCE}" ]] \
    || fail "non-Git path already occupies CUROBO_SOURCE: ${CUROBO_SOURCE}"
  mkdir -p "${CUROBO_SOURCE}"
  git -C "${CUROBO_SOURCE}" init
  git -C "${CUROBO_SOURCE}" remote add origin "${CUROBO_GIT_URL}"
fi
if git -C "${CUROBO_SOURCE}" remote get-url origin >/dev/null 2>&1; then
  git -C "${CUROBO_SOURCE}" remote set-url origin "${CUROBO_GIT_URL}"
else
  git -C "${CUROBO_SOURCE}" remote add origin "${CUROBO_GIT_URL}"
fi
git -C "${CUROBO_SOURCE}" config remote.origin.promisor true
git -C "${CUROBO_SOURCE}" config remote.origin.partialclonefilter blob:none
if ! git -C "${CUROBO_SOURCE}" cat-file \
  -e "${CUROBO_REVISION}^{commit}" 2>/dev/null
then
  retry_git -C "${CUROBO_SOURCE}" fetch \
    --filter=blob:none \
    --depth 1 \
    origin \
    "refs/tags/${CUROBO_TAG}:refs/tags/${CUROBO_TAG}" \
    || fail "failed to fetch sparse CuRobo ${CUROBO_TAG} after ${CUROBO_FETCH_ATTEMPTS} attempts"
fi
CUROBO_TAG_REVISION="$(
  git -C "${CUROBO_SOURCE}" rev-parse "${CUROBO_TAG}^{}"
)"
[[ "${CUROBO_TAG_REVISION}" == "${CUROBO_REVISION}" ]] \
  || fail "CuRobo tag ${CUROBO_TAG} resolves to ${CUROBO_TAG_REVISION}, expected ${CUROBO_REVISION}"
git -C "${CUROBO_SOURCE}" sparse-checkout init --no-cone
git -C "${CUROBO_SOURCE}" sparse-checkout set \
  --no-cone \
  '/*' \
  '!/src/curobo/content/assets/'
retry_git -C "${CUROBO_SOURCE}" checkout --detach "${CUROBO_REVISION}" \
  || fail "failed to materialize sparse CuRobo source"
[[ -z "$(git -C "${CUROBO_SOURCE}" status --porcelain)" ]] \
  || fail "CuRobo source checkout is dirty: ${CUROBO_SOURCE}"
[[ -d "${CUROBO_SOURCE}/src/curobo/content/configs" ]] \
  || fail "sparse CuRobo runtime configs are absent"
[[ -f "${CUROBO_SOURCE}/src/curobo/curobolib/cpp/kinematics_fused_kernel.cu" ]] \
  || fail "sparse CuRobo CUDA sources are absent"
[[ ! -e "${CUROBO_SOURCE}/src/curobo/content/assets" ]] \
  || fail "CuRobo example assets unexpectedly entered the minimal checkout"

# Build the exact official v0.7.8 source against WARM's inherited Torch 2.7.1.
# Constraints prevent dependency resolution from changing the checkpoint
# runtime.  MAX_JOBS keeps the one-time CUDA build bounded on CCI.
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
CUROBO_READY=false
if "${PYTHON_BIN}" - \
  "${WARM_RMBENCH_EVAL_ENV_DIR}/warm_rmbench_runtime.json" \
  "${CUROBO_REVISION}" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
expected_revision = sys.argv[2]
if not manifest.is_file():
    raise SystemExit(1)
value = json.loads(manifest.read_text(encoding="utf-8"))
if value.get("curobo_revision") != expected_revision:
    raise SystemExit(1)
if importlib.metadata.version("nvidia_curobo") != "0.7.8":
    raise SystemExit(1)
from curobo.types.math import Pose  # noqa: F401
from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
PY
then
  CUROBO_READY=true
fi
if [[ "${CUROBO_READY}" != "true" ]]; then
  "${PIP[@]}" install --force-reinstall --no-deps --no-build-isolation \
    -c "${CONSTRAINTS}" \
    "${CUROBO_SOURCE}"
fi

"${PYTHON_BIN}" - \
  "${WARM_RMBENCH_EVAL_ENV_DIR}/warm_rmbench_runtime.json" \
  "${RMBENCH_CODE_REVISION}" "${CUROBO_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
value = {
    "schema": "warm.rmbench-eval-runtime",
    "version": 2,
    "rmbench_revision": sys.argv[2],
    "curobo_revision": sys.argv[3],
    "construction": "venv-system-site-packages-over-warm",
    "dependency_profile": "rgb-only-minimal-v2",
    "open3d_provider": "warm-rgb-only-import-guard",
    "python_executable": str(Path(sys.executable).resolve()),
}
output.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"runtime_manifest={output}")
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/check_warm_rmbench_eval_runtime.py" \
  --rmbench-root "${RMBENCH_ROOT}" \
  --task "${WARM_RMBENCH_TASK:-blocks_ranking_try}" \
  --runtime-manifest \
  "${WARM_RMBENCH_EVAL_ENV_DIR}/warm_rmbench_runtime.json" \
  --apply-patches

printf '%s\n' \
  "RMBENCH_EVAL_ENV_READY" \
  "python=${PYTHON_BIN}" \
  "wheelhouse=${WARM_RMBENCH_WHEELHOUSE}" \
  "pip_cache=${PIP_CACHE_DIR}" \
  "runtime_manifest=${WARM_RMBENCH_EVAL_ENV_DIR}/warm_rmbench_runtime.json"
