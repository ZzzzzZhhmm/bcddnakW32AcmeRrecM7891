#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

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
CONSTRAINTS="${PROJECT_DIR}/scripts/constraints/warm_rmbench_eval.constraints"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

[[ -x "${WARM_BASE_ENV_DIR}/bin/python" ]] \
  || fail "base WARM Python is absent: ${WARM_BASE_ENV_DIR}/bin/python"
[[ -f "${PROJECT_DIR}/scripts/check_warm_rmbench_eval_runtime.py" ]] \
  || fail "runtime checker is absent from WARM checkout"
[[ -f "${CONSTRAINTS}" ]] \
  || fail "runtime constraints are absent: ${CONSTRAINTS}"
[[ -e "${RMBENCH_ROOT}/.git" ]] \
  || fail "pinned RMBench checkout is absent: ${RMBENCH_ROOT}"
command -v git >/dev/null 2>&1 || fail "git is required"
command -v flock >/dev/null 2>&1 || fail "flock is required"
git config --global --add safe.directory "${RMBENCH_ROOT}" 2>/dev/null || true

mkdir -p "$(dirname "${WARM_RMBENCH_EVAL_ENV_DIR}")" "${WARM_EXTERNAL_ROOT}"
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
"${PIP[@]}" install --upgrade "pip<26" "setuptools<81" wheel ninja

# This is intentionally not RMBench/script/requirements.txt: its torch==2.4.1
# pin is incompatible with the checkpoint-attested WARM runtime.
"${PIP[@]}" install -c "${CONSTRAINTS}" \
  "scipy==1.10.1" \
  "transforms3d==0.4.2" \
  "sapien==3.0.0b1" \
  "mplib==0.2.1" \
  "gymnasium==0.29.1" \
  "trimesh==4.4.3" \
  "open3d==0.18.0" \
  "pyglet==1.5.31" \
  "toppra==0.6.3" \
  "warp-lang==1.11.1" \
  "scikit-image==0.22.0" \
  "yourdfpy==0.0.60" \
  "numpy-quaternion==2024.0.13" \
  "pybind11==2.13.6" \
  "networkx==3.4.2" \
  "pyyaml==6.0.2" \
  "setuptools_scm==8.2.0" \
  "tqdm==4.66.5" \
  "importlib_resources==6.5.2"

# Ensure patched binary packages live in the evaluation overlay rather than
# being inherited from (and accidentally modifying) the base WARM environment.
"${PIP[@]}" install --force-reinstall --no-deps -c "${CONSTRAINTS}" \
  "sapien==3.0.0b1" \
  "mplib==0.2.1"

if [[ ! -e "${CUROBO_SOURCE}/.git" ]]; then
  [[ ! -e "${CUROBO_SOURCE}" ]] \
    || fail "non-Git path already occupies CUROBO_SOURCE: ${CUROBO_SOURCE}"
  git clone --no-checkout https://github.com/NVlabs/curobo.git \
    "${CUROBO_SOURCE}"
fi
git config --global --add safe.directory "${CUROBO_SOURCE}" 2>/dev/null || true
git -C "${CUROBO_SOURCE}" cat-file -e "${CUROBO_REVISION}^{commit}" 2>/dev/null \
  || git -C "${CUROBO_SOURCE}" fetch --depth 1 origin "${CUROBO_REVISION}"
git -C "${CUROBO_SOURCE}" checkout --detach "${CUROBO_REVISION}"
[[ -z "$(git -C "${CUROBO_SOURCE}" status --porcelain)" ]] \
  || fail "CuRobo source checkout is dirty: ${CUROBO_SOURCE}"

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
    "version": 1,
    "rmbench_revision": sys.argv[2],
    "curobo_revision": sys.argv[3],
    "construction": "venv-system-site-packages-over-warm",
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
  "runtime_manifest=${WARM_RMBENCH_EVAL_ENV_DIR}/warm_rmbench_runtime.json"
