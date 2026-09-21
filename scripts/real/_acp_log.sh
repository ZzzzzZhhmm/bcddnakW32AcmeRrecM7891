# Shared ACP logging and launch preflight for scripts/real/*.sh.
# After piper_acp_begin_logs, stdout/stderr go to the run console.log and
# ${PROJECT_DIR}/tmp/acp_logs/logs-acp-${RUN_ID}.log. EXIT gzips the ACP copy.

piper_resolve_conda_bins() {
  if [[ ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
    echo "ERROR: python is not executable: ${CONDA_ENV_DIR}/bin/python"
    return 2
  fi
  if [[ ! -x "${CONDA_ENV_DIR}/bin/accelerate" ]]; then
    echo "ERROR: accelerate is not executable: ${CONDA_ENV_DIR}/bin/accelerate"
    echo "ERROR: do not fall back to PATH; conda env is required"
    return 2
  fi
  export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
  PYTHON="${CONDA_ENV_DIR}/bin/python"
  ACCELERATE="${CONDA_ENV_DIR}/bin/accelerate"
  export PYTHON
  export ACCELERATE
}

piper_ensure_master_port() {
  local chosen
  chosen="$(
    "${PYTHON}" - "${MASTER_PORT:-0}" <<'PY'
import socket
import sys

requested = int(sys.argv[1])

def available(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True

if requested > 0 and available(requested):
    print(requested)
    raise SystemExit(0)
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
  )"
  if [[ "${MASTER_PORT:-}" != "${chosen}" && -n "${MASTER_PORT:-}" ]]; then
    echo "WARNING: MASTER_PORT=${MASTER_PORT} is busy; switching to ${chosen}"
  fi
  MASTER_PORT="${chosen}"
  export MASTER_PORT
}

piper_gpu_preflight() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; Stage B/WARM ACP requires GPUs"
    return 2
  fi
  echo "=== nvidia-smi -L ==="
  nvidia-smi -L
  echo "NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  piper_assert_gpu_count
}

piper_configure_job_local_caches() {
  local root="${TMPDIR:-/tmp}/warm-piper-${USER:-root}-${RUN_ID:-$$}"
  mkdir -p \
    "${root}/triton/autotune" \
    "${root}/torchinductor" \
    "${root}/cuda" \
    "${root}/xdg"
  export TRITON_CACHE_DIR="${root}/triton/autotune"
  export TORCHINDUCTOR_CACHE_DIR="${root}/torchinductor"
  export CUDA_CACHE_PATH="${root}/cuda"
  export XDG_CACHE_HOME="${root}/xdg"
}

piper_acp_begin_logs() {
  local run_log="$1"
  : "${PROJECT_DIR:?PROJECT_DIR is required}"
  : "${RUN_ID:?RUN_ID is required}"
  piper_configure_job_local_caches
  ACP_LOG_DIR="${ACP_LOG_DIR:-${PROJECT_DIR}/tmp/acp_logs}"
  mkdir -p "${ACP_LOG_DIR}" "$(dirname "${run_log}")"
  LOG="${run_log}"
  ACP_LOG="${ACP_LOG:-${ACP_LOG_DIR}/logs-acp-${RUN_ID}.log}"
  ACP_LOG_GZ="${ACP_LOG_DIR}/logs-acp-${RUN_ID}.txt.gz"
  if [[ "${ACP_APPEND_CONSOLE:-}" == "1" && -e "${LOG}" ]]; then
    printf '\n---- ACP resume %s ----\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG}"
  else
    : > "${LOG}"
  fi
  : > "${ACP_LOG}"
  export PYTHONUNBUFFERED=1
  export PYTHONFAULTHANDLER=1
  if command -v stdbuf >/dev/null 2>&1; then
    exec > >(stdbuf -oL -eL tee -a "${LOG}" "${ACP_LOG}") 2>&1
  else
    exec > >(tee -a "${LOG}" "${ACP_LOG}") 2>&1
  fi
  echo "ACP_LOG=${ACP_LOG}"
  echo "RUN_CONSOLE=${LOG}"
}

piper_acp_finish() {
  local rc="${1:-0}"
  trap - EXIT INT TERM
  echo "ACP_EXIT=${rc} ACP_LOG=${ACP_LOG:-} RUN_CONSOLE=${LOG:-}"
  if [[ "${rc}" -eq 0 && -n "${ACP_REQUIRE_TRAINING_COMPLETE:-}" && -n "${WARM_DIR:-}" && ! -f "${WARM_DIR}/training_complete.json" ]]; then
    echo "ERROR: ACP_EXIT was 0 but ${WARM_DIR}/training_complete.json is missing"
    rc=1
    echo "ACP_EXIT=${rc} ACP_LOG=${ACP_LOG:-} RUN_CONSOLE=${LOG:-}"
  fi
  if [[ -n "${ACP_LOG:-}" && -f "${ACP_LOG}" && -n "${ACP_LOG_DIR:-}" ]]; then
    gzip -c "${ACP_LOG}" > "${ACP_LOG_DIR}/logs-acp-${RUN_ID}.txt.gz" || true
    echo "ACP_LOG_GZ=${ACP_LOG_DIR}/logs-acp-${RUN_ID}.txt.gz"
  fi
}

piper_acp_install_traps() {
  trap 'ACP_STATUS=143; echo "ERROR: ACP received SIGTERM"; exit 143' TERM
  trap 'ACP_STATUS=130; echo "ERROR: ACP received SIGINT"; exit 130' INT
  trap 'rc=${ACP_STATUS-$?}; piper_acp_finish "${rc}"; exit "${rc}"' EXIT
}

piper_latest_training_state() {
  local output_dir="$1"
  local state_root="${output_dir}/checkpoints/state"
  local latest=""
  local latest_step=-1
  local child step
  [[ -d "${state_root}" ]] || return 1
  for child in "${state_root}"/step_*; do
    [[ -d "${child}" ]] || continue
    step="${child##*/step_}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( 10#${step} > latest_step )); then
      latest_step=$((10#${step}))
      latest="${child}"
    fi
  done
  [[ -n "${latest}" ]] || return 1
  printf '%s\n' "${latest}"
}

piper_find_latest_incomplete_warm_dir() {
  local parent="$1"
  local best=""
  local d
  [[ -d "${parent}" ]] || return 1
  for d in "${parent}"/run_*; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/training_complete.json" ]] && continue
    if piper_latest_training_state "${d}" >/dev/null; then
      best="${d}"
    fi
  done
  [[ -n "${best}" ]] || return 1
  printf '%s\n' "${best}"
}

piper_list_incomplete_warm_runs() {
  local parent="$1"
  local d
  [[ -d "${parent}" ]] || return 0
  for d in "${parent}"/run_*; do
    [[ -d "${d}" ]] || continue
    if [[ ! -f "${d}/training_complete.json" ]]; then
      echo "incomplete WARM run: ${d}"
    fi
  done
}

piper_acp_resolve_resume() {
  RESUME_ARGS=()
  if [[ -n "${RESUME:-}" ]]; then
    if [[ ! -d "${RESUME}" ]]; then
      echo "ERROR: RESUME state dir missing: ${RESUME}"
      return 2
    fi
    RESUME_ARGS=(--resume "${RESUME}")
    ACP_APPEND_CONSOLE=1
    return 0
  fi
  if [[ -n "${WARM_DIR:-}" && -d "${WARM_DIR}" && ! -f "${WARM_DIR}/training_complete.json" ]]; then
    local latest
    if latest="$(piper_latest_training_state "${WARM_DIR}")"; then
      RESUME="${latest}"
      RESUME_ARGS=(--resume "${RESUME}")
      ACP_APPEND_CONSOLE=1
    fi
  fi
}

piper_acp_require_training_complete() {
  local output_dir="$1"
  local launch_rc="${2:-1}"
  local marker="${output_dir}/training_complete.json"
  if [[ "${launch_rc}" -ne 0 ]]; then
    echo "ERROR: accelerate launch exited ${launch_rc} before training_complete.json"
    return "${launch_rc}"
  fi
  if [[ ! -f "${marker}" ]]; then
    echo "ERROR: accelerate exited 0 without ${marker}"
    echo "ERROR: Stage B is incomplete; do not treat ACP_EXIT=0 as success"
    return 1
  fi
  echo "training_complete=${marker}"
  return 0
}
