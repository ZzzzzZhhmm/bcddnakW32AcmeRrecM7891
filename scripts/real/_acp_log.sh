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
}

piper_acp_begin_logs() {
  local run_log="$1"
  : "${PROJECT_DIR:?PROJECT_DIR is required}"
  : "${RUN_ID:?RUN_ID is required}"
  ACP_LOG_DIR="${ACP_LOG_DIR:-${PROJECT_DIR}/tmp/acp_logs}"
  mkdir -p "${ACP_LOG_DIR}" "$(dirname "${run_log}")"
  LOG="${run_log}"
  ACP_LOG="${ACP_LOG:-${ACP_LOG_DIR}/logs-acp-${RUN_ID}.log}"
  ACP_LOG_GZ="${ACP_LOG_DIR}/logs-acp-${RUN_ID}.txt.gz"
  : > "${LOG}"
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
  trap - EXIT
  echo "ACP_EXIT=${rc} ACP_LOG=${ACP_LOG:-} RUN_CONSOLE=${LOG:-}"
  if [[ -n "${ACP_LOG:-}" && -f "${ACP_LOG}" && -n "${ACP_LOG_DIR:-}" ]]; then
    gzip -c "${ACP_LOG}" > "${ACP_LOG_DIR}/logs-acp-${RUN_ID}.txt.gz" || true
    echo "ACP_LOG_GZ=${ACP_LOG_DIR}/logs-acp-${RUN_ID}.txt.gz"
  fi
}
