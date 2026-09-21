# Shared GPU count for Piper ACP launchers. Source from scripts/real/*.sh.
# Controls: NUM_GPUS, CUDA_VISIBLE_DEVICES. If only one is set, the other is filled.

resolve_piper_gpus() {
  local i devices
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${NUM_GPUS:-}" ]]; then
    IFS=',' read -r -a _piper_devs <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS="${#_piper_devs[@]}"
  fi
  NUM_GPUS="${NUM_GPUS:-1}"
  if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_GPUS must be a positive integer, got ${NUM_GPUS}" >&2
    return 2
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    devices="0"
    for ((i = 1; i < NUM_GPUS; i++)); do
      devices+=",${i}"
    done
    CUDA_VISIBLE_DEVICES="${devices}"
  else
    IFS=',' read -r -a _piper_devs <<< "${CUDA_VISIBLE_DEVICES}"
    if (( ${#_piper_devs[@]} != NUM_GPUS )); then
      echo "ERROR: NUM_GPUS=${NUM_GPUS} but CUDA_VISIBLE_DEVICES has ${#_piper_devs[@]} entries (${CUDA_VISIBLE_DEVICES})" >&2
      return 2
    fi
  fi
  export NUM_GPUS
  export CUDA_VISIBLE_DEVICES
  # Single-node k8s usually has no IB. Reuse the official LIBERO ACP defaults
  # so DeepSpeed/NCCL does not probe dead interfaces.
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo,eth0,bond0}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$((29500 + RANDOM % 1000))"
  fi
  export MASTER_PORT
}

piper_epoch_steps() {
  local windows="${1}"
  local gpus="${NUM_GPUS:-1}"
  echo $(( (windows + gpus - 1) / gpus ))
}
