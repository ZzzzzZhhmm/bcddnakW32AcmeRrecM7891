#!/usr/bin/env bash

# Shared fail-closed guards for formal WARM GPU-server jobs.  This file is
# sourced by launchers; it intentionally performs no work at import time.

warm_die() {
  printf 'error: %s\n' "$*" >&2
  return 2
}

warm_require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      warm_die "${name} must be set" || return
    fi
  done
}

warm_require_file_or_directory() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      warm_die "required immutable input not found: ${path}" || return
    fi
  done
}

warm_require_sha40() {
  local value="$1"
  local label="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{40}$ ]]; then
    warm_die "${label} must be a lowercase 40-character commit" || return
  fi
}

warm_register_safe_directory() {
  local directory canonical existing
  directory="$1"
  if [[ ! -d "${directory}" ]]; then
    warm_die "Git checkout directory does not exist: ${directory}" || return
  fi
  canonical="$(cd -- "${directory}" && pwd -P)" || {
    warm_die "failed to resolve Git checkout directory: ${directory}"
    return
  }
  if [[ ! -e "${canonical}/.git" ]]; then
    warm_die "Git metadata is missing from checkout: ${canonical}" || return
  fi

  # ACP containers often run as root while the persistent AFS checkout belongs
  # to another uid.  Register only the exact, caller-derived checkout path; do
  # not use the unsafe '*' wildcard and do not fetch or mutate repository data.
  existing="$(git config --global --get-all safe.directory 2>/dev/null || true)"
  if ! printf '%s\n' "${existing}" | grep -Fqx -- "${canonical}"; then
    git config --global --add safe.directory "${canonical}" || {
      warm_die "failed to register exact Git safe.directory: ${canonical}"
      return
    }
  fi
}

warm_require_private_checkout() {
  local root
  root="$(pwd -P)"
  export WARM_REPOSITORY_ROOT="${root}"
  if [[ -z "${WARM_CODE_REVISION:-}" ]]; then
    export WARM_CODE_REVISION="0000000000000000000000000000000000000000"
  fi
}

warm_require_read_only_external_checkout() {
  local checkout="$1"
  if [[ ! -d "${checkout}" ]]; then
    warm_die "external checkout is missing: ${checkout}" || return
  fi
}

warm_configure_offline_logging() {
  export WANDB_MODE="${WANDB_MODE:-offline}"
  if [[ "${WANDB_MODE}" != "offline" && "${WANDB_MODE}" != "disabled" ]]; then
    if [[ "${WARM_ALLOW_NETWORK_LOGGING:-0}" != "1" ]]; then
      warm_die "network experiment logging requires WARM_ALLOW_NETWORK_LOGGING=1" || return
    fi
  fi
  # Formal runs consume local, revision-pinned snapshots.  Disallow accidental
  # model downloads even when the server has Internet access.
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
}

# Configure compiler/autotune caches on node-local storage before importing
# DeepSpeed.  DeepSpeed 0.18.5 probes TRITON_CACHE_DIR with `df` during import
# and does not first create the directory.  A fresh ACP container therefore
# exits before the training entrypoint is reached when the default
# ~/.triton/autotune path is absent.  Keeping compiler caches under /tmp also
# avoids Triton/Inductor lock contention and slow finalization on AFS/NFS.
warm_configure_job_local_caches() {
  local root="${1:-}"
  local directory probe available_kib fs_type
  if [[ -z "${root}" ]]; then
    warm_die "job-local cache root must be provided" || return
  fi
  case "${root}" in
    /*) ;;
    *) warm_die "job-local cache root must be absolute: ${root}" || return ;;
  esac

  # Formal ACP jobs must not put compiler caches on the persistent AFS mount.
  # The caller may use another node-local absolute path when /tmp is small.
  case "${root}" in
    /mnt/afs|/mnt/afs/*)
      warm_die "job-local compiler cache cannot be placed on AFS: ${root}" || return
      ;;
  esac

  umask 077
  for directory in \
    "${root}" \
    "${root}/triton/autotune" \
    "${root}/torchinductor" \
    "${root}/cuda" \
    "${root}/xdg" \
    "${root}/hf_datasets"; do
    mkdir -p -- "${directory}" || {
      warm_die "cannot create job-local cache directory: ${directory}"
      return
    }
    if [[ ! -d "${directory}" || ! -w "${directory}" ]]; then
      warm_die "job-local cache directory is not writable: ${directory}" || return
    fi
  done

  probe="${root}/.warm-write-probe.$$"
  if ! printf 'warm-cache-probe\n' > "${probe}"; then
    warm_die "cannot write job-local cache probe: ${root}" || return
  fi
  rm -f -- "${probe}"

  available_kib="$(df -Pk -- "${root}" | awk 'NR == 2 {print $4}')" || {
    warm_die "cannot inspect free space for job-local cache: ${root}"
    return
  }
  if [[ ! "${available_kib}" =~ ^[0-9]+$ ]]; then
    warm_die "invalid free-space result for job-local cache: ${available_kib}" || return
  fi
  if (( available_kib < 1048576 )); then
    warm_die "job-local cache has less than 1 GiB free: ${root}" || return
  fi
  fs_type="$(df -PT -- "${root}" | awk 'NR == 2 {print $2}')" || {
    warm_die "cannot inspect filesystem type for job-local cache: ${root}"
    return
  }
  case "${fs_type}" in
    nfs|nfs4|fuse.s3fs|fuse.*afs*)
      warm_die "job-local cache resolved to a network filesystem (${fs_type}): ${root}" || return
      ;;
  esac

  export WARM_JOB_LOCAL_CACHE_ROOT="${root}"
  export TRITON_CACHE_DIR="${root}/triton/autotune"
  export TORCHINDUCTOR_CACHE_DIR="${root}/torchinductor"
  export CUDA_CACHE_PATH="${root}/cuda"
  export XDG_CACHE_HOME="${root}/xdg"
  export HF_DATASETS_CACHE="${root}/hf_datasets"

  printf 'job_local_cache=%s fs_type=%s available_gib=%s triton_cache=%s\n' \
    "${root}" \
    "${fs_type}" \
    "$((available_kib / 1048576))" \
    "${TRITON_CACHE_DIR}"
}

warm_refuse_existing_output() {
  local path="$1"
  if [[ -e "${path}" || -L "${path}" ]]; then
    warm_die "immutable output already exists: ${path}" || return
  fi
}

# Formal GPU-server scripts must not rely on the container default ``python``.
# ACP containers often expose a bare miniconda interpreter without WARM deps.
warm_activate_conda_env() {
  local env_dir="${CONDA_ENV_DIR:-/mnt/afs/task3_2/L202500276_lwz/envs/warm}"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    warm_die "persistent WARM Python environment is incomplete: ${env_dir}" || return
  fi
  export CONDA_ENV_DIR="${env_dir}"
  export PATH="${env_dir}/bin:${PATH}"
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONUNBUFFERED=1
}
